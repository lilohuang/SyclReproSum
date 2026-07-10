# SyclReproSum

Bit-reproducible floating-point summation for SYCL - a single header
implementing the **Ahrens-Demmel-Nguyen (ADN)** *binned floating-point*
algorithm, with primitives adapted from the
[ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/) reference
implementation.

- **Bit-reproducible** - the sum is exactly the same bits on every run,
  for any input order, work-group size, or thread scheduling.
- **Cross-device** - CPU, NVIDIA GPU, and Intel GPU all produce
  *identical* results from the same input, across CUDA, Level-Zero,
  and OpenCL backends.
- **Single pass** - no max-scan or second sweep; the data is read once,
  reaching up to ~75% of the plain `sycl::reduction` bandwidth ceiling
  on GPU.
- **Robust** - Inf/NaN propagate deterministically using a fixed canonical
  quiet NaN, subnormals are handled, and accuracy is bounded (~2^-80 for
  double at the default K=3) even under catastrophic cancellation.
- **Easy to adopt** - one header, one function (`adn::sum`), `float`
  and `double`, C++17.

## Why?

Parallel floating-point reduction is **non-deterministic** - different
thread scheduling orders produce different rounding errors, and different
hardware (CPU vs GPU, NVIDIA vs Intel) produces different answers for the
same data. The Ahrens-Demmel-Nguyen binned format solves this by
depositing every value into K accumulators aligned to a fixed grid of
exponent bins, making the result **independent of execution order and of
the device it ran on** - whether that device is a multi-core CPU or a
discrete GPU.

```
Naive parallel sum of [1e16, 1.0, 1.0, ..., -1e16]  ->  unpredictable; varies
                                                        run-to-run and device-to-device
SyclReproSum of the same data                        ->  always the same bits, on
                                                        any CPU or GPU from any vendor
```

## Quick Start

A complete, runnable program (see [example.cpp](example.cpp), built by
`make example`):

```cpp
#include <sycl/sycl.hpp>
#include <vector>
#include "repro_sum.hpp"

int main() {
   // Works with any device: gpu_selector_v, cpu_selector_v, or default
   sycl::queue q{sycl::gpu_selector_v};

   // 1e16 +/- pairs cancel exactly; the true sum is the 10000 ones.
   std::vector<double> data;
   for (int i = 0; i < 5; ++i) {
      data.push_back(1e16);
      data.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      data.push_back(1.0);
   }

   // Single API: accepts device, shared, or host pointers
   double result = adn::sum(q, data.data(), data.size());
   return result == 10000.0 ? 0 : 1;
}
```

More API variants:

```cpp
// Device pointer (zero-copy, T deduced):
double *d_ptr = sycl::malloc_device<double>(N, q);
double result = adn::sum(q, d_ptr, N);

// Host pointer (auto-copied to device):
double result = adn::sum(q, host_ptr, N);

// Shared USM pointer (zero-copy):
float *s_ptr = sycl::malloc_shared<float>(N, q);
float result = adn::sum(q, s_ptr, N);

// More folds for higher accuracy:
double result = adn::sum<6>(q, ptr, N);

// Custom work-group size:
double result = adn::sum<3, 512>(q, ptr, N);
```

## Performance

Single pass over the data, 100M elements, average of 10 runs. Throughput
is measured in GElements/s (billions of input elements processed per
second, where 1 GElements/s = 10^9 elements/s), so `double` and `float`
can be compared without the element-size bias of GB/s:

| Device (backend) | `double` sum / ceiling* (GElements/s) | `float` sum / ceiling* (GElements/s) |
|---|---:|---:|
| NVIDIA RTX PRO 4500 (CUDA) | 42.4 / 78.4 (1.8x slower) | 96.7 / 131.9 (1.4x slower) |
| Intel Meteor Lake iGPU (OpenCL) | 1.7 / 6.0 (3.5x slower) | 3.7 / 11.6 (3.1x slower) |
| Intel Core Ultra 7 265 CPU (OpenCL) | 0.4 / 5.8 (14.5x slower) | 0.6 / 7.6 (12.7x slower) |

*Ceiling = plain (non-reproducible) `sycl::reduction` of the same data
type on the same device. The gap is the algorithm's inherent cost: each
deposit is a chain of K+1 dependent FP additions, which buys
bit-reproducibility. The slowdown factor is `ceiling / sum`.

> **Note:** The CPU backend uses the SYCL runtime's parallel thread pool
> (OpenCL CPU runtime), so it benefits from all available cores. The
> higher relative overhead on CPU vs GPU is expected: GPUs hide the
> deposit dependency chain with massive thread-level parallelism, while
> CPU threads have fewer opportunities to overlap dependent operations.

### Cross-device reproducibility

The sum is bit-identical **across CPUs, GPU vendors, and backends**, not
just across runs. Verified on this machine (single fat binary, device
code for all targets):

| | NVIDIA RTX PRO 4500 (CUDA) | Intel Meteor Lake iGPU (Level-Zero) | Intel Core Ultra 7 265 CPU (OpenCL) |
|---|---|---|---|
| `double`, 1M wide-range | `0x430FC878C605717F` | `0x430FC878C605717F` | `0x430FC878C605717F` |
| shuffled input order | same bits | same bits | same bits |

The test suite enforces this: `ADNSumCrossDevice` sums the same data on
every available device (CPU + all GPUs) - each seeing a *different* input
order - and requires bit-identical results. The parameterized correctness
suite runs every test on every device independently.

This makes results portable across heterogeneous clusters, between a
developer's laptop (CPU-only) and production GPU accelerators, and across
hardware upgrades - the answer never changes.

## How It Works

Every value is *deposited* into a K-fold **binned accumulator**: K primary
values aligned to a fixed, global grid of exponent bins (bin width W = 40
bits for double, 13 bits for float), plus K carry counters. Because every
deposit rounds on the same absolute grid, the result is independent of
summation order - no max-scan is needed, so the data is read exactly once:

1. **Deposit** - Each work-item strides over the input and adds each
   element to its private accumulator with a few plain FP additions
   (a sticky-bit trick makes the bin rounding order-independent).

2. **Renormalize** - After every 2^(mantissa_bits - W - 2) deposits, the
   drift of each primary value is shifted into its carry counter, so the
   accumulator never overflows.

3. **Merge** - Work-group accumulators are combined with an exact binned
   tree reduction; the per-group results are merged on the host and
   converted back to a single float/double.

```
Deposit of x (double, W=40, K=3), into an accumulator whose bins cover x:
   x's top 40 bits    -> primary value 0 (coarsest bin)
   next 40 bits       -> primary value 1 (one bin finer)
   remaining bits     -> primary value 2 (two bins finer)
```

Inf/NaN inputs propagate deterministically through the accumulator. Any NaN,
or a sum containing both +Inf and -Inf, produces a fixed positive quiet NaN
(`0x7fc00000` for float, `0x7ff8000000000000` for double), independent of
input payloads, order, reduction tree, and device conventions. Subnormals
deposit into the bottom bins; values below the format's absorption threshold
(~2^-1055 for double / ~2^-144 for float at K=3) round away deterministically,
per the documented ADN error bound ~N * 2^(-W*(K-1)) * max|x|.

## API Reference

```cpp
namespace adn {

/// Reproducible single-pass sum of N values.
/// Accepts device, shared, host USM, or plain host pointers.
/// Plain host pointers are automatically copied to device memory.
/// T is deduced from the pointer type (float or double).
/// @tparam K       Fold count (default 3)
/// @tparam WG_SIZE Work-group size (default 256, must be power of 2)
/// @tparam T       float or double (deduced)
template <int K = 3, int WG_SIZE = 256, typename T>
T sum(sycl::queue &q, const T *arr, size_t N);

}
```

### Pointer type handling

| Pointer source | Behavior |
|----------------|----------|
| `sycl::malloc_device` | Used directly by kernel (zero-copy) |
| `sycl::malloc_shared` | Used directly by kernel (zero-copy) |
| `sycl::malloc_host` | Used directly by kernel (zero-copy) |
| Plain host pointer (`new`, stack, `std::vector::data()`) | Auto-copied to device memory, freed after computation |

Detection is done at runtime via `sycl::get_pointer_type`.

### Template Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `K` | 3 | 2 to max fold | Fold count (bins held). More = higher accuracy, slower. |
| `WG_SIZE` | 256 | 2-1024 (power of 2) | Work-group size (threads per group on GPU, SIMD lanes on CPU). |
| `T` | (deduced) | `float` or `double` | Floating-point type. |

| Type | Mantissa bits | Bin width W | Max K | Accuracy at K=3 |
|------|--------------|-------------|-------|-----------------|
| `double` | 53 | 40 | 52 | ~2^-80 relative to max abs |
| `float` | 24 | 13 | 21 | ~2^-26 relative to max abs |

The bin width is fixed (ReproBLAS DBWIDTH/SBWIDTH); each additional fold
extends coverage by another W bits below the running maximum.
Invalid parameters trigger a `static_assert` at compile time.

### Compiler safety

The header rejects `-ffast-math`, `-ffinite-math-only`, `/fp:fast`, and
equivalent modes at compile time. These modes allow the compiler to discard
NaN, infinity, subnormal, signed-zero, and rounding-order semantics required
for reproducibility, and their individual assumptions cannot be reliably
undone inside a header.

With strict floating-point semantics enabled, compiler pragmas additionally
disable reassociation and FP contraction (FMA) in the accumulator logic.

## Building

### Prerequisites

- [Intel DPC++ compiler](https://github.com/intel/llvm) (clang++ with `-fsycl`, C++17 or later)
- For GPU: NVIDIA GPU + CUDA toolkit (for `nvptx64` target), and/or Intel GPU runtime (for `spir64`)
- For CPU: Intel OpenCL CPU runtime (`intel-oneapi-runtime-opencl`)
- CMake + g++ (for building Google Test)

### Build

```bash
# Clone with submodules
git clone --recursive https://github.com/lilohuang/SyclReproSum.git
cd SyclReproSum

# If you already cloned without --recursive:
git submodule update --init --recursive

# Build the usage example
make

# Build and run tests (gtest is built automatically on first run)
make test

# Build everything
make all
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make` / `make example` | Build the usage example (`repro_example`) |
| `make test` | Build and run Google Test suite |
| `make all` | Build both example and tests |
| `make run` | Build and run the example |
| `make clean` | Remove binaries and build artifacts |
| `make gtest` | Build Google Test from submodule (automatic) |

### Configuration

Override via environment or command line:

```bash
# Custom DPC++ location
make DPCPP_HOME=/opt/intel/oneapi/compiler/latest

# Target only NVIDIA GPU
make SYCL_TARGETS=nvptx64-nvidia-cuda
```

## Test Suite

The Google Test suite is value-parameterized over every device on the
system (CPUs + GPUs): each correctness case runs once per device from a
single fat binary (CUDA + SPIR-V), plus cross-device bit-identity tests
and throughput benchmarks. 437 tests on a system with 2 GPUs + 1 CPU.

| Category | Description |
|----------|-------------|
| Edge cases | Empty, single element, all zeros, negative zero, min/max values |
| Exact arithmetic | Integer sums, powers of two, geometric series, telescope sums |
| Cancellation stress | Multi-order cancellation, Kahan worst case, near-overflow cancellation |
| Large-scale | 1M to 20M elements, multi-work-group boundaries |
| Special values | Canonical NaN across payloads/devices, Inf propagation, subnormals, overflow to +Inf |
| Reproducibility | Multi-run bit-identity, shuffle order-independence, cross-WG_SIZE and cross-device (CPU vs CUDA vs Level-Zero) consistency |
| Configurations | K = 2..12, WG_SIZE = 64..1024, device-pointer and host-pointer APIs |
| Benchmarks | Throughput measurement on all devices (GPU + CPU) |

```bash
# Run all tests
make test

# Filter specific tests (e.g. float only)
LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib ./repro_test --gtest_filter="*Float*"

# Run only CPU tests
LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib ./repro_test --gtest_filter="CPUs/*"

# Run only benchmarks
LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib ./repro_test --gtest_filter="*Bench*"
```

## References

- J. Demmel, H.D. Nguyen. "Fast Reproducible Floating-Point Summation."
  *IEEE Symposium on Computer Arithmetic*, 2013.
- Ahrens, Demmel, Nguyen. "Algorithms for Efficient Reproducible Floating
  Point Summation." *ACM TOMS*, 2020.
- [ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/) - Reference
  implementation in C/MPI.

## License

BSD-3-Clause for original code in this repository (see [LICENSE](LICENSE)).

The binned accumulator primitives in `repro_sum.hpp` are adapted from
[ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/), Copyright (c) 2016,
University of California, used under its BSD-3-Clause-style Software
Development License - see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
Google Test (in `third_party/`) is under its own BSD-3-Clause license.
