# SyclReproSum

Bit-reproducible floating-point summation for SYCL - a single header
implementing the **Ahrens-Demmel-Nguyen (ADN)** *binned floating-point*
algorithm, with primitives adapted from the
[ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/) reference
implementation.

- **Bit-reproducible** - within the documented accumulator capacity, the sum
  is exactly the same bits on every run, for any input order, supported
  work-group size, or thread scheduling on a validated device.
- **Cross-device** - validated CPU, NVIDIA GPU, and Intel GPU devices produce
  *identical* results from the same input; this is tested here across CUDA,
  Level-Zero, and OpenCL backends.
- **Single pass** - no max-scan or second sweep; the accumulation kernel reads
  the device input once and approaches the plain `sycl::reduction` throughput
  baseline on the tested NVIDIA GPU.
- **Robust** - Inf/NaN propagate deterministically using a fixed canonical
  quiet NaN, subnormals are handled, and the dominant binned absolute-error
  term at the default K=3 is proportional to N * 2^-80 * max|x| for double.
- **Easy to adopt** - one header, one function (`adn::sum`), `float`
  and `double`, C++17.

## Why?

Parallel floating-point reduction is **non-deterministic** - different
thread scheduling orders produce different rounding errors, and different
hardware (CPU vs GPU, NVIDIA vs Intel) produces different answers for the
same data. Within its documented accumulator capacity, the
Ahrens-Demmel-Nguyen binned format solves this by depositing every value into
K accumulators aligned to a fixed grid of exponent bins, making the result
**independent of execution order and of the device it ran on** - whether that
device is a multi-core CPU or a discrete GPU.

```
Naive parallel sum  ->  varies by run and device
SyclReproSum        ->  same bits on every validated device
```

## Quick Start

A complete, runnable program (see [example.cpp](example.cpp), built by
`make example`):

```cpp
#include <sycl/sycl.hpp>
#include <vector>
#include "repro_sum.hpp"

int main() {
   // Use any validated fp64-capable SYCL device.
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
// Assume q, N, and initialized host_ptr are already available.

// Device pointer (no library-side copy, T deduced):
double *d_ptr = sycl::malloc_device<double>(N, q);
q.memcpy(d_ptr, host_ptr, N * sizeof(double)).wait();
double result = adn::sum(q, d_ptr, N);

// Host pointer (auto-copied to device):
double result = adn::sum(q, host_ptr, N);

// Shared USM pointer (populate before summing; no library-side copy):
float *s_ptr = sycl::malloc_shared<float>(N, q);
for (std::size_t i = 0; i < N; ++i) {
   s_ptr[i] = 1.0f;
}
float result = adn::sum(q, s_ptr, N);

// More folds for higher accuracy:
double result = adn::sum<6>(q, ptr, N);

// Custom work-group size:
double result = adn::sum<3, 512>(q, ptr, N);
```

## Performance

Single pass over the data, 100M elements. The values below are the mean
of three benchmark invocations, each measured as the average of 10 runs
after one warmup run. Throughput is measured in GElements/s (billions of
input elements processed per second, where 1 GElements/s = 10^9
elements/s), so `double` and `float` can be compared without the
element-size bias of GB/s:

| Device (backend) | `double` sum / baseline* (GElements/s) | `float` sum / baseline* (GElements/s) |
|---|---:|---:|
| NVIDIA RTX PRO 4500 Blackwell (CUDA) | 48.7 / 77.0 (1.6x slower) | 121.2 / 124.5 (1.0x slower) |
| Intel Graphics [0x7d67] (Level-Zero) | 1.7 / 6.6 (3.8x slower) | 4.1 / 12.5 (3.0x slower) |
| Intel Core Ultra 7 265 CPU (OpenCL) | 0.4 / 5.8 (13.6x slower) | 0.6 / 7.4 (12.3x slower) |

*Baseline = plain (non-reproducible) `sycl::reduction` of the same data
type on the same device. It is a comparison point, not a hard performance
ceiling. A binned deposit performs multiple dependent additions and
subtractions per fold; the observed gap also includes compiler, backend,
and runtime effects. The slowdown factor is `baseline / sum`.

> **Note:** The CPU backend uses the SYCL runtime's parallel thread pool
> (OpenCL CPU runtime), so it can use the available CPU cores. The
> higher relative overhead on CPU vs GPU is expected: GPUs hide the
> deposit dependency chain with massive thread-level parallelism, while
> CPU threads have fewer opportunities to overlap dependent operations.

### Cross-device reproducibility

The sum is bit-identical **across CPUs, GPU vendors, and backends**, not
just across runs. Verified on this machine (single fat binary, device
code for all targets):

| | NVIDIA RTX PRO 4500 Blackwell (CUDA) | Intel Graphics [0x7d67] (Level-Zero) | Intel Core Ultra 7 265 CPU (OpenCL) |
|---|---|---|---|
| `double`, 1M wide-range | `0x430FC878C605717F` | `0x430FC878C605717F` | `0x430FC878C605717F` |
| shuffled input order | same bits | same bits | same bits |

The test suite chooses one backend per distinct device name, preferring
Level-Zero, then CUDA, then other backends, with OpenCL as the fallback.
`ADNSumCrossDevice` sums the same data on every selected device - each seeing
a *different* input order - and requires bit-identical results. The
parameterized correctness suite runs every test on every selected device.

Within the documented capacity, this makes results portable across supported
devices that pass environment validation, including heterogeneous CPU and GPU
systems and hardware upgrades.

## How It Works

Every value is *deposited* into a K-fold **binned accumulator**: K primary
values aligned to a fixed, global grid of exponent bins (bin width W = 40
bits for double, 13 bits for float), plus K carry counters. Within the
accumulator capacity, every deposit rounds on the same absolute grid, so the
result is independent of summation order. No max-scan is needed, and the
accumulation kernel reads each device input element exactly once:

1. **Deposit** - Each work-item strides over the input and adds each
   element to its private accumulator with a few plain FP additions
   (a sticky-bit trick makes the bin rounding order-independent).

2. **Renormalize** - After every 2^(mantissa_bits - W - 2) deposits, the
   drift of each primary value is shifted into its carry counter, so the
   accumulator does not overflow within its documented capacity.

3. **Merge** - Work-group accumulators are combined with an exact binned
   tree reduction; a final device work-group merges the per-group results
   and converts the total back to a single float/double.

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
per the dominant binned error term ~N * 2^(-W*(K-1)) * max|x|. This is an
absolute error scale; cancellation can make error relative to the exact sum
much larger.

## API Reference

```cpp
namespace adn {

/// Conservative maximum input count for the binned accumulator.
template <typename T>
inline constexpr std::uint64_t max_reproducible_count = /* ... */;

/// Validate the queue's device floating-point environment.
template <typename T>
void validate_environment(sycl::queue &q);

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
| `sycl::malloc_device` | Used directly; no library-side copy |
| `sycl::malloc_shared` | Used directly; no library-side copy |
| `sycl::malloc_host` | Used directly; no library-side copy |
| Plain host pointer (`new`, stack, `std::vector::data()`) | Copied into a temporary device allocation; only the temporary is freed |

Detection is done at runtime via `sycl::get_pointer_type`. "No library-side
copy" does not guarantee a physical zero-copy implementation; the SYCL
runtime may migrate or otherwise manage USM storage.

### Accumulator capacity

The binned carry fields can represent a finite number of summands while
remaining exact. SyclReproSum uses the conservative capacities from ReproBLAS:

| Type | Maximum reproducible input count |
|------|---------------------------------:|
| `float` | 4,294,966,784 |
| `double` | 9,223,372,036,854,773,760 |

The limit is available to callers as
`adn::max_reproducible_count<float>` or
`adn::max_reproducible_count<double>`. `adn::sum` throws
`std::length_error` before inspecting the pointer or allocating memory when
`N` exceeds the corresponding limit. Splitting the input and adding the
scalar chunk results does not preserve the reproducibility guarantee.

### Template Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `K` | 3 | 2 to max fold | Fold count (bins held). More = higher accuracy, slower. |
| `WG_SIZE` | 256 | 2-1024 (power of 2) | Work-items per work-group; must also fit the target device limits. |
| `T` | (deduced) | `float` or `double` | Floating-point type. |

| Type | Mantissa bits | Bin width W | Max K | Dominant binned error term at K=3 |
|------|--------------|-------------|-------|-------------------------------------|
| `double` | 53 | 40 | 52 | ~N * 2^-80 * max abs input |
| `float` | 24 | 13 | 21 | ~N * 2^-26 * max abs input |

The bin width is fixed (ReproBLAS DBWIDTH/SBWIDTH); each additional fold
extends coverage by another W bits below the running maximum. The error term
above is absolute, not relative to the exact sum. Invalid template parameters
trigger a `static_assert`; a syntactically valid `WG_SIZE` can still exceed a
particular device's work-group or local-memory limits at runtime.

### Runtime environment validation

Every non-empty `adn::sum` call automatically validates the floating-point
environment before inspecting the input pointer or allocating the main work
buffer. All final merging and conversion runs on the device, so host rounding
and denormal modes do not affect the result. Validation is fail-closed and
covers:

- the IEEE binary32 and binary64 layouts used by the bit-level helpers;
- device fp64 support, including for float sums whose final conversion uses
  double arithmetic;
- SYCL round-to-nearest, denormal, and Inf/NaN capability declarations for
  every precision used;
- effective device arithmetic using runtime-loaded rounding, subnormal, and
  signed-zero probes.

A successful device arithmetic check for a root-enumerated device is shared by
all host threads and cached by backend, device, and floating-point type.
Dynamically created sub-devices are not in that fixed cache and are validated
on every use. Applications can optionally preflight a queue explicitly with
`adn::validate_environment<float>(q)` or
`adn::validate_environment<double>(q)`. Validation throws `std::runtime_error`
with the failed prerequisite when the guarantee cannot be established.

### Compiler safety

The header rejects `-ffast-math`, `-ffinite-math-only`, `/fp:fast`, and
equivalent modes at compile time. These modes allow the compiler to discard
NaN, infinity, subnormal, signed-zero, and rounding-order semantics required
for reproducibility, and their individual assumptions cannot be reliably
undone inside a header.

With strict floating-point semantics enabled, scoped compiler pragmas disable
reassociation, reciprocal transformations, and FP contraction (FMA) in the
accumulator logic. Individual options such as
`-fdenormal-fp-math=positive-zero` and `-fno-signed-zeros` are not consistently
exposed by a preprocessor macro, so runtime arithmetic probes verify their
effective device behavior and reject unsafe combinations.

## Building

### Prerequisites

- [Intel DPC++ compiler](https://github.com/intel/llvm)
  (`clang++` with `-fsycl`, C++17 or later)
- A SYCL device with fp64 support that passes runtime environment validation
  (required by both float and double sums)
- For GPU: NVIDIA GPU + CUDA toolkit (for `nvptx64` target), and/or Intel
  GPU runtime (for `spir64`)
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
| `make test-validation` | Run hostile host/device validation tests |
| `make all` | Build both example and tests |
| `make run` | Build and run the example |
| `make clean` | Remove binaries and build artifacts |
| `make gtest` | Build Google Test from submodule (automatic) |

### Configuration

Override via environment or command line:

```bash
# Custom DPC++ source/build workspace root. The Makefile appends
# /llvm/build/bin/clang++ and /llvm/build/lib.
make DPCPP_HOME=/path/to/sycl_workspace

# Target only NVIDIA GPU
make SYCL_TARGETS=nvptx64-nvidia-cuda
```

## Test Suite

The Google Test suite chooses one preferred backend per distinct device name
(Level-Zero, then CUDA, then other backends, with OpenCL as the fallback).
Each correctness case runs once per selected CPU or GPU from a single fat
binary (CUDA + SPIR-V), plus cross-device bit-identity tests and throughput
benchmarks. There are 148 correctness cases and 2 benchmarks per device, plus
7 cross-device cases and one version test: 458 tests on a system with 2 GPUs +
1 CPU. The `WG_SIZE=1024` cases skip on a device whose work-group or local
memory limits cannot support that configuration.

| Category | Description |
|----------|-------------|
| Edge cases | Empty, single element, all zeros, negative zero, min/max values |
| Exact arithmetic | Integer sums, powers of two, geometric series, telescope sums |
| Cancellation stress | Multi-order cancellation, Kahan worst case, near-overflow cancellation |
| Large-scale | 1M to 20M elements, multi-work-group boundaries |
| Special values | Canonical NaN across payloads/devices, Inf propagation, subnormals, overflow to +Inf |
| Reproducibility | Multi-run bit-identity, shuffle order-independence, cross-WG_SIZE and selected cross-device/backend consistency |
| Environment safety | Host rounding and FTZ/DAZ independence, shared validation across threads, unsafe device mode rejection |
| Configurations | K = 2, 3, 4, 5, 6, 8, 12; WG_SIZE = 64, 128, 256, 512, 1024; device- and host-pointer APIs |
| Benchmarks | Throughput measurement on all selected devices (GPU + CPU) |

```bash
# Run the normal suite, including benchmarks
make test

# Rebuild and test hostile host/device floating-point modes
make test-validation

# Run the normal suite and all validation variants
make test test-validation

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
