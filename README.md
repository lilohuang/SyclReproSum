# SyclReproSum

Bit-reproducible floating-point sums and cumulative sums for SYCL - a single
header implementing the **Ahrens-Demmel-Nguyen (ADN)** *binned
floating-point* algorithm, with primitives adapted from the
[ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/) reference
implementation.

- **Bit-reproducible** - within the documented accumulator capacity, each sum
  is exactly the same bits on every run, for any input order, supported
  work-group size, or thread scheduling on a validated device. Each cumulative
  sum has the same guarantee for the values in that prefix.
- **Cross-device** - validated CPU, NVIDIA GPU, AMD GPU, and Intel GPU devices
  produce *identical* results from the same input; this is tested here across
  CUDA, HIP, Level-Zero, and OpenCL backends.
- **Efficient algorithms** - `adn::sum` reads the device input once and
  approaches the plain `sycl::reduction` throughput baseline on the tested
  NVIDIA and AMD GPUs. `adn::cumsum` uses a hierarchical binned scan without
  a full accumulator per output element.
- **Robust** - Inf/NaN propagate deterministically using a fixed canonical
  quiet NaN, subnormals are handled, and the dominant binned absolute-error
  term at the default K=3 is proportional to N * 2^-80 * max|x| for double.
- **Easy to adopt** - one header, two functions (`adn::sum` and
  `adn::cumsum`), `float` and `double`, C++17.

## Why?

Parallel floating-point reduction is **non-deterministic** - different
thread scheduling orders produce different rounding errors, and different
hardware (CPU vs GPU, NVIDIA vs AMD vs Intel) produces different answers for
the same data. Within its documented accumulator capacity, the
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

   // Scalar reproducible sum: accepts device, shared, or host pointers.
   double result = adn::sum(q, data.data(), data.size());

   // Reproducible cumulative sums. Each output[i] sums data[0..i].
   std::vector<double> cumulative(data.size());
   adn::cumsum(q, data.data(), cumulative.data(), data.size());
   return result == 10000.0 && cumulative.back() == result ? 0 : 1;
}
```

More API variants:

```cpp
// Assume q, N, and initialized host_ptr are already available.

// Device pointer (no library-side copy, T deduced):
double *d_ptr = sycl::malloc_device<double>(N, q);
q.memcpy(d_ptr, host_ptr, N * sizeof(double)).wait();
double result = adn::sum(q, d_ptr, N);

// A producer event can be passed without a host-side wait:
sycl::event input_ready = q.memcpy(d_ptr, host_ptr, N * sizeof(double));
double dependent_result = adn::sum(q, d_ptr, N, {input_ready});

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

// Cumulative sums into a separate output array:
std::vector<double> output(N);
adn::cumsum(q, host_ptr, output.data(), N);

// Exact in-place cumulative sum is supported:
adn::cumsum(q, s_ptr, s_ptr, N);
```

## Performance

The throughput benchmarks process 100M elements with the defaults `K=3` and
`WG_SIZE=256`. Each algorithm gets one warmup call followed by 10 timed
synchronous calls. Each invocation converts their mean duration to throughput;
the tables average the throughputs from three separate invocations. Slowdown
factors use the unrounded averages. Timing uses device-USM arrays and excludes
the initial host-to-device input copy. Throughput is measured in GElements/s:
billions of input elements processed per second, where 1 GElements/s = 10^9
elements/s. This lets `double` and `float` be compared without the element-size
bias of GB/s.


| Measured devices | DPC++ / Clang | Compiler revision | oneDPL |
|---|---|---|---|
| NVIDIA GB10 | 7.1.0 / 23 | `ca38ac56d6e3` | 2022.12.0 |
| Intel Arc Pro B70 | 7.1.0 / 23 | `95ec82a14309` | 2022.13.0 |
| RTX PRO 4500, W7500, Core Ultra 7 265 CPU and iGPU | 7.2.0 / 24 | `f3aadb6ba275` | 2022.13.0 |

Once a default cumulative sum would require at least 32K of the original tiles
(about 67M elements), every device uses 16 contiguous elements per work-item
to reduce the number of tile totals and amortize the work-group scans. This
rule depends only on `N`, `K`, and `WG_SIZE`, never on a backend, vendor, model,
or device property. Smaller inputs and custom configurations retain the
previous path. The transformation merges binned accumulators without
reassociating scalar additions, so the reproducibility contract is unchanged.

### Small-N latency

The small-N baselines match the 100M throughput benchmarks: a plain
non-reproducible `sycl::reduction` for `sum`, and oneDPL `inclusive_scan` for
`cumsum`. For each N, both algorithms get three warmup calls followed by 201
synchronous calls in alternating baseline/ADN order. A test reports the
median call latency, and the tables report the median of five separate
test-binary invocations. Inputs and cumulative-sum outputs use device USM;
the scalar baseline uses the same shared-USM result as the 100M benchmark.
Allocation, initial data transfer, and warmup are outside the timed region.
Internal ADN allocation and synchronization remain included.

Each cell is `ADN / baseline` in microseconds, followed by ADN's relative
performance. The defaults `K=3` and `WG_SIZE=256` are used throughout. A
version-to-version A/B is useful as a regression check, but is not an
algorithm baseline and is therefore not reported here.

#### Scalar-sum latency on CUDA

| Device / type | N=1 | N=64 | N=1,024 | N=2,048 | N=2,049 |
|---|---:|---:|---:|---:|---:|
| NVIDIA GB10 (CUDA), `double` | 17.6 / 25.0 (1.42x faster) | 24.6 / 23.2 (1.06x slower) | 28.7 / 32.9 (1.15x faster) | 28.8 / 42.2 (1.46x faster) | 29.1 / 41.9 (1.44x faster) |
| NVIDIA GB10 (CUDA), `float` | 16.9 / 23.2 (1.37x faster) | 24.8 / 23.5 (1.05x slower) | 28.6 / 24.3 (1.18x slower) | 28.6 / 26.3 (1.09x slower) | 28.8 / 26.0 (1.11x slower) |
| NVIDIA RTX PRO 4500 Blackwell (CUDA), `double` | 14.5 / 18.8 (1.30x faster) | 22.2 / 18.7 (1.18x slower) | 25.2 / 26.6 (1.06x faster) | 25.1 / 34.9 (1.39x faster) | 25.1 / 34.8 (1.39x faster) |
| NVIDIA RTX PRO 4500 Blackwell (CUDA), `float` | 14.1 / 18.6 (1.31x faster) | 15.5 / 18.3 (1.19x faster) | 24.6 / 19.4 (1.27x slower) | 24.6 / 20.6 (1.19x slower) | 24.6 / 20.6 (1.19x slower) |

The CUDA scalar path can beat the generic reduction for several `double`
sizes because it keeps partials in device USM and copies back only the final
scalar. The advantage is modest and is not uniform across types or sizes.

#### Cumulative-sum latency on CUDA and Level Zero

| Device / type | N=1 | N=64 | N=1,024 | N=2,048 | N=2,049 |
|---|---:|---:|---:|---:|---:|
| NVIDIA GB10 (CUDA), `double` | 24.1 / 10.9 (2.21x slower) | 40.8 / 10.6 (3.85x slower) | 49.1 / 19.8 (2.48x slower) | 60.2 / 29.3 (2.05x slower) | 92.7 / 25.2 (3.67x slower) |
| NVIDIA GB10 (CUDA), `float` | 21.7 / 10.4 (2.10x slower) | 35.8 / 10.0 (3.56x slower) | 38.6 / 11.1 (3.49x slower) | 50.0 / 12.4 (4.04x slower) | 72.6 / 26.3 (2.77x slower) |
| NVIDIA RTX PRO 4500 Blackwell (CUDA), `double` | 21.3 / 9.2 (2.32x slower) | 36.1 / 8.8 (4.08x slower) | 44.5 / 17.2 (2.59x slower) | 54.9 / 26.4 (2.08x slower) | 82.3 / 21.8 (3.77x slower) |
| NVIDIA RTX PRO 4500 Blackwell (CUDA), `float` | 19.1 / 8.7 (2.20x slower) | 31.7 / 8.6 (3.68x slower) | 34.0 / 9.4 (3.61x slower) | 44.3 / 10.6 (4.20x slower) | 64.4 / 22.7 (2.84x slower) |
| Intel Arc Pro B70 (Level-Zero), `double` | 94.0 / 42.8 (2.20x slower) | 109.8 / 42.9 (2.56x slower) | 118.3 / 46.1 (2.56x slower) | 119.5 / 47.6 (2.51x slower) | 190.5 / 55.9 (3.41x slower) |
| Intel Arc Pro B70 (Level-Zero), `float` | 92.8 / 42.0 (2.21x slower) | 106.6 / 41.6 (2.56x slower) | 114.8 / 43.1 (2.66x slower) | 119.9 / 46.0 (2.61x slower) | 183.4 / 57.1 (3.21x slower) |
| Intel Core Ultra 7 265 iGPU (Level-Zero), `double` | 242.8 / 91.3 (2.66x slower) | 300.2 / 93.8 (3.20x slower) | 331.0 / 108.8 (3.04x slower) | 351.4 / 122.3 (2.87x slower) | 509.3 / 222.3 (2.29x slower) |
| Intel Core Ultra 7 265 iGPU (Level-Zero), `float` | 243.1 / 91.3 (2.66x slower) | 287.8 / 92.6 (3.11x slower) | 304.2 / 101.3 (3.00x slower) | 318.0 / 110.9 (2.87x slower) | 456.3 / 184.2 (2.48x slower) |

With the default work-group size, one tile contains 2,048 elements. CUDA and
Level Zero therefore use the fast path through `N=2048`; `N=2049` is the
intentional first multi-tile control. The fast path materially reduces ADN's
own fixed overhead, but the reproducibility guarantee still costs more than
a conventional oneDPL scan at these sizes.

### Scalar-sum throughput

| Device (backend) | `double` sum / baseline* (GElements/s) | `float` sum / baseline* (GElements/s) |
|---|---:|---:|
| NVIDIA GB10 (CUDA) | 25.5 / 31.2 (1.2x slower) | 55.4 / 62.1 (1.1x slower) |
| NVIDIA RTX PRO 4500 Blackwell (CUDA) | 50.5 / 78.2 (1.5x slower) | 135.1 / 126.4 (1.1x faster) |
| AMD Radeon Pro W7500 (HIP) | 9.3 / 14.5 (1.6x slower) | 16.0 / 19.1 (1.2x slower) |
| Intel Arc Pro B70 (Level-Zero) | 26.7 / 57.3 (2.1x slower) | 54.2 / 98.5 (1.8x slower) |
| Intel Core Ultra 7 265 iGPU (Xe-LPG, Level-Zero) | 1.8 / 6.6 (3.6x slower) | 5.0 / 12.9 (2.6x slower) |
| Intel Core Ultra 7 265 CPU (OpenCL) | 0.4 / 5.6 (13.8x slower) | 0.6 / 7.7 (13.1x slower) |

*Baseline = plain (non-reproducible) `sycl::reduction` of the same data
type on the same device. It is a comparison point, not a hard performance
ceiling. A binned deposit performs multiple dependent additions and
subtractions per fold; the observed gap also includes compiler, backend,
and runtime effects. Parenthetical factors compare the faster throughput to
the slower throughput.

> **Note:** The CPU backend uses the SYCL runtime's parallel thread pool
> (OpenCL CPU runtime), so it can use the available CPU cores. The
> higher relative overhead on CPU vs GPU is expected: GPUs hide the
> deposit dependency chain with massive thread-level parallelism, while
> CPU threads have fewer opportunities to overlap dependent operations.

### Cumulative-sum throughput

The cumulative-sum baseline is oneDPL `inclusive_scan` with a standard device
policy and `std::plus`. Both algorithms use the same device-USM input and
output arrays. The baseline is a throughput comparison only and does not
provide the reproducibility guarantees of `adn::cumsum`.

| Device (backend) | `double` cumsum / baseline* (GElements/s) | `float` cumsum / baseline* (GElements/s) |
|---|---:|---:|
| NVIDIA GB10 (CUDA) | 2.524 / 7.756 (3.1x slower) | 3.500 / 13.868 (4.0x slower) |
| NVIDIA RTX PRO 4500 Blackwell (CUDA) | 5.304 / 21.563 (4.1x slower) | 6.815 / 40.780 (6.0x slower) |
| AMD Radeon Pro W7500 (HIP) | 0.818 / 4.910 (6.0x slower) | 1.181 / 9.127 (7.7x slower) |
| Intel Arc Pro B70 (Level-Zero) | 3.926 / 19.280 (4.9x slower) | 7.181 / 43.274 (6.0x slower) |
| Intel Core Ultra 7 265 iGPU (Xe-LPG, Level-Zero) | 0.191 / 0.610 (3.2x slower) | 0.357 / 2.445 (6.9x slower) |
| Intel Core Ultra 7 265 CPU (OpenCL) | 0.144 / 1.497 (10.4x slower) | 0.173 / 2.525 (14.6x slower) |

*Baseline = oneDPL `inclusive_scan` of the same data type on the same device.
The slowdown factor is `baseline / cumsum`.

### Cross-device reproducibility

The sum is bit-identical **across CPUs, GPU vendors, and backends**, not
just across runs. Verified on the following devices and backends, with a
single fat binary providing device code for all targets on each system:

| Device (backend) | `double`, 1M wide-range | Shuffled input order |
|---|---|---|
| NVIDIA GB10 (CUDA) | `0x430FC878C605717F` | same bits |
| NVIDIA RTX PRO 4500 Blackwell (CUDA) | `0x430FC878C605717F` | same bits |
| AMD Radeon Pro W7500 (HIP) | `0x430FC878C605717F` | same bits |
| Intel Arc Pro B70 (Level-Zero) | `0x430FC878C605717F` | same bits |
| Intel Core Ultra 7 265 iGPU (Xe-LPG, Level-Zero) | `0x430FC878C605717F` | same bits |
| Intel Core Ultra 7 265 CPU (OpenCL) | `0x430FC878C605717F` | same bits |

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
`adn::sum` accumulation kernel reads each device input element exactly once:

1. **Deposit** - Each work-item strides over the input and adds each
   element to its private accumulator with a few plain FP additions
   (a sticky-bit trick makes the bin rounding order-independent).

2. **Renormalize** - After every 2^(mantissa_bits - W - 2) deposits, the
   drift of each primary value is shifted into its carry counter, so the
   accumulator does not overflow within its documented capacity.

3. **Merge** - Work-group accumulators are combined with an exact binned
   tree reduction; a final device work-group merges the per-group results
   and converts the total back to a single float/double.

For `adn::cumsum`, contiguous tiles are first reduced to binned totals.
Those totals undergo a recursive exclusive scan while remaining in binned
form. A final pass replays each tile from its incoming accumulator and converts
every prefix to `float` or `double`. Only tile totals and tile prefixes require
temporary global storage; the implementation does not store a full binned
accumulator for every output element.

Input order defines prefix membership, so shuffling the input generally
changes intermediate outputs. For a fixed input sequence, every output is
bit-identical across supported work-group sizes and validated devices. The
last output retains the full order-independent `adn::sum` guarantee.

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

/// Validate the queue's device floating-point environment and shared USM.
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

/// The dependency events complete before the library accesses arr.
template <int K = 3, int WG_SIZE = 256, typename T>
T sum(sycl::queue &q, const T *arr, size_t N,
   const std::vector<sycl::event> &dependencies);

/// Reproducible cumulative sums of N values.
/// output[i] is the reproducible sum of input[0] through input[i].
/// Accepts device, shared, host USM, or plain host pointers independently
/// for input and output. Plain host input is copied to device and plain host
/// output is copied back before this synchronous call returns.
/// Exact in-place operation (input == output) is supported; other overlapping
/// ranges are not supported.
template <int K = 3, int WG_SIZE = 256, typename T>
void cumsum(
   sycl::queue &q, const T *input, T *output, size_t N);

/// The dependency events complete before the library accesses either range.
template <int K = 3, int WG_SIZE = 256, typename T>
void cumsum(sycl::queue &q, const T *input, T *output, size_t N,
   const std::vector<sycl::event> &dependencies);

}
```

### Pointer type handling

| Pointer source | Input behavior | `cumsum` output behavior |
|----------------|----------------|-------------------------------|
| `sycl::malloc_device` | Used directly | Used directly |
| `sycl::malloc_shared` | Used directly | Used directly |
| `sycl::malloc_host` | Used directly | Used directly |
| Plain host pointer | Copied to temporary device storage | Temporary device output is copied back |

Detection is done at runtime via `sycl::get_pointer_type`. "No library-side
copy" does not guarantee a physical zero-copy implementation; the SYCL
runtime may migrate or otherwise manage USM storage.

Every valid, nonempty operation requires shared USM for internal accumulators.
`adn::cumsum` also requires device USM for its temporary arrays, while
`adn::sum` requires device USM only when copying a plain host pointer. The
library checks these capabilities and throws `std::runtime_error` with the
missing requirement instead of relying on an allocation failure.

For overloads without an explicit dependency list, the caller must ensure
that prior operations writing a USM input, or otherwise conflicting with a
USM input or output, have completed before the call.  The dependency overloads
accept events from those operations and order the first library access after
them.  Both overload forms are synchronous and return only after all work
submitted by the library has completed.

### Accumulator capacity

The binned carry fields can represent a finite number of summands while
remaining exact. SyclReproSum uses the conservative capacities from ReproBLAS:

| Type | Maximum reproducible input count |
|------|---------------------------------:|
| `float` | 4,294,966,784 |
| `double` | 9,223,372,036,854,773,760 |

The limit is available to callers as
`adn::max_reproducible_count<float>` or
`adn::max_reproducible_count<double>`. `adn::sum` and `adn::cumsum`
throw `std::length_error` before inspecting the pointer or allocating memory
when `N` exceeds the corresponding limit. Splitting the input and adding the
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

After input sizes and pointers have been checked, every non-empty `adn::sum`
or `adn::cumsum` call automatically validates shared USM and the
floating-point environment before classifying the pointer or allocating the
main work buffer. Path-specific device USM support is checked before the
corresponding temporary allocation. All merging and conversion runs on the
device, so host rounding and denormal modes do not affect the result.
Validation is fail-closed and covers:

- the IEEE binary32 and binary64 layouts used by the bit-level helpers;
- device fp64 support, including for float sums whose final conversion uses
  double arithmetic;
- shared USM support for all nonempty calls, plus device USM support for
  cumulative sums and plain-host-pointer sums;
- SYCL round-to-nearest, denormal, and Inf/NaN capability declarations for
  every precision used;
- effective device arithmetic using runtime-loaded rounding, subnormal, and
  signed-zero probes.

A successful device arithmetic check for a root-enumerated device is shared by
all host threads and cached by operation specialization, backend, device, and
floating-point type. Dynamically created sub-devices are not in that fixed
cache and are validated on every use. Applications can optionally preflight a
queue explicitly with `adn::validate_environment<float>(q)` or
`adn::validate_environment<double>(q)`. Operations still validate their own
specialization because translation units can use different device compiler
options. Validation throws `std::runtime_error` with the failed prerequisite
when the guarantee cannot be established.

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

Scalar conversion uses a volatile double running value for both float and
double results in the generic SPIR device image. This prevents observed CPU
OpenCL JIT miscompilations of conversion loops at higher fold counts and small
work-groups. Native NVIDIA and AMD images retain the ordinary running value.
The strict-FP pragmas remain necessary to prevent contraction within an
individual update. Both public APIs use this conversion path.

## Building

### Prerequisites

- [Intel DPC++ compiler](https://github.com/intel/llvm)
  (`clang++` with `-fsycl`, C++17 or later)
- oneDPL headers for the cumulative-sum throughput baseline
- A SYCL device with fp64 support that passes runtime environment validation
  (required by float and double sums and cumulative sums)
- Shared USM support, plus device USM support for cumulative sums and sums of
  plain host arrays
- For GPU: NVIDIA GPU + CUDA toolkit (for `nvptx64`), AMD GPU + ROCm (for
  `amdgcn`), and/or Intel GPU runtime (for `spir64`)
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

# Rebuild and test hostile host/device floating-point modes
make test-validation
```

CUDA, SPIR-V, and AMD HIP support are all enabled by default. The normal
`make`, `make all`, `make run`, `make test`, and `make test-validation`
targets build the same CUDA + SPIR-V + AMDGCN fat binaries. The test suite
therefore includes every enabled, visible Intel, NVIDIA, and AMD device.

Disable a backend when its toolkit or runtime is unavailable:

```bash
make test ENABLE_AMD=0
make test ENABLE_NVIDIA=0
make test ENABLE_SPIRV=0

# Intel/SPIR-V only
make test ENABLE_AMD=0 ENABLE_NVIDIA=0

# AMD/HIP only
make test ENABLE_NVIDIA=0 ENABLE_SPIRV=0
```

When NVIDIA or AMD support is enabled, the GPU architecture is auto-detected
with `nvidia-smi` or `rocm_agent_enumerator`, respectively. Set `CUDA_ARCH` or
`AMD_GPU_ARCH` explicitly when auto-detection is unavailable or when
cross-compiling. On a multi-GPU system, auto-detection uses the first
architecture reported by the corresponding tool; use an explicit override to
target a different installed architecture:

```bash
make test CUDA_ARCH=sm_86
make test-validation CUDA_ARCH=sm_86
make test AMD_GPU_ARCH=gfx1102
make test-validation AMD_GPU_ARCH=gfx1102
```

The test binary uses oneDPL as the cumulative-sum baseline in both latency
and throughput benchmarks. Until
[intel/llvm#22665](https://github.com/intel/llvm/pull/22665) is available in
the compiler, its AMD libspirv device library must be patched with the missing
group non-uniform shuffle builtins. The Makefile also supplies the libspirv
compatibility layout and the verified GlobalOffset LTO pipeline workaround
automatically.

The compatibility layout is created inside a configuration-specific
`build/clang-resource-*` directory. The Makefile re-emits the signed-character
libspirv bitcode with the `amdgcn-amd-amdhsa` target triple, avoiding a target
triple mismatch during device linking. It does not modify the selected
compiler installation. This is the local workaround for
[intel/llvm#19339](https://github.com/intel/llvm/issues/19339).

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

# Disable backends whose toolkits or runtimes are unavailable
make ENABLE_AMD=0
make ENABLE_NVIDIA=0
make ENABLE_SPIRV=0

# Custom oneDPL include directory for the benchmark baseline
make ONEDPL_INC=/path/to/oneDPL/include

# Override NVIDIA or AMD architecture, or AMD libspirv
make CUDA_ARCH=sm_86
make AMD_GPU_ARCH=gfx1102
make AMD_LIBSPIRV=/path/to/libspirv.l64.signed_char.bc

# Advanced target-list or runtime-device override
make SYCL_TARGETS=nvptx64-nvidia-cuda
make test ONEAPI_DEVICE_SELECTOR=hip:0
```

## Test Suite

The Google Test suite chooses one preferred backend per distinct device name
(Level-Zero, then CUDA, then other backends, with OpenCL as the fallback).
Each correctness case runs once per selected CPU or GPU from a single fat
binary (CUDA + SPIR-V + AMDGCN by default), plus cross-device bit-identity
tests and performance benchmarks. There are 189 correctness cases and 8
benchmarks per device, plus 11 cross-device cases and one version test: 800
tests on a system with three GPUs and one CPU. Four benchmarks measure 100M
element throughput and four measure small-N latency. The `WG_SIZE=1024` cases
skip on a device whose work-group or local memory limits cannot support that
configuration.

| Category | Description |
|----------|-------------|
| Edge cases | Empty, invalid pointers and sizes, single element, all zeros, negative zero, min/max values |
| Exact arithmetic | Integer sums, powers of two, geometric series, telescope sums |
| Cancellation stress | Multi-order cancellation, Kahan worst case, near-overflow cancellation |
| Large-scale | 1M to 20M elements, multi-work-group boundaries |
| Special values | Canonical NaN across payloads/devices, Inf propagation, subnormals, overflow to +Inf |
| Reproducibility | Multi-run bit-identity, shuffle order-independence, cross-WG_SIZE and selected cross-device/backend consistency |
| Cumulative sums | Prefix references, three-level tile scan, WG/K matrices, USM bounds, cross-device identity, and repeated-run stress cases |
| Environment safety | Host FP-mode independence, USM capability checks, shared validation, exception-safe USM cleanup, unsafe device mode rejection |
| Conversion regressions | Independent prefix references, signed/scaled rounding boundaries, all float folds, shuffles, in-place scans, and cross-device identity |
| Configurations | Float K = 2-21; selected double K up to 52; every power-of-two WG_SIZE from 2 to 1024 where supported; device- and host-pointer APIs |
| Benchmarks | `adn::sum` and `adn::cumsum` small-N latency and 100M throughput on every selected GPU and CPU |

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

# Reproduce the throughput tables with three invocations
for run in 1 2 3; do
   GTEST_FILTER='*Throughput*' \
      GTEST_OUTPUT="xml:build/throughput-${run}.xml" make test
done

# Reproduce the small-N tables with five invocations
for run in 1 2 3 4 5; do
   GTEST_FILTER='*Latency_*_SmallN*' \
      GTEST_OUTPUT="xml:build/latency-${run}.xml" make test
done

# Run only cumulative-sum benchmarks
LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib ./repro_test \
   --gtest_filter="*ADNCumsumBench*"
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
