# AGENTS.md - agent instructions for SyclReproSum

## Project overview

Header-only SYCL library (`repro_sum.hpp`) for bit-reproducible,
order-independent floating-point summation across CPU and GPU devices. It
implements the Ahrens-Demmel-Nguyen (ADN) binned floating-point format. The
binned primitives are adaptations of the ReproBLAS reference implementation
(see THIRD_PARTY_NOTICES); keep that attribution intact.

Key invariant: **the result must be bit-identical for any input order,
work-group size, and device.** Any change to the accumulator logic
(`update` / `deposit` / `renorm` / `merge` / `conv`) must preserve this.
Do not introduce shortcuts that round differently depending on
accumulation order (e.g. FMA in deposit paths, reassociated additions).

## Coding style

Enforced by `.clang-format` + CI (`.github/workflows/format.yml`):

- 3-space indentation, no tabs, 80-column limit, C++17.
- Pointer/reference symbols bind to the name: `int *p`, `int &r`.
- Every `if` / `else` / `for` / `while` body is braced, even
  single-statement ones.
- Use ASCII characters only in all project-authored files, filenames, comments,
  string literals, tests, documentation, and commit messages. Never introduce
  non-ASCII characters. Preserve vendored third-party content and required
  license notices unchanged.
- Run `clang-format-18 -i repro_sum.hpp repro_test.cpp example.cpp`
  before committing.

## Layout

- `repro_sum.hpp` - the entire library (public APIs: `adn::sum` and
  `adn::validate_environment`).
- `repro_test.cpp` - Google Test suite, value-parameterized over one preferred
  backend per distinct GPU or CPU name (149 correctness cases and 2
  `ADNSumBench` throughput benchmarks per device), plus 7 cross-device cases
  and a version test.
- `example.cpp` - minimal runnable usage example.
- `third_party/googletest` - git submodule; never edit.

## Building and testing

- Requires Intel DPC++ (`clang++ -fsycl`); set `DPCPP_HOME` if not in
  `~/sycl_workspace`. At least one fp64-capable SYCL CPU or GPU device is
  required to run the parameterized tests.
- `make all` builds example + tests; `make test` runs the suite.
- `make test-validation` rebuilds and runs the correctness suite with hostile
  host modes, then verifies unsafe device modes are rejected.
- `make test` includes the 100M-element throughput benchmarks. For a quicker
  correctness-only run, use `./repro_test "--gtest_filter=-*Bench*"`.
- CI currently checks formatting only; build and device testing are local.
- All tests must pass. New behavior needs new tests in
  `repro_test.cpp`, including a reproducibility check (shuffle the
  input, expect bit-identical results).
- Device code restrictions: no `std::` math that lacks a SYCL
  equivalent in kernels; use the bit-level helpers (`to_bits`,
  `from_bits`, `exp_field`) already provided.

## Commit conventions

- Keep commits focused; imperative-mood subject lines.
