# Contributing

## Coding style

C++ code follows the rules encoded in [.clang-format](.clang-format):

- **3-space indentation**, no tabs.
- Pointer and reference symbols bind to the **name**: `int *p`, `int &r`
  (not `int* p` / `int& r`).
- **Every** `if` / `else` / `for` / `while` body is braced, even
  single-statement ones.
- 80-column limit, C++17.
- **ASCII only:** Use ASCII characters in all project-authored files, filenames,
  comments, string literals, tests, documentation, and commit messages. Never
  introduce non-ASCII characters. Preserve vendored third-party content and
  required license notices unchanged.

Format before committing:

```bash
clang-format -i repro_sum.hpp repro_test.cpp example.cpp
```

CI rejects pull requests that are not clang-format clean.

## Building and testing

```bash
git clone --recursive https://github.com/lilohuang/SyclReproSum.git
cd SyclReproSum
make all          # build example + tests (see README for DPCPP_HOME)
make test         # run the Google Test suite
make test-validation  # run hostile host/device validation checks
```

All tests must pass on every available GPU (the suite is
value-parameterized over devices). New functionality needs new tests
in `repro_test.cpp`; performance-sensitive changes should be checked
against the `ADNSumBench` throughput benchmarks.

## Licensing

Original code is BSD-3-Clause. The binned accumulator primitives in
`repro_sum.hpp` are adapted from ReproBLAS - keep the attribution in
that file's header and in THIRD_PARTY_NOTICES intact when modifying it.
