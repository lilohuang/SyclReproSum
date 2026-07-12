// Copyright (c) 2026, Lilo Huang <kuso.cc@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file repro_test.cpp
 * @brief Google Test suite for Ahrens-Demmel-Nguyen K-Fold reproducible
 * summation.
 *
 * Tests correctness, reproducibility, and edge-case handling of adn::sum and
 * adn::cumsum across various input patterns, K configurations, and
 * floating-point types (double and float).
 */

#include <gtest/gtest.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>
#include <atomic>
#include <vector>
#include <cfenv>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <cctype>
#include <optional>
#include <string>
#include <random>
#include <algorithm>
#include <limits>
#include <mutex>
#include <thread>
#include <functional>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include "repro_sum.hpp"

// ============================================================
//  Strict bitwise equality
//
//  EXPECT_BIT_EQ compares raw IEEE-754 bit patterns instead of
//  using floating-point ==, so it distinguishes +0.0 from -0.0
//  and treats identical NaNs as equal.  All reproducibility
//  assertions (result vs. result) use it; comparisons against
//  mathematically expected constants keep value semantics.
// ============================================================

static std::uint32_t bits_of(float x) {
   std::uint32_t b;
   std::memcpy(&b, &x, sizeof(b));
   return b;
}

static std::uint64_t bits_of(double x) {
   std::uint64_t b;
   std::memcpy(&b, &x, sizeof(b));
   return b;
}

static float float_from_bits(std::uint32_t b) {
   float x;
   std::memcpy(&x, &b, sizeof(x));
   return x;
}

static double double_from_bits(std::uint64_t b) {
   double x;
   std::memcpy(&x, &b, sizeof(x));
   return x;
}

template <typename T>
static ::testing::AssertionResult assert_bit_equal(
   const char *a_expr, const char *b_expr, T a, T b) {
   if (bits_of(a) == bits_of(b)) {
      return ::testing::AssertionSuccess();
   }
   return ::testing::AssertionFailure()
      << a_expr << " and " << b_expr << " are not bit-identical:\n  "
      << std::setprecision(17) << a << " (0x" << std::hex << bits_of(a)
      << ") vs\n  " << std::setprecision(17) << b << " (0x" << std::hex
      << bits_of(b) << ")";
}

#define EXPECT_BIT_EQ(val, ref) EXPECT_PRED_FORMAT2(assert_bit_equal, val, ref)

template <typename T>
static void expect_bit_identical_arrays(const std::vector<T> &actual,
   const std::vector<T> &expected, const std::string &context) {
   ASSERT_EQ(actual.size(), expected.size()) << context;
   for (size_t i = 0; i < actual.size(); ++i) {
      if (bits_of(actual[i]) != bits_of(expected[i])) {
         EXPECT_BIT_EQ(actual[i], expected[i]) << context << ", prefix " << i;
      }
   }
}

// ============================================================
//  Device enumeration
//
//  All distinct GPUs on the system, deduplicated by name (the same
//  physical device can be exposed by several backends, e.g. an
//  Intel iGPU via both Level-Zero and OpenCL).  Prefer Level-Zero,
//  then CUDA, then other backends, with OpenCL as the fallback.
// ============================================================

static int backend_rank(sycl::backend backend) {
   if (backend == sycl::backend::ext_oneapi_level_zero) {
      return 0;
   }
   if (backend == sycl::backend::ext_oneapi_cuda) {
      return 1;
   }
   if (backend == sycl::backend::opencl) {
      return 3;
   }
   return 2;
}

static std::vector<sycl::device> preferred_devices(
   sycl::info::device_type type) {
   std::vector<sycl::device> result;
   for (const sycl::device &device : sycl::device::get_devices(type)) {
      const std::string name = device.get_info<sycl::info::device::name>();
      auto existing = result.end();
      for (auto it = result.begin(); it != result.end(); ++it) {
         if (it->get_info<sycl::info::device::name>() == name) {
            existing = it;
            break;
         }
      }

      if (existing == result.end()) {
         result.push_back(device);
      } else if (backend_rank(device.get_platform().get_backend()) <
         backend_rank(existing->get_platform().get_backend())) {
         *existing = device;
      }
   }
   return result;
}

static const std::vector<sycl::device> &all_gpus() {
   static const std::vector<sycl::device> devices =
      preferred_devices(sycl::info::device_type::gpu);
   return devices;
}

static bool has_level_zero_gpu(const std::string &name) {
   for (const sycl::device &device :
      sycl::device::get_devices(sycl::info::device_type::gpu)) {
      if (device.get_info<sycl::info::device::name>() == name &&
         device.get_platform().get_backend() ==
            sycl::backend::ext_oneapi_level_zero) {
         return true;
      }
   }
   return false;
}

static const std::vector<sycl::device> &all_cpus() {
   static const std::vector<sycl::device> devices =
      preferred_devices(sycl::info::device_type::cpu);
   return devices;
}

/// Preferred backend for each distinct GPU and CPU name.
static const std::vector<sycl::device> &all_devices() {
   static const std::vector<sycl::device> devices = [] {
      std::vector<sycl::device> result;
      std::vector<std::string> seen;
      for (auto *src : {&all_gpus(), &all_cpus()}) {
         for (auto &d : *src) {
            std::string key = d.get_info<sycl::info::device::name>();
            if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
               seen.push_back(key);
               result.push_back(d);
            }
         }
      }
      return result;
   }();
   return devices;
}

/// @brief Sanitize a device name into a valid gtest parameter label.
static std::string device_label(
   const ::testing::TestParamInfo<sycl::device> &info) {
   std::string name = info.param.get_info<sycl::info::device::name>();
   for (char &c : name) {
      if (!std::isalnum(static_cast<unsigned char>(c))) {
         c = '_';
      }
   }
   return name;
}

// ============================================================
//  Shared fixture, parameterized over one preferred backend for each
//  distinct GPU and CPU name.  Each test runs once per selected device
//  from a single fat binary (CUDA + SPIR-V device code at runtime).
// ============================================================

class ADNSumTest : public ::testing::TestWithParam<sycl::device> {
protected:
   sycl::queue &queue() {
      if (!q_.has_value()) {
         q_.emplace(GetParam());
      }
      return *q_;
   }

   /// @brief Shuffle data and call adn::sum (double).
   template <int K = 3>
   double shuffle_and_sum(std::vector<double> &v, unsigned seed = 42) {
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      return adn::sum<K>(queue(), v.data(), v.size());
   }

   /// @brief Shuffle data and call adn::sum (float).
   template <int K = 3>
   float shuffle_and_sum_f(std::vector<float> &v, unsigned seed = 42) {
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      return adn::sum<K>(queue(), v.data(), v.size());
   }

   /// @brief Verify 5 runs produce bit-identical results (double).
   template <int K = 3>
   void assert_reproducible(std::vector<double> &v, int runs = 5) {
      double first = adn::sum<K>(queue(), v.data(), v.size());
      for (int i = 1; i < runs; ++i) {
         EXPECT_BIT_EQ(adn::sum<K>(queue(), v.data(), v.size()), first)
            << "Run " << i << " differs from run 0";
      }
   }

   /// @brief Verify 5 runs produce bit-identical results (float).
   template <int K = 3>
   void assert_reproducible_f(std::vector<float> &v, int runs = 5) {
      float first = adn::sum<K>(queue(), v.data(), v.size());
      for (int i = 1; i < runs; ++i) {
         EXPECT_BIT_EQ(adn::sum<K>(queue(), v.data(), v.size()), first)
            << "Run " << i << " differs from run 0";
      }
   }

private:
   std::optional<sycl::queue> q_;
};

INSTANTIATE_TEST_SUITE_P(
   GPUs, ADNSumTest, ::testing::ValuesIn(all_gpus()), device_label);

INSTANTIATE_TEST_SUITE_P(
   CPUs, ADNSumTest, ::testing::ValuesIn(all_cpus()), device_label);

// ============================================================
//  Double-precision edge cases
// ============================================================

TEST_P(ADNSumTest, EmptyArray) {
   std::vector<double> v;
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0);
}

TEST_P(ADNSumTest, CapacityExceededRejected) {
   if constexpr (std::numeric_limits<size_t>::max() >
      adn::max_reproducible_count<float>) {
      float value = 1.0f;
      const size_t too_many =
         static_cast<size_t>(adn::max_reproducible_count<float> + 1);
      EXPECT_THROW(adn::sum(queue(), &value, too_many), std::length_error);
   }
}

TEST_P(ADNSumTest, DeviceEnvironmentValidated) {
   const sycl::device &device = queue().get_device();
   const std::string name = device.get_info<sycl::info::device::name>();
   if (device.is_gpu() && has_level_zero_gpu(name)) {
      EXPECT_EQ(queue().get_backend(), sycl::backend::ext_oneapi_level_zero);
   }

   if (!device.has(sycl::aspect::fp64)) {
      EXPECT_THROW(
         adn::validate_environment<float>(queue()), std::runtime_error);
      EXPECT_THROW(
         adn::validate_environment<double>(queue()), std::runtime_error);
      return;
   }

   EXPECT_NO_THROW(adn::validate_environment<float>(queue()));
   EXPECT_NO_THROW(adn::validate_environment<double>(queue()));
}

TEST_P(ADNSumTest, HostRoundingModeIndependent) {
   if (!queue().get_device().has(sycl::aspect::fp64)) {
      GTEST_SKIP() << "Device does not support fp64";
   }

   std::vector<float> values{1e20f, -1e20f, 3.5f, -2.0f, 0.25f};
   const float reference = adn::sum(queue(), values.data(), values.size());

   const int original_rounding = std::fegetround();
   ASSERT_NE(original_rounding, -1);
   ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);

   EXPECT_NO_THROW(adn::validate_environment<float>(queue()));
   float downward_result = 0.0f;
   EXPECT_NO_THROW(
      downward_result = adn::sum(queue(), values.data(), values.size()));

   EXPECT_EQ(std::fesetround(original_rounding), 0);
   EXPECT_BIT_EQ(downward_result, reference);
}

TEST_P(ADNSumTest, HostDenormalModeIndependent) {
#if defined(__SSE__)
   if (!queue().get_device().has(sycl::aspect::fp64)) {
      GTEST_SKIP() << "Device does not support fp64";
   }

   const double tiny = double_from_bits(UINT64_C(10000000000));
   std::vector<double> values(10000, tiny);
   const double reference = adn::sum(queue(), values.data(), values.size());

   const unsigned original_mxcsr = _mm_getcsr();
   _mm_setcsr(original_mxcsr | (1u << 15) | (1u << 6));

   EXPECT_NO_THROW(adn::validate_environment<double>(queue()));
   double hostile_result = 0.0;
   EXPECT_NO_THROW(
      hostile_result = adn::sum(queue(), values.data(), values.size()));

   _mm_setcsr(original_mxcsr);
   EXPECT_BIT_EQ(hostile_result, reference);
#else
   GTEST_SKIP() << "Host does not expose SSE MXCSR controls";
#endif
}

struct ConcurrentDeviceValidationTag {};

TEST_P(ADNSumTest, DeviceValidationSharedAcrossThreads) {
   if (!queue().get_device().has(sycl::aspect::fp64)) {
      GTEST_SKIP() << "Device does not support fp64";
   }

   sycl::queue &q = queue();
   auto *slot =
      adn::detail::find_device_validation_slot<ConcurrentDeviceValidationTag>(
         q.get_backend(), q.get_device());
   ASSERT_NE(slot, nullptr);

   constexpr int thread_count = 16;
   std::atomic<bool> start{false};
   std::atomic<int> probe_runs{0};
   std::atomic<int> failures{0};
   std::vector<std::thread> threads;
   threads.reserve(thread_count);

   for (int i = 0; i < thread_count; ++i) {
      threads.emplace_back([&] {
         while (!start.load(std::memory_order_acquire)) {
         }
         try {
            std::call_once(slot->once, [&] {
               probe_runs.fetch_add(1, std::memory_order_relaxed);
               adn::detail::validate_device_capabilities<double>(
                  q.get_device());
               adn::detail::run_device_environment_probe<double>(q);
               adn::detail::validate_device_capabilities<float>(q.get_device());
               adn::detail::run_device_environment_probe<float>(q);
            });
         } catch (...) {
            failures.fetch_add(1, std::memory_order_relaxed);
         }
      });
   }

   start.store(true, std::memory_order_release);
   for (std::thread &thread : threads) {
      thread.join();
   }

   EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
   EXPECT_EQ(probe_runs.load(std::memory_order_relaxed), 1);
}

TEST_P(ADNSumTest, USMRAIIWaitsAndReleasesOnException) {
   sycl::queue &q = queue();
   auto marker_owner = adn::detail::allocate_shared_usm<int>(1, q);
   int *marker = marker_owner.get();
   *marker = 0;

   int *released = nullptr;
   EXPECT_THROW(([&] {
      auto allocation = adn::detail::allocate_shared_usm<int>(1, q);
      released = allocation.get();
      sycl::event pending = q.submit([=](sycl::handler &h) {
         h.single_task([=] {
            released[0] = 7;
            marker[0] = 1;
         });
      });
      adn::detail::submit_or_wait_on_error(pending, []() -> sycl::event {
         throw std::runtime_error("injected submission failure");
      });
   }()),
      std::runtime_error);

   EXPECT_EQ(*marker, 1);
   EXPECT_EQ(sycl::get_pointer_type(released, q.get_context()),
      sycl::usm::alloc::unknown);
}

#if defined(ADN_TEST_EXPECT_DEVICE_REJECTION)
TEST_P(ADNSumTest, UnsafeDeviceEnvironmentRejected) {
   EXPECT_THROW(adn::validate_environment<float>(queue()), std::runtime_error);
   EXPECT_THROW(adn::validate_environment<double>(queue()), std::runtime_error);
}
#endif

TEST_P(ADNSumTest, SingleElement) {
   std::vector<double> v{42.0};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 42.0);
}

TEST_P(ADNSumTest, SingleNegative) {
   std::vector<double> v{-123.456};
   EXPECT_DOUBLE_EQ(adn::sum(queue(), v.data(), v.size()), -123.456);
}

TEST_P(ADNSumTest, TwoElementsCancellation) {
   std::vector<double> v{1e15, -1e15};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0);
}

TEST_P(ADNSumTest, AllZeros) {
   std::vector<double> v(10000, 0.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0);
}

// ============================================================
//  Double-precision exact arithmetic
// ============================================================

TEST_P(ADNSumTest, IntegerOnes) {
   std::vector<double> v(10000, 1.0);
   EXPECT_EQ(shuffle_and_sum(v), 10000.0);
}

TEST_P(ADNSumTest, NegativeDominated) {
   std::vector<double> v;
   for (int i = 0; i < 10000; ++i) {
      v.push_back(-3.0);
   }
   for (int i = 0; i < 5000; ++i) {
      v.push_back(1.0);
   }
   EXPECT_EQ(shuffle_and_sum(v), -25000.0);
}

TEST_P(ADNSumTest, PowersOfTwo) {
   std::vector<double> v;
   double expected = 0.0;
   for (int e = -20; e <= 20; ++e) {
      v.push_back(std::exp2(e));
      expected += std::exp2(e);
   }
   EXPECT_EQ(shuffle_and_sum(v), expected);
}

// ============================================================
//  Double-precision cancellation stress tests
// ============================================================

TEST_P(ADNSumTest, PerfectCancellation100K) {
   std::vector<double> v;
   for (int i = 0; i < 50000; ++i) {
      v.push_back(1e15);
      v.push_back(-1e15);
   }
   EXPECT_EQ(shuffle_and_sum(v), 0.0);
}

TEST_P(ADNSumTest, CatastrophicCancellation24Orders) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 50; ++i) {
      v.push_back(1e8);
      v.push_back(-1e8);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   for (int i = 0; i < 20000; ++i) {
      v.push_back(i % 2 == 0 ? 1e-8 : -1e-8);
   }

   EXPECT_NEAR(shuffle_and_sum(v), 10000.0, 1e-6);
}

TEST_P(ADNSumTest, KahanInterleave) {
   std::vector<double> v;
   v.push_back(1e16);
   for (int i = 0; i < 100000; ++i) {
      v.push_back(1.0);
   }
   v.push_back(-1e16);
   EXPECT_NEAR(shuffle_and_sum(v), 100000.0, 1e-6);
}

// ============================================================
//  Double-precision large-scale
// ============================================================

TEST_P(ADNSumTest, OneMillionSmallValues) {
   const int N = 1000000;
   std::vector<double> v(N, 1e-7);
   EXPECT_NEAR(shuffle_and_sum(v), N * 1e-7, 1e-6);
}

TEST_P(ADNSumTest, AllIdentical0_1) {
   std::vector<double> v(100000, 0.1);
   EXPECT_NEAR(shuffle_and_sum(v), 100000 * 0.1, 1e-6);
}

TEST_P(ADNSumTest, TenMillionElements) {
   const int N = 10000000;
   std::vector<double> v(N, 1.0);
   v[0] = 1e15;
   v[1] = -1e15;
   EXPECT_NEAR(shuffle_and_sum(v), N - 2.0, 1e-6);
}

// ============================================================
//  Double-precision irrational / non-representable signals
// ============================================================

TEST_P(ADNSumTest, PiSignalWithLargeCancellation) {
   const double pi = 3.14159265358979323846;
   std::vector<double> v;
   for (int i = 0; i < 3; ++i) {
      v.push_back(1e12);
      v.push_back(-1e12);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(pi);
   }
   EXPECT_NEAR(shuffle_and_sum(v), 10000.0 * pi, 1e-8);
}

// ============================================================
//  Double-precision subnormal fallback
// ============================================================

TEST_P(ADNSumTest, SubnormalValues) {
   // Binned-format (ADN/ReproBLAS) semantics: values below the format's
   // absorption threshold (ulp of the bottom bin, ~2^-1055 for double
   // K=3) round away deterministically.  N * denorm_min falls below it,
   // so we assert the documented error bound and bit-reproducibility
   // rather than an exact tiny sum.
   double tiny = 5e-324;
   std::vector<double> v(1000, tiny);
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result)) << "Subnormals must not produce NaN";
   EXPECT_NEAR(result, 1000.0 * tiny, 1000.0 * std::ldexp(1.0, -1055));
   assert_reproducible(v);
}

// ============================================================
//  Double-precision reproducibility across 5 runs
// ============================================================

TEST_P(ADNSumTest, ReproducibleStressTest) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   for (int i = 0; i < 20000; ++i) {
      v.push_back(i % 2 == 0 ? 1e-8 : -1e-8);
   }

   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible(v);
}

TEST_P(ADNSumTest, ReproducibleLargeScale) {
   std::vector<double> v(1000000, 1e-7);
   assert_reproducible(v);
}

// ============================================================
//  Double-precision different K configurations
// ============================================================

TEST_P(ADNSumTest, K2_Correct) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   EXPECT_NEAR(shuffle_and_sum<2>(v), 10000.0, 1e-6);
}

TEST_P(ADNSumTest, K4_Correct) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   EXPECT_NEAR(shuffle_and_sum<4>(v), 10000.0, 1e-6);
}

TEST_P(ADNSumTest, K5_Correct) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   EXPECT_NEAR(shuffle_and_sum<5>(v), 10000.0, 1e-6);
}

TEST_P(ADNSumTest, K6_Correct) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   EXPECT_NEAR(shuffle_and_sum<6>(v), 10000.0, 1e-6);
}

TEST_P(ADNSumTest, K2_Reproducible) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible<2>(v);
}

TEST_P(ADNSumTest, K6_Reproducible) {
   std::vector<double> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible<6>(v);
}

// ============================================================
// ============================================================
//  FLOAT PRECISION TESTS
// ============================================================
// ============================================================

// ============================================================
//  Float edge cases
// ============================================================

TEST_P(ADNSumTest, Float_EmptyArray) {
   std::vector<float> v;
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0f);
}

TEST_P(ADNSumTest, Float_SingleElement) {
   std::vector<float> v{42.0f};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 42.0f);
}

TEST_P(ADNSumTest, Float_SingleNegative) {
   std::vector<float> v{-123.456f};
   EXPECT_FLOAT_EQ(adn::sum(queue(), v.data(), v.size()), -123.456f);
}

TEST_P(ADNSumTest, Float_TwoElementsCancellation) {
   std::vector<float> v{1e7f, -1e7f};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0f);
}

TEST_P(ADNSumTest, Float_AllZeros) {
   std::vector<float> v(10000, 0.0f);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0f);
}

TEST_P(ADNSumTest, Float_SingleMaxValue) {
   std::vector<float> v{std::numeric_limits<float>::max()};
   EXPECT_EQ(
      adn::sum(queue(), v.data(), v.size()), std::numeric_limits<float>::max());
}

TEST_P(ADNSumTest, Float_SingleMinNormal) {
   std::vector<float> v{std::numeric_limits<float>::min()};
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FLOAT_EQ(result, std::numeric_limits<float>::min());
}

// ============================================================
//  Float exact arithmetic
// ============================================================

TEST_P(ADNSumTest, Float_IntegerOnes) {
   std::vector<float> v(10000, 1.0f);
   EXPECT_EQ(shuffle_and_sum_f(v), 10000.0f);
}

TEST_P(ADNSumTest, Float_NegativeDominated) {
   std::vector<float> v;
   for (int i = 0; i < 10000; ++i) {
      v.push_back(-3.0f);
   }
   for (int i = 0; i < 5000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_EQ(shuffle_and_sum_f(v), -25000.0f);
}

TEST_P(ADNSumTest, Float_PowersOfTwo) {
   std::vector<float> v;
   float expected = 0.0f;
   for (int e = -10; e <= 10; ++e) {
      v.push_back(std::ldexp(1.0f, e));
      expected += std::ldexp(1.0f, e);
   }
   EXPECT_EQ(shuffle_and_sum_f(v), expected);
}

TEST_P(ADNSumTest, Float_SmallIntegers) {
   std::vector<float> v;
   float expected = 0.0f;
   for (int i = 1; i <= 1000; ++i) {
      v.push_back(static_cast<float>(i));
      expected += static_cast<float>(i);
   }
   EXPECT_EQ(shuffle_and_sum_f(v), expected);
}

// ============================================================
//  Float cancellation stress tests
// ============================================================

TEST_P(ADNSumTest, Float_PerfectCancellation) {
   std::vector<float> v;
   for (int i = 0; i < 50000; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   EXPECT_EQ(shuffle_and_sum_f(v), 0.0f);
}

TEST_P(ADNSumTest, Float_CatastrophicCancellation) {
   // Float has ~7 decimal digits; use scales spanning that range
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 50; ++i) {
      v.push_back(1e4f);
      v.push_back(-1e4f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }

   EXPECT_NEAR(shuffle_and_sum_f(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_KahanInterleave) {
   std::vector<float> v;
   v.push_back(1e7f);
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   v.push_back(-1e7f);
   EXPECT_NEAR(shuffle_and_sum_f(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_AlternatingSign) {
   // Sum of +1, -1, +1, -1, ... (even count) should be 0
   std::vector<float> v;
   for (int i = 0; i < 100000; ++i) {
      v.push_back(i % 2 == 0 ? 1.0f : -1.0f);
   }
   EXPECT_EQ(shuffle_and_sum_f(v), 0.0f);
}

// ============================================================
//  Float large-scale
// ============================================================

TEST_P(ADNSumTest, Float_OneMillionSmallValues) {
   const int N = 1000000;
   std::vector<float> v(N, 1e-4f);
   float expected = static_cast<float>(N) * 1e-4f;
   // With K=3, W=8 for float, tree reduction introduces rounding
   // that exceeds 1e-5 relative error at 1M elements.
   EXPECT_NEAR(shuffle_and_sum_f(v), expected, expected * 1e-4f);
}

TEST_P(ADNSumTest, Float_AllIdentical0_1) {
   std::vector<float> v(100000, 0.1f);
   float expected = 100000.0f * 0.1f;
   EXPECT_NEAR(shuffle_and_sum_f(v), expected, std::fabs(expected) * 1e-5f);
}

TEST_P(ADNSumTest, Float_LargeArrayWithCancellation) {
   const int N = 1000000;
   std::vector<float> v(N, 1.0f);
   v[0] = 1e7f;
   v[1] = -1e7f;
   EXPECT_NEAR(shuffle_and_sum_f(v), static_cast<float>(N - 2), 1.0f);
}

// ============================================================
//  Float subnormal fallback
// ============================================================

TEST_P(ADNSumTest, Float_SubnormalValues) {
   // Same binned-format absorption semantics as SubnormalValues above:
   // float's bottom bin has ulp 2^-144, so N * denorm_min (2^-149 each)
   // is below the deterministic rounding threshold.
   float tiny = std::numeric_limits<float>::denorm_min();
   std::vector<float> v(1000, tiny);
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result)) << "Subnormals must not produce NaN";
   EXPECT_NEAR(result, 1000.0f * tiny, 1000.0f * std::ldexp(1.0f, -144));
   assert_reproducible_f(v);
}

TEST_P(ADNSumTest, Float_NearMinNormal) {
   float small = std::numeric_limits<float>::min(); // smallest normal
   std::vector<float> v(1000, small);
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_NEAR(result, 1000.0f * small, small);
}

// ============================================================
//  Float reproducibility across 5 runs
// ============================================================

TEST_P(ADNSumTest, Float_ReproducibleStressTest) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   for (int i = 0; i < 20000; ++i) {
      v.push_back(i % 2 == 0 ? 1e-4f : -1e-4f);
   }

   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible_f(v);
}

TEST_P(ADNSumTest, Float_ReproducibleLargeScale) {
   std::vector<float> v(1000000, 1e-4f);
   assert_reproducible_f(v);
}

TEST_P(ADNSumTest, Float_ReproducibleRandom) {
   std::mt19937 rng(123);
   std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
   std::vector<float> v(100000);
   for (auto &x : v) {
      x = dist(rng);
   }
   assert_reproducible_f(v);
}

// ============================================================
//  Float different K configurations
// ============================================================

TEST_P(ADNSumTest, Float_K2_Correct) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_NEAR(shuffle_and_sum_f<2>(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_K3_Correct) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_NEAR(shuffle_and_sum_f<3>(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_K4_Correct) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_NEAR(shuffle_and_sum_f<4>(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_K6_Correct) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_NEAR(shuffle_and_sum_f<6>(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_K8_Correct) {
   // K=8 with float: 24/8 = 3 bits per limb (valid)
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_NEAR(shuffle_and_sum_f<8>(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_K12_Correct) {
   // K=12 with float: 24/12 = 2 bits per limb (minimum valid)
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   EXPECT_NEAR(shuffle_and_sum_f<12>(v), 10000.0f, 1.0f);
}

TEST_P(ADNSumTest, Float_K2_Reproducible) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible_f<2>(v);
}

TEST_P(ADNSumTest, Float_K6_Reproducible) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible_f<6>(v);
}

TEST_P(ADNSumTest, Float_K12_Reproducible) {
   std::vector<float> v;
   for (int i = 0; i < 5; ++i) {
      v.push_back(1e7f);
      v.push_back(-1e7f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible_f<12>(v);
}

// ============================================================
//  Float precision boundary tests
// ============================================================

TEST_P(ADNSumTest, Float_MixedMagnitudes) {
   // Float only has 24 bits mantissa - test with values spanning
   // several orders within representable range
   std::vector<float> v;
   for (int i = 0; i < 100; ++i) {
      v.push_back(1e6f);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0f);
   }
   for (int i = 0; i < 100000; ++i) {
      v.push_back(1e-3f);
   }

   float expected = 100.0f * 1e6f + 10000.0f + 100000.0f * 1e-3f;
   float result = shuffle_and_sum_f(v);
   EXPECT_NEAR(result, expected, std::fabs(expected) * 1e-5f);
}

TEST_P(ADNSumTest, Float_ConsecutiveIntegers) {
   // Sum of 1+2+3+...+N = N*(N+1)/2; exact for small enough N
   const int N = 2048; // 2048*2049/2 = 2097152, fits in 24-bit mantissa
   std::vector<float> v;
   for (int i = 1; i <= N; ++i) {
      v.push_back(static_cast<float>(i));
   }
   float expected = static_cast<float>(N) * static_cast<float>(N + 1) / 2.0f;
   EXPECT_EQ(shuffle_and_sum_f(v), expected);
}

TEST_P(ADNSumTest, Float_LargeRandomReproducible) {
   // Random values with wide distribution - must be reproducible
   std::mt19937 rng(99);
   std::uniform_real_distribution<float> dist(-1e10f, 1e10f);
   std::vector<float> v(500000);
   for (auto &x : v) {
      x = dist(rng);
   }
   assert_reproducible_f(v);
}

TEST_P(ADNSumTest, Float_NearOverflow) {
   // Values near float max that cancel: verify reproducibility.
   // With K=3, W=8 the dynamic range between big-value residuals
   // (+/-2^102) and small addends (1.0) exceeds float precision in a
   // single limb, so we only verify order-independence here.
   float big = std::numeric_limits<float>::max() / 4.0f;
   std::vector<float> v;
   v.push_back(big);
   v.push_back(big);
   v.push_back(-big);
   v.push_back(-big);
   for (int i = 0; i < 1000; ++i) {
      v.push_back(1.0f);
   }
   assert_reproducible_f(v);
}

// ============================================================
// ============================================================
//  ADDITIONAL COVERAGE TESTS
// ============================================================
// ============================================================

// ============================================================
//  Device pointer API (raw pointer interface)
// ============================================================

TEST_P(ADNSumTest, DevicePointer_Double) {
   std::vector<double> host{1.0, 2.0, 3.0, 4.0, 5.0};
   double *d_arr = sycl::malloc_device<double>(host.size(), queue());
   queue().memcpy(d_arr, host.data(), host.size() * sizeof(double)).wait();

   double result = adn::sum(queue(), d_arr, host.size());
   EXPECT_EQ(result, 15.0);

   sycl::free(d_arr, queue());
}

TEST_P(ADNSumTest, DevicePointer_Float) {
   std::vector<float> host{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
   float *d_arr = sycl::malloc_device<float>(host.size(), queue());
   queue().memcpy(d_arr, host.data(), host.size() * sizeof(float)).wait();

   float result = adn::sum(queue(), d_arr, host.size());
   EXPECT_EQ(result, 15.0f);

   sycl::free(d_arr, queue());
}

TEST_P(ADNSumTest, DevicePointer_LargeArray) {
   const size_t N = 100000;
   std::vector<double> host(N, 1.0);
   double *d_arr = sycl::malloc_device<double>(N, queue());
   queue().memcpy(d_arr, host.data(), N * sizeof(double)).wait();

   double result = adn::sum(queue(), d_arr, N);
   EXPECT_EQ(result, static_cast<double>(N));

   sycl::free(d_arr, queue());
}

// ============================================================
//  Custom WG_SIZE configurations
// ============================================================

template <typename T, int K>
static bool supports_wg_size_1024(const sycl::device &device) {
   constexpr size_t wg_size = 1024;
   constexpr size_t local_bytes = wg_size * sizeof(adn::detail::Binned<T, K>);
   return device.get_info<sycl::info::device::max_work_group_size>() >=
      wg_size &&
      device.get_info<sycl::info::device::local_mem_size>() >= local_bytes;
}

TEST_P(ADNSumTest, WGSize64_Double) {
   std::vector<double> v(10000, 1.0);
   v[0] = 1e15;
   v[1] = -1e15;
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   double result = adn::sum<3, 64>(queue(), v.data(), v.size());
   EXPECT_NEAR(result, 9998.0, 1e-6);
}

TEST_P(ADNSumTest, WGSize128_Double) {
   std::vector<double> v(10000, 1.0);
   v[0] = 1e15;
   v[1] = -1e15;
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   double result = adn::sum<3, 128>(queue(), v.data(), v.size());
   EXPECT_NEAR(result, 9998.0, 1e-6);
}

TEST_P(ADNSumTest, WGSize512_Double) {
   std::vector<double> v(10000, 1.0);
   v[0] = 1e15;
   v[1] = -1e15;
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   double result = adn::sum<3, 512>(queue(), v.data(), v.size());
   EXPECT_NEAR(result, 9998.0, 1e-6);
}

TEST_P(ADNSumTest, WGSize1024_Double) {
   if (!supports_wg_size_1024<double, 3>(queue().get_device())) {
      GTEST_SKIP() << "Device resources do not support WG_SIZE=1024";
   }

   std::mt19937 gen(1024);
   std::uniform_real_distribution<double> dist(-1e8, 1e8);
   std::vector<double> v(16387);
   for (double &x : v) {
      x = dist(gen);
   }

   const double reference = adn::sum<3, 256>(queue(), v.data(), v.size());
   EXPECT_BIT_EQ((adn::sum<3, 1024>(queue(), v.data(), v.size())), reference);

   std::mt19937 shuffle_rng(7);
   std::shuffle(v.begin(), v.end(), shuffle_rng);
   EXPECT_BIT_EQ((adn::sum<3, 1024>(queue(), v.data(), v.size())), reference);
}

TEST_P(ADNSumTest, WGSize64_Float) {
   std::vector<float> v(10000, 1.0f);
   float result = adn::sum<3, 64>(queue(), v.data(), v.size());
   EXPECT_EQ(result, 10000.0f);
}

TEST_P(ADNSumTest, WGSize512_Float) {
   std::vector<float> v(10000, 1.0f);
   float result = adn::sum<3, 512>(queue(), v.data(), v.size());
   EXPECT_EQ(result, 10000.0f);
}

TEST_P(ADNSumTest, WGSize1024_Float) {
   if (!supports_wg_size_1024<float, 3>(queue().get_device())) {
      GTEST_SKIP() << "Device resources do not support WG_SIZE=1024";
   }

   std::mt19937 gen(1024);
   std::uniform_real_distribution<float> dist(-1e4f, 1e4f);
   std::vector<float> v(16387);
   for (float &x : v) {
      x = dist(gen);
   }

   const float reference = adn::sum<3, 256>(queue(), v.data(), v.size());
   EXPECT_BIT_EQ((adn::sum<3, 1024>(queue(), v.data(), v.size())), reference);

   std::mt19937 shuffle_rng(7);
   std::shuffle(v.begin(), v.end(), shuffle_rng);
   EXPECT_BIT_EQ((adn::sum<3, 1024>(queue(), v.data(), v.size())), reference);
}

// ============================================================
//  Array size boundary conditions (padding logic)
// ============================================================

TEST_P(ADNSumTest, SizeEqualWGSize) {
   // Exactly one full work-group (no padding, single group)
   std::vector<double> v(256, 1.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 256.0);
}

TEST_P(ADNSumTest, SizeWGSizeMinus1) {
   // One element short of a full work-group (padding with 0)
   std::vector<double> v(255, 1.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 255.0);
}

TEST_P(ADNSumTest, SizeWGSizePlus1) {
   // Two work-groups, second has only 1 real element
   std::vector<double> v(257, 1.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 257.0);
}

TEST_P(ADNSumTest, SizeTwoElements) {
   std::vector<double> v{1e15, -1e15 + 1.0};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 1.0);
}

TEST_P(ADNSumTest, SizeThreeElements) {
   std::vector<double> v{1e15, -1e15, 42.0};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 42.0);
}

TEST_P(ADNSumTest, SizePrimeNumber) {
   // 997 is prime - tests non-power-of-2 sizes with padding
   std::vector<double> v(997, 1.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 997.0);
}

TEST_P(ADNSumTest, Float_SizeEqualWGSize) {
   std::vector<float> v(256, 1.0f);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 256.0f);
}

TEST_P(ADNSumTest, Float_SizeWGSizeMinus1) {
   std::vector<float> v(255, 1.0f);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 255.0f);
}

TEST_P(ADNSumTest, Float_SizeWGSizePlus1) {
   std::vector<float> v(257, 1.0f);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 257.0f);
}

// ============================================================
//  Negative zero handling
// ============================================================

TEST_P(ADNSumTest, NegativeZero) {
   std::vector<double> v{-0.0, -0.0, -0.0};
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_EQ(result, 0.0);
   // Verify no NaN
   EXPECT_FALSE(std::isnan(result));
}

TEST_P(ADNSumTest, MixedZeros) {
   std::vector<double> v{0.0, -0.0, 0.0, -0.0};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0);
}

TEST_P(ADNSumTest, Float_NegativeZero) {
   std::vector<float> v{-0.0f, -0.0f, -0.0f};
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_EQ(result, 0.0f);
   EXPECT_FALSE(std::isnan(result));
}

// ============================================================
//  Double max value overflow guard
// ============================================================

TEST_P(ADNSumTest, Double_SingleMaxValue) {
   std::vector<double> v{std::numeric_limits<double>::max()};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()),
      std::numeric_limits<double>::max());
}

TEST_P(ADNSumTest, Double_TwoMaxOpposite) {
   double mx = std::numeric_limits<double>::max();
   std::vector<double> v{mx, -mx};
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_EQ(result, 0.0);
}

TEST_P(ADNSumTest, Float_TwoMaxOpposite) {
   float mx = std::numeric_limits<float>::max();
   std::vector<float> v{mx, static_cast<float>(-mx)};
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_EQ(result, 0.0f);
}

// ============================================================
//  Multi-seed reproducibility (order-independence)
// ============================================================

TEST_P(ADNSumTest, OrderIndependent_Double) {
   // Same data shuffled with 10 different seeds must give same result
   std::mt19937 gen(77);
   std::uniform_real_distribution<double> dist(-1e12, 1e12);
   std::vector<double> base(50000);
   for (auto &x : base) {
      x = dist(gen);
   }

   double reference = adn::sum(queue(), base.data(), base.size());
   for (unsigned seed = 1; seed <= 10; ++seed) {
      std::vector<double> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

TEST_P(ADNSumTest, OrderIndependent_Float) {
   std::mt19937 gen(77);
   std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
   std::vector<float> base(50000);
   for (auto &x : base) {
      x = dist(gen);
   }

   float reference = adn::sum(queue(), base.data(), base.size());
   for (unsigned seed = 1; seed <= 10; ++seed) {
      std::vector<float> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

TEST_P(ADNSumTest, OrderIndependent_WithCancellation) {
   // Large cancelling values + small signal - reproducible across shuffles
   std::vector<double> base;
   for (int i = 0; i < 100; ++i) {
      base.push_back(1e14);
      base.push_back(-1e14);
   }
   for (int i = 0; i < 10000; ++i) {
      base.push_back(1.0);
   }

   double reference = adn::sum(queue(), base.data(), base.size());
   for (unsigned seed = 1; seed <= 10; ++seed) {
      std::vector<double> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

// ============================================================
//  Carry propagation stress tests
// ============================================================

TEST_P(ADNSumTest, CarryPropagation_Double) {
   // Values designed so that the sum of limb[K-1] exceeds scale[K-2],
   // forcing carry propagation in the reduction.
   // With K=3, W=17 for double: scale[0]=2^(e-17), scale[1]=2^(e-34)
   // Many values near max_abs trigger carry from limb[2] to limb[1]
   std::vector<double> v;
   // All values are ~1.0, so e_max=1, scale[2]=2^(1-51)
   // 100000 values of 1.0: limb[2] per element is tiny, but accumulated
   // carry should fire after enough additions
   for (int i = 0; i < 100000; ++i) {
      v.push_back(1.0);
   }
   double result = shuffle_and_sum(v);
   EXPECT_EQ(result, 100000.0);
}

TEST_P(ADNSumTest, CarryPropagation_Float) {
   // With K=3, W=8 for float, force carry in tree reduction
   // Use values that produce non-zero limb[1] and limb[2] entries
   std::vector<float> v;
   // 0.7f doesn't split evenly, creating residuals in limb[1] and limb[2]
   for (int i = 0; i < 10000; ++i) {
      v.push_back(0.7f);
   }
   float result = shuffle_and_sum_f(v);
   float expected = 10000.0f * 0.7f;
   EXPECT_NEAR(result, expected, std::fabs(expected) * 1e-5f);
}

TEST_P(ADNSumTest, CarryChain_Double) {
   // Alternating values near +/-2^W boundary to trigger carry at each step
   std::vector<double> v;
   double val = 131071.999999; // Near 2^17 - 1 (W=17 for double K=3)
   for (int i = 0; i < 1000; ++i) {
      v.push_back(val);
   }
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_NEAR(result, 1000.0 * val, 1e-6);
}

// ============================================================
//  All-negative values
// ============================================================

TEST_P(ADNSumTest, AllNegative_Double) {
   std::vector<double> v(10000, -7.5);
   EXPECT_EQ(shuffle_and_sum(v), -75000.0);
}

TEST_P(ADNSumTest, AllNegative_Float) {
   std::vector<float> v(10000, -7.5f);
   EXPECT_EQ(shuffle_and_sum_f(v), -75000.0f);
}

TEST_P(ADNSumTest, AllNegativeLarge_Double) {
   std::vector<double> v(50000, -1e10);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), -5e14);
}

// ============================================================
//  Geometric / harmonic series with known sums
// ============================================================

TEST_P(ADNSumTest, GeometricSeries_Double) {
   // Sum of 1/2^k for k=0..52 -> exact in double (all powers of 2)
   // Exact sum = 2 - 2^(-52)
   std::vector<double> v;
   double expected = 0.0;
   for (int k = 0; k <= 52; ++k) {
      v.push_back(std::ldexp(1.0, -k));
      expected += std::ldexp(1.0, -k);
   }
   EXPECT_EQ(shuffle_and_sum(v), expected);
}

TEST_P(ADNSumTest, GeometricSeries_Float) {
   // Sum of 1/2^k for k=0..23 -> exact in float
   std::vector<float> v;
   float expected = 0.0f;
   for (int k = 0; k <= 23; ++k) {
      v.push_back(std::ldexp(1.0f, -k));
      expected += std::ldexp(1.0f, -k);
   }
   EXPECT_EQ(shuffle_and_sum_f(v), expected);
}

TEST_P(ADNSumTest, AlternatingGeometric_Double) {
   // Sum of (-1)^k / 2^k for k=0..40
   // Known sum: 2/3 (geometric series with ratio -1/2)
   std::vector<double> v;
   for (int k = 0; k <= 40; ++k) {
      v.push_back(std::ldexp(k % 2 == 0 ? 1.0 : -1.0, -k));
   }
   double result = shuffle_and_sum(v);
   EXPECT_NEAR(result, 2.0 / 3.0, 1e-12);
}

// ============================================================
//  Monotone sequences (sorted input)
// ============================================================

TEST_P(ADNSumTest, SortedAscending_Double) {
   // Sorted ascending should give same result as shuffled
   std::vector<double> v;
   for (int i = 1; i <= 10000; ++i) {
      v.push_back(static_cast<double>(i));
   }
   double sorted_result = adn::sum(queue(), v.data(), v.size());
   std::mt19937 rng(123);
   std::shuffle(v.begin(), v.end(), rng);
   double shuffled_result = adn::sum(queue(), v.data(), v.size());
   EXPECT_BIT_EQ(sorted_result, shuffled_result);
}

TEST_P(ADNSumTest, SortedDescending_Double) {
   std::vector<double> v;
   for (int i = 10000; i >= 1; --i) {
      v.push_back(static_cast<double>(i));
   }
   double desc_result = adn::sum(queue(), v.data(), v.size());
   std::sort(v.begin(), v.end());
   double asc_result = adn::sum(queue(), v.data(), v.size());
   EXPECT_BIT_EQ(desc_result, asc_result);
}

// ============================================================
//  Naive sum fallback paths
// ============================================================

TEST_P(ADNSumTest, SubnormalFallback_ManyElements) {
   // Enough subnormals to require multi-work-group reduction
   double tiny = double_from_bits(UINT64_C(10000000000));
   std::vector<double> v(10000, tiny);
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FALSE(std::isinf(result));
   EXPECT_NE(bits_of(result), UINT64_C(0));
}

TEST_P(ADNSumTest, Float_SubnormalFallback_ManyElements) {
   float tiny = float_from_bits(UINT32_C(100000));
   std::vector<float> v(10000, tiny);
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FALSE(std::isinf(result));
   EXPECT_NE(bits_of(result), UINT32_C(0));
}

TEST_P(ADNSumTest, OverflowFallback_MultipleMaxValues) {
   // Multiple max-value elements: tests that the overflow fallback
   // handles more than 1 element correctly
   float mx = std::numeric_limits<float>::max();
   std::vector<float> v{
      mx, static_cast<float>(-mx), mx, static_cast<float>(-mx)};
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_EQ(result, 0.0f);
}

TEST_P(ADNSumTest, Double_OverflowFallback_Multiple) {
   double mx = std::numeric_limits<double>::max();
   std::vector<double> v{mx, -mx, mx, -mx, 1.0, 2.0, 3.0};
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   // naive_sum may not be perfectly accurate here, but should not be NaN/inf
   EXPECT_FALSE(std::isinf(result));
}

// ============================================================
//  WG_SIZE reproducibility (same input, different WG_SIZE)
// ============================================================

TEST_P(ADNSumTest, WGSizeConsistency_Double) {
   // Different WG_SIZE should give the same reproducible result
   std::mt19937 gen(55);
   std::uniform_real_distribution<double> dist(-1e8, 1e8);
   std::vector<double> v(10000);
   for (auto &x : v) {
      x = dist(gen);
   }

   double r256 = adn::sum<3, 256>(queue(), v.data(), v.size());
   double r128 = adn::sum<3, 128>(queue(), v.data(), v.size());
   double r64 = adn::sum<3, 64>(queue(), v.data(), v.size());
   // All should produce the same bit-exact result
   EXPECT_BIT_EQ(r256, r128);
   EXPECT_BIT_EQ(r256, r64);
}

TEST_P(ADNSumTest, WGSizeConsistency_Float) {
   std::mt19937 gen(55);
   std::uniform_real_distribution<float> dist(-1e4f, 1e4f);
   std::vector<float> v(10000);
   for (auto &x : v) {
      x = dist(gen);
   }

   float r256 = adn::sum<3, 256>(queue(), v.data(), v.size());
   float r128 = adn::sum<3, 128>(queue(), v.data(), v.size());
   float r64 = adn::sum<3, 64>(queue(), v.data(), v.size());
   EXPECT_BIT_EQ(r256, r128);
   EXPECT_BIT_EQ(r256, r64);
}

// ============================================================
//  Single-element edge cases
// ============================================================

TEST_P(ADNSumTest, SingleLargePositive) {
   std::vector<double> v{1e308};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 1e308);
}

TEST_P(ADNSumTest, SingleLargeNegative) {
   std::vector<double> v{-1e308};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), -1e308);
}

TEST_P(ADNSumTest, SingleSmallPositive) {
   std::vector<double> v{1e-300};
   EXPECT_DOUBLE_EQ(adn::sum(queue(), v.data(), v.size()), 1e-300);
}

TEST_P(ADNSumTest, Float_SingleOne) {
   std::vector<float> v{1.0f};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 1.0f);
}

// ============================================================
//  Precision: high-K gives better accuracy for double
// ============================================================

TEST_P(ADNSumTest, HighK_MoreAccurate_Double) {
   // With more limbs, the result should be at least as accurate
   std::vector<double> v;
   for (int i = 0; i < 100; ++i) {
      v.push_back(1e15);
      v.push_back(-1e15);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);

   double k3 = adn::sum<3>(queue(), v.data(), v.size());
   double k6 = adn::sum<6>(queue(), v.data(), v.size());
   double expected = 10000.0;
   EXPECT_NEAR(k3, expected, 1e-6);
   EXPECT_NEAR(k6, expected, 1e-6);
   // Both should be accurate but K=6 error should be <= K=3 error
   EXPECT_LE(std::fabs(k6 - expected), std::fabs(k3 - expected) + 1e-15);
}

// ============================================================
//  Multiple data distributions
// ============================================================

TEST_P(ADNSumTest, NormalDistribution_Reproducible) {
   std::mt19937 gen(42);
   std::normal_distribution<double> dist(0.0, 1e6);
   std::vector<double> v(100000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible(v);
}

TEST_P(ADNSumTest, ExponentialDistribution_Reproducible) {
   std::mt19937 gen(42);
   std::exponential_distribution<double> dist(1e-6);
   std::vector<double> v(100000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible(v);
}

TEST_P(ADNSumTest, Float_NormalDistribution_Reproducible) {
   std::mt19937 gen(42);
   std::normal_distribution<float> dist(0.0f, 1e3f);
   std::vector<float> v(100000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible_f(v);
}

// ============================================================
//  Worst-case: conditioned sum (ill-conditioned inputs)
// ============================================================

TEST_P(ADNSumTest, IllConditioned_Double) {
   // Condition number ~ 10^15: large values that nearly cancel
   // Result should still be reproducible
   std::vector<double> v;
   double base = 1e15;
   for (int i = 0; i < 1000; ++i) {
      v.push_back(base + static_cast<double>(i));
      v.push_back(-(base + static_cast<double>(i)));
   }
   v.push_back(42.0);
   // Exact sum is 42.0
   double result = shuffle_and_sum(v);
   EXPECT_NEAR(result, 42.0, 1e-6);
}

TEST_P(ADNSumTest, IllConditioned_Reproducible) {
   std::vector<double> v;
   double base = 1e15;
   for (int i = 0; i < 1000; ++i) {
      v.push_back(base + static_cast<double>(i));
      v.push_back(-(base + static_cast<double>(i)));
   }
   v.push_back(42.0);
   std::mt19937 rng(42);
   std::shuffle(v.begin(), v.end(), rng);
   assert_reproducible(v);
}

// ============================================================
//  Sparse arrays (mostly zeros with few non-zero)
// ============================================================

TEST_P(ADNSumTest, Sparse_Double) {
   std::vector<double> v(100000, 0.0);
   v[0] = 1.0;
   v[999] = 2.0;
   v[50000] = 3.0;
   v[99999] = 4.0;
   EXPECT_EQ(shuffle_and_sum(v), 10.0);
}

TEST_P(ADNSumTest, Sparse_Float) {
   std::vector<float> v(100000, 0.0f);
   v[0] = 1.0f;
   v[999] = 2.0f;
   v[50000] = 3.0f;
   v[99999] = 4.0f;
   EXPECT_EQ(shuffle_and_sum_f(v), 10.0f);
}

// ============================================================
//  Denormalized boundary: mix of normal and subnormal
// ============================================================

TEST_P(ADNSumTest, MixNormalSubnormal_Double) {
   double normal = std::numeric_limits<double>::min(); // smallest normal
   double subnormal = normal / 2.0;                    // subnormal
   std::vector<double> v;
   for (int i = 0; i < 1000; ++i) {
      v.push_back(normal);
   }
   for (int i = 0; i < 1000; ++i) {
      v.push_back(subnormal);
   }
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_GT(result, 0.0);
}

TEST_P(ADNSumTest, MixNormalSubnormal_Float) {
   float normal = std::numeric_limits<float>::min();
   float subnormal = normal / 2.0f;
   std::vector<float> v;
   for (int i = 0; i < 1000; ++i) {
      v.push_back(normal);
   }
   for (int i = 0; i < 1000; ++i) {
      v.push_back(subnormal);
   }
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_GT(result, 0.0f);
}

// ============================================================
//  Large N with multiple work-groups
// ============================================================

TEST_P(ADNSumTest, MultiWorkGroup_ExactBoundary) {
   // 4 * 256 = 1024 elements: exactly 4 work-groups
   std::vector<double> v(1024, 1.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 1024.0);
}

TEST_P(ADNSumTest, MultiWorkGroup_OddCount) {
   // 4 * 256 + 1 = 1025: 5th work-group has 1 element
   std::vector<double> v(1025, 1.0);
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 1025.0);
}

TEST_P(ADNSumTest, ManyWorkGroups_Reproducible) {
   // ~400 work-groups: stresses the device-side final reduction
   std::mt19937 gen(42);
   std::uniform_real_distribution<double> dist(-1e10, 1e10);
   std::vector<double> v(100000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible(v);
}

// ============================================================
//  Telescope sum (partial sums that cancel pair-wise)
// ============================================================

TEST_P(ADNSumTest, TelescopeSum_Double) {
   // v[i] = (i+1) - i = 1 for each pair -> sum = N
   // But encoded as a[0]=1, a[1]=2, a[2]=3,...; b[0]=-1, b[1]=-2,...
   // sum = N (only the last term survives)
   const int N = 10000;
   std::vector<double> v;
   for (int i = 0; i <= N; ++i) {
      v.push_back(static_cast<double>(i));
   }
   for (int i = 0; i <= N; ++i) {
      v.push_back(-static_cast<double>(i));
   }
   v.push_back(static_cast<double>(N)); // extra copy of N
   // sum = 0 + N = N
   EXPECT_EQ(shuffle_and_sum(v), static_cast<double>(N));
}

// ============================================================
//  Reproducibility under K variation
// ============================================================

TEST_P(ADNSumTest, K3_Reproducible_Random) {
   std::mt19937 gen(88);
   std::uniform_real_distribution<double> dist(-1e14, 1e14);
   std::vector<double> v(200000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible<3>(v);
}

TEST_P(ADNSumTest, K4_Reproducible_Random) {
   std::mt19937 gen(88);
   std::uniform_real_distribution<double> dist(-1e14, 1e14);
   std::vector<double> v(200000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible<4>(v);
}

TEST_P(ADNSumTest, K5_Reproducible_Random) {
   std::mt19937 gen(88);
   std::uniform_real_distribution<double> dist(-1e14, 1e14);
   std::vector<double> v(200000);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible<5>(v);
}

// ============================================================
//  Float: values at representability boundaries
// ============================================================

TEST_P(ADNSumTest, Float_LargestExactInteger) {
   // 2^24 = 16777216 is the largest integer exactly representable in float
   float big_int = 16777216.0f;
   std::vector<float> v{big_int, big_int, static_cast<float>(-big_int)};
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), big_int);
}

TEST_P(ADNSumTest, Float_PowersOfTwoSpan) {
   // Powers of 2 spanning the full normal float range
   std::vector<float> v;
   float expected = 0.0f;
   for (int e = -50; e <= 50; ++e) {
      float val = std::ldexp(1.0f, e);
      v.push_back(val);
      expected += val;
   }
   float result = shuffle_and_sum_f(v);
   EXPECT_EQ(result, expected);
}

// ============================================================
//  Stress: very large arrays
// ============================================================

TEST_P(ADNSumTest, TwentyMillion_Reproducible) {
   const int N = 20000000;
   std::vector<double> v(N, 1.0);
   // Inject cancellation
   v[0] = 1e15;
   v[1] = -1e15;
   assert_reproducible(v);
}

TEST_P(ADNSumTest, Float_FiveMillion_Reproducible) {
   const int N = 5000000;
   std::mt19937 gen(42);
   std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
   std::vector<float> v(N);
   for (auto &x : v) {
      x = dist(gen);
   }
   assert_reproducible_f(v);
}

// ============================================================
//  Regression: intermediate overflow must not produce NaN
// ============================================================

TEST_P(ADNSumTest, ManyLargeCancelling_Double_NoNaN) {
   // 300 x 1e307 + 300 x (-1e307): partial sums used to overflow to
   // +/-Inf inside the work-group reduction, yielding NaN.
   std::vector<double> v;
   for (int i = 0; i < 300; ++i) {
      v.push_back(1e307);
   }
   for (int i = 0; i < 300; ++i) {
      v.push_back(-1e307);
   }
   double result = shuffle_and_sum(v);
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FALSE(std::isinf(result));
   EXPECT_EQ(result, 0.0);
}

TEST_P(ADNSumTest, ManyLargeCancelling_Float_NoNaN) {
   std::vector<float> v;
   for (int i = 0; i < 300; ++i) {
      v.push_back(1e38f);
   }
   for (int i = 0; i < 300; ++i) {
      v.push_back(-1e38f);
   }
   float result = shuffle_and_sum_f(v);
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FALSE(std::isinf(result));
   EXPECT_EQ(result, 0.0f);
}

TEST_P(ADNSumTest, ManyLargeCancelling_Double_Reproducible) {
   std::vector<double> v;
   for (int i = 0; i < 300; ++i) {
      v.push_back(1e307);
   }
   for (int i = 0; i < 300; ++i) {
      v.push_back(-1e307);
   }
   assert_reproducible(v);
}

TEST_P(ADNSumTest, LargeSameSign_Double_Overflow) {
   // True sum overflows: result must be +Inf, not NaN.
   std::vector<double> v(300, 1e308);
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_TRUE(std::isinf(result) && result > 0.0);
}

// ============================================================
//  Regression: Inf/NaN inputs must propagate, not corrupt
// ============================================================

TEST_P(ADNSumTest, InfInput_Propagates) {
   std::vector<double> v{1.0, std::numeric_limits<double>::infinity(), 2.0};
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_TRUE(std::isinf(result) && result > 0.0);
}

TEST_P(ADNSumTest, NaNInput_Propagates) {
   std::vector<double> v{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
   EXPECT_BIT_EQ(adn::sum(queue(), v.data(), v.size()),
      double_from_bits(UINT64_C(0x7ff8000000000000)));
}

TEST_P(ADNSumTest, Double_NaNPayloadsCanonicalAndReproducible) {
   const double expected = double_from_bits(UINT64_C(0x7ff8000000000000));
   std::vector<double> base(4099, 1.0);
   base[17] = double_from_bits(UINT64_C(0x7ff8000000000001));
   base[1025] = double_from_bits(UINT64_C(0xfff8000000001234));
   base[3073] = double_from_bits(UINT64_C(0x7ff0000000000001));

   for (unsigned seed = 0; seed < 10; ++seed) {
      auto v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ((adn::sum<3, 64>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 128>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 256>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 512>(queue(), v.data(), v.size())), expected);
   }
}

TEST_P(ADNSumTest, Float_NaNPayloadsCanonicalAndReproducible) {
   const float expected = float_from_bits(UINT32_C(0x7fc00000));
   std::vector<float> base(4099, 1.0f);
   base[17] = float_from_bits(UINT32_C(0x7fc00001));
   base[1025] = float_from_bits(UINT32_C(0xffc01234));
   base[3073] = float_from_bits(UINT32_C(0x7f800001));

   for (unsigned seed = 0; seed < 10; ++seed) {
      auto v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ((adn::sum<3, 64>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 128>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 256>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 512>(queue(), v.data(), v.size())), expected);
   }
}

TEST_P(ADNSumTest, OppositeInfinitiesCanonicalAndReproducible) {
   const double expected = double_from_bits(UINT64_C(0x7ff8000000000000));
   std::vector<double> base(4099, 1.0);
   base[17] = std::numeric_limits<double>::infinity();
   base[3073] = -std::numeric_limits<double>::infinity();

   for (unsigned seed = 0; seed < 10; ++seed) {
      auto v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ((adn::sum<3, 64>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 256>(queue(), v.data(), v.size())), expected);
      EXPECT_BIT_EQ((adn::sum<3, 512>(queue(), v.data(), v.size())), expected);
   }
}

// ============================================================
//  Regression: float order-independence at large N
//  (the final limb merge used to round in float, breaking
//   bit-reproducibility beyond ~256 work-groups)
// ============================================================

TEST_P(ADNSumTest, OrderIndependent_Float_OneMillion) {
   std::mt19937 gen(1);
   std::uniform_real_distribution<float> dist(1.0f, 100.0f);
   std::vector<float> base(1000000);
   for (auto &x : base) {
      x = dist(gen);
   }

   float reference = adn::sum(queue(), base.data(), base.size());
   for (unsigned seed = 1; seed <= 5; ++seed) {
      std::vector<float> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

// ============================================================
//  Regression: small-magnitude inputs must stay reproducible
//  (the old underflow guard fell back to a non-deterministic
//   naive reduction, silently losing order-independence)
// ============================================================

TEST_P(ADNSumTest, OrderIndependent_TinyNormals_Double) {
   std::mt19937 gen(3);
   std::uniform_real_distribution<double> dist(1.0, 2.0);
   std::vector<double> base(100000);
   for (auto &x : base) {
      x = std::ldexp(dist(gen), -1000);
   }

   double reference = adn::sum<6>(queue(), base.data(), base.size());
   for (unsigned seed = 1; seed <= 5; ++seed) {
      std::vector<double> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum<6>(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

TEST_P(ADNSumTest, OrderIndependent_SubnormalRegion_Double) {
   std::mt19937 gen(4);
   std::uniform_real_distribution<double> dist(1.0, 2.0);
   std::vector<double> base(100000);
   for (auto &x : base) {
      x = std::ldexp(dist(gen), -1030);
   }

   double reference = adn::sum(queue(), base.data(), base.size());
   EXPECT_FALSE(std::isnan(reference));
   EXPECT_GT(reference, 0.0);
   for (unsigned seed = 1; seed <= 5; ++seed) {
      std::vector<double> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

TEST_P(ADNSumTest, OrderIndependent_TinyNormals_Float) {
   std::mt19937 gen(5);
   std::uniform_real_distribution<float> dist(1.0f, 2.0f);
   std::vector<float> base(100000);
   for (auto &x : base) {
      x = std::ldexp(dist(gen), -120);
   }

   float reference = adn::sum<8>(queue(), base.data(), base.size());
   for (unsigned seed = 1; seed <= 5; ++seed) {
      std::vector<float> v = base;
      std::mt19937 rng(seed);
      std::shuffle(v.begin(), v.end(), rng);
      EXPECT_BIT_EQ(adn::sum<8>(queue(), v.data(), v.size()), reference)
         << "Seed " << seed << " produced different result";
   }
}

// ============================================================
//  Reproducible cumulative sums
// ============================================================

template <int K = 3, typename T, typename IndexRange>
static void expect_cumsum_matches_sum(sycl::queue &q,
   const std::vector<T> &input, const std::vector<T> &output,
   const IndexRange &indices) {
   ASSERT_EQ(output.size(), input.size());
   for (size_t i : indices) {
      ASSERT_LT(i, input.size());
      EXPECT_BIT_EQ(output[i], adn::sum<K>(q, input.data(), i + 1))
         << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_EmptyAndInvalidPointers) {
   double output = 7.0;
   EXPECT_NO_THROW(
      adn::cumsum(queue(), static_cast<const double *>(nullptr), &output, 0));
   EXPECT_EQ(output, 7.0);
   EXPECT_THROW(
      adn::cumsum(queue(), static_cast<const double *>(nullptr), &output, 1),
      std::invalid_argument);
   EXPECT_THROW(
      adn::cumsum(queue(), &output, static_cast<double *>(nullptr), 1),
      std::invalid_argument);
}

TEST_P(ADNSumTest, Cumsum_CapacityExceededRejected) {
   if constexpr (std::numeric_limits<size_t>::max() >
      adn::max_reproducible_count<float>) {
      float input = 1.0f;
      float output = 0.0f;
      const size_t too_many =
         static_cast<size_t>(adn::max_reproducible_count<float> + 1);
      EXPECT_THROW(
         adn::cumsum(queue(), &input, &output, too_many), std::length_error);
   }
}

TEST_P(ADNSumTest, Cumsum_DoubleMatchesEveryPrefixSum) {
   std::vector<double> input{
      1e16, 1.0, -1e16, 3.0, -2.0, 1e-200, -1e-200, 0.25};
   std::vector<double> output(input.size());

   adn::cumsum(queue(), input.data(), output.data(), input.size());

   for (size_t i = 0; i < input.size(); ++i) {
      const double expected = adn::sum(queue(), input.data(), i + 1);
      EXPECT_BIT_EQ(output[i], expected) << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_FloatMatchesEveryPrefixSum) {
   std::vector<float> input{
      1e8f, 1.0f, -1e8f, 3.0f, -2.0f, 1e-30f, -1e-30f, 0.25f};
   std::vector<float> output(input.size());

   adn::cumsum(queue(), input.data(), output.data(), input.size());

   for (size_t i = 0; i < input.size(); ++i) {
      const float expected = adn::sum(queue(), input.data(), i + 1);
      EXPECT_BIT_EQ(output[i], expected) << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_SizeBoundaryMatrix) {
   const size_t wg64_sizes[] = {1, 2, 7, 8, 9, 63, 64, 65, 511, 512, 513, 4095,
      4096, 4097, 32767, 32768, 32769};
   for (size_t N : wg64_sizes) {
      std::vector<double> input(N, 1.0);
      std::vector<double> output(N);
      adn::cumsum<3, 64>(queue(), input.data(), output.data(), N);
      for (size_t i = 0; i < N; ++i) {
         EXPECT_EQ(output[i], static_cast<double>(i + 1)) << "N=" << N;
      }
   }

   const size_t default_sizes[] = {2047, 2048, 2049};
   for (size_t N : default_sizes) {
      std::vector<float> input(N, 1.0f);
      std::vector<float> output(N);
      adn::cumsum(queue(), input.data(), output.data(), N);
      for (size_t i = 0; i < N; ++i) {
         EXPECT_EQ(output[i], static_cast<float>(i + 1)) << "N=" << N;
      }
   }
}

TEST_P(ADNSumTest, Cumsum_WorkGroupSizeIndependent) {
   std::mt19937 gen(91);
   std::uniform_real_distribution<double> dist(-1e12, 1e12);
   std::vector<double> input(16387);
   for (double &x : input) {
      x = dist(gen);
   }
   for (size_t i = 0; i + 2 < input.size(); i += 97) {
      input[i] = 1e16;
      input[i + 1] = -1e16;
      input[i + 2] = 1.0;
   }

   std::vector<double> wg64(input.size());
   std::vector<double> wg128(input.size());
   std::vector<double> wg256(input.size());
   std::vector<double> wg512(input.size());
   adn::cumsum<3, 64>(queue(), input.data(), wg64.data(), input.size());
   adn::cumsum<3, 128>(queue(), input.data(), wg128.data(), input.size());
   adn::cumsum<3, 256>(queue(), input.data(), wg256.data(), input.size());
   adn::cumsum<3, 512>(queue(), input.data(), wg512.data(), input.size());

   for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_BIT_EQ(wg128[i], wg64[i]) << "WG_SIZE=128, prefix " << i;
      EXPECT_BIT_EQ(wg256[i], wg64[i]) << "WG_SIZE=256, prefix " << i;
      EXPECT_BIT_EQ(wg512[i], wg64[i]) << "WG_SIZE=512, prefix " << i;
   }

   if (supports_wg_size_1024<double, 3>(queue().get_device())) {
      std::vector<double> wg1024(input.size());
      adn::cumsum<3, 1024>(queue(), input.data(), wg1024.data(), input.size());
      for (size_t i = 0; i < input.size(); ++i) {
         EXPECT_BIT_EQ(wg1024[i], wg64[i]) << "WG_SIZE=1024, prefix " << i;
      }
   }
   EXPECT_BIT_EQ(wg64.back(), adn::sum(queue(), input.data(), input.size()));
}

TEST_P(ADNSumTest, Cumsum_RecursiveTileScanMatchesSum) {
   constexpr size_t N = 40003;
   std::mt19937 gen(17);
   std::uniform_real_distribution<double> dist(-1e8, 1e8);
   std::vector<double> input(N);
   for (double &x : input) {
      x = dist(gen);
   }
   std::vector<double> output(N);

   // WG_SIZE=64 gives 79 tiles, forcing a recursive scan of tile totals.
   adn::cumsum<3, 64>(queue(), input.data(), output.data(), input.size());

   const size_t checkpoints[] = {0, 1, 510, 511, 512, 32767, 32768, N - 1};
   for (size_t i : checkpoints) {
      EXPECT_BIT_EQ(output[i], adn::sum(queue(), input.data(), i + 1))
         << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_ThreeLevelRecursiveTileScan) {
   // WG_SIZE=64 has 512 elements per tile.  This creates 4097 tiles, whose
   // group totals require 65 groups and then 2 groups: three recursive levels.
   constexpr size_t N = size_t(64) * 64 * 64 * 8 + 17;
   std::mt19937 gen(117);
   std::uniform_real_distribution<double> dist(-1e6, 1e6);
   std::vector<double> input(N);
   for (double &x : input) {
      x = dist(gen);
   }
   for (size_t i = 0; i + 2 < N; i += 8191) {
      input[i] = 1e16;
      input[i + 1] = -1e16;
      input[i + 2] = 1.0;
   }

   std::vector<double> wg64(N);
   std::vector<double> wg256(N);
   adn::cumsum<3, 64>(queue(), input.data(), wg64.data(), N);
   adn::cumsum<3, 256>(queue(), input.data(), wg256.data(), N);

   ASSERT_EQ(std::memcmp(wg64.data(), wg256.data(), N * sizeof(double)), 0);
   const size_t checkpoints[] = {
      0, 7, 8, 510, 511, 512, 32767, 32768, 2097151, 2097152, N - 1};
   expect_cumsum_matches_sum(queue(), input, wg64, checkpoints);
}

TEST_P(ADNSumTest, Cumsum_InPlaceHostPointer) {
   std::vector<double> values(4099, 1.0);
   values[0] = 1e16;
   values[1] = -1e16;
   std::vector<double> expected(values.size());
   adn::cumsum(queue(), values.data(), expected.data(), values.size());

   adn::cumsum(queue(), values.data(), values.data(), values.size());

   for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_BIT_EQ(values[i], expected[i]) << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_DeviceUSMAndInPlace) {
   constexpr size_t N = 4099;
   std::vector<float> input(N);
   for (size_t i = 0; i < N; ++i) {
      input[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.25f;
   }
   std::vector<float> expected(N);
   adn::cumsum(queue(), input.data(), expected.data(), N);

   float *device = sycl::malloc_device<float>(N, queue());
   ASSERT_NE(device, nullptr);
   queue().memcpy(device, input.data(), N * sizeof(float)).wait_and_throw();
   adn::cumsum(queue(), device, device, N);

   std::vector<float> actual(N);
   queue().memcpy(actual.data(), device, N * sizeof(float)).wait_and_throw();
   sycl::free(device, queue());

   for (size_t i = 0; i < N; ++i) {
      EXPECT_BIT_EQ(actual[i], expected[i]) << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_MixedHostAndDevicePointers) {
   constexpr size_t N = 2051;
   std::vector<double> input(N);
   for (size_t i = 0; i < N; ++i) {
      input[i] = static_cast<double>(static_cast<int>(i % 23) - 11) * 0.125;
   }
   std::vector<double> expected(N);
   adn::cumsum(queue(), input.data(), expected.data(), N);

   auto device_input = adn::detail::allocate_device_usm<double>(N, queue());
   auto device_output = adn::detail::allocate_device_usm<double>(N, queue());
   queue()
      .memcpy(device_input.get(), input.data(), N * sizeof(double))
      .wait_and_throw();

   std::vector<double> host_output(N);
   adn::cumsum(
      queue(), device_input.get(), host_output.data(), host_output.size());
   adn::cumsum(queue(), input.data(), device_output.get(), input.size());

   std::vector<double> copied_output(N);
   queue()
      .memcpy(copied_output.data(), device_output.get(), N * sizeof(double))
      .wait_and_throw();
   for (size_t i = 0; i < N; ++i) {
      EXPECT_BIT_EQ(host_output[i], expected[i]) << "host output " << i;
      EXPECT_BIT_EQ(copied_output[i], expected[i]) << "device output " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_OutputBoundsAndHostSharedUSM) {
   constexpr size_t N = 2053;
   constexpr double LEFT_CANARY = 1234567.0;
   constexpr double RIGHT_CANARY = -7654321.0;
   std::vector<double> input(N);
   for (size_t i = 0; i < N; ++i) {
      input[i] = static_cast<double>(static_cast<int>(i % 31) - 15) * 0.5;
   }
   std::vector<double> expected(N);
   adn::cumsum(queue(), input.data(), expected.data(), N);

   std::vector<double> guarded(N + 2);
   guarded.front() = LEFT_CANARY;
   guarded.back() = RIGHT_CANARY;
   adn::cumsum(queue(), input.data(), guarded.data() + 1, N);
   EXPECT_BIT_EQ(guarded.front(), LEFT_CANARY);
   EXPECT_BIT_EQ(guarded.back(), RIGHT_CANARY);
   EXPECT_EQ(
      std::memcmp(guarded.data() + 1, expected.data(), N * sizeof(double)), 0);

   const sycl::device &device = queue().get_device();
   if (device.has(sycl::aspect::usm_shared_allocations)) {
      auto shared_input = adn::detail::allocate_shared_usm<double>(N, queue());
      auto shared_output =
         adn::detail::allocate_shared_usm<double>(N + 2, queue());
      std::copy(input.begin(), input.end(), shared_input.get());
      shared_output.get()[0] = LEFT_CANARY;
      shared_output.get()[N + 1] = RIGHT_CANARY;
      adn::cumsum(queue(), shared_input.get(), shared_output.get() + 1, N);
      EXPECT_BIT_EQ(shared_output.get()[0], LEFT_CANARY);
      EXPECT_BIT_EQ(shared_output.get()[N + 1], RIGHT_CANARY);
      EXPECT_EQ(std::memcmp(shared_output.get() + 1, expected.data(),
                   N * sizeof(double)),
         0);
   }

   if (device.has(sycl::aspect::usm_host_allocations)) {
      adn::detail::UsmUniquePtr<double> host_input(
         sycl::malloc_host<double>(N, queue()),
         adn::detail::UsmDeleter<double>{queue().get_context()});
      adn::detail::UsmUniquePtr<double> host_output(
         sycl::malloc_host<double>(N, queue()),
         adn::detail::UsmDeleter<double>{queue().get_context()});
      ASSERT_NE(host_input.get(), nullptr);
      ASSERT_NE(host_output.get(), nullptr);
      std::copy(input.begin(), input.end(), host_input.get());
      adn::cumsum(queue(), host_input.get(), host_output.get(), N);
      EXPECT_EQ(
         std::memcmp(host_output.get(), expected.data(), N * sizeof(double)),
         0);
   }
}

TEST_P(ADNSumTest, Cumsum_MoreFoldsMatchPrefixSum) {
   std::mt19937 gen(61);
   std::uniform_real_distribution<double> dist(-1e100, 1e100);
   std::vector<double> input(1031);
   for (double &x : input) {
      x = dist(gen);
   }
   std::vector<double> output(input.size());
   adn::cumsum<6>(queue(), input.data(), output.data(), input.size());

   const size_t checkpoints[] = {0, 7, 8, 511, 1023, 1030};
   for (size_t i : checkpoints) {
      EXPECT_BIT_EQ(output[i], adn::sum<6>(queue(), input.data(), i + 1))
         << "prefix " << i;
   }
}

TEST_P(ADNSumTest, Cumsum_FoldConfigurationsMatchPrefixSum) {
   std::mt19937 gen(211);
   std::uniform_real_distribution<double> double_dist(-1e100, 1e100);
   std::vector<double> double_input(2057);
   for (double &x : double_input) {
      x = double_dist(gen);
   }
   std::vector<double> double_k2(double_input.size());
   adn::cumsum<2>(
      queue(), double_input.data(), double_k2.data(), double_input.size());

   const size_t double_checkpoints[] = {0, 7, 8, 511, 512, 2047, 2056};
   expect_cumsum_matches_sum<2>(
      queue(), double_input, double_k2, double_checkpoints);

   std::uniform_real_distribution<float> float_dist(-1e20f, 1e20f);
   std::vector<float> float_input(4099);
   for (float &x : float_input) {
      x = float_dist(gen);
   }
   std::vector<float> float_k2(float_input.size());
   std::vector<float> float_k8(float_input.size());
   std::vector<float> float_k12(float_input.size());
   adn::cumsum<2>(
      queue(), float_input.data(), float_k2.data(), float_input.size());
   adn::cumsum<8>(
      queue(), float_input.data(), float_k8.data(), float_input.size());
   adn::cumsum<12>(
      queue(), float_input.data(), float_k12.data(), float_input.size());

   const size_t float_checkpoints[] = {
      0, 7, 8, 511, 512, 2047, 2048, 4095, 4098};
   expect_cumsum_matches_sum<2>(
      queue(), float_input, float_k2, float_checkpoints);
   expect_cumsum_matches_sum<8>(
      queue(), float_input, float_k8, float_checkpoints);
   expect_cumsum_matches_sum<12>(
      queue(), float_input, float_k12, float_checkpoints);
}

TEST_P(ADNSumTest, Cumsum_DoubleExtremeNumericPrefixes) {
   constexpr size_t N = 8195;
   std::vector<double> input(N);
   for (size_t i = 0; i < N; ++i) {
      const int exponent = static_cast<int>((i * 37) % 2001) - 1000;
      const double mantissa = 1.0 + static_cast<double>(i % 17) / 32.0;
      input[i] = std::ldexp(i % 2 == 0 ? mantissa : -mantissa, exponent);
   }

   const double denorm = std::numeric_limits<double>::denorm_min();
   const double max = std::numeric_limits<double>::max();
   input[7] = denorm;
   input[8] = -denorm;
   input[511] = max;
   input[512] = -max;
   input[2047] = 1e308;
   input[2048] = 1e308;
   input[2049] = -1e308;
   input[2050] = -1e308;
   input[4095] = 1e16;
   input[4096] = 1.0;
   input[4097] = -1e16;
   input[8191] = std::numeric_limits<double>::min();

   std::vector<double> output(N);
   adn::cumsum(queue(), input.data(), output.data(), N);
   const size_t checkpoints[] = {0, 6, 7, 8, 9, 510, 511, 512, 513, 2046, 2047,
      2048, 2049, 2050, 2051, 4094, 4095, 4096, 4097, 4098, 8190, 8191, 8192,
      N - 1};
   expect_cumsum_matches_sum(queue(), input, output, checkpoints);
}

TEST_P(ADNSumTest, Cumsum_FloatExtremeNumericPrefixes) {
   constexpr size_t N = 4099;
   std::vector<float> input(N);
   for (size_t i = 0; i < N; ++i) {
      const int exponent = static_cast<int>((i * 29) % 241) - 120;
      const float mantissa = 1.0f + static_cast<float>(i % 13) / 32.0f;
      input[i] = std::ldexp(i % 2 == 0 ? mantissa : -mantissa, exponent);
   }

   const float denorm = std::numeric_limits<float>::denorm_min();
   const float max = std::numeric_limits<float>::max();
   input[7] = denorm;
   input[8] = -denorm;
   input[511] = max;
   input[512] = -max;
   input[2047] = 1e30f;
   input[2048] = 1.0f;
   input[2049] = -1e30f;
   input[4095] = std::numeric_limits<float>::min();

   std::vector<float> output(N);
   adn::cumsum(queue(), input.data(), output.data(), N);
   const size_t checkpoints[] = {0, 6, 7, 8, 9, 510, 511, 512, 513, 2046, 2047,
      2048, 2049, 2050, 4094, 4095, 4096, N - 1};
   expect_cumsum_matches_sum(queue(), input, output, checkpoints);
}

TEST_P(ADNSumTest, Cumsum_RandomizedDifferentialAndRepeatability) {
   for (unsigned seed = 1; seed <= 8; ++seed) {
      const size_t N = 3000 + size_t(seed) * 733;
      std::mt19937 gen(seed * 101);
      std::uniform_real_distribution<double> mantissa_dist(0.5, 1.0);
      std::uniform_int_distribution<int> exponent_dist(-1000, 1000);
      std::uniform_int_distribution<int> sign_dist(0, 1);
      std::vector<double> input(N);
      for (double &x : input) {
         const double mantissa = mantissa_dist(gen);
         x = std::ldexp(
            sign_dist(gen) == 0 ? mantissa : -mantissa, exponent_dist(gen));
      }
      for (size_t i = 0; i + 2 < N; i += 257) {
         input[i] = 1e200;
         input[i + 1] = -1e200;
         input[i + 2] = 1.0;
      }

      std::vector<double> first(N);
      std::vector<double> second(N);
      adn::cumsum(queue(), input.data(), first.data(), N);
      adn::cumsum(queue(), input.data(), second.data(), N);
      ASSERT_EQ(std::memcmp(first.data(), second.data(), N * sizeof(double)), 0)
         << "seed " << seed;

      std::vector<size_t> checkpoints{
         0, 1, 6, 7, 8, 9, 510, 511, 512, 513, 2046, 2047, 2048, 2049, N - 1};
      std::uniform_int_distribution<size_t> index_dist(0, N - 1);
      for (int i = 0; i < 12; ++i) {
         checkpoints.push_back(index_dist(gen));
      }
      std::sort(checkpoints.begin(), checkpoints.end());
      checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()),
         checkpoints.end());
      expect_cumsum_matches_sum(queue(), input, first, checkpoints);
   }
}

TEST_P(ADNSumTest, Cumsum_UniformDecimalReproducible) {
   // Require every prefix to remain identical across repeated runs and
   // work-group sizes for a long uniform decimal input.
   constexpr size_t N = 250000;
   constexpr int RUNS = 100;
   std::vector<double> input(N, 0.1);
   auto device_input = adn::detail::allocate_device_usm<double>(N, queue());
   auto device_output = adn::detail::allocate_device_usm<double>(N, queue());
   queue()
      .memcpy(device_input.get(), input.data(), N * sizeof(double))
      .wait_and_throw();

   adn::cumsum(queue(), device_input.get(), device_output.get(), input.size());
   std::vector<double> reference(N);
   queue()
      .memcpy(reference.data(), device_output.get(), N * sizeof(double))
      .wait_and_throw();
   EXPECT_BIT_EQ(
      reference.back(), adn::sum(queue(), device_input.get(), input.size()));

   std::vector<double> actual(N);
   for (int run = 0; run < RUNS; ++run) {
      switch (run % 4) {
         case 0:
            adn::cumsum<3, 64>(
               queue(), device_input.get(), device_output.get(), N);
            break;
         case 1:
            adn::cumsum<3, 128>(
               queue(), device_input.get(), device_output.get(), N);
            break;
         case 2:
            adn::cumsum<3, 256>(
               queue(), device_input.get(), device_output.get(), N);
            break;
         default:
            adn::cumsum<3, 512>(
               queue(), device_input.get(), device_output.get(), N);
            break;
      }
      queue()
         .memcpy(actual.data(), device_output.get(), N * sizeof(double))
         .wait_and_throw();
      ASSERT_EQ(
         std::memcmp(actual.data(), reference.data(), N * sizeof(double)), 0)
         << "run " << run;
   }
}

TEST_P(ADNSumTest, Cumsum_SpecialValuesMatchPrefixSum) {
   const double inf = std::numeric_limits<double>::infinity();
   std::vector<double> nan_input(4099, 1.0);
   nan_input[2047] = double_from_bits(UINT64_C(0xfff8000000001234));
   std::vector<double> nan_output(nan_input.size());
   adn::cumsum(queue(), nan_input.data(), nan_output.data(), nan_input.size());
   const size_t nan_checkpoints[] = {0, 2046, 2047, 2048, 4095, 4098};
   expect_cumsum_matches_sum(queue(), nan_input, nan_output, nan_checkpoints);
   EXPECT_EQ(bits_of(nan_output[2047]), UINT64_C(0x7ff8000000000000));
   EXPECT_EQ(bits_of(nan_output.back()), UINT64_C(0x7ff8000000000000));

   std::vector<double> inf_input(4099, 1.0);
   inf_input[2047] = inf;
   inf_input[2048] = -inf;
   std::vector<double> inf_output(inf_input.size());
   adn::cumsum(queue(), inf_input.data(), inf_output.data(), inf_input.size());
   const size_t inf_checkpoints[] = {0, 2046, 2047, 2048, 2049, 4098};
   expect_cumsum_matches_sum(queue(), inf_input, inf_output, inf_checkpoints);
   EXPECT_EQ(inf_output[2047], inf);
   EXPECT_EQ(bits_of(inf_output[2048]), UINT64_C(0x7ff8000000000000));
}

TEST_P(ADNSumTest, Cumsum_ShuffledBlocksKeepBoundaryBits) {
   constexpr size_t BLOCK_SIZE = 257;
   constexpr size_t BLOCKS = 8;
   std::vector<double> base(BLOCK_SIZE * BLOCKS);
   std::mt19937 gen(73);
   std::uniform_real_distribution<double> dist(-1e10, 1e10);
   for (double &x : base) {
      x = dist(gen);
   }

   std::vector<double> reference(base.size());
   adn::cumsum(queue(), base.data(), reference.data(), base.size());

   for (unsigned seed = 1; seed <= 4; ++seed) {
      std::vector<double> shuffled = base;
      std::mt19937 rng(seed);
      for (size_t block = 0; block < BLOCKS; ++block) {
         const auto first = shuffled.begin() + block * BLOCK_SIZE;
         std::shuffle(first, first + BLOCK_SIZE, rng);
      }
      std::vector<double> output(shuffled.size());
      adn::cumsum(queue(), shuffled.data(), output.data(), shuffled.size());
      for (size_t block = 0; block < BLOCKS; ++block) {
         const size_t boundary = (block + 1) * BLOCK_SIZE - 1;
         EXPECT_BIT_EQ(output[boundary], reference[boundary])
            << "seed " << seed << ", block " << block;
      }
   }
}

// ============================================================
//  Throughput benchmarks (informational)
//
//  Measures adn::sum against a plain (non-reproducible)
//  sycl::reduction sum, which serves as the device memory
//  throughput ceiling for a single-pass reduction.  Results are
//  reported via GTest RecordProperty and stdout; the only hard
//  assertion is that both sums complete and the reproducible
//  throughput is a sane fraction of the naive one.
// ============================================================

class ADNSumBench : public ADNSumTest {
protected:
   struct Result {
      double naive_gelem_s;
      double binned_gelem_s;
   };

   /// @brief Time f() over @p runs iterations, return avg milliseconds.
   template <typename F> static double time_ms(F &&f, int runs = 10) {
      f(); // warmup
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < runs; ++i) {
         f();
      }
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
   }

   /// @brief Benchmark naive sycl::reduction vs adn::sum on N randoms.
   template <typename T> Result run(size_t N) {
      std::mt19937 gen(7);
      std::uniform_real_distribution<T> dist(T(-1e6), T(1e6));
      std::vector<T> v(N);
      for (auto &x : v) {
         x = dist(gen);
      }

      T *d_arr = sycl::malloc_device<T>(N, queue());
      queue().memcpy(d_arr, v.data(), N * sizeof(T)).wait();
      T *d_out = sycl::malloc_shared<T>(1, queue());

      const double naive_ms = time_ms([&] {
         *d_out = T(0);
         queue()
            .submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(N),
               sycl::reduction(d_out, sycl::plus<T>()),
               [=](sycl::id<1> i, auto &s) { s.combine(d_arr[i]); });
         }).wait();
      });

      volatile T sink;
      const double binned_ms =
         time_ms([&] { sink = adn::sum(queue(), d_arr, N); });
      (void)sink;

      sycl::free(d_arr, queue());
      sycl::free(d_out, queue());

      const double billions = static_cast<double>(N) / 1e9;
      return {billions / (naive_ms / 1e3), billions / (binned_ms / 1e3)};
   }

   template <typename T> void report(const char *type_name, size_t N) {
      const auto r = run<T>(N);
      const double slowdown = r.naive_gelem_s / r.binned_gelem_s;
      RecordProperty("naive_GElemPerSec", r.naive_gelem_s);
      RecordProperty("binned_GElemPerSec", r.binned_gelem_s);
      std::printf("  [bench] %s: %-6s N=%zu  naive %7.1f G elements/s | "
                  "adn::sum %7.1f G elements/s (%.1fx slower)\n",
         GetParam().get_info<sycl::info::device::name>().c_str(), type_name, N,
         r.naive_gelem_s, r.binned_gelem_s, slowdown);
   }
};

TEST_P(ADNSumBench, Throughput_Double_100M) {
   report<double>("double", 100000000);
}

TEST_P(ADNSumBench, Throughput_Float_100M) {
   report<float>("float", 100000000);
}

INSTANTIATE_TEST_SUITE_P(
   GPUs, ADNSumBench, ::testing::ValuesIn(all_gpus()), device_label);

INSTANTIATE_TEST_SUITE_P(
   CPUs, ADNSumBench, ::testing::ValuesIn(all_cpus()), device_label);

template <typename T> class ADNCumsumBaselinePolicy;

class ADNCumsumBench : public ADNSumTest {
protected:
   struct Result {
      double baseline_gelem_s;
      double binned_gelem_s;
   };

   /// @brief Time f() over @p runs iterations, return avg milliseconds.
   template <typename F> static double time_ms(F &&f, int runs = 10) {
      f(); // warmup
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < runs; ++i) {
         f();
      }
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
   }

   /// @brief Benchmark oneDPL inclusive_scan vs adn::cumsum on N randoms.
   template <typename T> Result run(size_t N) {
      std::mt19937 gen(7);
      std::uniform_real_distribution<T> dist(T(-1e6), T(1e6));
      std::vector<T> input(N);
      for (T &x : input) {
         x = dist(gen);
      }

      auto device_input = adn::detail::allocate_device_usm<T>(N, queue());
      auto device_output = adn::detail::allocate_device_usm<T>(N, queue());
      queue()
         .memcpy(device_input.get(), input.data(), N * sizeof(T))
         .wait_and_throw();
      auto policy =
         oneapi::dpl::execution::make_device_policy<ADNCumsumBaselinePolicy<T>>(
            queue());

      const double baseline_ms = time_ms([&] {
         oneapi::dpl::inclusive_scan(policy, device_input.get(),
            device_input.get() + N, device_output.get(), std::plus<T>{});
         queue().wait_and_throw();
      });
      T baseline_last;
      queue()
         .memcpy(&baseline_last, device_output.get() + N - 1, sizeof(T))
         .wait_and_throw();

      const double binned_ms = time_ms([&] {
         adn::cumsum(queue(), device_input.get(), device_output.get(), N);
      });
      T binned_last;
      queue()
         .memcpy(&binned_last, device_output.get() + N - 1, sizeof(T))
         .wait_and_throw();
      EXPECT_TRUE(std::isfinite(baseline_last));
      EXPECT_TRUE(std::isfinite(binned_last));

      const double billions = static_cast<double>(N) / 1e9;
      return {billions / (baseline_ms / 1e3), billions / (binned_ms / 1e3)};
   }

   template <typename T> void report(const char *type_name, size_t N) {
      const Result result = run<T>(N);
      const double slowdown = result.baseline_gelem_s / result.binned_gelem_s;
      RecordProperty("oneDPL_GElemPerSec", result.baseline_gelem_s);
      RecordProperty("binned_GElemPerSec", result.binned_gelem_s);
      std::printf("  [bench] %s: %-6s N=%zu  oneDPL scan %7.3f G "
                  "elements/s | adn::cumsum %7.3f G elements/s "
                  "(%.1fx slower)\n",
         GetParam().get_info<sycl::info::device::name>().c_str(), type_name, N,
         result.baseline_gelem_s, result.binned_gelem_s, slowdown);
   }
};

TEST_P(ADNCumsumBench, Throughput_Double_100M) {
   report<double>("double", 100000000);
}

TEST_P(ADNCumsumBench, Throughput_Float_100M) {
   report<float>("float", 100000000);
}

INSTANTIATE_TEST_SUITE_P(
   GPUs, ADNCumsumBench, ::testing::ValuesIn(all_gpus()), device_label);

INSTANTIATE_TEST_SUITE_P(
   CPUs, ADNCumsumBench, ::testing::ValuesIn(all_cpus()), device_label);

// ============================================================
//  Version macro sanity
// ============================================================

TEST(Version, MacroEncoding) {
   EXPECT_EQ(SYCL_REPRO_SUM_VERSION,
      SYCL_REPRO_SUM_VERSION_MAJOR * 10000 +
         SYCL_REPRO_SUM_VERSION_MINOR * 100 + SYCL_REPRO_SUM_VERSION_PATCH);
   EXPECT_GE(SYCL_REPRO_SUM_VERSION, 10100); // at least 1.1.0
   EXPECT_EQ(adn::max_reproducible_count<float>, UINT64_C(4294966784));
   EXPECT_EQ(
      adn::max_reproducible_count<double>, UINT64_C(9223372036854773760));
}

// ============================================================
//  Cross-device / cross-backend bit-identity
//
//  The strongest reproducibility guarantee: the same data must
//  produce a bit-identical sum on every selected GPU and CPU,
//  even when each device sees the input in a different order.
//  Skipped when fewer than two distinct devices are present.
// ============================================================

class ADNSumCrossDevice : public ::testing::Test {
protected:
   /// @brief Sum @p v on every selected device, shuffle differently per
   ///        device, and require bit-identical results across all of them.
   template <typename T> static void expect_identical(std::vector<T> v) {
      auto &devices = all_devices();
      if (devices.size() < 2) {
         GTEST_SKIP() << "fewer than two distinct devices available";
      }

      bool have_ref = false;
      T ref{};
      std::string ref_name;
      for (size_t i = 0; i < devices.size(); ++i) {
         const sycl::device &dev = devices[i];
         if (std::is_same_v<T, double> && !dev.has(sycl::aspect::fp64)) {
            continue;
         }
         sycl::queue q(dev);
         std::mt19937 rng(static_cast<unsigned>(100 + i));
         std::shuffle(v.begin(), v.end(), rng);
         T r = adn::sum(q, v.data(), v.size());
         if (!have_ref) {
            ref = r;
            ref_name = dev.get_info<sycl::info::device::name>();
            have_ref = true;
         } else {
            EXPECT_EQ(std::memcmp(&r, &ref, sizeof(T)), 0)
               << dev.get_info<sycl::info::device::name>() << " produced " << r
               << " but " << ref_name << " produced " << ref;
         }
      }
   }

   /// @brief Run cumsum with different work-group sizes on every selected
   ///        device and compare the complete output arrays bit-for-bit.
   template <typename T>
   static void expect_cumsum_identical(const std::vector<T> &input) {
      auto &devices = all_devices();
      if (devices.size() < 2) {
         GTEST_SKIP() << "fewer than two distinct devices available";
      }

      size_t validated_devices = 0;
      std::vector<T> reference;
      std::string reference_name;
      for (size_t device_index = 0; device_index < devices.size();
           ++device_index) {
         const sycl::device &device = devices[device_index];
         if (!device.has(sycl::aspect::fp64)) {
            continue;
         }

         sycl::queue q(device);
         std::vector<T> output(input.size());
         if (device_index % 2 == 0) {
            adn::cumsum<3, 64>(q, input.data(), output.data(), input.size());
         } else {
            adn::cumsum<3, 256>(q, input.data(), output.data(), input.size());
         }
         EXPECT_BIT_EQ(output.back(), adn::sum(q, input.data(), input.size()));
         ++validated_devices;

         if (reference.empty()) {
            reference = output;
            reference_name = device.get_info<sycl::info::device::name>();
            continue;
         }

         const std::string device_name =
            device.get_info<sycl::info::device::name>();
         expect_bit_identical_arrays(
            output, reference, device_name + " vs " + reference_name);
      }

      if (validated_devices < 2) {
         GTEST_SKIP() << "fewer than two fp64-capable devices available";
      }
   }
};

TEST_F(ADNSumCrossDevice, Double_WideRange) {
   std::mt19937 gen(1);
   std::uniform_real_distribution<double> dist(-1e12, 1e12);
   std::vector<double> v(1000000);
   for (auto &x : v) {
      x = dist(gen);
   }
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, Float_WideRange) {
   std::mt19937 gen(2);
   std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
   std::vector<float> v(1000000);
   for (auto &x : v) {
      x = dist(gen);
   }
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, Double_CancellationWithSignal) {
   std::vector<double> v;
   for (int i = 0; i < 500; ++i) {
      v.push_back(1e16);
      v.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      v.push_back(1.0);
   }
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, Double_TinyNormals) {
   std::mt19937 gen(3);
   std::uniform_real_distribution<double> dist(1.0, 2.0);
   std::vector<double> v(100000);
   for (auto &x : v) {
      x = std::ldexp(dist(gen), -1000);
   }
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, Double_NaNPayloads) {
   std::vector<double> v(4099, 1.0);
   v[17] = double_from_bits(UINT64_C(0x7ff8000000000001));
   v[1025] = double_from_bits(UINT64_C(0xfff8000000001234));
   v[3073] = double_from_bits(UINT64_C(0x7ff0000000000001));
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, Float_NaNPayloads) {
   std::vector<float> v(4099, 1.0f);
   v[17] = float_from_bits(UINT32_C(0x7fc00001));
   v[1025] = float_from_bits(UINT32_C(0xffc01234));
   v[3073] = float_from_bits(UINT32_C(0x7f800001));
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, OppositeInfinities) {
   std::vector<double> v(4099, 1.0);
   v[17] = std::numeric_limits<double>::infinity();
   v[3073] = -std::numeric_limits<double>::infinity();
   expect_identical(v);
}

TEST_F(ADNSumCrossDevice, Cumsum_Double) {
   std::mt19937 gen(29);
   std::uniform_real_distribution<double> dist(-1e12, 1e12);
   std::vector<double> input(16387);
   for (double &x : input) {
      x = dist(gen);
   }
   input[17] = 1e16;
   input[18] = -1e16;
   input[2047] = std::numeric_limits<double>::denorm_min();
   input[2048] = -std::numeric_limits<double>::denorm_min();
   expect_cumsum_identical(input);
}

TEST_F(ADNSumCrossDevice, Cumsum_Float) {
   std::mt19937 gen(31);
   std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
   std::vector<float> input(16387);
   for (float &x : input) {
      x = dist(gen);
   }
   input[7] = 1e30f;
   input[8] = -1e30f;
   input[2047] = std::numeric_limits<float>::denorm_min();
   input[2048] = -std::numeric_limits<float>::denorm_min();
   expect_cumsum_identical(input);
}

TEST_F(ADNSumCrossDevice, Cumsum_ShuffledCancellationFullArray) {
   constexpr size_t N = 2048;
   constexpr int RUNS = 100;
   const auto &devices = all_devices();
   const size_t supported_devices = static_cast<size_t>(std::count_if(
      devices.begin(), devices.end(), [](const sycl::device &device) {
      return device.has(sycl::aspect::fp64);
   }));
   if (supported_devices < 2) {
      GTEST_SKIP() << "fewer than two fp64-capable devices available";
   }

   std::vector<double> input;
   input.reserve(N);
   for (size_t i = 0; i < 682; ++i) {
      input.push_back(1e16);
      input.push_back(-1e16);
      input.push_back(1.0);
   }
   input.push_back(1.0);
   input.push_back(1.0);
   ASSERT_EQ(input.size(), N);

   // Keep the permutation identical across standard-library implementations.
   std::mt19937 rng(3);
   for (size_t i = input.size() - 1; i > 0; --i) {
      const size_t j = static_cast<std::uint64_t>(rng()) % (i + 1);
      std::swap(input[i], input[j]);
   }

   std::vector<double> cross_device_reference;
   std::string reference_name;
   for (const sycl::device &device : devices) {
      if (!device.has(sycl::aspect::fp64)) {
         continue;
      }

      sycl::queue q(device);
      const std::string device_name =
         device.get_info<sycl::info::device::name>();
      auto device_input = adn::detail::allocate_device_usm<double>(N, q);
      auto device_output = adn::detail::allocate_device_usm<double>(N, q);
      q.memcpy(device_input.get(), input.data(), N * sizeof(double))
         .wait_and_throw();

      adn::cumsum(q, device_input.get(), device_output.get(), N);
      std::vector<double> device_reference(N);
      q.memcpy(device_reference.data(), device_output.get(), N * sizeof(double))
         .wait_and_throw();
      EXPECT_BIT_EQ(device_reference.back(), 684.0) << device_name;
      EXPECT_BIT_EQ(device_reference.back(), adn::sum(q, device_input.get(), N))
         << device_name;

      if (cross_device_reference.empty()) {
         cross_device_reference = device_reference;
         reference_name = device_name;
      } else {
         expect_bit_identical_arrays(device_reference, cross_device_reference,
            device_name + " vs " + reference_name);
      }

      std::vector<double> actual(N);
      for (int run = 0; run < RUNS; ++run) {
         switch (run % 4) {
            case 0:
               adn::cumsum<3, 64>(
                  q, device_input.get(), device_output.get(), N);
               break;
            case 1:
               adn::cumsum<3, 128>(
                  q, device_input.get(), device_output.get(), N);
               break;
            case 2:
               adn::cumsum<3, 256>(
                  q, device_input.get(), device_output.get(), N);
               break;
            default:
               adn::cumsum<3, 512>(
                  q, device_input.get(), device_output.get(), N);
               break;
         }
         q.memcpy(actual.data(), device_output.get(), N * sizeof(double))
            .wait_and_throw();
         expect_bit_identical_arrays(actual, device_reference,
            device_name + ", run " + std::to_string(run));
      }
   }
}
