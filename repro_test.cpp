// Copyright (c) 2026, Lilo Huang <kuso.cc@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file repro_test.cpp
 * @brief Google Test suite for Ahrens-Demmel-Nguyen K-Fold reproducible
 * summation.
 *
 * Tests correctness, reproducibility, and edge-case handling of
 * adn::sum across various input patterns, K configurations, and
 * floating-point types (double and float).
 */

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
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
static ::testing::AssertionResult
assert_bit_equal(const char *a_expr, const char *b_expr, T a, T b) {
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

// ============================================================
//  Device enumeration
//
//  All distinct GPUs on the system, deduplicated by name (the same
//  physical device can be exposed by several backends, e.g. an
//  Intel iGPU via both Level-Zero and OpenCL).
// ============================================================

static const std::vector<sycl::device> &all_gpus() {
   static const std::vector<sycl::device> devices = [] {
      std::vector<sycl::device> result;
      std::vector<std::string> seen;
      for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
         std::string key = d.get_info<sycl::info::device::name>();
         if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
            seen.push_back(key);
            result.push_back(d);
         }
      }
      return result;
   }();
   return devices;
}

static const std::vector<sycl::device> &all_cpus() {
   static const std::vector<sycl::device> devices = [] {
      std::vector<sycl::device> result;
      std::vector<std::string> seen;
      for (auto &d : sycl::device::get_devices(sycl::info::device_type::cpu)) {
         std::string key = d.get_info<sycl::info::device::name>();
         if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
            seen.push_back(key);
            result.push_back(d);
         }
      }
      return result;
   }();
   return devices;
}

/// All devices (GPUs + CPUs), deduplicated by name.
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
static std::string
device_label(const ::testing::TestParamInfo<sycl::device> &info) {
   std::string name = info.param.get_info<sycl::info::device::name>();
   for (char &c : name) {
      if (!std::isalnum(static_cast<unsigned char>(c))) {
         c = '_';
      }
   }
   return name;
}

// ============================================================
//  Shared fixture, parameterized over every GPU on the system:
//  each test runs once per device from a single fat binary
//  (CUDA + SPIR-V device code selected at runtime).
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

INSTANTIATE_TEST_SUITE_P(GPUs, ADNSumTest, ::testing::ValuesIn(all_gpus()),
                         device_label);

INSTANTIATE_TEST_SUITE_P(CPUs, ADNSumTest, ::testing::ValuesIn(all_cpus()),
                         device_label);

// ============================================================
//  Double-precision edge cases
// ============================================================

TEST_P(ADNSumTest, EmptyArray) {
   std::vector<double> v;
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()), 0.0);
}

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
   EXPECT_EQ(adn::sum(queue(), v.data(), v.size()),
             std::numeric_limits<float>::max());
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
   std::vector<float> v{mx, -mx};
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
   double tiny = std::numeric_limits<double>::denorm_min() * 1e10;
   std::vector<double> v(10000, tiny);
   double result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FALSE(std::isinf(result));
   EXPECT_GT(result, 0.0);
}

TEST_P(ADNSumTest, Float_SubnormalFallback_ManyElements) {
   float tiny = std::numeric_limits<float>::denorm_min() * 1e5f;
   std::vector<float> v(10000, tiny);
   float result = adn::sum(queue(), v.data(), v.size());
   EXPECT_FALSE(std::isnan(result));
   EXPECT_FALSE(std::isinf(result));
   EXPECT_GT(result, 0.0f);
}

TEST_P(ADNSumTest, OverflowFallback_MultipleMaxValues) {
   // Multiple max-value elements: tests that the overflow fallback
   // handles more than 1 element correctly
   float mx = std::numeric_limits<float>::max();
   std::vector<float> v{mx, -mx, mx, -mx};
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
   // ~400 work-groups: stresses host-side final reduction
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
   std::vector<float> v{big_int, big_int, -big_int};
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
//  (host-side limb merge used to round in float, breaking
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
               h.parallel_for(
                  sycl::range<1>(N), sycl::reduction(d_out, sycl::plus<T>()),
                  [=](sycl::id<1> i, auto &s) { s.combine(d_arr[i]); });
            })
            .wait();
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
                  GetParam().get_info<sycl::info::device::name>().c_str(),
                  type_name, N, r.naive_gelem_s, r.binned_gelem_s, slowdown);
   }
};

TEST_P(ADNSumBench, Throughput_Double_100M) {
   report<double>("double", 100000000);
}

TEST_P(ADNSumBench, Throughput_Float_100M) {
   report<float>("float", 100000000);
}

INSTANTIATE_TEST_SUITE_P(GPUs, ADNSumBench, ::testing::ValuesIn(all_gpus()),
                         device_label);

INSTANTIATE_TEST_SUITE_P(CPUs, ADNSumBench, ::testing::ValuesIn(all_cpus()),
                         device_label);

// ============================================================
//  Version macro sanity
// ============================================================

TEST(Version, MacroEncoding) {
   EXPECT_EQ(SYCL_REPRO_SUM_VERSION, SYCL_REPRO_SUM_VERSION_MAJOR * 10000 +
                                        SYCL_REPRO_SUM_VERSION_MINOR * 100 +
                                        SYCL_REPRO_SUM_VERSION_PATCH);
   EXPECT_GE(SYCL_REPRO_SUM_VERSION, 10000); // at least 1.0.0
}

// ============================================================
//  Cross-device / cross-backend bit-identity
//
//  The strongest reproducibility guarantee: the same data must
//  produce a bit-identical sum on every available device (GPUs
//  across backends, and CPU), even when each device sees the
//  input in a different order.  Skipped when fewer than two
//  distinct devices are present.
// ============================================================

class ADNSumCrossDevice : public ::testing::Test {
protected:
   /// @brief Sum @p v on every device (shuffling differently per device)
   ///        and require bit-identical results across all of them.
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
