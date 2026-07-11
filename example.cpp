// Copyright (c) 2026, Lilo Huang <kuso.cc@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file example.cpp
 * @brief Minimal usage example for adn::sum.
 *
 * Sums data that defeats naive floating-point summation (large values
 * cancelling around a small signal) and shows that the result is
 * bit-identical no matter how the input is ordered.
 */

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "repro_sum.hpp"

int main() {
   sycl::queue q{sycl::gpu_selector_v};
   std::printf("Device: %s\n",
      q.get_device().get_info<sycl::info::device::name>().c_str());

   // 1e16 +/- pairs cancel exactly; the true sum is the 10000 ones.
   std::vector<double> data;
   for (int i = 0; i < 5; ++i) {
      data.push_back(1e16);
      data.push_back(-1e16);
   }
   for (int i = 0; i < 10000; ++i) {
      data.push_back(1.0);
   }

   // Host pointer overload (copies to device internally):
   double result = adn::sum(q, data.data(), data.size());
   std::printf("adn::sum       = %.17g (expected 10000)\n", result);

   // Same data, shuffled - the reproducible sum is bit-identical:
   std::mt19937 rng(42);
   std::shuffle(data.begin(), data.end(), rng);
   double shuffled = adn::sum(q, data.data(), data.size());
   std::printf("shuffled input = %.17g (%s)\n", shuffled,
      shuffled == result ? "bit-identical" : "MISMATCH");

   // Device-pointer overload with more folds for higher accuracy:
   double *d_arr = sycl::malloc_device<double>(data.size(), q);
   q.memcpy(d_arr, data.data(), data.size() * sizeof(double)).wait();
   double k6 = adn::sum<6>(q, d_arr, data.size());
   std::printf("K=6 folds      = %.17g\n", k6);
   sycl::free(d_arr, q);

   return shuffled == result ? 0 : 1;
}
