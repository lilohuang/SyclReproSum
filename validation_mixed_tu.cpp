// Copyright (c) 2026, Lilo Huang <kuso.cc@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "repro_sum.hpp"

#if defined(ADN_VALIDATION_STRICT_TU)

void strict_environment_preflight(sycl::queue &q) {
   adn::validate_environment<float>(q);
}

#else

#include <cstdio>
#include <cstring>
#include <exception>

void strict_environment_preflight(sycl::queue &q);

static bool is_environment_rejection(const std::runtime_error &error) {
   constexpr const char prefix[] = "adn::validate_environment:";
   return std::strncmp(error.what(), prefix, sizeof(prefix) - 1) == 0;
}

int main() {
   size_t checked = 0;
   size_t failures = 0;

   for (const sycl::platform &platform : sycl::platform::get_platforms()) {
      for (const sycl::device &device : platform.get_devices()) {
         sycl::queue q(device);
         try {
            strict_environment_preflight(q);
         } catch (const std::runtime_error &) {
            continue;
         }

         ++checked;
         float *input = sycl::malloc_shared<float>(1, q);
         if (input == nullptr) {
            std::fprintf(stderr, "Unable to allocate shared USM on %s\n",
               device.get_info<sycl::info::device::name>().c_str());
            ++failures;
            continue;
         }
         input[0] = 1.0f;

         try {
            (void)adn::sum<6, 64>(q, input, 1);
            std::fprintf(stderr,
               "Hostile translation unit was not rejected on %s\n",
               device.get_info<sycl::info::device::name>().c_str());
            ++failures;
         } catch (const std::runtime_error &error) {
            if (!is_environment_rejection(error)) {
               std::fprintf(stderr, "Unexpected rejection on %s: %s\n",
                  device.get_info<sycl::info::device::name>().c_str(),
                  error.what());
               ++failures;
            }
         } catch (const std::exception &error) {
            std::fprintf(stderr, "Unexpected exception on %s: %s\n",
               device.get_info<sycl::info::device::name>().c_str(),
               error.what());
            ++failures;
         }
         sycl::free(input, q);
      }
   }

   if (checked == 0) {
      std::fprintf(stderr, "No device passed the strict preflight\n");
      return 1;
   }
   return failures == 0 ? 0 : 1;
}

#endif
