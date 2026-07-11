// Copyright (c) 2026, Lilo Huang <kuso.cc@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause
//
// Portions of the binned accumulator primitives are adapted from ReproBLAS.
// Copyright (c) 2016, University of California.
// See THIRD_PARTY_NOTICES for complete attribution and license terms.

#pragma once
/**
 * @file repro_sum.hpp
 * @brief Single-pass reproducible floating-point summation for SYCL GPU,
 *        based on the Ahrens-Demmel-Nguyen (ADN) binned ("indexed")
 *        floating-point format as implemented by ReproBLAS.
 *
 * Each scalar is deposited into a K-fold *binned accumulator*: K primary
 * values aligned to a fixed, global grid of exponent bins (bin width
 * W = 40 bits for double, 13 bits for float), plus K carry counters.
 * Because every deposit rounds on the same absolute grid, sums within the
 * documented accumulator capacity are independent of summation order -
 * bit-reproducible across runs, work-group sizes, and devices - without ever
 * scanning for max|x| first.
 *
 * Compared to a two-pass max-then-split scheme, this needs only ONE pass
 * over the data.  Special values need no fallback path: Inf/NaN propagate
 * through the accumulator with a fixed canonical quiet NaN, and subnormals
 * deposit into the bottom bins (values below the format's absorption threshold,
 * ~2^-1055 for double / ~2^-144 for float with K=3, round away --
 * deterministically - per the documented ADN error bound).
 *
 * @par Usage
 * @code
 *   #include "repro_sum.hpp"
 *
 *   // Double precision (default K=3):
 *   double result = adn::sum(queue, d_ptr, N);
 *
 *   // Float precision - T deduced from pointer:
 *   float result = adn::sum(queue, f_ptr, N);
 *
 *   // Host pointer (automatic memcpy to device):
 *   double result = adn::sum(queue, host_ptr, N);
 *
 *   // More folds for higher accuracy:
 *   double result = adn::sum<6>(queue, d_ptr, N);
 *
 *   // Custom work-group size:
 *   double result = adn::sum<3, 512>(queue, d_ptr, N);
 * @endcode
 *
 * @tparam K       Fold count (number of bins held, default 3).
 * @tparam WG_SIZE Work-group size (default 256).
 * @tparam T       Floating-point type (float or double), deduced.
 *
 * @par References
 *  - Ahrens, Demmel, Nguyen. "Algorithms for Efficient Reproducible
 *    Floating Point Summation." ACM TOMS, 2020.
 *  - ReproBLAS (https://bebop.cs.berkeley.edu/reproblas/) - the binned
 *    primitive routines here are SYCL adaptations of its binned_dm /
 *    binned_sm reference implementations.
 *    Copyright (c) 2016, University of California.  Used under the
 *    ReproBLAS Software Development License (BSD-3-Clause style); see
 *    THIRD_PARTY_NOTICES for the full license text.
 */

/// @name Library version
/// SYCL_REPRO_SUM_VERSION encodes MAJOR * 10000 + MINOR * 100 + PATCH,
/// so version checks can be done with a single comparison:
/// @code
///   #if SYCL_REPRO_SUM_VERSION >= 10200  // require >= 1.2.0
/// @endcode
/// @{
#define SYCL_REPRO_SUM_VERSION_MAJOR 1
#define SYCL_REPRO_SUM_VERSION_MINOR 0
#define SYCL_REPRO_SUM_VERSION_PATCH 0
#define SYCL_REPRO_SUM_VERSION                                                 \
   (SYCL_REPRO_SUM_VERSION_MAJOR * 10000 +                                     \
      SYCL_REPRO_SUM_VERSION_MINOR * 100 + SYCL_REPRO_SUM_VERSION_PATCH)
/// @}

// === Strict IEEE 754 enforcement ==========================================
// Fast-math permits the compiler to discard NaN, infinity, subnormal, signed
// zero, and rounding-order semantics required by the binned accumulator.  Its
// individual assumptions cannot be reliably undone by local pragmas, so reject
// it instead of silently returning non-reproducible results.
#if defined(__FAST_MATH__) ||                                                  \
   (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0) ||              \
   (defined(_MSC_VER) && defined(_M_FP_FAST))
#error "SyclReproSum requires strict IEEE 754 semantics; disable fast-math"
#endif

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__SYCL_DEVICE_ONLY__)
static_assert(FLT_EVAL_METHOD == 0,
   "device floating-point expressions must evaluate in their "
   "declared type");
#endif

// Prevent contraction, reassociation, and reciprocal transformations in the
// arithmetic primitives.  Runtime device probes reject unsafe modes that
// cannot be overridden by a local pragma.
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#define SYCL_REPRO_SUM_DETAIL_STRICT_FP                                        \
   _Pragma("clang fp reassociate(off) reciprocal(off) contract(off)")
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math")
#define SYCL_REPRO_SUM_DETAIL_STRICT_FP
#elif defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#define SYCL_REPRO_SUM_DETAIL_STRICT_FP
#else
#define SYCL_REPRO_SUM_DETAIL_STRICT_FP
#endif
// ==========================================================================

namespace adn {

namespace detail {

/**
 * @brief Check whether an integer is a power of two.
 * @param n The integer to check.
 * @return true if @p n is a positive power of two.
 */
constexpr bool is_power_of_2(int n) {
   return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Verify the binary floating-point representation at compile time.
 */
template <typename T> constexpr void validate_fp_type() {
   constexpr bool supported =
      std::is_same_v<T, float> || std::is_same_v<T, double>;
   static_assert(supported, "T must be float or double");

   if constexpr (supported) {
      static_assert(std::numeric_limits<T>::is_iec559,
         "T must implement IEC 60559 (IEEE 754) semantics");
      static_assert(
         std::numeric_limits<T>::radix == 2, "T must use a binary radix");
      static_assert(std::numeric_limits<T>::has_denorm == std::denorm_present,
         "T must support gradual underflow");

      if constexpr (std::is_same_v<T, float>) {
         static_assert(sizeof(T) == 4 && std::numeric_limits<T>::digits == 24 &&
               std::numeric_limits<T>::min_exponent == -125 &&
               std::numeric_limits<T>::max_exponent == 128,
            "float must use the IEEE 754 binary32 format");
      } else {
         static_assert(sizeof(T) == 8 && std::numeric_limits<T>::digits == 53 &&
               std::numeric_limits<T>::min_exponent == -1021 &&
               std::numeric_limits<T>::max_exponent == 1024,
            "double must use the IEEE 754 binary64 format");
      }
   }
}

/**
 * @brief IEEE-754 layout traits and ADN bin parameters per type.
 *
 * Everything is derived from std::numeric_limits except bin_width,
 * which follows ReproBLAS: 40 bits for double, 13 for float.
 */
template <typename T> struct fp {
   using bits_t =
      std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;

   static constexpr int mant_dig = std::numeric_limits<T>::digits;
   static constexpr int max_exp = std::numeric_limits<T>::max_exponent;
   static constexpr int min_exp = std::numeric_limits<T>::min_exponent;

   /// ReproBLAS DBWIDTH / SBWIDTH.
   static constexpr int bin_width = std::is_same_v<T, double> ? 40 : 13;

   /// Exponent-field bias used by the index arithmetic (EXP_BIAS).
   static constexpr int exp_bias = max_exp - 2;

   /// Largest valid bin index (binned_DBMAXINDEX / binned_SBMAXINDEX).
   static constexpr int max_index =
      ((max_exp - min_exp + mant_dig - 1) / bin_width) - 1;

   /// Largest supported fold count (K).
   static constexpr int max_fold = max_index + 1;

   /// Deposits allowed between renormalizations (*ENDURANCE).
   static constexpr int endurance = 1 << (mant_dig - bin_width - 2);

   /// Maximum supported summand count (ReproBLAS *CAPACITY).
   static constexpr std::uint64_t capacity = std::uint64_t(endurance) *
      ((std::uint64_t(1) << (mant_dig - 1)) - std::uint64_t(1));

   /// Scale-down factor for deposits into bin 0 (*COMPRESSION).
   static constexpr T compression =
      T(1.0 / (bits_t(1) << (mant_dig - bin_width + 1)));

   /// Inverse of compression (*EXPANSION).
   static constexpr T expansion =
      T(1.0 * (bits_t(1) << (mant_dig - bin_width + 1)));
};

/// @return The raw IEEE-754 bit pattern of @p x.
template <typename T> inline typename fp<T>::bits_t to_bits(T x) {
   return sycl::bit_cast<typename fp<T>::bits_t>(x);
}

/// @return The value whose IEEE-754 bit pattern is @p b.
template <typename T> inline T from_bits(typename fp<T>::bits_t b) {
   return sycl::bit_cast<T>(b);
}

/// @return The biased exponent field of @p x (0 for zero/subnormal).
template <typename T> inline int exp_field(T x) {
   return static_cast<int>(
      (to_bits(x) >> (fp<T>::mant_dig - 1)) & (2u * fp<T>::max_exp - 1));
}

/// @return true if @p x is NaN or +/-Inf.
template <typename T> inline bool is_nan_inf(T x) {
   return exp_field(x) == 2 * fp<T>::max_exp - 1;
}

/// @return true if @p x is NaN.
template <typename T> inline bool is_nan(T x) {
   using B = typename fp<T>::bits_t;
   constexpr B fraction_mask = (B(1) << (fp<T>::mant_dig - 1)) - 1;
   return is_nan_inf(x) && (to_bits(x) & fraction_mask) != 0;
}

/// @return true if @p x is +/-Inf.
template <typename T> inline bool is_inf(T x) {
   using B = typename fp<T>::bits_t;
   constexpr B fraction_mask = (B(1) << (fp<T>::mant_dig - 1)) - 1;
   return is_nan_inf(x) && (to_bits(x) & fraction_mask) == 0;
}

/// @return A fixed positive quiet NaN, independent of device conventions.
template <typename T> inline T canonical_nan() {
   using B = typename fp<T>::bits_t;
   constexpr B exponent = B(2 * fp<T>::max_exp - 1) << (fp<T>::mant_dig - 1);
   constexpr B quiet_bit = B(1) << (fp<T>::mant_dig - 2);
   return from_bits<T>(exponent | quiet_bit);
}

/**
 * @brief Reproducibly combine values when at least one is NaN or Inf.
 *
 * NaN is absorbing and canonicalized.  Equal infinities remain unchanged;
 * opposite infinities produce the canonical NaN.  This operation is
 * commutative and associative, so its result is independent of input order,
 * reduction-tree shape, and device-specific NaN propagation rules.
 */
template <typename T> inline T combine_special(T x, T y) {
   if (is_nan(x) || is_nan(y)) {
      return canonical_nan<T>();
   }
   if (is_inf(x) && is_inf(y)) {
      using B = typename fp<T>::bits_t;
      constexpr B sign_mask = B(1) << (sizeof(T) * 8 - 1);
      if (((to_bits(x) ^ to_bits(y)) & sign_mask) != 0) {
         return canonical_nan<T>();
      }
      return x;
   }
   return is_inf(x) ? x : y;
}

/**
 * @brief Reference value of exponent bin @p index (ReproBLAS binned_*mbins).
 *
 * bin(0)   = 0.75 * 2^max_exp
 * bin(i)   = 0.75 * 2^(max_exp + mant_dig - W + 1 - i*W)  for 1 <= i <=
 * max_index Indices beyond max_index clamp to bin(max_index).
 *
 * Constructed directly from bits (0.75 * 2^q == 1.1b * 2^(q-1)), so it is
 * usable in both host and device code with no math-library calls.
 */
template <typename T> inline T bin_value(int index) {
   using B = typename fp<T>::bits_t;
   if (index > fp<T>::max_index) {
      index = fp<T>::max_index;
   }
   const int q = (index == 0) ? fp<T>::max_exp
                              : fp<T>::max_exp + fp<T>::mant_dig -
         fp<T>::bin_width + 1 - index * fp<T>::bin_width;
   const B b = (B(q + fp<T>::exp_bias) << (fp<T>::mant_dig - 1)) |
      (B(1) << (fp<T>::mant_dig - 2));
   return from_bits<T>(b);
}

/**
 * @brief Bin index a scalar would need to be summed reproducibly
 *        (ReproBLAS binned_dindex / binned_sindex).
 *
 * Higher indices correspond to smaller bins.  Subnormals are normalized
 * with an exact power-of-two multiply before the exponent is read.
 */
template <typename T> inline int scalar_index(T x) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   int e = exp_field(x);
   if (e == 0) {
      if (x == T(0)) {
         return fp<T>::max_index;
      }
      constexpr T norm = T(typename fp<T>::bits_t(1) << fp<T>::mant_dig);
      e = exp_field(x * norm) - fp<T>::mant_dig;
   }
   const int r = ((fp<T>::max_exp + fp<T>::exp_bias) - e) / fp<T>::bin_width;
   return r < fp<T>::max_index ? r : fp<T>::max_index;
}

/**
 * @brief K-fold binned accumulator: K primary values + K carry counters.
 *
 * Zero-initialize with `Binned<T,K> acc{};`.  pri[0] == 0 denotes the
 * empty accumulator.
 */
template <typename T, int K> struct Binned {
   T pri[K];
   T car[K];
};

/// Bin index the accumulator currently holds (ReproBLAS binned_dmindex).
template <typename T, int K> inline int accum_index(const Binned<T, K> &a) {
   const int top_exp =
      fp<T>::max_exp + fp<T>::mant_dig - fp<T>::bin_width + 1 + fp<T>::exp_bias;
   return (top_exp - exp_field(a.pri[0])) / fp<T>::bin_width;
}

/// @return true if the accumulator's index is 0 (ReproBLAS binned_dmindex0).
template <typename T, int K> inline bool accum_index0(const Binned<T, K> &a) {
   return exp_field(a.pri[0]) == fp<T>::max_exp + fp<T>::exp_bias;
}

/**
 * @brief Rebin the accumulator so scalars with |v| <= |x| can be deposited
 *        (ReproBLAS binned_dmdupdate).
 *
 * Must be called on a renormalized (or empty) accumulator.
 */
template <typename T, int K> inline void update(Binned<T, K> &a, T x) {
   if (is_nan_inf(a.pri[0])) {
      return;
   }

   const int xi = scalar_index(x);
   if (a.pri[0] == T(0)) {
      for (int i = 0; i < K; ++i) {
         a.pri[i] = bin_value<T>(xi + i);
         a.car[i] = T(0);
      }
      return;
   }

   const int shift = accum_index(a) - xi;
   if (shift > 0) {
      const int fill = shift < K ? shift : K;
      for (int i = K - 1; i >= fill; --i) {
         a.pri[i] = a.pri[i - shift];
         a.car[i] = a.car[i - shift];
      }
      for (int j = 0; j < fill; ++j) {
         a.pri[j] = bin_value<T>(xi + j);
         a.car[j] = T(0);
      }
   }
}

/**
 * @brief Deposit scalar @p x into a suitably binned accumulator
 *        (ReproBLAS binned_dmddeposit).
 *
 * Requires accum_index(a) <= scalar_index(x) (ensured by @ref update) and
 * at most @ref endurance deposits since the last @ref renorm.  The sticky
 * low-mantissa bit (`|= 1`) makes the bin additions round identically
 * regardless of accumulation order.
 */
template <typename T, int K> inline void deposit(Binned<T, K> &a, T x) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   if (is_nan_inf(x) || is_nan_inf(a.pri[0])) {
      a.pri[0] = combine_special(a.pri[0], x);
      return;
   }

   if (accum_index0(a)) {
      // Bin 0 holds values near the overflow threshold; deposits are
      // compressed by 2^-(mant_dig - W + 1) to create headroom, and the
      // residual is re-expanded before flowing into the finer bins.
      T M = a.pri[0];
      T qd = from_bits<T>(to_bits(x * fp<T>::compression) | 1);
      qd += M;
      a.pri[0] = qd;
      M -= qd;
      M *= fp<T>::expansion * T(0.5);
      x += M;
      x += M;
      for (int i = 1; i < K - 1; ++i) {
         M = a.pri[i];
         qd = from_bits<T>(to_bits(x) | 1);
         qd += M;
         a.pri[i] = qd;
         M -= qd;
         x += M;
      }
      a.pri[K - 1] += from_bits<T>(to_bits(x) | 1);
   } else {
      for (int i = 0; i < K - 1; ++i) {
         T M = a.pri[i];
         T qd = from_bits<T>(to_bits(x) | 1);
         qd += M;
         a.pri[i] = qd;
         M -= qd;
         x += M;
      }
      a.pri[K - 1] += from_bits<T>(to_bits(x) | 1);
   }
}

/**
 * @brief Renormalize: shift primary-value drift into the carry counters
 *        (ReproBLAS binned_dmrenorm).
 *
 * Restores each primary to the canonical [1.5, 1.75) * 2^bin window so
 * another @ref endurance deposits can be absorbed.
 */
template <typename T, int K> inline void renorm(Binned<T, K> &a) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   using B = typename fp<T>::bits_t;
   if (a.pri[0] == T(0) || is_nan_inf(a.pri[0])) {
      return;
   }

   for (int i = 0; i < K; ++i) {
      B b = to_bits(a.pri[i]);
      a.car[i] += T(static_cast<int>((b >> (fp<T>::mant_dig - 3)) & 3) - 2);
      b &= ~(B(1) << (fp<T>::mant_dig - 3));
      b |= B(1) << (fp<T>::mant_dig - 2);
      a.pri[i] = from_bits<T>(b);
   }
}

/**
 * @brief Merge two binned accumulators: y += x (ReproBLAS binned_dmdmadd).
 *
 * Exact and order-independent while the total summand count is within the
 * accumulator capacity; used for both device tree-reduction stages.  Both
 * operands must be renormalized.
 */
template <typename T, int K>
inline void merge(Binned<T, K> &y, const Binned<T, K> &x) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   if (x.pri[0] == T(0)) {
      return;
   }
   if (y.pri[0] == T(0)) {
      y = x;
      return;
   }
   if (is_nan_inf(x.pri[0]) || is_nan_inf(y.pri[0])) {
      y.pri[0] = combine_special(y.pri[0], x.pri[0]);
      return;
   }

   const int xi = accum_index(x);
   const int yi = accum_index(y);
   const int shift = yi - xi;
   if (shift > 0) {
      // x has the coarser (smaller) index: realign y to x's bins.
      for (int i = K - 1; i >= shift; --i) {
         y.pri[i] =
            x.pri[i] + (y.pri[i - shift] - bin_value<T>(yi + i - shift));
         y.car[i] = x.car[i] + y.car[i - shift];
      }
      const int fill = shift < K ? shift : K;
      for (int i = 0; i < fill; ++i) {
         y.pri[i] = x.pri[i];
         y.car[i] = x.car[i];
      }
   } else {
      // y's index is already <= x's: fold x's bins into y.
      for (int i = -shift; i < K; ++i) {
         y.pri[i] += x.pri[i + shift] - bin_value<T>(xi + i + shift);
         y.car[i] += x.car[i + shift];
      }
   }

   renorm(y);
}

/**
 * @brief Convert a binned accumulator to its floating-point value
 *        (ReproBLAS binned_ddmconv / binned_ssmconv).
 *
 * Terms are summed in decreasing exponent order.  The double flavour
 * rescales near-overflow indices; the float flavour accumulates in
 * double, which provides the necessary headroom directly.
 */
template <typename T, int K> inline T conv(const Binned<T, K> &a) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   if (is_nan_inf(a.pri[0])) {
      return a.pri[0];
   }
   if (a.pri[0] == T(0)) {
      return T(0);
   }

   const int xi = accum_index(a);

   if constexpr (std::is_same_v<T, float>) {
      double Y = 0.0;
      int i;
      if (xi == 0) {
         Y += double(a.car[0]) * (double(bin_value<float>(0)) / 6.0) *
            double(fp<float>::expansion);
         Y += double(a.car[1]) * (double(bin_value<float>(1)) / 6.0);
         Y += double(a.pri[0] - bin_value<float>(0)) *
            double(fp<float>::expansion);
         i = 2;
      } else {
         Y += double(a.car[0]) * (double(bin_value<float>(xi)) / 6.0);
         i = 1;
      }
      for (; i < K; ++i) {
         Y += double(a.car[i]) * (double(bin_value<float>(xi + i)) / 6.0);
         Y += double(a.pri[i - 1] - bin_value<float>(xi + i - 1));
      }
      Y += double(a.pri[K - 1] - bin_value<float>(xi + K - 1));
      return static_cast<float>(Y);
   } else {
      constexpr int mant = fp<double>::mant_dig;
      constexpr int bw = fp<double>::bin_width;
      double Y = 0.0;
      int i = 0;
      if (xi <= (3 * mant) / bw) {
         // Indices near overflow: accumulate scaled down by 2^-(2m-W),
         // then scale back up (may correctly produce +/-Inf).
         using B = typename fp<double>::bits_t;
         constexpr int scale_exp = 2 * mant - bw;
         constexpr int ieee_bias = fp<double>::max_exp - 1;
         const double scale_down = from_bits<double>(
            B(ieee_bias - scale_exp) << (fp<double>::mant_dig - 1));
         const double scale_up = from_bits<double>(
            B(ieee_bias + scale_exp) << (fp<double>::mant_dig - 1));
         const int scaled = std::max(std::min(K, (3 * mant) / bw - xi), 0);
         if (xi == 0) {
            Y += a.car[0] *
               ((bin_value<double>(0) / 6.0) * scale_down *
                  fp<double>::expansion);
            Y += a.car[1] * ((bin_value<double>(1) / 6.0) * scale_down);
            Y += (a.pri[0] - bin_value<double>(0)) * scale_down *
               fp<double>::expansion;
            i = 2;
         } else {
            Y += a.car[0] * ((bin_value<double>(xi) / 6.0) * scale_down);
            i = 1;
         }
         for (; i < scaled; ++i) {
            Y += a.car[i] * ((bin_value<double>(xi + i) / 6.0) * scale_down);
            Y += (a.pri[i - 1] - bin_value<double>(xi + i - 1)) * scale_down;
         }
         if (i == K) {
            Y += (a.pri[K - 1] - bin_value<double>(xi + K - 1)) * scale_down;
            return Y * scale_up;
         }
         if (is_inf(Y * scale_up)) {
            return Y * scale_up;
         }
         Y *= scale_up;
         for (; i < K; ++i) {
            Y += a.car[i] * (bin_value<double>(xi + i) / 6.0);
            Y += a.pri[i - 1] - bin_value<double>(xi + i - 1);
         }
         Y += a.pri[K - 1] - bin_value<double>(xi + K - 1);
      } else {
         Y += a.car[0] * (bin_value<double>(xi) / 6.0);
         for (i = 1; i < K; ++i) {
            Y += a.car[i] * (bin_value<double>(xi + i) / 6.0);
            Y += a.pri[i - 1] - bin_value<double>(xi + i - 1);
         }
         Y += a.pri[K - 1] - bin_value<double>(xi + K - 1);
      }
      return Y;
   }
}

/**
 * @brief Per-element accumulation step for the device kernel.
 *
 * Deposits @p x into @p acc, rebinning when x's exponent field exceeds
 * the cached threshold and renormalizing every @ref fp::endurance
 * deposits.  The threshold caching keeps the hot path free of integer
 * divisions: scalar_index(x) < accum_index(acc) is equivalent to
 * exp_field(x) > e_threshold for the nonzero exponent fields tested
 * here, subnormals (exp field 0) map to the bottom bin and never force
 * a rebin, and the initial threshold of -1 makes the first element
 * initialize the empty accumulator.
 */
template <typename T, int K>
inline void accumulate(
   Binned<T, K> &acc, int &e_threshold, int &since_renorm, T x) {
   if (exp_field(x) > e_threshold) {
      renorm(acc);
      update(acc, x);
      e_threshold =
         fp<T>::max_exp + fp<T>::exp_bias - accum_index(acc) * fp<T>::bin_width;
      since_renorm = 0;
   }
   deposit(acc, x);
   if (++since_renorm >= fp<T>::endurance) {
      renorm(acc);
      since_renorm = 0;
   }
}

/**
 * @brief Single-pass input kernel plus device-side final reduction.
 *
 * A fixed grid of work-items strides over the input.  Each work-item
 * keeps a private binned accumulator in registers and deposits each
 * element via @ref accumulate (rebinning is division-free thanks to
 * threshold caching).  Work-group accumulators are combined with an
 * exact binned tree reduction.  A final device work-group merges the
 * per-group results and converts the total to T.
 *
 * @tparam T       Floating-point type.
 * @tparam K       Fold count.
 * @tparam WG_SIZE Work-group size.
 * @param  q       SYCL queue.
 * @param  d_arr   Device pointer to the input array.
 * @param  N       Number of elements.
 * @return         The reproducible sum.
 */
template <typename T, int K, int WG_SIZE>
T sum_impl(sycl::queue &q, const T *d_arr, size_t N) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   using Acc = Binned<T, K>;

   constexpr size_t MAX_GROUPS = 2048;
   const size_t groups_needed = (N + WG_SIZE - 1) / WG_SIZE;
   const size_t num_groups =
      groups_needed < MAX_GROUPS ? groups_needed : MAX_GROUPS;

   Acc *d_partial = sycl::malloc_shared<Acc>(num_groups, q);

   sycl::event partials_ready = q.submit([&](sycl::handler &h) {
      sycl::local_accessor<Acc, 1> loc(sycl::range<1>(WG_SIZE), h);

      auto accumulate_partial = [=](sycl::nd_item<1> item) {
         const size_t stride = num_groups * WG_SIZE;
         const int lid = item.get_local_id(0);

         Acc acc{};
         int e_threshold = -1;
         int since_renorm = 0;
         for (size_t i = item.get_global_id(0); i < N; i += stride) {
            accumulate(acc, e_threshold, since_renorm, d_arr[i]);
         }
         renorm(acc);

         loc[lid] = acc;
         sycl::group_barrier(item.get_group());

         for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
            if (lid < s) {
               Acc lhs = loc[lid];
               merge(lhs, loc[lid + s]);
               loc[lid] = lhs;
            }
            sycl::group_barrier(item.get_group());
         }

         if (lid == 0) {
            if (num_groups == 1) {
               d_partial[0].pri[0] = conv(loc[0]);
            } else {
               d_partial[item.get_group(0)] = loc[0];
            }
         }
      };

      h.parallel_for(
         sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE), accumulate_partial);
   });

   if (num_groups > 1) {
      sycl::event result_ready = q.submit([&](sycl::handler &h) {
         h.depends_on(partials_ready);
         sycl::local_accessor<Acc, 1> loc(sycl::range<1>(WG_SIZE), h);

         auto finalize = [=](sycl::nd_item<1> item) {
            const int lid = item.get_local_id(0);
            Acc acc{};
            for (size_t g = lid; g < num_groups; g += WG_SIZE) {
               merge(acc, d_partial[g]);
            }

            loc[lid] = acc;
            sycl::group_barrier(item.get_group());

            for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
               if (lid < s) {
                  Acc lhs = loc[lid];
                  merge(lhs, loc[lid + s]);
                  loc[lid] = lhs;
               }
               sycl::group_barrier(item.get_group());
            }

            if (lid == 0) {
               d_partial[0].pri[0] = conv(loc[0]);
            }
         };

         h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), finalize);
      });
      result_ready.wait_and_throw();
   } else {
      partials_ready.wait_and_throw();
   }

   const T result = from_bits<T>(to_bits(d_partial[0].pri[0]));
   sycl::free(d_partial, q);
   return result;
}

/**
 * @brief Runtime values and results for the device floating-point probe.
 */
template <typename T> struct EnvironmentProbeData {
   T min_normal;
   T half;
   T denorm_min;
   T two;
   T one;
   T epsilon;
   T quarter;
   T three_quarters;
   T negative_zero;
   T compiled_half_min_normal;
   T half_min_normal;
   T twice_denorm_min;
   T below_half_ulp;
   T above_half_ulp;
   T preserved_negative_zero;
   T negative_zero_plus_zero;
};

template <typename T> class EnvironmentProbeKernel;

inline bool has_fp_config(const std::vector<sycl::info::fp_config> &config,
   sycl::info::fp_config value) {
   return std::find(config.begin(), config.end(), value) != config.end();
}

/**
 * @brief Check the floating-point capabilities advertised by a SYCL device.
 */
template <typename T>
inline void validate_device_capabilities(const sycl::device &device) {
   if constexpr (std::is_same_v<T, double>) {
      if (!device.has(sycl::aspect::fp64)) {
         throw std::runtime_error(
            "adn::validate_environment: device lacks fp64 support");
      }
   }

   std::vector<sycl::info::fp_config> config;
   if constexpr (std::is_same_v<T, float>) {
      config = device.get_info<sycl::info::device::single_fp_config>();
   } else {
      config = device.get_info<sycl::info::device::double_fp_config>();
   }

   if (!has_fp_config(config, sycl::info::fp_config::round_to_nearest)) {
      throw std::runtime_error(
         "adn::validate_environment: device does not report "
         "round-to-nearest support");
   }
   if (!has_fp_config(config, sycl::info::fp_config::denorm)) {
      throw std::runtime_error(
         "adn::validate_environment: device does not report denormal "
         "support");
   }
   if (!has_fp_config(config, sycl::info::fp_config::inf_nan)) {
      throw std::runtime_error(
         "adn::validate_environment: device does not report Inf/NaN "
         "support");
   }
}

/**
 * @brief Run arithmetic on the device to verify its effective FP semantics.
 */
template <typename T> inline void run_device_environment_probe(sycl::queue &q) {
   SYCL_REPRO_SUM_DETAIL_STRICT_FP
   using B = typename fp<T>::bits_t;
   constexpr B half_min_normal_bits = B(1) << (fp<T>::mant_dig - 2);
   constexpr B negative_zero_bits = B(1) << (sizeof(T) * 8 - 1);

   EnvironmentProbeData<T> *probe =
      sycl::malloc_shared<EnvironmentProbeData<T>>(1, q);
   if (probe == nullptr) {
      throw std::bad_alloc();
   }

   probe->min_normal = std::numeric_limits<T>::min();
   probe->half = T(0.5);
   probe->denorm_min = std::numeric_limits<T>::denorm_min();
   probe->two = T(2);
   probe->one = T(1);
   probe->epsilon = std::numeric_limits<T>::epsilon();
   probe->quarter = T(0.25);
   probe->three_quarters = T(0.75);
   probe->negative_zero = from_bits<T>(negative_zero_bits);

   try {
      q.submit([=](sycl::handler &h) {
         h.single_task<EnvironmentProbeKernel<T>>([=]() {
            probe->compiled_half_min_normal =
               std::numeric_limits<T>::min() * T(0.5);
            probe->half_min_normal = probe->min_normal * probe->half;
            probe->twice_denorm_min = probe->denorm_min * probe->two;
            probe->below_half_ulp =
               probe->one + probe->epsilon * probe->quarter;
            probe->above_half_ulp =
               probe->one + probe->epsilon * probe->three_quarters;
            probe->preserved_negative_zero = probe->negative_zero * probe->one;
            probe->negative_zero_plus_zero = probe->negative_zero + T(0);
         });
      }).wait_and_throw();
   } catch (...) {
      sycl::free(probe, q);
      throw;
   }

   const bool gradual_underflow =
      to_bits(probe->compiled_half_min_normal) == half_min_normal_bits &&
      to_bits(probe->half_min_normal) == half_min_normal_bits &&
      to_bits(probe->twice_denorm_min) == B(2);
   const bool round_to_nearest =
      to_bits(probe->below_half_ulp) == to_bits(T(1)) &&
      to_bits(probe->above_half_ulp) == to_bits(T(1)) + B(1);
   const bool signed_zero =
      to_bits(probe->preserved_negative_zero) == negative_zero_bits &&
      to_bits(probe->negative_zero_plus_zero) == B(0);
   sycl::free(probe, q);

   if (!gradual_underflow) {
      throw std::runtime_error(
         "adn::validate_environment: device lacks gradual underflow");
   }
   if (!round_to_nearest) {
      throw std::runtime_error(
         "adn::validate_environment: device arithmetic is not "
         "round-to-nearest");
   }
   if (!signed_zero) {
      throw std::runtime_error(
         "adn::validate_environment: device does not preserve signed zero");
   }
}

template <typename T> struct DeviceValidationSlot {
   sycl::backend backend;
   sycl::device device;
   std::once_flag once;

   DeviceValidationSlot(
      sycl::backend backend_value, const sycl::device &device_value)
      : backend(backend_value), device(device_value) {}
};

template <typename T>
inline const std::vector<std::unique_ptr<DeviceValidationSlot<T>>> &
device_validation_slots() {
   static const auto slots = [] {
      std::vector<std::unique_ptr<DeviceValidationSlot<T>>> result;
      for (const sycl::platform &platform : sycl::platform::get_platforms()) {
         const sycl::backend backend = platform.get_backend();
         for (const sycl::device &device : platform.get_devices()) {
            result.push_back(
               std::make_unique<DeviceValidationSlot<T>>(backend, device));
         }
      }
      return result;
   }();
   return slots;
}

template <typename T>
inline DeviceValidationSlot<T> *find_device_validation_slot(
   sycl::backend backend, const sycl::device &device) {
   for (const auto &slot : device_validation_slots<T>()) {
      if (slot->backend == backend && slot->device == device) {
         return slot.get();
      }
   }
   return nullptr;
}

/**
 * @brief Validate and cache one backend/device/type combination.
 */
template <typename T> inline void validate_device_environment(sycl::queue &q) {
   const sycl::backend backend = q.get_backend();
   const sycl::device device = q.get_device();
   DeviceValidationSlot<T> *slot =
      find_device_validation_slot<T>(backend, device);
   auto validate = [&] {
      validate_device_capabilities<T>(device);
      run_device_environment_probe<T>(q);
   };

   if (slot == nullptr) {
      // Dynamically created sub-devices are not part of root enumeration.
      // Validate them on every use rather than caching them unsafely.
      validate();
      return;
   }
   std::call_once(slot->once, validate);
}

} // namespace detail

/**
 * @brief Maximum input count supported by the reproducible accumulator.
 *
 * Exceeding this conservative ReproBLAS capacity can make carry additions
 * inexact and therefore order-dependent.
 *
 * @tparam T Floating-point type (`float` or `double`).
 */
template <typename T>
inline constexpr std::uint64_t max_reproducible_count = [] {
   static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
      "T must be float or double");
   return detail::fp<T>::capacity;
}();

/**
 * @brief Validate the device floating-point environment.
 *
 * A successful check is shared by all host threads and cached per backend,
 * device, and floating-point type.  Float sums also validate double precision
 * because their final conversion accumulates in double on the device.
 *
 * @tparam T Floating-point type (`float` or `double`).
 * @param q  SYCL queue bound to the target device.
 * @throws std::runtime_error if the required IEEE 754 semantics are absent.
 */
template <typename T> void validate_environment(sycl::queue &q) {
   detail::validate_fp_type<T>();
   detail::validate_device_environment<double>(q);
   if constexpr (std::is_same_v<T, float>) {
      detail::validate_device_environment<float>(q);
   }
}

/**
 * @brief Compile-time validation of the template parameters.
 *
 * Fires a static_assert with a descriptive message if any parameter
 * falls outside the valid range.
 *
 * @tparam T       Floating-point type (float or double).
 * @tparam K       Fold count.  Must be in [2, max_fold] (52 for double,
 *                 21 for float).
 * @tparam WG_SIZE Work-group size.  Must be a power of 2 in [2, 1024].
 */
template <typename T, int K, int WG_SIZE> constexpr void validate_params() {
   detail::validate_fp_type<T>();
   static_assert(
      K >= 2, "K must be >= 2 (at least 2 folds needed for binned summation)");
   static_assert(K <= detail::fp<T>::max_fold,
      "K exceeds the binned format's maximum fold for this type");
   static_assert(detail::is_power_of_2(WG_SIZE),
      "WG_SIZE must be a power of 2 (required by tree reduction)");
   static_assert(
      WG_SIZE >= 2 && WG_SIZE <= 1024, "WG_SIZE must be between 2 and 1024");
}

/**
 * @brief Compute a bit-reproducible sum of N floating-point values in
 *        device memory, in a single pass.
 *
 * For @p N within @ref max_reproducible_count, the result is independent of
 * work-item execution order, work-group size, and device, making it
 * deterministic across runs and GPUs.
 *
 * @tparam K       Fold count (default 3).  More folds = higher accuracy.
 * @tparam WG_SIZE Work-group size (default 256).
 * @tparam T       Floating-point type (deduced from pointer).
 * @param  q       SYCL queue bound to the target device.
 * @param  arr     Pointer to the input array.  Accepts device, shared,
 *                 host USM, or plain host pointers.  Plain host pointers
 *                 are automatically copied to device memory.
 * @param  N       Number of elements.
 * @return         Bit-reproducible sum regardless of execution order within
 *                 the documented accumulator capacity.
 * @throws std::length_error if @p N exceeds @ref max_reproducible_count.
 * @throws std::runtime_error if the device lacks the required IEEE 754
 *                             floating-point semantics.
 *
 * @note Both float and double sums require device fp64 support because the
 *       final float conversion accumulates in double precision.
 *
 * @note NaN inputs and sums containing both +Inf and -Inf return a fixed
 *       positive quiet NaN (0x7fc00000 for float, 0x7ff8000000000000 for
 *       double), independent of input NaN payloads and device conventions.
 *       Values smaller than the binned format's absorption threshold relative
 *       to the running maximum (~2^(-W*(K-1)) * max|x|, and at absolute
 *       minimum ~2^-1055 for double / ~2^-144 for float with K=3) are rounded
 *       away deterministically, per the ADN accuracy bound.
 */
template <int K = 3, int WG_SIZE = 256, typename T>
T sum(sycl::queue &q, const T *arr, size_t N) {
   validate_params<T, K, WG_SIZE>();
   if (N == 0) {
      return T(0);
   }
   if (N > max_reproducible_count<T>) {
      throw std::length_error(
         "adn::sum input count exceeds max_reproducible_count<T>");
   }
   validate_environment<T>(q);

   // If the pointer was allocated by SYCL (device/shared/host USM),
   // use it directly.  Otherwise it is a plain host pointer: allocate
   // device memory, copy, compute, and free.
   auto ptr_type = sycl::get_pointer_type(arr, q.get_context());
   if (ptr_type != sycl::usm::alloc::unknown) {
      return detail::sum_impl<T, K, WG_SIZE>(q, arr, N);
   }

   T *d_arr = sycl::malloc_device<T>(N, q);
   q.memcpy(d_arr, arr, N * sizeof(T)).wait();

   T result = detail::sum_impl<T, K, WG_SIZE>(q, d_arr, N);

   sycl::free(d_arr, q);
   return result;
}

} // namespace adn

// === Restore caller's floating-point options ===============================
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
#pragma GCC pop_options
#elif defined(_MSC_VER)
#pragma float_control(pop)
#endif

#undef SYCL_REPRO_SUM_DETAIL_STRICT_FP
