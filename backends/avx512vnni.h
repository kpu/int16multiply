#pragma once

#include "intgemm_config.h"

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
#include "avx512.h"
#include "../types.h"
#include "backend.h"

namespace intgemm {

template <>
struct Backend<CPUType::AVX512VNNI, int8_t> : public Backend<CPUType::AVX512BW, int8_t> {
  static inline const char* const Name() { return "8-bit AVX512VNNI"; };

  template <typename Callback>
  INTGEMM_AVX512VNNI static void Multiply(const int8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    using Integer = __m512i;
    assert(width % sizeof(Integer) == 0);
    assert(B_cols % 8 == 0);
    assert(reinterpret_cast<uintptr_t>(A) % sizeof(Integer) == 0);
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(Integer) == 0);
    auto callback_impl = callbacks::CallbackImpl<CPUType::AVX2, Callback>(callback);
    const int simd_width = width / sizeof(Integer);
    const Integer *B0_col = reinterpret_cast<const Integer*>(B);
    Integer zeros = setzero_si<Integer>();
    // Go over 8 columns of B at a time.
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
      // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
      for (Index A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
        // Iterate over shared (inner) dimension.
        const Integer *A_live = reinterpret_cast<const Integer *>(A + A_rowidx * width);
        const Integer *A_end = A_live + simd_width;
        const Integer *B_live = B0_col;
        // TODO: separate first step.
        Integer sum0 = zeros, sum1 = zeros, sum2 = zeros, sum3 = zeros, sum4 = zeros, sum5 = zeros, sum6 = zeros, sum7 = zeros;
        for (; A_live != A_end; ++A_live, B_live += 8) {
          Integer a = *A_live;
          // Retrieve the conveniently consecutive values of B.
          Integer b0 = *B_live;
          Integer b1 = *(B_live + 1);
          Integer b2 = *(B_live + 2);
          Integer b3 = *(B_live + 3);
          Integer b4 = *(B_live + 4);
          Integer b5 = *(B_live + 5);
          Integer b6 = *(B_live + 6);
          Integer b7 = *(B_live + 7);
          // Get a mask where a is negative.
          __mmask64 neg_mask = _mm512_test_epi8_mask(a, _mm512_set1_epi8(-128));
          Integer a_positive = _mm512_abs_epi8(a);
          // Negate by subtracting from zero with a mask.
          b0 = _mm512_mask_sub_epi8(b0, neg_mask, zeros, b0);
          b1 = _mm512_mask_sub_epi8(b1, neg_mask, zeros, b1);
          b2 = _mm512_mask_sub_epi8(b2, neg_mask, zeros, b2);
          b3 = _mm512_mask_sub_epi8(b3, neg_mask, zeros, b3);
          b4 = _mm512_mask_sub_epi8(b4, neg_mask, zeros, b4);
          b5 = _mm512_mask_sub_epi8(b5, neg_mask, zeros, b5);
          b6 = _mm512_mask_sub_epi8(b6, neg_mask, zeros, b6);
          b7 = _mm512_mask_sub_epi8(b7, neg_mask, zeros, b7);
          sum0 = _mm512_dpbusds_epi32(sum0, a_positive, b0);
          sum1 = _mm512_dpbusds_epi32(sum1, a_positive, b1);
          sum2 = _mm512_dpbusds_epi32(sum2, a_positive, b2);
          sum3 = _mm512_dpbusds_epi32(sum3, a_positive, b3);
          sum4 = _mm512_dpbusds_epi32(sum4, a_positive, b4);
          sum5 = _mm512_dpbusds_epi32(sum5, a_positive, b5);
          sum6 = _mm512_dpbusds_epi32(sum6, a_positive, b6);
          sum7 = _mm512_dpbusds_epi32(sum7, a_positive, b7);
        }
        Integer pack0123 = Pack0123(sum0, sum1, sum2, sum3);
        Integer pack4567 = Pack0123(sum4, sum5, sum6, sum7);
        auto total = PermuteSummer(pack0123, pack4567);
        callback_impl(total, callbacks::OutputBufferInfo(A_rowidx, B0_colidx, A_rows, B_cols));
      }
    }
  }

  template <typename Callback>
  INTGEMM_AVX512VNNI static void Multiply8Shift(const uint8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    using Integer = __m512i;
    assert(width % sizeof(Integer) == 0);
    assert(B_cols % 8 == 0);
    assert(reinterpret_cast<uintptr_t>(A) % sizeof(Integer) == 0);
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(Integer) == 0);
    auto callback_impl = callbacks::CallbackImpl<CPUType::AVX2, Callback>(callback);
    const int simd_width = width / sizeof(Integer);
    const Integer *B0_col = reinterpret_cast<const Integer*>(B);
    Integer zeros = setzero_si<Integer>();
    // Go over 8 columns of B at a time.
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
      // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
      for (Index A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
        // Iterate over shared (inner) dimension.
        const Integer *A_live = reinterpret_cast<const Integer *>(A + A_rowidx * width);
        const Integer *A_end = A_live + simd_width;
        const Integer *B_live = B0_col;
        // TODO: separate first step.
        Integer sum0 = zeros, sum1 = zeros, sum2 = zeros, sum3 = zeros, sum4 = zeros, sum5 = zeros, sum6 = zeros, sum7 = zeros;
        for (; A_live != A_end; ++A_live, B_live += 8) {
          Integer a = *A_live;
          //MultiplyAdd
          sum0 = _mm512_dpbusds_epi32(sum0, a, *B_live);
          sum1 = _mm512_dpbusds_epi32(sum1, a, *(B_live + 1));
          sum2 = _mm512_dpbusds_epi32(sum2, a, *(B_live + 2));
          sum3 = _mm512_dpbusds_epi32(sum3, a, *(B_live + 3));
          sum4 = _mm512_dpbusds_epi32(sum4, a, *(B_live + 4));
          sum5 = _mm512_dpbusds_epi32(sum5, a, *(B_live + 5));
          sum6 = _mm512_dpbusds_epi32(sum6, a, *(B_live + 6));
          sum7 = _mm512_dpbusds_epi32(sum7, a, *(B_live + 7));
        }
        Integer pack0123 = Pack0123(sum0, sum1, sum2, sum3);
        Integer pack4567 = Pack0123(sum4, sum5, sum6, sum7);
        auto total = PermuteSummer(pack0123, pack4567);
        callback_impl(total, callbacks::OutputBufferInfo(A_rowidx, B0_colidx, A_rows, B_cols));
      }
    }
  }

  template <typename Callback>
  INTGEMM_AVX512VNNI static void PrepareBiasFor8(const int8_t *B, Index width, Index B_cols, Callback callback) {
    using Integer = __m512i;
    assert(width % sizeof(Integer) == 0);
    assert(B_cols % 8 == 0);
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(Integer) == 0);
    auto callback_impl = callbacks::CallbackImpl<CPUType::AVX2, Callback>(callback);
    const int simd_width = width / sizeof(Integer);
    const Integer *B0_col = reinterpret_cast<const Integer*>(B);
    Integer zeros = setzero_si<Integer>();
    const Integer a = set1_epi8<Integer>(1);
    // Go over 8 columns of B at a time.
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
      const Integer *B_live = B0_col; //In order to make the code look as much as possible as the above function
      const Integer *B_end = B_live + simd_width*8;

      // TODO: separate first step.
      Integer sum0 = zeros, sum1 = zeros, sum2 = zeros, sum3 = zeros, sum4 = zeros, sum5 = zeros, sum6 = zeros, sum7 = zeros;
      for (; B_live != B_end; B_live += 8) {
        // Retrieve the conveniently consecutive values of B.
        sum0 = _mm512_dpbusds_epi32(sum0, a, *B_live);
        sum1 = _mm512_dpbusds_epi32(sum1, a, *(B_live + 1));
        sum2 = _mm512_dpbusds_epi32(sum2, a, *(B_live + 2));
        sum3 = _mm512_dpbusds_epi32(sum3, a, *(B_live + 3));
        sum4 = _mm512_dpbusds_epi32(sum4, a, *(B_live + 4));
        sum5 = _mm512_dpbusds_epi32(sum5, a, *(B_live + 5));
        sum6 = _mm512_dpbusds_epi32(sum6, a, *(B_live + 6));
        sum7 = _mm512_dpbusds_epi32(sum7, a, *(B_live + 7));
      }
      Integer pack0123 = Pack0123(sum0, sum1, sum2, sum3);
      Integer pack4567 = Pack0123(sum4, sum5, sum6, sum7);
      auto total = PermuteSummer(pack0123, pack4567);
      callback_impl(total, callbacks::OutputBufferInfo(0, B0_colidx, 1, B_cols));
    }
  }
};

} // namespace intgemm

#endif
