#pragma once

#include "../interleave.h"
#include "../kernels.h"
#include "../multiply.h"
#include "../types.h"
#include "backend.h"

#include <cstdint>
#include <stdint.h>

// 16-bit is in sse2.h

namespace intgemm {

namespace ssse3 {

INTGEMM_SSSE3 inline __m128i QuantizerGrab(const float *input, const __m128 quant_mult_reg) {
  return kernels::quantize(loadu_ps<__m128>(input), quant_mult_reg);
}

INTGEMM_SELECT_COL_B(INTGEMM_SSSE3, __m128i)

class QuantizeTile8 {
  public:
    using Integer = __m128i;

    INTGEMM_SSSE3 explicit QuantizeTile8(float mult) : mult_reg_(_mm_set1_ps(mult)) {}

    INTGEMM_SSSE3 inline __m128i ForReshape(const float *input, Index cols) {
      // Skip a row.
      return Tile(input, input + 2 * cols);
    }

    INTGEMM_SSSE3 inline __m128i Consecutive(const float *input) {
      return Tile(input, input + 8);
    }

    INTGEMM_SSSE3 inline __m128i ConsecutiveU(const float *input) {
      return TileU(input, input + 8);
    }


  private:
    // Quantize 16xfloat into 16xint8_t
    INTGEMM_SSSE3 inline __m128i Tile(const float *input0, const float *input1) {
      const __m128i neg128 = _mm_set1_epi8(-128);
      __m128i g0 = QuantizerGrab(input0, mult_reg_);
      __m128i g1 = QuantizerGrab(input0 + 4, mult_reg_);
      __m128i g2 = QuantizerGrab(input1, mult_reg_);
      __m128i g3 = QuantizerGrab(input1 + 4, mult_reg_);
      __m128i packed0 = _mm_packs_epi32(g0, g1);
      __m128i packed1 = _mm_packs_epi32(g2, g3);
      __m128i packed = _mm_packs_epi16(packed0, packed1);
      /* Ban -128.
       * Don't use the SSE4.1 instruction _mm_max_epi8(packed, neg127).  Instead,
       * use SSE2 instructions _mm_cmpeq_epi8 and _mm_sub_epi8.
       * The first generates 0xff for fields -128.
       * The second subtracts 0xff from -128 which has the effect of converting
       * to -127.
       */
      // packed = _mm_max_epi8(packed, neg127);
      __m128i evils = _mm_cmpeq_epi8(packed, neg128);
      return _mm_sub_epi8(packed, evils);
      // No permute needed.  packs is in order for SSE.
    }

    INTGEMM_SSSE3 inline __m128i TileU(const float *input0, const float *input1) {
      const __m128i neg128 = _mm_set1_epi8(-128);
      const __m128i pos127 = _mm_set1_epi8(127);
      __m128i g0 = QuantizerGrab(input0, mult_reg_);
      __m128i g1 = QuantizerGrab(input0 + 4, mult_reg_);
      __m128i g2 = QuantizerGrab(input1, mult_reg_);
      __m128i g3 = QuantizerGrab(input1 + 4, mult_reg_);
      __m128i packed0 = _mm_packs_epi32(g0, g1);
      __m128i packed1 = _mm_packs_epi32(g2, g3);
      __m128i packed = _mm_packs_epi16(packed0, packed1);
      /* Ban -128.
       * Don't use the SSE4.1 instruction _mm_max_epi8(packed, neg127).  Instead,
       * use SSE2 instructions _mm_cmpeq_epi8 and _mm_sub_epi8.
       * The first generates 0xff for fields -128.
       * The second subtracts 0xff from -128 which has the effect of converting
       * to -127.
       */
      // packed = _mm_max_epi8(packed, neg127);
      __m128i evils = _mm_cmpeq_epi8(packed, neg128);
      return _mm_add_epi8(_mm_sub_epi8(packed, evils), pos127);
      // No permute needed.  packs is in order for SSE.
    }

  private:
    const __m128 mult_reg_;
};

} // namespace

// pmaddubsw (the 8-bit multiply) is INTGEMM_SSSE3, so pedantically that's the version we need.
template <>
struct Backend<CPUType::SSSE3, int8_t> {
  static inline const char* const Name() { return "8-bit SSSE3"; };

  // Currently A is prepared by quantization but this could theoretically change.
  INTGEMM_SSSE3 static inline void PrepareA(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  INTGEMM_SSSE3 static void Quantize(const float *input, int8_t *output, float quant_mult, Index size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(output) % 16 == 0);
    ssse3::QuantizeTile8 q(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 16, output += 16) {
      *reinterpret_cast<__m128i*>(output) = q.Consecutive(input);
    }
  }

  // Version with unsigned int + 127
  // Currently A is prepared by quantization but this could theoretically change.
  INTGEMM_SSSE3 static inline void PrepareA(const float *input, uint8_t *output, float quant_mult, Index rows, Index cols) {
    QuantizeU(input, output, quant_mult, rows * cols);
  }

  INTGEMM_SSSE3 static void QuantizeU(const float *input, uint8_t *output, float quant_mult, Index size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(output) % 16 == 0);
    ssse3::QuantizeTile8 q(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 16, output += 16) {
      *reinterpret_cast<__m128i*>(output) = q.ConsecutiveU(input);
    }
  }

  // Tile size for B; B must be a multiple of this block size.
  static const Index kBTileRow = 16;
  static const Index kBTileCol = 8;

  INTGEMM_PREPARE_B_8(INTGEMM_SSSE3, ssse3::QuantizeTile8)

  INTGEMM_SSSE3 static void SelectColumnsB(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    ssse3::SelectColumnsOfB((const __m128i*)input, (__m128i*)output, rows, cols_begin, cols_end);
  }

  INTGEMM_MULTIPLY8(__m128i, INTGEMM_SSSE3, CPUType::SSE2)

  INTGEMM_MULTIPLY8SHIFT(__m128i, INTGEMM_SSSE3, CPUType::SSE2)
  
  INTGEMM_PREPAREBIASFOR8(__m128i, INTGEMM_SSSE3, CPUType::SSE2)
};

} // namespace intgemm
