#pragma once

#include "backends/unsupported.h"
#include "backends/sse2.h"
#include "backends/ssse3.h"
#include "backends/avx2.h"
#include "backends/avx512.h"
#include "backends/avx512vnni.h"

namespace intgemm {

using Unsupported_8bit  = Backend<CPUType::UNSUPPORTED, int8_t>;
using Unsupported_16bit = Backend<CPUType::UNSUPPORTED, int16_t>;
using SSE2_16bit        = Backend<CPUType::SSE2, int16_t>;
using SSSE3_8bit        = Backend<CPUType::SSSE3, int8_t>;
using AVX2_8bit         = Backend<CPUType::AVX2, int8_t>;
using AVX2_16bit        = Backend<CPUType::AVX2, int16_t>;

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
using AVX512_8bit       = Backend<CPUType::AVX512BW, int8_t>;
using AVX512_16bit      = Backend<CPUType::AVX512BW, int16_t>;
#else
using AVX512_8bit       = Unsupported_8bit;
using AVX512_16bit      = Unsupported_16bit;
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
using AVX512VNNI_8bit   = Backend<CPUType::AVX512VNNI, int8_t>;
#else
using AVX512VNNI_8bit   = Unsupported_8bit;
#endif

}
