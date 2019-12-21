#include "intgemm.h"

namespace intgemm {

Int8::BackendInterface Int8::interface = ChooseCPU<Int8::BackendInterface>(
  BackendInterface::create<AVX512VNNI_8bit>(),
  BackendInterface::create<AVX512_8bit>(),
  BackendInterface::create<AVX2_8bit>(),
  BackendInterface::create<SSSE3_8bit>(),
  BackendInterface::create<Unsupported_8bit>(),
  BackendInterface::create<Unsupported_8bit>()
);

Int8Shift::BackendInterface Int8Shift::interface = ChooseCPU<Int8Shift::BackendInterface>(
  BackendInterface::create<AVX512VNNI_8bit>(),
  BackendInterface::create<AVX512_8bit>(),
  BackendInterface::create<AVX2_8bit>(),
  BackendInterface::create<SSSE3_8bit>(),
  BackendInterface::create<Unsupported_8bit>(),
  BackendInterface::create<Unsupported_8bit>()
);

Int16::BackendInterface Int16::interface = ChooseCPU<Int16::BackendInterface>(
  BackendInterface::create<AVX512_16bit>(),
  BackendInterface::create<AVX512_16bit>(),
  BackendInterface::create<AVX2_16bit>(),
  BackendInterface::create<SSE2_16bit>(),
  BackendInterface::create<SSE2_16bit>(),
  BackendInterface::create<Unsupported_16bit>()
);

const CPUType kCPU = ChooseCPU(CPUType::AVX512VNNI, CPUType::AVX512BW, CPUType::AVX2, CPUType::SSSE3, CPUType::SSE2, CPUType::UNSUPPORTED);

float Unsupported_MaxAbsolute(const float *begin, const float *end) {
  throw UnsupportedCPU();
}

float (*MaxAbsolute)(const float *begin, const float *end) = ChooseCPU(avx512f::MaxAbsolute, avx512f::MaxAbsolute, avx2::MaxAbsolute, sse2::MaxAbsolute, sse2::MaxAbsolute, Unsupported_MaxAbsolute);

}
