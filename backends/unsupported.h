#pragma once

#include "backend.h"
#include <cstdint>

namespace intgemm {

template <>
struct Backend<CPUType::UNSUPPORTED, int8_t> {
  static void Quantize(const float *, int8_t *, float, Index) {
    throw UnsupportedCPU();
  }
  static void QuantizeU(const float *, uint8_t *, float, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareB(const float *, int8_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  template<class Callback>
  static void PrepareBiasFor8(const int8_t, const int8_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int8_t *, int8_t *, Index, const Index *, const Index *) {
    throw UnsupportedCPU();
  }
  template <typename Callback>
  static void Multiply(const int8_t *, const int8_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  template<class Callback>
  static void Multiply8Shift(const uint8_t *, const int8_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  static inline const char* const Name() { return "8-bit Unsupported"; };
};

template <>
struct Backend<CPUType::UNSUPPORTED, int16_t> {
  static void Quantize(const float *, int16_t *, float, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareB(const float *, int16_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int16_t *, int16_t *, Index, const Index *, const Index *) {
    throw UnsupportedCPU();
  }
  template <typename Callback>
  static void Multiply(const int16_t *, const int16_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  static inline const char* const Name() { return "16-bit Unsupported"; };
};

}
