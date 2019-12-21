#include "test.h"
#include "../aligned.h"
#include "../interleave.h"
#include "../intgemm.h"
#include "../multiply.h"
#include "../callbacks.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

namespace intgemm {

// Rearrange a tile of simd x unroll entries.
template <class V> void SlowRearrangeTile(const V *from, V *to, int simd, int unroll, Index cols) {
  for (int i = 0; i < unroll; ++i) {
    for (int j = 0; j < simd; ++j) {
      to[simd * i + j] = from[cols * j + i];
    }
  }
}

template <class V> void SlowRearrange(const V *from, V *to, int simd, int unroll, Index rows, Index cols) {
  for (Index c = 0; c < cols; c += unroll) {
    for (Index r = 0; r < rows; r += simd) {
      SlowRearrangeTile(from + cols * r + c, to, simd, unroll, cols);
      to += unroll * simd;
    }
  }
}

template <class V> void SlowTranspose(const V *from, V *to, Index rows, Index cols) {
  for (Index r = 0; r < rows; ++r) {
    for (Index c = 0; c < cols; ++c) {
      to[rows * c + r] = from[cols * r + c];
    }
  }
}

INTGEMM_SSE2 TEST_CASE("Transpose 16", "[transpose]") {
  if (kCPU < CPUType::SSE2) return;
  const unsigned N = 8;
  AlignedVector<int16_t> input(N * N);
  std::iota(input.begin(), input.end(), 0);

  AlignedVector<int16_t> ref(N * N);
  SlowTranspose(input.begin(), ref.begin(), N, N);

  // Overwrite input.
  __m128i *t = input.as<__m128i>();
  Transpose16InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

  for (int16_t i = 0; i < input.size(); ++i) {
  	CHECK_MESSAGE(ref[i] == input[i], "16-bit transpose failure at: " << i << ": " << ref[i] << " != " << input[i]);
  }
}

INTGEMM_SSSE3 TEST_CASE("Transpose 8", "[transpose]") {
  if (kCPU < CPUType::SSSE3) return;
  const unsigned N = 16;
  AlignedVector<int8_t> input(N * N);
  std::iota(input.begin(), input.end(), 0);

  AlignedVector<int8_t> ref(input.size());
  SlowTranspose(input.begin(), ref.begin(), N, N);

  // Overwrite input.
  __m128i *t = input.as<__m128i>();
  Transpose8InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]);

  for (int i = 0; i < input.size(); ++i) {
    CHECK_MESSAGE(ref[i] == input[i], "8-bit transpose failure at " << i << ": " << (int16_t)ref[i] << " != " << (int16_t)input[i]);
  }
}

template <class T> std::string PrintMatrix(const T *mem, Index rows, Index cols) {
  std::ostringstream out;
  for (Index r = 0; r < rows; ++r) {
    for (Index c = 0; c < cols; ++c) {
      out << std::setw(4) << (int64_t) mem[r * cols + c] << ' ';
    }
    out << '\n';
  }
  return out.str();
}

template <class Routine> void TestPrepare(Index rows = 32, Index cols = 16) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-129.0, 129.0);
  // Create array.
  AlignedVector<float> input(rows * cols);
  for (auto& it : input) {
    it = dist(gen);
  }

  using Integer = typename BackendInfo<Routine>::Integer;
  // Call Prepare
  AlignedVector<Integer> test(input.size());
  Routine::PrepareB(input.begin(), test.begin(), 1, rows, cols);

  // Compute reference output.
  AlignedVector<Integer> quantized(input.size());
  Routine::Quantize(input.begin(), quantized.begin(), 1, input.size());
  AlignedVector<Integer> reference(input.size());
  // Note this won't work for Int8/Int16 generic routines because tile sizes vary.
  SlowRearrange<Integer>(quantized.begin(), reference.begin(), Routine::kBTileRow, Routine::kBTileCol, rows, cols);
  CHECK_MESSAGE(memcmp(reference.begin(), test.begin(), test.size() * sizeof(Integer)) == 0, Routine::Name() << " Mismatch:\n" <<
  	"Quantized Input" << '\n' << PrintMatrix(quantized.begin(), rows, cols) << "Reference" << '\n' <<
  	 PrintMatrix(reference.begin(), rows, cols) << "Routine" << '\n' << PrintMatrix(test.begin(), rows, cols));
}

TEST_CASE("Prepare AVX512", "[prepare]") {
  if (kCPU < CPUType::AVX512BW) return;
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
	TestPrepare<AVX512_8bit>(64, 8);
	TestPrepare<AVX512_8bit>(256, 32);
    TestPrepare<AVX512_16bit>(64, 8);
    TestPrepare<AVX512_16bit>(256, 32);
#endif
}

TEST_CASE("Prepare AVX2", "[prepare]") {
  if (kCPU < CPUType::AVX2) return;
  TestPrepare<AVX2_8bit>(64, 32);
  TestPrepare<AVX2_16bit>(64, 32);
}

TEST_CASE("Prepare SSSE3", "[prepare]") {
  if (kCPU < CPUType::SSSE3) return;
  TestPrepare<SSSE3_8bit>(16, 8);
  TestPrepare<SSSE3_8bit>(32, 16);
  TestPrepare<SSSE3_8bit>(32, 32);
}

TEST_CASE("Prepare SSE2", "[prepare]") {
  if (kCPU < CPUType::SSE2) return;
  TestPrepare<SSE2_16bit>(8, 8);
  TestPrepare<SSE2_16bit>(32, 32);
}

template <class Routine> void TestSelectColumnsB(Index rows = 64, Index cols = 16) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-129.0, 129.0);
  AlignedVector<float> input(rows * cols);
  for (auto& it : input) {
    it = dist(gen);
  }
  using Integer = typename BackendInfo<Routine>::Integer;
  AlignedVector<Integer> prepared(input.size());
  Routine::PrepareB(input.begin(), prepared.begin(), 1, rows, cols);

  int kSelectCols = 24;
  Index select_cols[kSelectCols];
  std::uniform_int_distribution<Index> col_dist(0, cols - 1);
  for (auto& it : select_cols) {
    it = col_dist(gen);
  }

  AlignedVector<Integer> test(rows * kSelectCols);
  Routine::SelectColumnsB(prepared.begin(), test.begin(), rows, select_cols, select_cols + kSelectCols);

  // Select columns manually in float space.
  AlignedVector<float> selected(rows * kSelectCols);
  for (Index r = 0; r < rows; ++r) {
    for (int c = 0; c < kSelectCols; ++c) {
      assert(c + r * kSelectCols < rows * kSelectCols);
      selected[c + r * kSelectCols] = input[select_cols[c] + r * cols];
    }
  }
  AlignedVector<Integer> ref(rows * kSelectCols);
  Routine::PrepareB(selected.begin(), ref.begin(), 1, rows, kSelectCols);
  CHECK_MESSAGE(memcmp(ref.begin(), test.begin(), sizeof(Integer) * rows * kSelectCols) == 0, "Reference:\n" <<
  	PrintMatrix(ref.begin(), rows, kSelectCols) << PrintMatrix(test.begin(), rows, kSelectCols));
}

TEST_CASE("SelectColumnsB AVX512", "[select]") {
  if (kCPU < CPUType::AVX512BW) return;
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
    TestSelectColumnsB<AVX512_8bit>();
    TestSelectColumnsB<AVX512_16bit>(256, 256);
#endif
}

TEST_CASE("SelectColumnsB AVX2", "[select]") {
  if (kCPU < CPUType::AVX2) return;
  TestSelectColumnsB<AVX2_8bit>(256, 256);
  TestSelectColumnsB<AVX2_16bit>(256, 256);
}

TEST_CASE("SelectColumnsB SSSE3", "[select]") {
  if (kCPU < CPUType::SSSE3) return;
  TestSelectColumnsB<SSSE3_8bit>();
  TestSelectColumnsB<SSSE3_8bit>(256, 256);
}

TEST_CASE("SelectColumnsB SSE2", "[select]") {
  if (kCPU < CPUType::SSE2) return;
  TestSelectColumnsB<SSE2_16bit>();
  TestSelectColumnsB<SSE2_16bit>(256, 256);
}

template <class Register> void TestMax() {
  Register r = set1_ps<Register>(-2.0);
  for (std::size_t i = 0; i < sizeof(Register) / sizeof(float); ++i) {
    Register c = r;
    reinterpret_cast<float*>(&c)[i] = -1.0;
    CHECK_MESSAGE((MaxFloat32(c) == -1.0), "MaxFloat32 produced " << MaxFloat32(c));
  }
}

TEST_CASE("Max", "[max]") {
  TestMax<__m128>();
}

void CompareMaxAbs(const float *begin, const float *end, float test) {
  float largest = fabs(*std::max_element(begin, end));
  float smallest = fabs(*std::min_element(begin, end));
  largest = std::max(largest, smallest);
  CHECK_MESSAGE(largest == test, "Error: " << largest << " versus " << test);
}

template <float (*Backend) (const float *, const float *)> void TestMaxAbsolute() {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-8.0, 8.0);
  AlignedVector<float> test(64);
  // 64 tries.
  for (int t = 0; t < 64; ++t) {
    // Fill with [-8, 8).
    for (auto& it : test) {
      it = dist(gen);
    }
    CompareMaxAbs(test.begin(), test.end(), Backend(test.begin(), test.end()));
    test[t] = -32.0;
    CompareMaxAbs(test.begin(), test.end(), Backend(test.begin(), test.end()));
    test[t] = 32.0;
    CompareMaxAbs(test.begin(), test.end(), Backend(test.begin(), test.end()));
  }
}

TEST_CASE("MaxAbsolute SSE2", "[max]") {
  if (kCPU < CPUType::SSE2) return;
  TestMaxAbsolute<sse2::MaxAbsolute>();
}

TEST_CASE("MaxAbsolute AVX2", "[max]") {
  if (kCPU < CPUType::AVX2) return;
  TestMaxAbsolute<avx2::MaxAbsolute>();
}

TEST_CASE("MaxAbsolute AVX512F", "[max]") {
  if (kCPU < CPUType::AVX512BW) return;
  #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
  TestMaxAbsolute<avx512f::MaxAbsolute>();
  #endif
}

// Based on https://arxiv.org/abs/1705.01991

// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Compute A*B slowly in floats.

template <class Routine> void TestMultiply(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  using Integer = typename BackendInfo<Routine>::Integer;
  std::ostringstream info;
  info << Routine::Name() << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }
  
  float quant_mult = (sizeof(Integer) == 2) ? 1024 : 64;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<Integer> A_prep(A.size());
  AlignedVector<Integer> B_prep(B.size());
  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);
  Routine::Multiply(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndWrite(unquant_mult, test_C.begin()));
  // Routine::Multiply(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::Sequence(
  //   callbacks::Unquantize(unquant_mult),
  //   callbacks::Write<float>(test_C.begin())
  // ));

  AlignedVector<Integer> B_quant(B.size());
  Routine::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());
  AlignedVector<float> slowint_C(test_C.size());
  // Assuming A is just quantization here.
  SlowRefInt(A_prep.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols);

  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols);

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

//Code duplication may be avoided through some use of variadic templates, as the different WriteC symbols
//Require different number of arguments. I don't think the refactoring is worth it.
template <class Routine> void TestMultiplyBias(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  using Integer = typename BackendInfo<Routine>::Integer;
  std::ostringstream info;
  info << Routine::Name() << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  AlignedVector<float> bias(B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }
  for (auto& it : bias) {
    it = dist(gen);
  }
  
  float quant_mult = (sizeof(Integer) == 2) ? 1024 : 64;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<Integer> A_prep(A.size());
  AlignedVector<Integer> B_prep(B.size());
  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);

  Routine::Multiply(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias.begin(), test_C.begin()));

  AlignedVector<Integer> B_quant(B.size());
  Routine::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());
  AlignedVector<float> slowint_C(test_C.size());
  // Assuming A is just quantization here.
  SlowRefInt(A_prep.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, bias.begin());

  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols, bias.begin());

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

TEST_CASE ("Multiply SSE2 16bit", "[multiply]") {
  if (kCPU < CPUType::SSE2) return;
  TestMultiply<SSE2_16bit>(8, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit>(8, 2048, 256, .1, 1, 0.02);
  TestMultiply<SSE2_16bit>(320, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit>(472, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit>(248, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit>(200, 256, 256, .1, 1, 0.01);
}

TEST_CASE ("Multiply SSE2 16bit with bias", "[biased_multiply]") {
  if (kCPU < CPUType::SSE2) return;
  TestMultiplyBias<SSE2_16bit>(8, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<SSE2_16bit>(8, 2048, 256, .1, 1, 0.02);
  TestMultiplyBias<SSE2_16bit>(320, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<SSE2_16bit>(472, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<SSE2_16bit>(248, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<SSE2_16bit>(200, 256, 256, .1, 1, 0.01);
}

TEST_CASE ("Multiply SSSE3 8bit", "[multiply]") {
  if (kCPU < CPUType::SSSE3) return;
  TestMultiply<SSSE3_8bit>(8, 256, 256, 1.2, 1.2, 0.064, 0.026);
  TestMultiply<SSSE3_8bit>(8, 2048, 256, 33, 33, 4.4, 4.4);
  TestMultiply<SSSE3_8bit>(320, 256, 256, 1.9, 1.9, 0.1, 0.01);
  TestMultiply<SSSE3_8bit>(472, 256, 256, 2.1, 2.1, 0.1, 0.011);
  TestMultiply<SSSE3_8bit>(248, 256, 256, 1.7, 1.7, 0.1, 0.012);
  TestMultiply<SSSE3_8bit>(200, 256, 256, 1.8, 1.9, 0.1, 0.011);
}

TEST_CASE ("Multiply SSSE3 8bit with bias", "[biased_multiply]") {
  if (kCPU < CPUType::SSSE3) return;
  TestMultiplyBias<SSSE3_8bit>(8, 256, 256, 1.2, 1.2, 0.064, 0.026);
  TestMultiplyBias<SSSE3_8bit>(8, 2048, 256, 33, 33, 4.4, 4.4);
  TestMultiplyBias<SSSE3_8bit>(320, 256, 256, 1.9, 1.9, 0.1, 0.01);
  TestMultiplyBias<SSSE3_8bit>(472, 256, 256, 2.1, 2.1, 0.1, 0.011);
  TestMultiplyBias<SSSE3_8bit>(248, 256, 256, 1.7, 1.7, 0.1, 0.012);
  TestMultiplyBias<SSSE3_8bit>(200, 256, 256, 1.8, 1.9, 0.1, 0.011);
}

TEST_CASE ("Multiply AVX2 8bit", "[multiply]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiply<AVX2_8bit>(8, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit>(8, 2048, 256, 19, 19, 1.8, 1.8);
  TestMultiply<AVX2_8bit>(320, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit>(472, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit>(248, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit>(200, 256, 256, .1, 1, 0.1);
}

TEST_CASE ("Multiply AVX2 8bit with bias", "[biased_multiply]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiplyBias<AVX2_8bit>(8, 256, 256, .1, 1, 0.1);
  TestMultiplyBias<AVX2_8bit>(8, 2048, 256, 19, 19, 1.8, 1.8);
  TestMultiplyBias<AVX2_8bit>(320, 256, 256, .1, 1, 0.1);
  TestMultiplyBias<AVX2_8bit>(472, 256, 256, .1, 1, 0.1);
  TestMultiplyBias<AVX2_8bit>(248, 256, 256, .1, 1, 0.1);
  TestMultiplyBias<AVX2_8bit>(200, 256, 256, .1, 1, 0.1);
}

TEST_CASE ("Multiply AVX2 16bit", "[multiply]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiply<AVX2_16bit>(8, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit>(8, 2048, 256, .1, 1, 0.02);
  TestMultiply<AVX2_16bit>(320, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit>(472, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit>(248, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit>(200, 256, 256, .1, 1, 0.01);
}

TEST_CASE ("Multiply AVX2 16bit with bias", "[biased_multiply]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiplyBias<AVX2_16bit>(8, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<AVX2_16bit>(8, 2048, 256, .1, 1, 0.02);
  TestMultiplyBias<AVX2_16bit>(320, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<AVX2_16bit>(472, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<AVX2_16bit>(248, 256, 256, .1, 1, 0.01);
  TestMultiplyBias<AVX2_16bit>(200, 256, 256, .1, 1, 0.01);
}

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
  TEST_CASE ("Multiply AVX512 8bit", "[multiply]") {
    if (kCPU < CPUType::AVX512BW) return;
    TestMultiply<AVX512_8bit>(8, 256, 256, 0, 0.25, 0.062);
    TestMultiply<AVX512_8bit>(8, 2048, 256, 3.7, 4, 0.37, 0.33);
    TestMultiply<AVX512_8bit>(320, 256, 256, 0, 0.26, 0.059);
    TestMultiply<AVX512_8bit>(472, 256, 256, 0, 0.29, 0.059);
    TestMultiply<AVX512_8bit>(248, 256, 256, 0, 0.29, 0.059);
    TestMultiply<AVX512_8bit>(200, 256, 256, 0, 0.28, 0.06);
  }

  TEST_CASE ("Multiply AVX512 8bit with bias", "[biased_multiply]") {
    if (kCPU < CPUType::AVX512BW) return;
    TestMultiplyBias<AVX512_8bit>(8, 256, 256, 0, 0.25, 0.062);
    TestMultiplyBias<AVX512_8bit>(8, 2048, 256, 3.7, 4, 0.37, 0.33);
    TestMultiplyBias<AVX512_8bit>(320, 256, 256, 0, 0.26, 0.059);
    TestMultiplyBias<AVX512_8bit>(472, 256, 256, 0, 0.29, 0.059);
    TestMultiplyBias<AVX512_8bit>(248, 256, 256, 0, 0.29, 0.059);
    TestMultiplyBias<AVX512_8bit>(200, 256, 256, 0, 0.28, 0.06);
  }

  #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
    TEST_CASE ("Multiply AVX512VNNI 8bit", "[multiply]") {
      if (kCPU < CPUType::AVX512VNNI) return;
      TestMultiply<AVX512VNNI_8bit>(8, 256, 256, 0, 0.25, 0.062);
      TestMultiply<AVX512VNNI_8bit>(8, 2048, 256, 0, 0.55, 0.25);
      TestMultiply<AVX512VNNI_8bit>(320, 256, 256, 0, 0.26, 0.059);
      TestMultiply<AVX512VNNI_8bit>(472, 256, 256, 0, 0.29, 0.059);
      TestMultiply<AVX512VNNI_8bit>(248, 256, 256, 0, 0.29, 0.059);
      TestMultiply<AVX512VNNI_8bit>(200, 256, 256, 0, 0.28, 0.06);
    }

    TEST_CASE ("Multiply AVX512VNNI 8bit with bias", "[biased_multiply]") {
      if (kCPU < CPUType::AVX512VNNI) return;
      TestMultiplyBias<AVX512VNNI_8bit>(8, 256, 256, 0, 0.25, 0.062);
      TestMultiplyBias<AVX512VNNI_8bit>(8, 2048, 256, 0, 0.55, 0.25);
      TestMultiplyBias<AVX512VNNI_8bit>(320, 256, 256, 0, 0.26, 0.059);
      TestMultiplyBias<AVX512VNNI_8bit>(472, 256, 256, 0, 0.29, 0.059);
      TestMultiplyBias<AVX512VNNI_8bit>(248, 256, 256, 0, 0.29, 0.059);
      TestMultiplyBias<AVX512VNNI_8bit>(200, 256, 256, 0, 0.28, 0.06);
    }
  #endif

  TEST_CASE ("Multiply AVX512 16bit", "[multiply]") {
    if (kCPU < CPUType::AVX512BW) return;
    TestMultiply<AVX512_16bit>(8, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit>(8, 2048, 256, .1, 1, 0.011);
    TestMultiply<AVX512_16bit>(320, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit>(472, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit>(248, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit>(200, 256, 256, .1, 1, 0.01);
  }

  TEST_CASE ("Multiply AVX512 16bit with bias", "[biased_multiply]") {
    if (kCPU < CPUType::AVX512BW) return;
    TestMultiplyBias<AVX512_16bit>(8, 256, 256, .1, 1, 0.01);
    TestMultiplyBias<AVX512_16bit>(8, 2048, 256, .1, 1, 0.011);
    TestMultiplyBias<AVX512_16bit>(320, 256, 256, .1, 1, 0.01);
    TestMultiplyBias<AVX512_16bit>(472, 256, 256, .1, 1, 0.01);
    TestMultiplyBias<AVX512_16bit>(248, 256, 256, .1, 1, 0.01);
    TestMultiplyBias<AVX512_16bit>(200, 256, 256, .1, 1, 0.01);
  }
#endif

} // namespace intgemm
