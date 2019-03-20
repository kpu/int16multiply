#include "avx512_gemm.h"
#include "avx2_gemm.h"
#include "ssse3_gemm.h"
#include "sse2_gemm.h"
#include "intgemm.h"
#include "aligned.h"
#include "interleave.h"
#include "multiply.h"

#define CATCH_CONFIG_RUNNER 
#include "3rd_party/catch.hpp"
#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while((void)0, 0)
#define CHECK_FALSE_MESSAGE(cond, msg) do { INFO(msg); CHECK_FALSE(cond); } while((void)0, 0)
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)
#define REQUIRE_FALSE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE_FALSE(cond); } while((void)0, 0)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <sstream>
#include <iostream>
#include <iomanip>

namespace intgemm {

struct Result {
  bool result;
  std::string text;
};

// Rearrange a tile of simd x unroll entries.
template <class V> void SlowRearrangeTile(const V *from, V *to, int simd, int unroll, Index cols) {
  for (int i = 0; i < unroll; ++i) {
    for (int j = 0; j < simd; ++j) {
      to[simd * i + j] = from[cols * j + i];
    }
  }
}

template <class V> void SlowRearrange(const V *from, V *to, int simd, int unroll, Index rows, Index cols) {
  for (int c = 0; c < cols; c += unroll) {
    for (int r = 0; r < rows; r += simd) {
      SlowRearrangeTile(from + cols * r + c, to, simd, unroll, cols);
      to += unroll * simd;
    }
  }
}

template <class V> void SlowTranspose(const V *from, V *to, Index rows, Index cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      to[rows * c + r] = from[cols * r + c];
    }
  }
}


TEST_CASE("Transpose 16", "[transpose]") {
  if (kCPU < CPU_SSE2) return;
  AlignedVector<int16_t> input(8 * 8);
  for (int16_t i = 0; i < 64; ++i) {
    input.get()[i] = i;
  }
  AlignedVector<int16_t> ref(8 * 8);
  SlowTranspose(input.get(), ref.get(), 8, 8);

  // Overwrite input.
  __m128i *t = reinterpret_cast<__m128i*>(input.get());
  Transpose16InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

  for (int16_t i = 0; i < 64; ++i) {
  	CHECK_MESSAGE(ref.get()[i] == input.get()[i], "16-bit transpose failure at: " << i << ": " << ref.get()[i] << " != " << input.get()[i]);
  }
}

TEST_CASE("Transpose 8", "[transpose]") {
  if (kCPU < CPU_SSSE3) return;
  AlignedVector<int8_t> input(16 * 16);
  for (int i = 0; i < 16 * 16; ++i) {
    input.get()[i] = i;
  }
  AlignedVector<int8_t> ref(16 * 16);
  SlowTranspose(input.get(), ref.get(), 16, 16);

  // Overwrite input.
  __m128i *t = reinterpret_cast<__m128i*>(input.get());
  Transpose8InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]);

  for (int i = 0; i < 16 * 16; ++i) {
    CHECK_MESSAGE(ref.get()[i] == input.get()[i], "8-bit transpose failure at " << i << ": " << (int16_t)ref.get()[i] << " != " << (int16_t)input.get()[i]);
  }
}

template <class T> void PrintMatrix(const T *mem, Index rows, Index cols, std::ostringstream& out) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out << std::setw(4) << (int64_t) mem[r * cols + c] << ' ';
    }
    out << '\n';
  }
}

template <class Routine> std::string TestPrepare(Index rows = 32, Index cols = 16) {
  std::ostringstream out;

  // Create array.
  AlignedVector<float> input(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    input.get()[i] = //(i > 127) ? (i - 256) : i;
      (float)rand() / (float)RAND_MAX * 256.0 - 127.0;
  }

  typedef typename Routine::Integer Integer;
  // Call Prepare
  AlignedVector<Integer> test(rows * cols);
  Routine::PrepareB(input.get(), test.get(), 1, rows, cols);

  // Compute reference output.
  AlignedVector<Integer> quantized(rows * cols);
  Routine::Quantize(input.get(), quantized.get(), 1, rows * cols);
  AlignedVector<Integer> reference(rows * cols);
  // Note this won't work for Int8/Int16 generic routines because tile sizes vary.
  SlowRearrange<Integer>(quantized.get(), reference.get(), Routine::kBTileRow, Routine::kBTileCol, rows, cols);
  if (memcmp(reference.get(), test.get(), rows * cols * sizeof(Integer))) {
    out << "TestPrepare " << Routine::kName << " Mismatch:\n";
    out << "Quantized Input" << '\n';
    PrintMatrix(quantized.get(), rows, cols, out);
    out << "Reference" << '\n';
    PrintMatrix(reference.get(), rows, cols, out);
    out << "Routine" << '\n';
    PrintMatrix(test.get(), rows, cols, out);
  }
  return out.str();
}

TEST_CASE("Test prepare AVX512", "[prepare]") {
  if (kCPU < CPU_AVX512BW) return;
#ifndef INTGEMM_NO_AVX512
    std::string one = TestPrepare<AVX512_8bit>(64, 8);
    CHECK(one == "");

    std::string two = TestPrepare<AVX512_8bit>(256, 32);
    CHECK(two == "");

    std::string three = TestPrepare<AVX512_16bit>(256, 32);
    CHECK(three == "");

    std::string four = TestPrepare<AVX512_16bit>(256, 32);
    CHECK(four == "");
#endif
}

TEST_CASE("Test prepare AVX2", "[prepare]") {
  if (kCPU < CPU_AVX2) return;
  std::string one = TestPrepare<AVX2_8bit>(64, 32);
  CHECK(one == "");

  std::string two = TestPrepare<AVX2_16bit>(64, 32);
  CHECK(two == "");
}

TEST_CASE("Test prepare SSSE3", "[prepare]") {
  if (kCPU < CPU_SSSE3) return;
  std::string one = TestPrepare<SSSE3_8bit>(16, 8);
  CHECK(one == "");

  std::string two = TestPrepare<SSSE3_8bit>(32, 16);
  CHECK(two == "");

  std::string three = TestPrepare<SSSE3_8bit>(32, 32);
  CHECK(three == "");
}

TEST_CASE("Test prepare SSE2", "[prepare]") {
  if (kCPU < CPU_SSE2) return;
  std::string one = TestPrepare<SSE2_16bit>(8, 8);
  CHECK(one == "");

  std::string two = TestPrepare<SSE2_16bit>(32, 32);
  CHECK(two == "");
}


template <class Routine> std::string TestSelectColumnsB(Index rows = 32, Index cols = 16) {
  std::ostringstream out;

  AlignedVector<float> input(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    input.get()[i] = (float)rand() / (float)RAND_MAX * 256.0 - 127.0;
  }
  typedef typename Routine::Integer Integer;
  AlignedVector<Integer> prepared(rows * cols);
  Routine::PrepareB(input.get(), prepared.get(), 1, rows, cols);

  int kSelectCols = 24;
  Index select_cols[kSelectCols];
  for (int i = 0; i < kSelectCols; ++i) {
    select_cols[i] = rand() % cols;
  }

  AlignedVector<Integer> test(rows * kSelectCols);
  Routine::SelectColumnsB(prepared.get(), test.get(), rows, select_cols, select_cols + kSelectCols);

  // Select columns manually in float space.
  AlignedVector<float> selected(rows * kSelectCols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < kSelectCols; ++c) {
      assert(c + r * kSelectCols < rows * kSelectCols);
      selected[c + r * kSelectCols] = input[select_cols[c] + r * cols];
    }
  }
  AlignedVector<Integer> ref(rows * kSelectCols);
  Routine::PrepareB(selected.get(), ref.get(), 1, rows, kSelectCols);

  if (memcmp(ref.get(), test.get(), sizeof(Integer) * rows * kSelectCols)) {
    out << "SelectColumnsB failed.\nReference:\n";
    PrintMatrix(ref.get(), rows, kSelectCols, out);
    out << "Routine:\n";
    PrintMatrix(test.get(), rows, kSelectCols, out);
  }
  return out.str();
}

TEST_CASE("Test SelectColumnsB AVX512", "[select]") {
  if (kCPU < CPU_AVX512BW) return;
#ifndef INTGEMM_NO_AVX512
    std::string one = TestSelectColumnsB<AVX512_8bit>();
    CHECK(one == "");

    std::string two = TestSelectColumnsB<AVX512_16bit>(256, 256);
    CHECK(two == "");
#endif
}

TEST_CASE("Test SelectColumnsB AVX2", "[select]") {
  if (kCPU < CPU_AVX2) return;
  std::string one = TestSelectColumnsB<AVX2_8bit>(256, 256);
  CHECK(one == "");

  std::string two = TestSelectColumnsB<AVX2_16bit>(256, 256);
  CHECK(two == "");
}

TEST_CASE("Test SelectColumnsB SSSE3", "[select]") {
  if (kCPU < CPU_SSSE3) return;
  std::string one = TestSelectColumnsB<SSSE3_8bit>();
  CHECK(one == "");

  std::string two = TestSelectColumnsB<SSSE3_8bit>(256, 256);
  CHECK(two == "");
}

TEST_CASE("Test SelectColumnsB SSE2", "[select]") {
  if (kCPU < CPU_SSE2) return;
  std::string one = TestSelectColumnsB<SSE2_16bit>();
  CHECK(one == "");

  std::string two = TestSelectColumnsB<SSE2_16bit>(256, 256);
  CHECK(two == "");
}

template <class Register> std::string TestMax() {
  std::ostringstream out;
  Register r = set1_ps<Register>(-2.0);
  for (int i = 0; i < sizeof(Register) / sizeof(float); ++i) {
    Register c = r;
    reinterpret_cast<float*>(&c)[i] = -1.0;
    if (MaxFloat32(c) != -1.0) {
      out << "MaxFloat32 produced " << MaxFloat32(c) << std::endl;
    }
  }
  return out.str();
}

TEST_CASE("Test Max", "[max]") {
  std::string error = TestMax<__m128>();
  CHECK(error == "");
}

void CompareMaxAbs(const float *begin, const float *end, float test, std::ostringstream& out) {
  float largest = fabs(*std::max_element(begin, end));
  float smallest = fabs(*std::min_element(begin, end));
  largest = std::max(largest, smallest);
  if (largest != test) out << "TestMaxAbsolute error: " << largest << " versus " << test << "\n";
}

template <float (*Backend) (const float *, const float *)> std::string TestMaxAbsolute() {
  std::ostringstream out;
  const int kLength = 64;
  AlignedVector<float> test(kLength);
  // 64 tries.
  for (int t = 0; t < 64; ++t) {
    for (int i = 0; i < kLength; ++i)
      test[i] = rand() / (float)RAND_MAX * 16.0 - 8.0;
    CompareMaxAbs(test.get(), test.get() + kLength, Backend(test.get(), test.get() + kLength), out);
    test[t] = -32.0;
    CompareMaxAbs(test.get(), test.get() + kLength, Backend(test.get(), test.get() + kLength), out);
    test[t] = 32.0;
    CompareMaxAbs(test.get(), test.get() + kLength, Backend(test.get(), test.get() + kLength), out);
  }
  return out.str();
}

TEST_CASE("Test Max absolute SSE2", "[max]") {
  if (kCPU < CPU_SSE2) return;
  std::string error = TestMaxAbsolute<SSE2_MaxAbsolute>();
  CHECK(error == "");
}

TEST_CASE("Test Max absolute AVX2", "[max]") {
  if (kCPU < CPU_AVX2) return;
  std::string error = TestMaxAbsolute<AVX2_MaxAbsolute>();;
  CHECK(error == "");
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
void SlowRefFloat(const float *A, const float *B, float *C, Index A_rows, Index width, Index B_cols) {
  for (int r = 0; r < A_rows; ++r) {
    for (int c = 0; c < B_cols; ++c) {
      float sum = 0.0f;
      for (int k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      C[r * B_cols + c] = sum;
    }
  }
}

// Compute A*B slowly from integers.
template <class Integer> void SlowRefInt(const Integer *A, const Integer *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols) {
  for (int r = 0; r < A_rows; ++r) {
    for (int c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (int k = 0; k < width; ++k) {
        sum += static_cast<int16_t>(A[r * width + k]) * static_cast<int16_t>(B[k * B_cols + c]);
      }
      C[r * B_cols + c] = sum * unquant_mult;
    }
  }
}


bool Compare(const float *float_ref, const float *int_ref, const float *int_test, std::size_t size, std::ostringstream& out) {
  bool res = true;
  float int_sum = 0.0, float_sum = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    float int_diff = int_ref[i] - int_test[i];
    float float_diff = float_ref[i] - int_test[i];
    if (fabs(int_diff) > .1 || fabs(float_diff) > 1) {
      out << "Inaccurate at " << i << ' ' << float_ref[i] << ' ' << int_ref[i] << ' ' << int_test[i] << '\n';
      res = false;
    }
    int_sum += int_diff * int_diff;
    float_sum += float_diff * float_diff;
  }
  out << "Float MSE = " << sqrt(float_sum / size) << "\tInt MSE = " << sqrt(int_sum / size) << std::endl;
  return res;
}

template <class Routine> std::string TestMultiply(Index A_rows, Index width, Index B_cols) {
  typedef typename Routine::Integer Integer;
  std::ostringstream out;
  out << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  for (int i = 0; i < A_rows * width; i++) {
    A.get()[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
  }
  for (int i = 0; i < width * B_cols; ++i) {
    B.get()[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
  }
  
  float quant_mult = (sizeof(Integer) == 2) ? 1024 : 64;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<Integer> A_prep(A_rows * width), B_prep(width * B_cols);
  Routine::PrepareA(A.get(), A_prep.get(), quant_mult, A_rows, width);
  Routine::PrepareB(B.get(), B_prep.get(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);
  Routine::Multiply(A_prep.get(), B_prep.get(), test_C.get(), unquant_mult, A_rows, width, B_cols);

  AlignedVector<Integer> B_quant(width * B_cols);
  Routine::Quantize(B.get(), B_quant.get(), quant_mult, width * B_cols);
  AlignedVector<float> slowint_C(A_rows * B_cols);
  // Assuming A is just quantization here.
  SlowRefInt(A_prep.get(), B_quant.get(), slowint_C.get(), unquant_mult, A_rows, width, B_cols);

  AlignedVector<float> float_C(A_rows * B_cols);
  SlowRefFloat(A.get(), B.get(), float_C.get(), A_rows, width, B_cols);

  bool res = Compare(float_C.get(), slowint_C.get(), test_C.get(), A_rows * B_cols, out);
  if (res) {
    return std::string("");
  } else {
    return out.str();
  }
}

TEST_CASE ("Test Multiply SSE2 16bit", "[multiply]") {
  if (kCPU < CPU_SSE2) return;
  std::string one = TestMultiply<SSE2_16bit>(8, 256, 256);
  CHECK(one == "");

  std::string two = TestMultiply<SSE2_16bit>(8, 2048, 256);
  CHECK(two == "");

  std::string three = TestMultiply<SSE2_16bit>(320, 256, 256);
  CHECK(three == "");

  std::string four = TestMultiply<SSE2_16bit>(472, 256, 256);
  CHECK(four == "");

  std::string five = TestMultiply<SSE2_16bit>(248, 256, 256);
  CHECK(five == "");

  std::string six = TestMultiply<SSE2_16bit>(200, 256, 256);
  CHECK(six == "");
}

TEST_CASE ("Test Multiply SSSE3 8bit", "[multiply]") {
  if (kCPU < CPU_SSSE3) return;
  std::string one = TestMultiply<SSSE3_8bit>(8, 256, 256);
  CHECK(one == "");

  std::string two = TestMultiply<SSSE3_8bit>(8, 2048, 256);
  CHECK_FALSE(two == ""); //Expected failiure due to int saturation

  std::string three = TestMultiply<SSSE3_8bit>(320, 256, 256);
  CHECK(three == "");

  std::string four = TestMultiply<SSSE3_8bit>(472, 256, 256);
  CHECK(four == "");

  std::string five = TestMultiply<SSSE3_8bit>(248, 256, 256);
  CHECK(five == "");

  std::string six = TestMultiply<SSSE3_8bit>(200, 256, 256);
  CHECK(six == "");
}

TEST_CASE ("Test Multiply AVX2 8bit", "[multiply]") {
  if (kCPU < CPU_AVX2) return;
  std::string one = TestMultiply<AVX2_8bit>(8, 256, 256);
  CHECK(one == "");

  std::string two = TestMultiply<AVX2_8bit>(8, 2048, 256);
  CHECK_FALSE(two == ""); //Expected failiure due to int saturation

  std::string three = TestMultiply<AVX2_8bit>(320, 256, 256);
  CHECK(three == "");

  std::string four = TestMultiply<AVX2_8bit>(472, 256, 256);
  CHECK(four == "");

  std::string five = TestMultiply<AVX2_8bit>(248, 256, 256);
  CHECK(five == "");

  std::string six = TestMultiply<AVX2_8bit>(200, 256, 256);
  CHECK(six == "");
}

TEST_CASE ("Test Multiply AVX2 16bit", "[multiply]") {
  if (kCPU < CPU_AVX2) return;
  std::string one = TestMultiply<AVX2_16bit>(8, 256, 256);
  CHECK(one == "");

  std::string two = TestMultiply<AVX2_16bit>(8, 2048, 256);
  CHECK(two == "");

  std::string three = TestMultiply<AVX2_16bit>(320, 256, 256);
  CHECK(three == "");

  std::string four = TestMultiply<AVX2_16bit>(472, 256, 256);
  CHECK(four == "");

  std::string five = TestMultiply<AVX2_16bit>(248, 256, 256);
  CHECK(five == "");

  std::string six = TestMultiply<AVX2_16bit>(200, 256, 256);
  CHECK(six == "");
}

#ifndef INTGEMM_NO_AVX512
  TEST_CASE ("Test Multiply AVX512 8bit", "[multiply]") {
    if (kCPU < CPU_AVX512BW) return;
    std::string one = TestMultiply<AVX512_8bit>(8, 256, 256);
    CHECK(one == "");

    std::string two = TestMultiply<AVX512_8bit>(8, 2048, 256);
    CHECK(two == "");

    std::string three = TestMultiply<AVX512_8bit>(320, 256, 256);
    CHECK(three == "");

    std::string four = TestMultiply<AVX512_8bit>(472, 256, 256);
    CHECK(four == "");

    std::string five = TestMultiply<AVX512_8bit>(248, 256, 256);
    CHECK(five == "");

    std::string six = TestMultiply<AVX512_8bit>(200, 256, 256);
    CHECK(six == "");
  }

  TEST_CASE ("Test Multiply AVX512 16bit", "[multiply]") {
    if (kCPU < CPU_AVX512BW) return;
    std::string one = TestMultiply<AVX512_16bit>(8, 256, 256);
    CHECK(one == "");

    std::string two = TestMultiply<AVX512_16bit>(8, 2048, 256);
    CHECK(two == "");

    std::string three = TestMultiply<AVX512_16bit>(320, 256, 256);
    CHECK(three == "");

    std::string four = TestMultiply<AVX512_16bit>(472, 256, 256);
    CHECK(four == "");

    std::string five = TestMultiply<AVX512_16bit>(248, 256, 256);
    CHECK(five == "");

    std::string six = TestMultiply<AVX512_16bit>(200, 256, 256);
    CHECK(six == "");
  }
#endif

} // namespace intgemm

int main(int argc, char ** argv) {
    std::srand(45678); //If the random seed is not crucial we can forego the main alltogether
    int result = Catch::Session().run( argc, argv );
}

/*
    // Top matrix sizes from Marian
    TestBoth(8, 256, 256);
    TestBoth(8, 2048, 256);
    TestBoth(8, 2048, 256);
    TestBoth(320, 256, 256);
    TestBoth(472, 256, 256);
    TestBoth(248, 256, 256);
    TestBoth(200, 256, 256);
    return 0;
}
*/
