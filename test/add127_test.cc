#include "test/common.h"

namespace intgemm {

void SlowSumB(const float * input, float * bias, float* output, float alpha, Index rows, Index cols) {
	for (Index r = 0; r<rows; r++) {
		for (Index c = 0; c<cols; c++) {
			output[c] += input[r * cols + c];
		}
	}

	for (Index c = 0; c<cols; c++) {
		output[c] = bias[c] - output[c]*alpha;
	}
}

void CompareAs(int8_t * output_old, uint8_t * output_new, Index rows, Index cols) {
	for (Index r = 0; r<rows; r++) {
		for (Index c = 0; c<cols; c++) {
			int a = int(output_old[rows*c + r]);
			int b = int(output_new[rows*c + r]);
			INFO("Inaccurate at row: " << r << " column " << c << ' '
			 << a << ' ' << b);
			CHECK(a+127 == b);
		}
	}
}

void CompareBiases(const float *bias_ref, const float *bias, Index cols) {
  for (std::size_t i = 0; i < cols; ++i) {
  	INFO("Inaccurate at " << i << ' ' << bias_ref[i] << ' ' << bias[i]);
    CHECK(fabs(bias_ref[i] - bias[i]) < 0.1);
  }
}

template <class Routine> void TestPrepareA(Index rows, Index cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-2, 2);
  // Create array.
  AlignedVector<float> inputA(rows * cols);
  for (auto& it : inputA) {
    it = dist(gen);
  }
  AlignedVector<int8_t> oldA(rows * cols);
  AlignedVector<uint8_t> newA(rows * cols);
  float quant_mult = 64; //From example
  Routine::PrepareA(inputA.begin(), oldA.begin(), quant_mult, rows, cols);
  Routine::PrepareA(inputA.begin(), newA.begin(), quant_mult, rows, cols);
  CompareAs(oldA.begin(), newA.begin(), rows, cols);
}

template <class Routine> void TestPrepareBias(Index rows, Index cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-1000.0, 1000.0);
  // Create array.
  AlignedVector<float> inputB(rows * cols);
  for (auto& it : inputB) {
    it = dist(gen);
  }

  AlignedVector<float> inputBias(cols);
  AlignedVector<float> goldBias(cols);
  //goldBias.begin()[0] = 25;
  for (auto& it : goldBias) {
  	it = 0;
  }
  for (auto& it : inputBias) {
    it = dist(gen);
  }
  float alpha = dist(gen);
  SlowSumB(inputB.begin(), inputBias.begin(), goldBias.begin(), alpha, rows, cols);

  Routine::PrepareBiasFor8(inputB.begin(), inputBias.begin(), alpha, rows, cols);

  CompareBiases(goldBias.begin(), inputBias.begin(), cols);
}

template <class Routine> void TestMultiplyBiasNew(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  std::ostringstream info;
  info << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

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
  
  float alpha = 2.0f;
  float quant_mult = 127/alpha;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<uint8_t> A_prep(A.size());
  AlignedVector<int8_t> B_prep(B.size());
  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);

  /*REFERENCE MULTIPLICATION
  *
  *
  */
  AlignedVector<int8_t> B_quant(B.size());
  Routine::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());
  AlignedVector<float> slowint_C(test_C.size());
  // Taking the original A_preparation which means A would be int8_t
  AlignedVector<int8_t> A_prep2(A.size());
  Routine::PrepareA(A.begin(), A_prep2.begin(), quant_mult, A_rows, width);
  SlowRefInt(A_prep2.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, bias.begin());

  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols, bias.begin());

  /*ACTUAL MULTIPLICATION
  *
  *
  */
  Routine::PrepareBiasFor8(B.begin(), bias.begin(), alpha, width, B_cols);
  Routine::Multiply8new(A_prep.begin(), B_prep.begin(), BiasAddUnquantizeC(test_C.begin(), bias.begin(), unquant_mult), A_rows, width, B_cols);

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

// Bias
TEST_CASE("PrepareBias SSSE3", "[Add127]") {
	if (kCPU < CPU_SSSE3) return;
	TestPrepareBias<SSSE3_8bit>(8,8);
	TestPrepareBias<SSSE3_8bit>(256,256);
	TestPrepareBias<SSSE3_8bit>(2048,256);
	TestPrepareBias<SSSE3_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX2", "[Add127]") {
	if (kCPU < CPU_AVX2) return;
	TestPrepareBias<AVX2_8bit>(8,8);
	TestPrepareBias<AVX2_8bit>(256,256);
	TestPrepareBias<AVX2_8bit>(2048,256);
	TestPrepareBias<AVX2_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX512F", "[Add127]") {
	if (kCPU < CPU_AVX512BW) return;
	#ifndef INTGEMM_NO_AVX512
	TestPrepareBias<AVX512_8bit>(8,8);
	TestPrepareBias<AVX512_8bit>(256,256);
	TestPrepareBias<AVX512_8bit>(2048,256);
	TestPrepareBias<AVX512_8bit>(512,512);
	#endif
}

//A
TEST_CASE("PrepareA SSSE3", "[Add127]") {
	if (kCPU < CPU_SSSE3) return;
	TestPrepareA<SSSE3_8bit>(8,8);
	TestPrepareA<SSSE3_8bit>(256,256);
	TestPrepareA<SSSE3_8bit>(2048,256);
	TestPrepareA<SSSE3_8bit>(512,512);
}

TEST_CASE("PrepareA AVX2", "[Add127]") {
	if (kCPU < CPU_AVX2) return;
	TestPrepareA<AVX2_8bit>(8,8);
	TestPrepareA<AVX2_8bit>(256,256);
	TestPrepareA<AVX2_8bit>(2048,256);
	TestPrepareA<AVX2_8bit>(512,512);
}

TEST_CASE("PrepareA AVX512F", "[Add127]") {
	if (kCPU < CPU_AVX512BW) return;
	#ifndef INTGEMM_NO_AVX512
	TestPrepareA<AVX512_8bit>(8,8);
	TestPrepareA<AVX512_8bit>(256,256);
	TestPrepareA<AVX512_8bit>(2048,256);
	TestPrepareA<AVX512_8bit>(512,512);
	#endif
}

// Multiply

TEST_CASE ("Multiply SSSE3 8bit with new bias", "[Add127]") {
  if (kCPU < CPU_SSSE3) return;
  TestMultiplyBiasNew<SSSE3_8bit>(1, 64, 8, 1.2, 1.2, 0.064, 0.05);
  TestMultiplyBiasNew<SSSE3_8bit>(8, 256, 256, 30, 30, 7.3, 7.3); // 0.064, 0.026);
  TestMultiplyBiasNew<SSSE3_8bit>(8, 2048, 256, 158, 158, 46.0, 46.0); // 4.4, 4.4);
  TestMultiplyBiasNew<SSSE3_8bit>(320, 256, 256, 31, 31, 7.4, 7.4); // 0.1, 0.01);
  TestMultiplyBiasNew<SSSE3_8bit>(472, 256, 256, 43, 43, 8.3, 8.3); // 0.1, 0.011);
  TestMultiplyBiasNew<SSSE3_8bit>(248, 256, 256, 42, 42, 7.4, 7.4); // 0.1, 0.012);
  TestMultiplyBiasNew<SSSE3_8bit>(200, 256, 256, 36, 36, 7.3, 7.3); // 0.1, 0.011);
}

TEST_CASE ("Multiply AVX2 8bit with new bias", "[Add127]") {
  if (kCPU < CPU_AVX2) return;
  TestMultiplyBiasNew<AVX2_8bit>(1, 64, 8, 1.2, 1.2, 0.064, 0.05);
  TestMultiplyBiasNew<AVX2_8bit>(8, 256, 256, 17, 17, 3.6, 3.6); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(8, 2048, 256, 132, 132, 41.0, 41.0); //1.8, 1.8);
  TestMultiplyBiasNew<AVX2_8bit>(320, 256, 256, 18, 18, 3.7, 3.7); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(472, 256, 256, 28, 28, 3.9, 3.9); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(248, 256, 256, 25, 25, 3.9, 3.9); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(200, 256, 256, 19, 19, 3.6, 3.6); //0.1, 0);
}

TEST_CASE ("Multiply AVX512F 8bit with new bias", "[Add127]") {
  if (kCPU < CPU_AVX512BW) return;
  TestMultiplyBiasNew<AVX512_8bit>(1, 64, 8, 0.11, 0.11, 0.06, 0.05);
  TestMultiplyBiasNew<AVX512_8bit>(8, 256, 256, 7.5, 7.5, 0.99, 0.99); //, 1.6, 1.6); //0.1, 0);
  TestMultiplyBiasNew<AVX512_8bit>(8, 2048, 256, 109, 109, 31.0, 31.0); //1.8, 1.8);
  TestMultiplyBiasNew<AVX512_8bit>(320, 256, 256, 9, 9, 1.1, 1.1); //0.1, 0);
  TestMultiplyBiasNew<AVX512_8bit>(472, 256, 256, 10, 10, 1.2, 1.2); //0.1, 0);
  TestMultiplyBiasNew<AVX512_8bit>(248, 256, 256, 8.2, 8.2, 1.1, 1.1); //0.1, 0);
  TestMultiplyBiasNew<AVX512_8bit>(200, 256, 256, 8.3, 8.3, 1.2, 1.2); //0.1, 0);
}

} //namespace intgemm
