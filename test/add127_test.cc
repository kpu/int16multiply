#include "test.h"

namespace intgemm {

void newBias(const float * input, float * bias, float* output, float alpha, Index rows, Index width, Index cols) {
  AlignedVector<float> intermediate(rows*cols);
  AlignedVector<float> ones(rows*width);
  for (auto&& it : ones) {
    it = 1;
  }
  SlowRefFloat(ones.begin(), input, intermediate.begin(), rows, width, cols);
  for (auto&& it : intermediate) {
    it = it*alpha;
  }


  for (Index c = 0; c<cols; c++) {
    output[c] = bias[c] - intermediate.begin()[rows*c];
  }

}

void SlowSumB(const int8_t * input, float * bias, float* output, float alpha, Index rows, Index cols) {
	for (Index r = 0; r<rows; r++) {
		for (Index c = 0; c<cols; c++) {
			output[c] += input[r * cols + c];
		}
	}

	for (Index c = 0; c<cols; c++) {
		output[c] = bias[c] + output[c]*alpha;
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
    CHECK(fabs(bias_ref[i] - bias[i]) < 0.0001);
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
  std::uniform_real_distribution<float> dist(-30.0, 30.0);
  // Create array.
  AlignedVector<float> inputB(rows * cols);
  for (auto& it : inputB) {
    it = dist(gen);
  }

  float alpha = 25;
  float quant_mult = 127/alpha;

  AlignedVector<int8_t> B_prep(inputB.size());
  AlignedVector<int8_t> B_quant(inputB.size());
  Routine::PrepareB(inputB.begin(), B_prep.begin(), quant_mult, rows, cols);
  Routine::Quantize(inputB.begin(), B_quant.begin(), quant_mult, inputB.size());


  AlignedVector<float> inputBias(cols);
  AlignedVector<float> goldBias(cols);

  for (auto& it : goldBias) {
  	it = 0;
  }
  for (auto& it : inputBias) {
    it = dist(gen);
  }

  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f);

  SlowSumB(B_quant.begin(), inputBias.begin(), goldBias.begin(), alpha, rows, cols);

  Routine::PrepareBiasFor8(1, B_prep.begin(), 1, rows, cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, inputBias.begin(), inputBias.begin()));

  CompareBiases(goldBias.begin(), inputBias.begin(), cols);
}

template <class Routine> void TestMultiplyBiasNew(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
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
  */
  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f); //Minus one to invert add_ps later on
  Routine::PrepareBiasFor8(B_prep.begin(), width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, bias.begin(), bias.begin()));
  //Routine::PrepareBiasFor8(B.begin(), bias.begin(), alpha, width, B_cols);
  Routine::Multiply8Shift(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias.begin(), test_C.begin()));

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

/*
// Bias
TEST_CASE("PrepareBias SSSE3", "[Add127]") {
	if (kCPU < CPUType::SSSE3) return;
	TestPrepareBias<SSSE3_8bit>(8,8);
	TestPrepareBias<SSSE3_8bit>(256,256);
	TestPrepareBias<SSSE3_8bit>(2048,256);
	TestPrepareBias<SSSE3_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX2", "[Add127]") {
	if (kCPU < CPUType::AVX2) return;
	TestPrepareBias<AVX2_8bit>(8,8);
	TestPrepareBias<AVX2_8bit>(256,256);
	TestPrepareBias<AVX2_8bit>(2048,256);
	TestPrepareBias<AVX2_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX512F", "[Add127]") {
	if (kCPU < CPUType::AVX512BW) return;
	#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
	TestPrepareBias<AVX512_8bit>(8,8);
	TestPrepareBias<AVX512_8bit>(256,256);
	TestPrepareBias<AVX512_8bit>(2048,256);
	TestPrepareBias<AVX512_8bit>(512,512);
	#endif
}*/

//A
TEST_CASE("PrepareA SSSE3", "[Add127]") {
	if (kCPU < CPUType::SSSE3) return;
	TestPrepareA<SSSE3_8bit>(64,64);
	TestPrepareA<SSSE3_8bit>(256,256);
	TestPrepareA<SSSE3_8bit>(512,512);
  TestPrepareA<SSSE3_8bit>(2048,256);
}

TEST_CASE("PrepareA AVX2", "[Add127]") {
	if (kCPU < CPUType::AVX2) return;
	TestPrepareA<AVX2_8bit>(64,64);
	TestPrepareA<AVX2_8bit>(256,256);
	TestPrepareA<AVX2_8bit>(512,512);
  TestPrepareA<AVX2_8bit>(2048,256);
}

TEST_CASE("PrepareA AVX512F", "[Add127]") {
	if (kCPU < CPUType::AVX512BW) return;
	#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
	TestPrepareA<AVX512_8bit>(64,64);
	TestPrepareA<AVX512_8bit>(256,256);
	TestPrepareA<AVX512_8bit>(512,512);
  TestPrepareA<AVX512_8bit>(2048,256);
	#endif
}

// Multiply

TEST_CASE ("Multiply SSSE3 8bit Shift with bias", "[Add127]") {
  if (kCPU < CPUType::SSSE3) return;
  TestMultiplyBiasNew<SSSE3_8bit>(1, 64, 8, 0.11, 0.1, 0.06, 0.05);
  TestMultiplyBiasNew<SSSE3_8bit>(8, 256, 256, 0.45, 0.54, 0.17, 0.16); // 0.064, 0.026);
  TestMultiplyBiasNew<SSSE3_8bit>(8, 2048, 256, 1.7, 1.7, 0.46, 0.43); // 4.4, 4.4);
  TestMultiplyBiasNew<SSSE3_8bit>(320, 256, 256, 0.56, 0.64, 0.16, 0.15); // 0.1, 0.01);
  TestMultiplyBiasNew<SSSE3_8bit>(472, 256, 256, 0.46, 0.62, 0.17, 0.16); // 0.1, 0.011);
  TestMultiplyBiasNew<SSSE3_8bit>(248, 256, 256, 0.48, 0.64, 0.16, 0.15); // 0.1, 0.012);
  TestMultiplyBiasNew<SSSE3_8bit>(200, 256, 256, 0.55, 0.74, 0.17, 0.16); // 0.1, 0.011);
}

TEST_CASE ("Multiply AVX2 8bit Shift with bias", "[Add127]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiplyBiasNew<AVX2_8bit>(1, 64, 8, 0.11, 0.11, 0.06, 0.05);
  TestMultiplyBiasNew<AVX2_8bit>(8, 256, 256, 0.49, 0.54, 0.17, 0.16); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(8, 2048, 256, 1.57, 1.66, 0.46, 0.46); //1.8, 1.8);
  TestMultiplyBiasNew<AVX2_8bit>(320, 256, 256, 0.49, 0.64, 0.16, 0.15); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(472, 256, 256, 0.46, 0.62, 0.17, 0.16); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(248, 256, 256, 0.48, 0.64, 0.16, 0.15); //0.1, 0);
  TestMultiplyBiasNew<AVX2_8bit>(200, 256, 256, 0.55, 0.74, 0.17, 0.16); //0.1, 0);
}
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
TEST_CASE ("Multiply AVX512F 8bit Shift with bias", "[Add127]") {
  if (kCPU < CPUType::AVX512BW) return;
  TestMultiplyBiasNew<AVX512_8bit>(1, 64, 8, 0.0001, 0.05, 0.03, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(8, 256, 256, 0.0001, 0.22, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(8, 2048, 256, 0.0001, 0.61, 0.17, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(320, 256, 256, 0.0001, 0.27, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(472, 256, 256, 0.0001, 0.33, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(248, 256, 256, 0.0001, 0.27, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(200, 256, 256, 0.0001, 0.28, 0.06, 0.001);
}
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  TEST_CASE ("Multiply AVX512VNNI 8bit Shift with bias", "[Add127]") {
    if (kCPU < CPUType::AVX512VNNI) return;
    TestMultiplyBiasNew<AVX512VNNI_8bit>(1, 64, 8, 0.0001, 0.05, 0.03, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(8, 256, 256, 0.0001, 0.22, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(8, 2048, 256, 0.0001, 0.61, 0.17, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(320, 256, 256, 0.0001, 0.27, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(472, 256, 256, 0.0001, 0.33, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(248, 256, 256, 0.0001, 0.27, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(200, 256, 256, 0.0001, 0.28, 0.06, 0.001);
  }
#endif

} //namespace intgemm
