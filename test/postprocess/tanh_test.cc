#include "test/test.h"
#include "aligned.h"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_AVX2 TEST_CASE("Tanh AVX2",) {
  if (kCPU < CPUType::AVX2)
    return;

  const float error_tolerance = 0.001f;

  AlignedVector<float> input(8);
  AlignedVector<float> output(8);

  std::generate(input.begin(), input.end(), [] () { static int n = -4; return n++ / 4.f; });

  auto postproc = PostprocessImpl<Tanh, CPUType::AVX2>(Tanh());
  *output.as<__m256>() = postproc.run(*input.as<__m256>(), 0, 0);

  CHECK_EPS(output[0], -0.7615942f, error_tolerance); // input = -1
  CHECK_EPS(output[1], -0.6351490f, error_tolerance); // input = -0.75
  CHECK_EPS(output[2], -0.4621172f, error_tolerance); // input = -0.5
  CHECK_EPS(output[3], -0.2449187f, error_tolerance); // input = -0.25
  CHECK_EPS(output[4],  0.0f      , error_tolerance); // input =  0
  CHECK_EPS(output[5],  0.2449187f, error_tolerance); // input =  0.25
  CHECK_EPS(output[6],  0.4621172f, error_tolerance); // input =  0.5
  CHECK_EPS(output[7],  0.6351490f, error_tolerance); // input =  0.75
}

}
