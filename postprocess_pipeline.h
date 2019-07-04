#pragma once

#include "intrinsics.h"
#include "types.h"
#include "utils.h"

#include <tuple>

namespace intgemm {

template <typename... Stages>
using PostprocessPipeline = std::tuple<Stages...>;

template <typename... Stages>
constexpr std::tuple<Stages...> CreatePostprocessPipeline(Stages&&... stages) {
  return std::make_tuple(std::forward<Stages>(stages)...);
}

template <typename Postprocess, CPUType CpuType>
class PostprocessImpl;

namespace { // anonymous namespace

template <typename... Stages>
using input_register_type = typename std::tuple_element<
    0,
    std::tuple<Stages...>
  >::type::InputRegister;

template <typename... Stages>
using output_register_type = typename std::tuple_element<
    std::tuple_size<std::tuple<Stages...>>::value - 1,
    std::tuple<Stages...>
  >::type::OutputRegister;

template <typename FirstStage, typename... RestStages>
constexpr std::tuple<RestStages...> DropFirstStage(const std::tuple<FirstStage, RestStages...>& pipeline) {
  return make_subtuple(pipeline, sequence_popfront<make_sequence<sizeof...(RestStages) + 1>>());
}

template <CPUType CpuType>
constexpr std::tuple<> InitPostprocessPipelineImpl(std::tuple<> pipeline) {
  return std::tuple<>();
}

template <CPUType CpuType, typename FirstStage, typename... RestStages>
constexpr std::tuple<PostprocessImpl<FirstStage, CpuType>, PostprocessImpl<RestStages, CpuType>...> InitPostprocessPipelineImpl(std::tuple<FirstStage, RestStages...> pipeline) {
  return std::tuple_cat(
    std::tuple<PostprocessImpl<FirstStage, CpuType>>(PostprocessImpl<FirstStage, CpuType>(std::get<0>(pipeline))),
    InitPostprocessPipelineImpl<CpuType, RestStages...>(DropFirstStage(pipeline))
  );
}

template <CPUType CpuType>
struct RunPostprocessPipelineImpl;

#define RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(attribute, cpu_type) \
  template <> \
  struct RunPostprocessPipelineImpl<cpu_type> { \
    template <typename Stage> \
    attribute static constexpr output_register_type<Stage> \
    run(std::tuple<Stage> pipeline, input_register_type<Stage> input, Index row, Index col) { \
      return std::get<0>(pipeline).run(input, row, col); \
    } \
    template <typename... Stages> \
    attribute static constexpr output_register_type<Stages...> \
    run(std::tuple<Stages...> pipeline, input_register_type<Stages...> input, Index row, Index col) { \
      return run( \
        DropFirstStage(pipeline), \
        std::get<0>(pipeline).run(input, row, col), row, col); \
    } \
  };

RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_SSE2, CPUType::SSE2)
RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_SSSE3, CPUType::SSSE3)
RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_AVX2, CPUType::AVX2)
RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_AVX512BW, CPUType::AVX512BW)

} // anonymous namespace

template <CPUType CpuType, typename... Stages>
class InitedPostprocessPipeline {};

template <CPUType CpuType, typename... Stages>
constexpr InitedPostprocessPipeline<CpuType, Stages...> InitPostprocessPipeline(std::tuple<Stages...> pipeline) {
  return InitedPostprocessPipeline<CpuType, Stages...>(pipeline);
}

#define INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(attribute, cpu_type) \
  template <typename... Stages> \
  class InitedPostprocessPipeline<cpu_type, Stages...> { \
  public: \
    using InputRegister = input_register_type<PostprocessImpl<Stages, cpu_type>...>; \
    using OutputRegister = output_register_type<PostprocessImpl<Stages, cpu_type>...>; \
    InitedPostprocessPipeline(std::tuple<Stages...> pipeline) \
        : inited_pipeline(InitPostprocessPipelineImpl<cpu_type, Stages...>(pipeline)) {} \
    attribute inline OutputRegister run(InputRegister input, Index row, Index col) { \
      return RunPostprocessPipelineImpl<cpu_type>::run(inited_pipeline, input, row, col); \
    } \
    attribute inline void run(const InputRegister* input, Index rows, Index cols, OutputRegister* output) { \
      for (Index r = 0; r < rows; ++r) { \
        Index i = r * cols; \
        for (Index c = 0; c < cols; ++c, ++i) \
          output[i] = RunPostprocessPipelineImpl<cpu_type>::run(inited_pipeline, input[i], r, c); \
      } \
    } \
  private: \
    const std::tuple<PostprocessImpl<Stages, cpu_type>...> inited_pipeline; \
  };

INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_SSE2, CPUType::SSE2)
INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_SSSE3, CPUType::SSSE3)
INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_AVX2, CPUType::AVX2)
INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_AVX512BW, CPUType::AVX512BW)

}
