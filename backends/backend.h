#pragma once

#include "../types.h"

namespace intgemm {

template <CPUType Cpu, typename Integer>
struct Backend;

template <typename Backend>
struct BackendInfo;

template <CPUType Cpu_, typename Integer_>
struct BackendInfo<Backend<Cpu_, Integer_>> {
  static constexpr CPUType Cpu = Cpu_;
  using Integer = Integer_;
};

}
