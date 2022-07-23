// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <filesystem>
#include <fstream>

#include "benchmark/benchmark.h"
#include "xtensor/xrandom.hpp"

#include "ppu/device/processor.h"
#include "ppu/device/processor_test_utils.h"
#include "ppu/mpc/util/simulate.h"
#include "ppu/utils/exception.h"

namespace {

std::string readFileContent(const std::filesystem::path &in) {
  if (!std::filesystem::exists(in)) {
    spdlog::error("File {} does not exist!", in.c_str());
    assert(false);
    return {};
  }

  std::ifstream in_stream(in);
  std::string contents{std::istreambuf_iterator<char>{in_stream}, {}};
  return contents;
}

ppu::ExecutableProto loadCode(const std::filesystem::path &p, size_t num_inputs,
                              size_t num_outputs) {
  ppu::ExecutableProto exec;

  exec.set_code(readFileContent(p));

  for (size_t idx = 0; idx < num_inputs; ++idx) {
    exec.add_input_names(fmt::format("input{}", idx));
  }

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    exec.add_output_names(fmt::format("output{}", idx));
  }

  return exec;
}

} // namespace

#define CREATE_FLOAT_INPUT(INDEX, SHAPE, VIS)                                  \
  {                                                                            \
    xt::xarray<float> input##INDEX = xt::random::rand<float>(SHAPE);           \
    io.InFeed(exec.input_names(INDEX), input##INDEX, VIS);                     \
  }

#define CREATE_INT_INPUT(INDEX, SHAPE, VIS)                                    \
  {                                                                            \
    xt::xarray<int> input##INDEX = xt::random::randint<int>(SHAPE);            \
    io.InFeed(exec.input_names(INDEX), input##INDEX, VIS);                     \
  }

static void BM_PPHLO(benchmark::State &state) {
  ppu::ExecutableProto exec =
      loadCode("ppu/compiler/test_data/jit_keras.mlir", 13, 6);

  ppu::RuntimeConfig config;
  config.set_field(ppu::FieldType::FM128);
  config.set_protocol(ppu::ProtocolKind::SEMI2K);

  ppu::device::LocalIo io(2, config);

  // Setup inputs
  //%arg0: tensor<1024x16x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(0, (std::vector<size_t>{1024, 16}),
                     ppu::Visibility::VIS_PUBLIC)

  // %arg1: tensor<1024x7x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(1, (std::vector<size_t>{1024, 7}),
                     ppu::Visibility::VIS_PUBLIC)
  // %arg2: tensor<1024x1x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(2, (std::vector<size_t>{1024, 1}),
                     ppu::Visibility::VIS_PUBLIC)
  // %arg3: tensor<16x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(3, (std::vector<size_t>{16}), ppu::Visibility::VIS_PUBLIC)
  //%arg4: tensor<16x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(4, (std::vector<size_t>{16}), ppu::Visibility::VIS_PUBLIC)
  // %arg5: tensor<7x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(5, (std::vector<size_t>{7}), ppu::Visibility::VIS_PUBLIC)
  // %arg6: tensor<7x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(6, (std::vector<size_t>{7}), ppu::Visibility::VIS_PUBLIC)
  //%arg7: tensor<23x1x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(7, (std::vector<size_t>{23, 1}),
                     ppu::Visibility::VIS_PUBLIC)
  // %arg8: tensor<1x!pphlo.pfxp>
  CREATE_FLOAT_INPUT(8, (std::vector<size_t>{1}), ppu::Visibility::VIS_PUBLIC)
  // %arg9: tensor<!pphlo.pfxp>
  CREATE_FLOAT_INPUT(9, (std::vector<size_t>{}), ppu::Visibility::VIS_PUBLIC)
  // %arg10: tensor<!pphlo.pfxp>
  CREATE_FLOAT_INPUT(10, (std::vector<size_t>{}), ppu::Visibility::VIS_PUBLIC)
  // %arg11: tensor<!pphlo.pint>
  CREATE_INT_INPUT(11, (std::vector<size_t>{}), ppu::Visibility::VIS_PUBLIC)
  // %arg12: tensor<!pphlo.pfxp>
  CREATE_FLOAT_INPUT(12, (std::vector<size_t>{}), ppu::Visibility::VIS_PUBLIC)

  for (const auto _ : state) {
    size_t num_iter = state.range(0);

    for (size_t i = 0; i < num_iter; ++i) {
      ::ppu::mpc::util::simulate(
          2, [&](const std::shared_ptr<::ppu::link::Context> &lctx) {
            ppu::device::Processor processor(config, lctx);
            processor.runWithEnv(exec, io.GetSymbolTable(lctx->Rank()));
          });
    }
  }
}

BENCHMARK(BM_PPHLO)->Unit(benchmark::kMillisecond)->Arg(10);

BENCHMARK_MAIN();
