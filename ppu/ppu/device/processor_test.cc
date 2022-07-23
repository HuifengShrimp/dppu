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


#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"

#include "ppu/device/io_accessor.h"
#include "ppu/device/processor.h"
#include "ppu/device/processor_test_utils.h"
#include "ppu/device/symbol_table.h"
#include "ppu/mpc/ref2k/ref2k.h"
#include "ppu/mpc/util/simulate.h"

namespace ppu::device {
namespace {

class Runner {
public:
  Runner(size_t world_size, FieldType field, ProtocolKind protocol)
      : world_size_(world_size) {
    config_.set_field(field);
    config_.set_protocol(protocol);
    config_.set_enable_type_checker(true);
    // config_.set_enable_action_trace(true);
    io_ = std::make_unique<LocalIo>(world_size_, config_);
  }

  void disableSizeCheck() { config_.set_enable_type_checker(false); }

  auto &getConfig() { return config_; }

  template <typename T>
  void addInput(const T &input, Visibility vis = Visibility::VIS_PUBLIC) {
    const std::string name = fmt::format("input{}", input_idx_++);
    io_->InFeed(name, input, vis);
    exec_.add_input_names(name);
  }

  void run(const std::string &mlir, size_t num_output = 1) {
    for (size_t idx = 0; idx < num_output; ++idx) {
      exec_.add_output_names(fmt::format("output{}", idx));
    }
    exec_.set_code(mlir);
    ::ppu::mpc::util::simulate(
        world_size_, [&](const std::shared_ptr<link::Context> &lctx) {
          Processor processor(config_, lctx);
          processor.runWithEnv(exec_, io_->GetSymbolTable(lctx->Rank()));
        });
  }

  template <typename T>
  void verifyOutput(const T *expected, size_t idx = 0) {
    PtType output_type =
        std::is_integral_v<T> ? PtType::PT_I32 : PtType::PT_F32;

    const auto &out = io_->OutFeed(fmt::format("output{}", idx), output_type);

    size_t numel = out.numel();
    const auto *in_ptr = static_cast<const T *>(out.data());

    // TODO: handle strides
    for (size_t idx = 0; idx < numel; ++idx) {
      if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(in_ptr[idx], expected[idx]) << "idx = " << idx << "\n";
      } else {
        EXPECT_FLOAT_EQ(in_ptr[idx], expected[idx]) << "idx = " << idx << "\n";
      }
    }
  }

  template <typename T, std::enable_if_t<std::is_scalar_v<T>, bool> = true>
  void verifyScalarOutput(T expected, size_t idx = 0) {
    verifyOutput(&expected, idx);
  }

private:
  size_t world_size_;
  RuntimeConfig config_;
  std::unique_ptr<LocalIo> io_;
  size_t input_idx_{0};
  ExecutableProto exec_;
};

} // namespace

class ProcessorTest : public ::testing::TestWithParam<
                          std::tuple<size_t, FieldType, ProtocolKind>> {};

TEST_P(ProcessorTest, Basic) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(2);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pint>, %arg1: tensor<!pphlo.pint>) -> (tensor<!pphlo.pint>) {
  %0 = "pphlo.add"(%arg0, %arg1) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  return %0 : tensor<!pphlo.pint>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ProcessorTest, WithConst) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 1}, {1, 1}}));

  r.run(R"(
func @main(%arg0: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    %0 = "pphlo.constant"() {value = dense<[[1,2],[3,4]]> : tensor<2x2xi64>} : () -> tensor<2x2x!pphlo.pint>
    %1 = "pphlo.add"(%arg0, %0) : (tensor<2x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %1 : tensor<2x2x!pphlo.pint>
})");

  std::array<int, 4> expect = {2, 3, 4, 5};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, RowConcate) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 2, 3}, {4, 5, 6}}));
  r.addInput(xt::xarray<int>({{7, 8, 9}, {10, 11, 12}}));

  r.run(R"(
func @main(%arg0: tensor<2x3x!pphlo.pint>, %arg1: tensor<2x3x!pphlo.pint>) -> (tensor<4x3x!pphlo.pint>) {
  %0 = "pphlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<2x3x!pphlo.pint>, tensor<2x3x!pphlo.pint>) -> tensor<4x3x!pphlo.pint>
  return %0 : tensor<4x3x!pphlo.pint>
})");

  std::array<int, 12> expect = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ColConcate) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{1, 2, 3}, {4, 5, 6}}));
  r.addInput(xt::xarray<int>({{7, 8, 9}, {10, 11, 12}}));

  r.run(R"(
func @main(%arg0: tensor<2x3x!pphlo.pint>, %arg1: tensor<2x3x!pphlo.pint>) -> (tensor<2x6x!pphlo.pint>) {
  %0 = "pphlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<2x3x!pphlo.pint>, tensor<2x3x!pphlo.pint>) -> tensor<2x6x!pphlo.pint>
  return %0 : tensor<2x6x!pphlo.pint>
}
  )");

  std::array<int, 12> expect = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, VariadicConcate) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({1, 2, 3}));

  r.run(R"(
func @main(%arg0: tensor<3x!pphlo.pint>) -> (tensor<9x!pphlo.pint>) {
  %0 = "pphlo.concatenate"(%arg0, %arg0,%arg0) {dimension = 0 : i64} : (tensor<3x!pphlo.pint>, tensor<3x!pphlo.pint>, tensor<3x!pphlo.pint>) -> tensor<9x!pphlo.pint>
  return %0 : tensor<9x!pphlo.pint>
})");

  std::array<int, 12> expect = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, Slice) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  r.run(R"(
func @main(%arg0: tensor<4x3x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
  %0 = "pphlo.slice"(%arg0) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<[2, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<4x3x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
  return %0 : tensor<2x2x!pphlo.pint>
})");

  std::array<int, 4> expect = {7, 8, 10, 11};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, SliceStride) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({                          //
                              {0, 1, 2, 3, 4, 5},       //
                              {6, 7, 8, 9, 10, 11},     //
                              {12, 13, 14, 15, 16, 17}, //
                              {18, 19, 20, 21, 22, 23}}));

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pint>) -> (tensor<2x3x!pphlo.pint>) {
  %0 = "pphlo.slice"(%arg0) {limit_indices = dense<[4, 7]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<[2, 2]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pint>) -> tensor<2x3x!pphlo.pint>
  return %0 : tensor<2x3x!pphlo.pint>
})");

  std::array<int, 6> expect = {0,  2,  4, //
                               12, 14, 16};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, Reshape) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(xt::xarray<int>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}));

  // Reshape to 2x6
  r.run(R"(
func @main(%arg0: tensor<4x3x!pphlo.pint>) -> (tensor<2x6x!pphlo.pint>) {
  %0 = "pphlo.reshape"(%arg0) : (tensor<4x3x!pphlo.pint>) -> tensor<2x6x!pphlo.pint>
  return %0 : tensor<2x6x!pphlo.pint>
}
  )");

  std::array<int, 12> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, While) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));
  r.addInput(1);
  r.addInput(3);

  // while(x < y) { x = x + 1; }
  r.run(R"(
func @main(%arg0: tensor<!pphlo.pint>, %arg1: tensor<!pphlo.pint>) -> tensor<!pphlo.pint> {
  %0, %1 = "pphlo.while"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<!pphlo.pint>, %arg3: tensor<!pphlo.pint>):  // no predecessors
    %2 = "pphlo.less"(%arg2, %arg3) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
  },  {
  ^bb0(%arg2: tensor<!pphlo.pint>, %arg3: tensor<!pphlo.pint>):  // no predecessors
    %2 = "pphlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<!pphlo.pint>
    %3 = "pphlo.add"(%arg2, %2) {name = "compare.0"} : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
    "pphlo.return"(%3, %arg3) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> ()
  }) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> (tensor<!pphlo.pint>, tensor<!pphlo.pint>)
  return %0 : tensor<!pphlo.pint>
})");

  r.verifyScalarOutput(3);
}

TEST_P(ProcessorTest, Reduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  // FIXME: figure out why
  r.disableSizeCheck();

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<10x!pphlo.pint>) -> (tensor<!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce"(%arg0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
         "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<10x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  return %1 :  tensor<!pphlo.pint>
})");

  xt::xarray<int> expect = xt::sum(in1);
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, VReduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  // FIXME: figure out why
  r.disableSizeCheck();

  const xt::xarray<int> in1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<10x!pphlo.pint>) -> (tensor<!pphlo.pint>, tensor<!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1:2 = "pphlo.reduce"(%arg0, %arg0, %0, %0) ( {
        ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>, %arg3: tensor<!pphlo.pint>, %arg4: tensor<!pphlo.pint>): // no predecessors
         %2 = "pphlo.add"(%arg1, %arg3) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
         %3 = "pphlo.maximum"(%arg2, %arg4) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
         "pphlo.return"(%2, %3) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<10x!pphlo.pint>, tensor<10x!pphlo.pint>, tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> (tensor<!pphlo.pint>, tensor<!pphlo.pint>)
  return %1#0, %1#1 : tensor<!pphlo.pint>, tensor<!pphlo.pint>
})",
        2);

  xt::xarray<int> expect0 = xt::sum(in1);
  r.verifyOutput(expect0.data(), 0);
  xt::xarray<int> expect1 = {10};
  r.verifyOutput(expect1.data(), 1);
}

TEST_P(ProcessorTest, MaxReduce) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  // FIXME: figure out why
  r.disableSizeCheck();

  const xt::xarray<float> in1({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<1x10x!pphlo.pfxp>) -> (tensor<1x!pphlo.pfxp>) {
  // Initial value is -inf
  %0 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
  %1 = "pphlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<!pphlo.pfxp>, %arg2: tensor<!pphlo.pfxp>):  // no predecessors
    %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    "pphlo.return"(%2) : (tensor<!pphlo.pfxp>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10x!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<1x!pphlo.pfxp>
  return %1 :  tensor<1x!pphlo.pfxp>
})");

  xt::xarray<float> expect = {0};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindow) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{-7, 6, 1, -14, -7, 5},
                               {-13, -14, -11, 13, -13, -7},
                               {8, -11, 12, -2, 14, 4},
                               {0, 13, 3, -13, -7, -3}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.add"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<[2,3]> : tensor<2xi64>} : (tensor<4x6x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    
  return %1 :  tensor<2x2x!pphlo.pint>
})");

  xt::xarray<int> expect = {{-38, -23}, {25, -7}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowDefaultStrides) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {{-7, 6, 1, -14, -7, 5},
                               {-13, -14, -11, 13, -13, -7},
                               {8, -11, 12, -2, 14, 4},
                               {0, 13, 3, -13, -7, -3}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x6x!pphlo.pint>) -> (tensor<3x4x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[2,3]> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<4x6x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<3x4x!pphlo.pint>
    
  return %1 :  tensor<3x4x!pphlo.pint>
})");

  xt::xarray<int> expect = {
      {6, 13, 13, 13}, {12, 13, 14, 14}, {13, 13, 14, 14}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowIotaWindowDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<2> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<4x4x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    
  return %1 :  tensor<2x2x!pphlo.pint>
})");

  xt::xarray<int> expect = {{10, 11}, {14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowIotaStrideWindowDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pint>) -> (tensor<1x1x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<2> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<4x4x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<1x1x!pphlo.pint>
    
  return %1 :  tensor<1x1x!pphlo.pint>
})");

  xt::xarray<int> expect = {10};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowMaxIotaBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pint>) -> (tensor<6x6x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<4x4x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<6x6x!pphlo.pint>
    
  return %1 :  tensor<6x6x!pphlo.pint>
})");

  xt::xarray<int> expect = {{0, 1, 1, 2, 2, 3},    {4, 5, 5, 6, 6, 7},
                            {4, 5, 5, 6, 6, 7},    {8, 9, 9, 10, 10, 11},
                            {8, 9, 9, 10, 10, 11}, {12, 13, 13, 14, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowMaxIotaStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pint>) -> (tensor<3x3x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<4x4x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<3x3x!pphlo.pint>
    
  return %1 :  tensor<3x3x!pphlo.pint>
})");

  xt::xarray<int> expect = {{0, 1, 2}, {4, 5, 6}, {8, 9, 10}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowMaxIotaStrideBothDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pint>) -> (tensor<3x3x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<0> : tensor<2x2xi64>, window_dilations = dense<2> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<4x4x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<3x3x!pphlo.pint>
    
  return %1 :  tensor<3x3x!pphlo.pint>
})");

  xt::xarray<int> expect = {{5, 6, 7}, {9, 10, 11}, {13, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, ReduceWindowMaxIotaPaddingStrideBaseDilation) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in1 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  r.addInput(in1);

  r.run(R"(
func @main(%arg0: tensor<4x4x!pphlo.pint>) -> (tensor<3x3x!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.reduce_window"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<!pphlo.pint>, %arg2: tensor<!pphlo.pint>):  // no predecessors
      %2 = "pphlo.maximum"(%arg1, %arg2) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
      "pphlo.return"(%2) : (tensor<!pphlo.pint>) -> ()
    }) {base_dilations = dense<2> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<3> : tensor<2xi64>, window_strides = dense<3> : tensor<2xi64>} : (tensor<4x4x!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<3x3x!pphlo.pint>
    
  return %1 :  tensor<3x3x!pphlo.pint>
})");

  xt::xarray<int> expect = {{0, 2, 3}, {8, 10, 11}, {12, 14, 15}};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, If) {
  const auto *prog = R"(
 func @main(%arg0: tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp> {
  %0 = "pphlo.constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
  %1 = "pphlo.less"(%arg0, %0) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
  %2 = "pphlo.if"(%1, %arg0, %arg0) ( {
  ^bb0(%arg1: tensor<!pphlo.pfxp>):  // no predecessors
    %3 = "pphlo.multiply"(%arg1, %arg1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    "pphlo.return"(%3) : (tensor<!pphlo.pfxp>) -> ()
  },  {
  ^bb0(%arg1: tensor<!pphlo.pfxp>):  // no predecessors
    %3 = "pphlo.add"(%arg1, %arg1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    "pphlo.return"(%3) : (tensor<!pphlo.pfxp>) -> ()
  }) {operand_segment_sizes = dense<1> : vector<3xi32>} : (tensor<!pphlo.pint>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
  return %2 : tensor<!pphlo.pfxp>
}
)";
  {
    // True case
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(2.5F);

    r.run(prog);

    r.verifyScalarOutput(2.5f * 2.5f);
  }

  {
    // False case
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(12.5F);

    r.run(prog);

    r.verifyScalarOutput(12.5f + 12.5f);
  }
}

TEST_P(ProcessorTest, SecretControlflow) {
  const auto *prog = R"(
func @main(%arg0: tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp> {
  %0 = "pphlo.constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<!pphlo.pfxp>
  %1 = "pphlo.protect"(%arg0) : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.sfxp>
  %2 = "pphlo.less"(%1, %0) : (tensor<!pphlo.sfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.sint>
  %3 = "pphlo.if"(%2, %arg0, %arg0) ( {
  ^bb0(%arg1: tensor<!pphlo.pfxp>):  // no predecessors
    %4 = "pphlo.multiply"(%arg1, %arg1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    "pphlo.return"(%4) : (tensor<!pphlo.pfxp>) -> ()
  },  {
  ^bb0(%arg1: tensor<!pphlo.pfxp>):  // no predecessors
    %4 = "pphlo.add"(%arg1, %arg1) : (tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
    "pphlo.return"(%4) : (tensor<!pphlo.pfxp>) -> ()
  }) {operand_segment_sizes = dense<1> : vector<3xi32>} : (tensor<!pphlo.sint>, tensor<!pphlo.pfxp>, tensor<!pphlo.pfxp>) -> tensor<!pphlo.pfxp>
  return %3 : tensor<!pphlo.pfxp>
}
)";
  // default
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.addInput(2.5F);

    EXPECT_THROW(r.run(prog), EnforceNotMet);
  }
  // reveal
  {
    Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
             std::get<2>(GetParam()));

    r.getConfig().set_reveal_secret_condition(true);

    r.addInput(2.5F);

    r.run(prog);

    r.verifyScalarOutput(2.5f * 2.5f);
  }
}

TEST_P(ProcessorTest, Iota1D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func @main() -> (tensor<4x!pphlo.pint>) {
    %0 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4x!pphlo.pint>
    return %0 : tensor<4x!pphlo.pint>
})");

  std::array<int, 4> expect = {0, 1, 2, 3};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, Iota2D) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.run(R"(
func @main() -> (tensor<4x2x!pphlo.pint>) {
    %0 = "pphlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4x2x!pphlo.pint>
    return %0 : tensor<4x2x!pphlo.pint>
})");

  std::array<int, 8> expect = {0, 1, 0, 1, 0, 1, 0, 1};
  r.verifyOutput(expect.data());
}

TEST_P(ProcessorTest, SimpleBitcast) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  float in = 2.0F;
  r.addInput(in);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pfxp>) -> (tensor<!pphlo.pint>) {
    %0 = "pphlo.bitcast_convert"(%arg0) {elsize = 32 : i64} : (tensor<!pphlo.pfxp>) -> tensor<!pphlo.pint>
    return %0 : tensor<!pphlo.pint>
})");

  r.verifyOutput(reinterpret_cast<int32_t *>(&in));
}

TEST_P(ProcessorTest, Gather1) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // Start indices
  r.addInput(xt::xarray<int>{0, 2});

  r.run(R"(
func @main(%arg0: tensor<3x3x!pphlo.pint>, %arg1: tensor<2x!pphlo.pint>) -> (tensor<2x3x!pphlo.pint>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x3x!pphlo.pint>
    return %0 : tensor<2x3x!pphlo.pint>
})");

  xt::xarray<int> expected = {{1, 2, 3}, {7, 8, 9}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Gather2) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // Start indices
  r.addInput(xt::xarray<int>{0, 2});

  r.run(R"(
func @main(%arg0: tensor<3x3x!pphlo.pint>, %arg1: tensor<2x!pphlo.pint>) -> (tensor<3x2x!pphlo.pint>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[3,1]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<3x2x!pphlo.pint>
    return %0 : tensor<3x2x!pphlo.pint>
})");

  xt::xarray<int> expected = {{1, 3}, {4, 6}, {7, 9}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, GatherBatch) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(xt::xarray<int>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // Start indices
  r.addInput(xt::xarray<int>{{0, 2}, {2, 1}});

  r.run(R"(
func @main(%arg0: tensor<3x3x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x3x2x!pphlo.pint>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[3,1]> : tensor<2xi64>} : (tensor<3x3x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x3x2x!pphlo.pint>
    return %0 : tensor<2x3x2x!pphlo.pint>
})");

  xt::xarray<int> expected = {{{1, 3}, {4, 6}, {7, 9}},
                              {{3, 2}, {6, 5}, {9, 8}}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, GatherNd) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  const xt::xarray<int> in = {{{-1, 1}, {-2, 2}, {-3, 3}},
                              {{-4, 4}, {-5, 5}, {-6, 6}},
                              {{-7, 7}, {-8, 8}, {-9, 9}}};
  r.addInput(in);
  // Start indices
  r.addInput(xt::xarray<int>{{0, 0}, {1, 0}});

  r.run(R"(
func @main(%arg0: tensor<3x3x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0,1], start_index_map = [0,1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1,1,2]> : tensor<3xi64>} : (tensor<3x3x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
})");

  xt::xarray<int> expected = {{-1, 1}, {-4, 4}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, GatherNdNonDefaultIndexVectorDim) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> in = {{{-1, 1}, {-2, 2}, {-3, 3}},
                        {{-4, 4}, {-5, 5}, {-6, 6}},
                        {{-7, 7}, {-8, 8}, {-9, 9}}};
  r.addInput(in);
  // Start indices
  r.addInput(xt::xarray<int>{{0, 0}, {1, 0}});

  r.run(R"(
func @main(%arg0: tensor<3x3x2x!pphlo.pint>, %arg1: tensor<2x2x!pphlo.pint>) -> (tensor<2x2x!pphlo.pint>) {
    %0 = "pphlo.gather"(%arg0, %arg1) {dimension_numbers = #pphlo.gather<offset_dims = [1], collapsed_slice_dims = [0,1], start_index_map = [0,1], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = dense<[1,1,2]> : tensor<3xi64>} : (tensor<3x3x2x!pphlo.pint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.pint>
    return %0 : tensor<2x2x!pphlo.pint>
})");

  xt::xarray<int> expected = {{-2, 2}, {-1, 1}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Simple4x4Conv2DWith2x2Kernel) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {{{
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  }}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{
      {5, 6},
      {7, 8},
  }}};
  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<1x1x4x4x!pphlo.pfxp>, %arg1: tensor<1x1x2x2x!pphlo.pfxp>) -> (tensor<1x1x4x4x!pphlo.pfxp>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x4x4x!pphlo.pfxp>, tensor<1x1x2x2x!pphlo.pfxp>) -> tensor<1x1x4x4x!pphlo.pfxp>
    return %0 : tensor<1x1x4x4x!pphlo.pfxp>
})");

  xt::xarray<float> expected = {{{
      {100, 126, 152, 76},
      {204, 230, 256, 124},
      {308, 334, 360, 172},
      {149, 160, 171, 80},
  }}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Conv2DGeneralDimensions) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<float> lhs = {
      {{{1, 2, 3, 4}}, {{5, 6, 7, 8}}, {{9, 10, 11, 12}}},
      {{{13, 14, 15, 16}}, {{17, 18, 19, 20}}, {{21, 22, 23, 24}}}};
  r.addInput(lhs);

  xt::xarray<float> rhs = {{{{1, 7, 13}, {4, 10, 16}},
                            {{2, 8, 14}, {5, 11, 17}},
                            {{3, 9, 15}, {6, 12, 18}}}};

  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<2x3x1x4x!pphlo.pfxp>, %arg1: tensor<1x3x2x3x!pphlo.pfxp>) -> (tensor<1x1x1x2x!pphlo.pfxp>) {
    %0 = pphlo.convolution(%arg0, %arg1) dim_numbers = [f, 0, b, 1]x[o, 1, i, 0]->[f, 0, b, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x1x4x!pphlo.pfxp>, tensor<1x3x2x3x!pphlo.pfxp>) -> tensor<1x1x1x2x!pphlo.pfxp>
    return %0 : tensor<1x1x1x2x!pphlo.pfxp>
})");

  xt::xarray<float> expected = {{{{2514, 2685}}}};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, ShiftLeft) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> lhs = {1, 1};
  r.addInput(lhs);

  xt::xarray<int> rhs = {1, 2};
  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<2x!pphlo.pint>, %arg1: tensor<2x!pphlo.pint>) -> (tensor<2x!pphlo.pint>) {
    %0 = "pphlo.shift_left"(%arg0, %arg1) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    return %0 : tensor<2x!pphlo.pint>
})");

  xt::xarray<int> expected = {1 << 1, 1 << 2};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, RightShiftLogical) {
  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  xt::xarray<int> lhs = {1 << 4, 1 << 4};
  r.addInput(lhs);

  xt::xarray<int> rhs = {1, 2};
  r.addInput(rhs);

  r.run(R"(
func @main(%arg0: tensor<2x!pphlo.pint>, %arg1: tensor<2x!pphlo.pint>) -> (tensor<2x!pphlo.pint>) {
    %0 = "pphlo.shift_right_logical"(%arg0, %arg1) : (tensor<2x!pphlo.pint>, tensor<2x!pphlo.pint>) -> tensor<2x!pphlo.pint>
    return %0 : tensor<2x!pphlo.pint>
})");

  xt::xarray<int> expected = {1 << 3, 1 << 2};
  r.verifyOutput(expected.data());
}

TEST_P(ProcessorTest, Maximum) {
  if (std::get<1>(GetParam()) == FM32) {
    return; // Ring type is not large enough to hold value
  }

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(10);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pint>) -> (tensor<!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<-2147483648> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.maximum"(%0, %arg0) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  return %1 :  tensor<!pphlo.pint>
})");

  int expected = 10;
  r.verifyOutput(&expected);
}

TEST_P(ProcessorTest, Minimum) {
  if (std::get<1>(GetParam()) == FM32) {
    return; // Ring type is not large enough to hold value
  }

  Runner r(std::get<0>(GetParam()), std::get<1>(GetParam()),
           std::get<2>(GetParam()));

  r.addInput(10);

  r.run(R"(
func @main(%arg0: tensor<!pphlo.pint>) -> (tensor<!pphlo.pint>) {
  %0 = "pphlo.constant"() {value = dense<2147483647> : tensor<i32>} : () -> tensor<!pphlo.pint> 
  %1 = "pphlo.minimum"(%0, %arg0) : (tensor<!pphlo.pint>, tensor<!pphlo.pint>) -> tensor<!pphlo.pint>
  return %1 :  tensor<!pphlo.pint>
})");

  int expected = 10;
  r.verifyOutput(&expected);
}

INSTANTIATE_TEST_SUITE_P(
    ProcessorTestInstances, ProcessorTest,
    testing::Combine(
        testing::Values(4, 3, 2),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<ProcessorTest::ParamType> &info) {
      return fmt::format("{}x{}x{}", std::get<0>(info.param),
                         std::get<1>(info.param), std::get<2>(info.param));
    });

// NOTE(junfeng): ABY3 is 3pc only.
INSTANTIATE_TEST_SUITE_P(
    ProcessorTestABY3Instances, ProcessorTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<ProcessorTest::ParamType> &info) {
      return fmt::format("{}x{}x{}", std::get<0>(info.param),
                         std::get<1>(info.param), std::get<2>(info.param));
    });

} // namespace ppu::device
