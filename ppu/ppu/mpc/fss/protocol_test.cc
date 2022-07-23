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


#include "ppu/mpc/fss/protocol.h"

#include "ppu/mpc/arithmetic_test.h"
#include "ppu/mpc/boolean_test.h"
#include "ppu/mpc/compute_test.h"

namespace ppu::mpc::test {

INSTANTIATE_TEST_SUITE_P(
    fssComputeTest, ComputeTest,
    testing::Combine(testing::Values(makeFSSProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<ComputeTest::ParamType>& info) {
      return fmt::format("{}x{}", std::get<1>(info.param),
                         std::get<2>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    fssArithmeticTest, ArithmeticTest,
    testing::Combine(testing::Values(makeFSSProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& info) {
      return fmt::format("{}x{}", std::get<1>(info.param),
                         std::get<2>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    fssBooleanTest, BooleanTest,
    testing::Combine(testing::Values(makeFSSProtocol),  //
                     testing::Values(2, 3, 5),             //
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<BooleanTest::ParamType>& info) {
      return fmt::format("{}x{}", std::get<1>(info.param),
                         std::get<2>(info.param));
    });

}  // namespace ppu::mpc::test
