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


#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "ppu/device/io_accessor.h"

namespace ppu::device {

class IoAccessorTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, ProtocolKind, FieldType, Visibility>> {};

TEST_P(IoAccessorTest, Float) {
  const size_t kWorldSize = std::get<0>(GetParam());
  const Visibility kVisibility = std::get<3>(GetParam());

  RuntimeConfig config;
  config.set_protocol(std::get<1>(GetParam()));
  config.set_field(std::get<2>(GetParam()));
  IoAccessor sym(kWorldSize, config);

  xt::xarray<float> input({{1, -2, 3, 0}});

  auto shares = sym.makeShares(kVisibility, input);
  EXPECT_EQ(shares.size(), kWorldSize);

  auto reconstruct = sym.combineShares(shares, PtType::PT_F64);
  auto output = xt_adapt<double>(reconstruct);
  EXPECT_EQ(input, output);
}

TEST_P(IoAccessorTest, Int) {
  const size_t kWorldSize = std::get<0>(GetParam());
  const Visibility kVisibility = std::get<3>(GetParam());

  RuntimeConfig config;
  config.set_protocol(std::get<1>(GetParam()));
  config.set_field(std::get<2>(GetParam()));
  IoAccessor sym(kWorldSize, config);

  xt::xarray<int> input({{1, -2, 3, 0}});

  auto shares = sym.makeShares(kVisibility, input);
  EXPECT_EQ(shares.size(), kWorldSize);

  auto reconstruct = sym.combineShares(shares, PtType::PT_I32);
  auto output = xt_adapt<int>(reconstruct);
  EXPECT_EQ(input, output);
}

INSTANTIATE_TEST_SUITE_P(
    IoAccessorTestInstance, IoAccessorTest,
    testing::Combine(
        testing::Values(4, 3, 2),
        testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Visibility::VIS_PUBLIC, Visibility::VIS_SECRET)),
    [](const testing::TestParamInfo<IoAccessorTest::ParamType> &info) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(info.param),
                         std::get<1>(info.param), std::get<2>(info.param),
                         std::get<3>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    IoAccessorTestABY3Instance, IoAccessorTest,
    testing::Combine(
        testing::Values(3), testing::Values(ProtocolKind::ABY3),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Visibility::VIS_PUBLIC, Visibility::VIS_SECRET)),
    [](const testing::TestParamInfo<IoAccessorTest::ParamType> &info) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(info.param),
                         std::get<1>(info.param), std::get<2>(info.param),
                         std::get<3>(info.param));
    });

} // namespace ppu::device
