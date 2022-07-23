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


#include "ppu/hal/conv.h"

#include <cstdint>
#include <utility>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "ppu/hal/test_util.h"

namespace ppu::hal {
namespace {

using secret_v = std::integral_constant<Visibility, VIS_SECRET>;
using public_v = std::integral_constant<Visibility, VIS_PUBLIC>;

template <typename S>
class ConvTest : public ::testing::Test {};

using ConvTestTypes = ::testing::Types<
    // ss
    std::tuple<secret_v, secret_v>,
    // sp
    std::tuple<secret_v, public_v>,
    // pp
    std::tuple<public_v, public_v>,
    // ps
    std::tuple<public_v, secret_v>>;

TYPED_TEST_SUITE(ConvTest, ConvTestTypes);

// (<1x1x3x1>, <1x3x1x1>) -> <1x1x1x1>, no padding, no strides
TYPED_TEST(ConvTest, conv2d_b01f_01io_b01f_1111_1111) {
  using INPUT_VT = typename std::tuple_element<0, TypeParam>::type;
  using KERNEL_VT = typename std::tuple_element<1, TypeParam>::type;

  // GIVEN
  const xt::xtensor<float, 4> x = {{{{1.0}}, {{2.0}}, {{3.0}}}};
  const xt::xtensor<float, 4> y = {{{{0.0}}}, {{{1.0}}}, {{{0.5}}}};

  // WHAT
  auto conv_wrapper = [](HalContext* ctx, const Value& input,
                         const Value& kernel) {
    return conv2d_b01f_01io_b01f(ctx, input, kernel);
  };

  // THEN
  auto z =
      test::EvalBinaryOp<float>(INPUT_VT(), KERNEL_VT(), conv_wrapper,
                                xt::xarray<float>(x), xt::xarray<float>(y));
  const xt::xtensor<float, 4> expected_z = {{{{3.5}}}};
  EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
      << expected_z << std::endl
      << z << std::endl;
}

// (<2x4x3x2>, <3x3x2x2>) -> <2x4x3x2>, padding is {1,1}, no strides
TYPED_TEST(ConvTest, conv2d_b01f_01io_b01f_2432_3322_11) {
  using INPUT_VT = typename std::tuple_element<0, TypeParam>::type;
  using KERNEL_VT = typename std::tuple_element<1, TypeParam>::type;

  // GIVEN
  const xt::xtensor<int, 4> x = {{{{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}}},
                                 {{{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}}}};
  const xt::xtensor<float, 4> y = {
      {{{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}},
      {{{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}},
      {{{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}}};

  // WHAT
  auto conv_wrapper = [](HalContext* ctx, const Value& input,
                         const Value& kernel) {
    return conv2d_b01f_01io_b01f(ctx, input, kernel, {1, 1}, {{1, 1}, {1, 1}});
  };

  // THEN
  auto z = test::EvalBinaryOp<int>(INPUT_VT(), KERNEL_VT(), conv_wrapper,
                                   xt::xarray<int>(x), xt::xarray<float>(y));
  const xt::xtensor<float, 4> expected_z = {
      {{{44.0, 44.0}, {66.0, 66.0}, {44.0, 44.0}},
       {{66.0, 66.0}, {99.0, 99.0}, {66.0, 66.0}},
       {{66.0, 66.0}, {99.0, 99.0}, {66.0, 66.0}},
       {{44.0, 44.0}, {66.0, 66.0}, {44.0, 44.0}}},
      {{{44.0, 44.0}, {66.0, 66.0}, {44.0, 44.0}},
       {{66.0, 66.0}, {99.0, 99.0}, {66.0, 66.0}},
       {{66.0, 66.0}, {99.0, 99.0}, {66.0, 66.0}},
       {{44.0, 44.0}, {66.0, 66.0}, {44.0, 44.0}}}};

  EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
      << expected_z << std::endl
      << z << std::endl;
}

// (<2x4x3x2>, <3x3x2x2>) -> <2x4x3x2>, padding is {1,1}, no strides
TYPED_TEST(ConvTest, conv2d_b01f_01io_b01f_2432_3322_21_21) {
  using INPUT_VT = typename std::tuple_element<0, TypeParam>::type;
  using KERNEL_VT = typename std::tuple_element<1, TypeParam>::type;

  // GIVEN
  const xt::xtensor<int, 4> x = {{{{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}}},
                                 {{{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}},
                                  {{1, 2}, {1, 2}, {1, 2}}}};
  const xt::xtensor<float, 4> y = {
      {{{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}},
      {{{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}},
      {{{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}, {{3, 3}, {4, 4}}}};

  // WHAT
  auto conv_wrapper = [](HalContext* ctx, const Value& input,
                         const Value& kernel) {
    return conv2d_b01f_01io_b01f(ctx, input, kernel, {2, 1}, {{2, 1}, {1, 1}});
  };

  // THEN
  auto z = test::EvalBinaryOp<int>(INPUT_VT(), KERNEL_VT(), conv_wrapper,
                                   xt::xarray<int>(x), xt::xarray<float>(y));
  const xt::xtensor<float, 4> expected_z = {{{{{22, 22}, {33, 33}, {22, 22}},
                                              {{66, 66}, {99, 99}, {66, 66}},
                                              {{44, 44}, {66, 66}, {44, 44}}},
                                             {{{22, 22}, {33, 33}, {22, 22}},
                                              {{66, 66}, {99, 99}, {66, 66}},
                                              {{44, 44}, {66, 66}, {44, 44}}}}};

  EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
      << expected_z << std::endl
      << z << std::endl;
}

}  // namespace
}  // namespace ppu::hal
