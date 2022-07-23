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


#include "ppu/hal/reduce.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"

#include "ppu/hal/polymorphic.h"
#include "ppu/hal/test_util.h"

namespace ppu::hal {

using secret_v = std::integral_constant<Visibility, VIS_SECRET>;
using public_v = std::integral_constant<Visibility, VIS_PUBLIC>;

using MathUnaryTestTypes = ::testing::Types<
    // s
    std::tuple<float, secret_v, float>,      // (sfxp)
    std::tuple<int32_t, secret_v, int64_t>,  // (sint)
    // p
    std::tuple<float, public_v, float>,     // (pfxp)
    std::tuple<int32_t, public_v, int64_t>  // (pint)
    >;

template <typename S>
class MathUnaryTest : public ::testing::Test {};
TYPED_TEST_SUITE(MathUnaryTest, MathUnaryTestTypes);

TYPED_TEST(MathUnaryTest, ReduceSum) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6});

  auto reduce_sum_wrapper = [](HalContext* ctx, const Value& in) {
    return reduce(
        ctx, in, make_public(ctx, 0u), std::vector<size_t>{1},
        [&ctx](const Value& a, const Value& b) { return add(ctx, a, b); });
  };

  // WHAT
  auto z = test::EvalUnaryOp<RES_DT>(IN_VT(), reduce_sum_wrapper, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::sum(x, std::vector<size_t>{1}), z, 0.01, 0.001))
      << xt::sum(x, std::vector<size_t>{1}) << std::endl
      << z << std::endl;
}

TYPED_TEST(MathUnaryTest, ReduceMax) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6});

  auto reduce_sum_wrapper = [](HalContext* ctx, const Value& in) {
    return reduce(
        ctx, in, make_public(ctx, IN_DT(0)), std::vector<size_t>{1},
        [&ctx](const Value& a, const Value& b) { return max(ctx, a, b); });
  };

  // WHAT
  auto z = test::EvalUnaryOp<RES_DT>(IN_VT(), reduce_sum_wrapper, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::amax(x, std::vector<size_t>{1}), z, 0.01, 0.001))
      << xt::amax(x, std::vector<size_t>{1}) << std::endl
      << z << std::endl;
}

TEST(ReduceAndTest, ReduceAnd) {
  // GIVEN
  xt::xarray<int32_t> x = test::xt_random<int32_t>({5, 6});

  {
    using VT = public_v::type;
    auto reduce_and_wrapper = [](HalContext* ctx, const Value& in) {
      return reduce(ctx, in, make_public(ctx, 1u), std::vector<size_t>{1},
                    [&ctx](const Value& a, const Value& b) {
                      return bitwise_and(ctx, a, b);
                    });
    };

    // WHAT
    auto z = test::EvalUnaryOp<int64_t>(VT(), reduce_and_wrapper, x);

    // expected
    auto logical_and = [](const int32_t& left, const int32_t& right) {
      return left & right;
    };
    auto and_functor = xt::make_xreducer_functor(logical_and);
    xt::xarray<int32_t> expected = xt::reduce(and_functor, x, {1});
    // THEN
    EXPECT_EQ(expected, z) << expected << std::endl << z << std::endl;
  }

  {
    using VT = secret_v::type;

    auto reduce_and_wrapper = [](HalContext* ctx, const Value& in) {
      return reduce(ctx, in, make_public(ctx, 1u), std::vector<size_t>{1},
                    [&ctx](const Value& a, const Value& b) {
                      return bitwise_and(ctx, a, b);
                    });
    };

    // WHAT
    auto z = test::EvalUnaryOp<int64_t>(VT(), reduce_and_wrapper, x);

    // expected
    auto logical_and = [](const int32_t& left, const int32_t& right) {
      return left & right;
    };
    auto and_functor = xt::make_xreducer_functor(logical_and);
    xt::xarray<int32_t> expected = xt::reduce(and_functor, x, {1});
    // THEN
    EXPECT_EQ(expected, z) << expected << std::endl << z << std::endl;
  }
}

}  // namespace ppu::hal
