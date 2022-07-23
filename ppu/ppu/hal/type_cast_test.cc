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


#include "ppu/hal/type_cast.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "ppu/hal/test_util.h"

namespace ppu::hal {
namespace {

TEST(TypeCastTest, int2fxp) {
  const xt::xarray<int32_t> x = test::xt_random<int32_t>({5, 6});
  auto expected = xt::cast<int32_t>(x);
  HalContext ctx = test::MakeRefHalContext();

  {
    Value a = make_public(&ctx, x);
    Value c = int2fxp(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(expected, y, 0.1, 0.5)) << x << std::endl
                                                     << expected << std::endl
                                                     << y;
    ;
  }

  {
    Value a = make_secret(&ctx, x);
    Value c = int2fxp(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    EXPECT_TRUE(xt::allclose(expected, y, 0.1, 0.5)) << x << std::endl
                                                     << expected << std::endl
                                                     << y;
  }
}

TEST(TypeCastTest, fxp2int) {
  const xt::xarray<float> x = {{0.0, 1.0, 5.0, 10.0, 100.0, 1000.0},
                               {0.1, 0.5, 0.7, 10.1, 10.5, 10.7},
                               {-0.0, -1.0, -5.0, -10.0, -100.0, -1000.0},
                               {-0.1, -0.5, -0.7, -10.1, -10.5, -10.7}};
  auto expected = xt::cast<int32_t>(x);
  HalContext ctx = test::MakeRefHalContext();

  {
    Value a = make_public(&ctx, x);
    Value c = fxp2int(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_INT);

    auto y = test::dump_public_as<int>(&ctx, c);
    EXPECT_TRUE(xt::allclose(expected, y, 0.1, 0.5)) << x << std::endl
                                                     << expected << std::endl
                                                     << y;
  }

  {
    Value a = make_secret(&ctx, x);
    Value c = fxp2int(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_INT);

    auto y = test::dump_public_as<int>(&ctx, _s2p(&ctx, c).as_int());
    EXPECT_TRUE(xt::allclose(expected, y, 0.1, 0.5)) << x << std::endl
                                                     << expected << std::endl
                                                     << y;
  }
}

}  // namespace
}  // namespace ppu::hal
