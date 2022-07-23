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


#include "ppu/hal/fxp.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "ppu/hal/io_ops.h"
#include "ppu/hal/test_util.h"

namespace ppu::hal {

TEST(FxpTest, Reciprocal) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{1.0, -2.0, -15000}, {-0.5, 3.14, 15000}};

  // public reciprocal
  {
    Value a = make_public(&ctx, x);
    Value c = f_reciprocal(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(1.0f / x, y, 0.001, 0.0001))
        << (1.0 / x) << std::endl
        << y;
  }

  // secret reciprocal
  {
    Value a = make_secret(&ctx, x);
    Value c = f_reciprocal(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    EXPECT_TRUE(xt::allclose(1.0f / x, y, 0.001, 0.0001))
        << (1.0 / x) << std::endl
        << y;
  }
}

TEST(FxpTest, Exponential) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{1.0, 2.0}, {0.5, 1.8}};

  // public exp
  {
    Value a = make_public(&ctx, x);
    Value c = f_exp(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
        << xt::exp(x) << std::endl
        << y;
  }

  // secret exp
  {
    Value a = make_secret(&ctx, x);
    Value c = f_exp(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    EXPECT_TRUE(xt::allclose(xt::exp(x), y, 0.01, 0.001))
        << xt::exp(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Log) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{0.05, 0.5}, {5, 50}};
  // public log
  {
    Value a = make_public(&ctx, x);
    Value c = f_log(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log(x), y, 0.01, 0.001))
        << xt::log(x) << std::endl
        << y;
  }

  // secret log
  {
    Value a = make_secret(&ctx, x);
    Value c = f_log(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log(x), y, 0.01, 0.001))
        << xt::log(x) << std::endl
        << y;
  }
}

TEST(FxpTest, Log1p) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{0.5, 2.0}, {0.9, 1.8}};

  // public log1p
  {
    Value a = make_public(&ctx, x);
    Value c = f_log1p(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::log1p(x), y, 0.01, 0.001))
        << xt::log1p(x) << std::endl;
  }

  // secret log1p
  {
    Value a = make_secret(&ctx, x);
    Value c = f_log1p(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::log1p(x), y, 0.01, 0.001))
        << xt::log1p(x) << std::endl
        << y;
  }
}

TEST(FxpTest, abs) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{0.5, -2.0}, {0.9, -1.8}};

  // public abs
  {
    Value a = make_public(&ctx, x);
    Value c = f_abs(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::abs(x), y, 0.01, 0.05))
        << xt::abs(x) << std::endl
        << y;
  }

  // secret abs
  {
    Value a = make_secret(&ctx, x);
    Value c = f_abs(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::abs(x), y, 0.1, 0.5))
        << xt::abs(x) << std::endl
        << y;
  }
}

TEST(FxpTest, floor) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{0.5, -0.5}, {-20.0, 31.8}, {0, 5.0}, {-5.0, -31.8}};

  // public floor
  {
    Value a = make_public(&ctx, x);
    Value c = f_floor(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::floor(x), y, 0.01, 0.001))
        << xt::floor(x) << std::endl
        << y;
  }

  // secret floor
  {
    Value a = make_secret(&ctx, x);
    Value c = f_floor(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::floor(x), y, 0.01, 0.001))
        << xt::floor(x) << std::endl
        << y;
  }
}

TEST(FxpTest, ceil) {
  // GIVEN
  HalContext ctx = test::MakeRefHalContext();

  xt::xarray<float> x{{0.5, -0.5}, {-20.0, 31.8}, {0, 5.0}, {-5.0, -31.8}};

  // public ceil
  {
    Value a = make_public(&ctx, x);
    Value c = f_ceil(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(xt::ceil(x), y, 0.01, 0.001))
        << xt::ceil(x) << std::endl
        << y;
  }

  // secret ceil
  {
    Value a = make_secret(&ctx, x);
    Value c = f_ceil(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_FXP);

    auto y = test::dump_public_as<float>(&ctx, _s2p(&ctx, c).as_fxp());
    // low precision
    EXPECT_TRUE(xt::allclose(xt::ceil(x), y, 0.01, 0.001))
        << xt::ceil(x) << std::endl
        << y;
  }
}

}  // namespace ppu::hal
