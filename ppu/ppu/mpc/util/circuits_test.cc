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


#include "ppu/mpc/util/circuits.h"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"

#include "ppu/utils/int128.h"

namespace ppu::mpc {

TEST(KoggleStoneAdder, Scalar) {
  {
    int64_t x = 42;
    int64_t y = 17;

    int64_t z = KoggleStoneAdder(x, y);
    EXPECT_EQ(x + y, z);
  }

  {
    int32_t x = 17;
    int32_t y = 116;

    int32_t z = KoggleStoneAdder(x, y);
    EXPECT_EQ(x + y, z);
  }

  {
    int128_t x = 42;
    int128_t y = 17;

    int128_t z = KoggleStoneAdder(x, y);
    EXPECT_EQ(x + y, z);
  }
}

TEST(KoggleStoneAdder, Xtensor) {
  const std::vector<int> shape = {20, 20};

  using scalar_t = int64_t;
  using tensor_t = xt::xarray<scalar_t>;

  CircuitBasicBlock<tensor_t> cbb;
  {
    cbb.num_bits = sizeof(scalar_t) * 8;
    cbb._xor = [](tensor_t const& lhs, tensor_t const& rhs) -> tensor_t {
      return lhs ^ rhs;
    };
    cbb._and = [](tensor_t const& lhs, tensor_t const& rhs) -> tensor_t {
      return lhs & rhs;
    };
    cbb.lshift = [](tensor_t const& x, size_t bits) -> tensor_t {
      return x << bits;
    };
    cbb.rshift = [](tensor_t const& x, size_t bits) -> tensor_t {
      return x >> bits;
    };
  }

  tensor_t x = xt::random::randint<scalar_t>(shape, 1, 1000);
  tensor_t y = xt::random::randint<scalar_t>(shape, 1, 1000);

  auto z = KoggleStoneAdder<tensor_t>(x, y, cbb);

  EXPECT_EQ(x + y, z);
}

}  // namespace ppu::mpc
