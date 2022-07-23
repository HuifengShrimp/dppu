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


#include "ppu/mpc/dealer/dealer_test.h"

#include "xtensor/xarray.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/type_util.h"
#include "ppu/mpc/util/ring_ops.h"
#include "ppu/mpc/util/test_util.h"

namespace ppu::mpc {

TEST_P(DealerTest, Mul) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;

  std::vector<Dealer::Triple> triples;
  triples.resize(kWorldSize);

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    auto dealer = factory(lctx);
    triples[lctx->Rank()] = dealer->Mul(kField, kNumel);
  });

  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }
  EXPECT_EQ(ring_mul(sum_a, sum_b), sum_c) << sum_a << sum_b << sum_c;
}
}  // namespace ppu::mpc


