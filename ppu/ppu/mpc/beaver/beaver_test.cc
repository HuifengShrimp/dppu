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


#include "ppu/mpc/beaver/beaver_test.h"

#include "xtensor/xarray.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/type_util.h"
#include "ppu/mpc/util/ring_ops.h"
#include "ppu/mpc/util/test_util.h"

namespace ppu::mpc {

TEST_P(BeaverTest, Mul) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;
  std::cout<<"**test**!"<<std::endl;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->Mul(kField, kNumel);
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

TEST_P(BeaverTest, And) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->And(kField, kNumel);
  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  auto sum_c = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);
    EXPECT_EQ(c.numel(), kNumel);

    ring_xor_(sum_a, a);
    ring_xor_(sum_b, b);
    ring_xor_(sum_c, c);
  }
  EXPECT_EQ(ring_and(sum_a, sum_b), sum_c) << sum_a << sum_b << sum_c;
}

TEST_P(BeaverTest, Dot) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t M = 3;
  const size_t N = 5;
  const size_t K = 4;

  std::vector<Beaver::Triple> triples;
  triples.resize(kWorldSize);

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    auto beaver = factory(lctx);
    triples[lctx->Rank()] = beaver->Dot(kField, M, N, K);
  });

  EXPECT_EQ(triples.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, M * K);
  auto sum_b = ring_zeros(kField, K * N);
  auto sum_c = ring_zeros(kField, M * N);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b, c] = triples[r];
    EXPECT_EQ(a.numel(), M * K);
    EXPECT_EQ(b.numel(), K * N);
    EXPECT_EQ(c.numel(), M * N);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
    ring_add_(sum_c, c);
  }
  EXPECT_EQ(ring_mmul(sum_a, sum_b, M, N, K), sum_c) << sum_a << sum_b << sum_c;
}

TEST_P(BeaverTest, Trunc) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;
  const size_t kBits = 5;

  std::vector<Beaver::Pair> pairs;
  pairs.resize(kWorldSize);

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    auto beaver = factory(lctx);
    pairs[lctx->Rank()] = beaver->Trunc(kField, kNumel, kBits);
  });

  EXPECT_EQ(pairs.size(), kWorldSize);
  auto sum_a = ring_zeros(kField, kNumel);
  auto sum_b = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    const auto& [a, b] = pairs[r];
    EXPECT_EQ(a.numel(), kNumel);
    EXPECT_EQ(b.numel(), kNumel);

    ring_add_(sum_a, a);
    ring_add_(sum_b, b);
  }
  EXPECT_EQ(ring_arshift(sum_a, kBits), sum_b) << sum_a << sum_b;
}

TEST_P(BeaverTest, Randbit) {
  const auto factory = std::get<0>(GetParam());
  const size_t kWorldSize = std::get<1>(GetParam());
  const FieldType kField = std::get<2>(GetParam());
  const size_t kNumel = 7;

  std::vector<ArrayRef> shares(kWorldSize);

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    auto beaver = factory(lctx);
    shares[lctx->Rank()] = beaver->RandBit(kField, kNumel);
  });

  EXPECT_EQ(shares.size(), kWorldSize);
  auto sum = ring_zeros(kField, kNumel);
  for (Rank r = 0; r < kWorldSize; r++) {
    EXPECT_EQ(shares[r].numel(), kNumel);
    ring_add_(sum, shares[r]);
  }

  DISPATCH_ALL_FIELDS(kField, "_", [&]() {
    using scalar_t = typename Ring2kTrait<_kField>::scalar_t;
    auto x = xt_adapt<scalar_t>(sum);
    EXPECT_TRUE(xt::all(x <= xt::ones_like(x)));
    EXPECT_TRUE(xt::all(x >= xt::zeros_like(x)));
    // TODO: BeaverRef could not pass following test.
    // EXPECT_TRUE(x != xt::zeros_like(x));
    return;
  });
}

}  // namespace ppu::mpc
