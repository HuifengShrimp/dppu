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


#include "ppu/mpc/io_test.h"

#include "gtest/gtest.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/mpc/util/test_util.h"

namespace ppu::mpc {
namespace {

bool RingEqual(const NdArrayRef& a, const NdArrayRef& b, size_t abs_err = 0) {
  PPU_ENFORCE(a.eltype() == b.eltype(), "type mismatch, a={}, b={}", a.eltype(),
              b.eltype());
  PPU_ENFORCE(a.eltype().isa<Ring2k>());

  const auto field = a.eltype().as<Ring2k>()->field();

  return DISPATCH_ALL_FIELDS(field, "RingEqual", [&]() {
    auto err = xt_adapt<ring2k_t>(a) - xt_adapt<ring2k_t>(b);
    return xt::all(xt::abs(err) <= abs_err);
  });
}

const std::vector<int64_t> kShape = {3, 1};
NdArrayRef RingRand(FieldType field,
                    const std::vector<int64_t>& shape = kShape) {
  Type pt_ty = makePtType(GetStorageType(field));
  return randint(pt_ty, shape).as(makeType<RingTy>(field));
}

}  // namespace

TEST_P(IoTest, MakePublicAndReconstruct) {
  const auto create_io = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  auto io = create_io(npc);

  auto raw = RingRand(field);
  auto shares = io->makePublic(raw);
  auto result = io->reconstruct(shares);

  EXPECT_TRUE(RingEqual(raw, result));
}

TEST_P(IoTest, MakeSecretAndReconstruct) {
  const auto create_io = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  auto io = create_io(npc);

  auto raw = RingRand(field);
  auto shares = io->makeSecret(raw);
  auto result = io->reconstruct(shares);

  EXPECT_TRUE(RingEqual(raw, result));
}

}  // namespace ppu::mpc
