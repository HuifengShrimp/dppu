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


#include "ppu/mpc/arithmetic_test.h"

#include "gtest/gtest.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/test_util.h"

namespace ppu::mpc::test {
namespace {

const std::vector<int64_t> kShape = {3, 1};

const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

}  // namespace

#define TEST_ARITHMETIC_BINARY_OP_AA(OP)                                \
  TEST_P(ArithmeticTest, OP##AA) {                                      \
    const auto factory = std::get<0>(GetParam());                       \
    const size_t npc = std::get<1>(GetParam());                         \
    const FieldType field = std::get<2>(GetParam());                    \
                                                                        \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {          \
      auto obj = factory(lctx);                                         \
      auto arithmetic = obj->getInterface<IArithmetic>();               \
      auto compute = obj->getInterface<ICompute>();                     \
      auto rnd = obj->getInterface<IRandom>();                          \
                                                                        \
      /* GIVEN */                                                       \
      auto p0 = rnd->RandP(field, numel(kShape));                       \
      auto p1 = rnd->RandP(field, numel(kShape));                       \
                                                                        \
      /* WHEN */                                                        \
      auto a0 = arithmetic->P2A(p0);                                    \
      auto a1 = arithmetic->P2A(p1);                                    \
      auto prev = obj->getState<Communicator>()->getStats();            \
      auto tmp = arithmetic->OP##AA(a0, a1);                            \
      auto cost = obj->getState<Communicator>()->getStats() - prev;     \
      auto re = arithmetic->A2P(tmp);                                   \
      auto rp = compute->OP##PP(p0, p1);                                \
                                                                        \
      /* THEN */                                                        \
      EXPECT_TRUE(RingEqual(re, rp));                                   \
      EXPECT_TRUE(VerifyCost(obj->getKernel(#OP "AA"), #OP "AA", field, \
                             numel(kShape), npc, cost));                \
    });                                                                 \
  }

#define TEST_ARITHMETIC_BINARY_OP_AP(OP)                                \
  TEST_P(ArithmeticTest, OP##AP) {                                      \
    const auto factory = std::get<0>(GetParam());                       \
    const size_t npc = std::get<1>(GetParam());                         \
    const FieldType field = std::get<2>(GetParam());                    \
                                                                        \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {          \
      auto obj = factory(lctx);                                         \
      auto arithmetic = obj->getInterface<IArithmetic>();               \
      auto compute = obj->getInterface<ICompute>();                     \
      auto rnd = obj->getInterface<IRandom>();                          \
                                                                        \
      /* GIVEN */                                                       \
      auto p0 = rnd->RandP(field, numel(kShape));                       \
      auto p1 = rnd->RandP(field, numel(kShape));                       \
                                                                        \
      /* WHEN */                                                        \
      auto a0 = arithmetic->P2A(p0);                                    \
      auto prev = obj->getState<Communicator>()->getStats();            \
      auto tmp = arithmetic->OP##AP(a0, p1);                            \
      auto cost = obj->getState<Communicator>()->getStats() - prev;     \
      auto re = arithmetic->A2P(tmp);                                   \
      auto rp = compute->OP##PP(p0, p1);                                \
                                                                        \
      /* THEN */                                                        \
      EXPECT_TRUE(RingEqual(re, rp));                                   \
      EXPECT_TRUE(VerifyCost(obj->getKernel(#OP "AP"), #OP "AP", field, \
                             numel(kShape), npc, cost));                \
    });                                                                 \
  }

#define TEST_ARITHMETIC_BINARY_OP(OP) \
  TEST_ARITHMETIC_BINARY_OP_AA(OP)    \
  TEST_ARITHMETIC_BINARY_OP_AP(OP)

TEST_ARITHMETIC_BINARY_OP(Add)
TEST_ARITHMETIC_BINARY_OP(Mul)

}  // namespace ppu::mpc::test
