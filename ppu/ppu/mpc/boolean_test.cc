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


#include "ppu/mpc/boolean_test.h"

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

TEST_P(BooleanTest, A2B_B2A) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
    auto obj = factory(lctx);
    auto compute = obj->getInterface<ICompute>();
    auto arithmetic = obj->getInterface<IArithmetic>();
    auto boolean = obj->getInterface<IBoolean>();
    auto rnd = obj->getInterface<IRandom>();

    /* GIVEN */
    auto p0 = rnd->RandP(field, numel(kShape));
    auto a0 = arithmetic->P2A(p0);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto b1 = boolean->A2B(a0);
    auto cost_0 = obj->getState<Communicator>()->getStats() - prev;
    auto a1 = boolean->B2A(b1);
    auto cost_1 = obj->getState<Communicator>()->getStats() - cost_0;

    /* THEN */
    EXPECT_TRUE(VerifyCost(obj->getKernel("A2B"), "A2B", field, numel(kShape),
                           npc, cost_0));
    EXPECT_TRUE(VerifyCost(obj->getKernel("B2A"), "B2A", field, numel(kShape),
                           npc, cost_1));
    EXPECT_TRUE(RingEqual(p0, arithmetic->A2P(a1)));
  });
}

}  // namespace ppu::mpc::test
