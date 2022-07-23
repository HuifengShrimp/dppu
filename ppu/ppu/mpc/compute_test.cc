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


#include "ppu/mpc/compute_test.h"

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

#define TEST_BINARY_OP_SS(OP)                                         \
  TEST_P(ComputeTest, OP##SS) {                                       \
    const auto factory = std::get<0>(GetParam());                     \
    const size_t npc = std::get<1>(GetParam());                       \
    const FieldType field = std::get<2>(GetParam());                  \
                                                                      \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {        \
      auto obj = factory(lctx);                                       \
      auto compute = obj->getInterface<ICompute>();                   \
      auto rnd = obj->getInterface<IRandom>();                        \
                                                                      \
      /* GIVEN */                                                     \
      auto p0 = rnd->RandP(field, numel(kShape));                     \
      auto p1 = rnd->RandP(field, numel(kShape));                     \
                                                                      \
      /* WHEN */                                                      \
      auto tmp = compute->OP##SS(compute->P2S(p0), compute->P2S(p1)); \
      auto re = compute->S2P(tmp);                                    \
      auto rp = compute->OP##PP(p0, p1);                              \
                                                                      \
      /* THEN */                                                      \
      EXPECT_TRUE(RingEqual(re, rp));                                 \
    });                                                               \
  }

#define TEST_BINARY_OP_SP(OP)                                  \
  TEST_P(ComputeTest, OP##SP) {                                \
    const auto factory = std::get<0>(GetParam());              \
    const size_t npc = std::get<1>(GetParam());                \
    const FieldType field = std::get<2>(GetParam());           \
                                                               \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
      auto obj = factory(lctx);                                \
      auto compute = obj->getInterface<ICompute>();            \
      auto rnd = obj->getInterface<IRandom>();                 \
                                                               \
      /* GIVEN */                                              \
      auto p0 = rnd->RandP(field, numel(kShape));              \
      auto p1 = rnd->RandP(field, numel(kShape));              \
                                                               \
      /* WHEN */                                               \
      auto tmp = compute->OP##SP(compute->P2S(p0), p1);        \
      auto re = compute->S2P(tmp);                             \
      auto rp = compute->OP##PP(p0, p1);                       \
                                                               \
      /* THEN */                                               \
      EXPECT_TRUE(RingEqual(re, rp));                          \
    });                                                        \
  }

#define TEST_BINARY_OP(OP) \
  TEST_BINARY_OP_SS(OP)    \
  TEST_BINARY_OP_SP(OP)

TEST_BINARY_OP(Add)
TEST_BINARY_OP(Mul)
TEST_BINARY_OP(And)
TEST_BINARY_OP(Xor)

#define TEST_UNARY_OP_S(OP)                                      \
  TEST_P(ComputeTest, OP##S) {                                   \
    const auto factory = std::get<0>(GetParam());                \
    const size_t npc = std::get<1>(GetParam());                  \
    const FieldType field = std::get<2>(GetParam());             \
                                                                 \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {   \
      auto obj = factory(lctx);                                  \
      auto compute = obj->getInterface<ICompute>();              \
      auto rnd = obj->getInterface<IRandom>();                   \
                                                                 \
      /* GIVEN */                                                \
      auto p0 = rnd->RandP(field, numel(kShape));                \
                                                                 \
      /* WHEN */                                                 \
      auto r_s = compute->S2P(compute->OP##S(compute->P2S(p0))); \
      auto r_p = compute->OP##P(p0);                             \
                                                                 \
      /* THEN */                                                 \
      EXPECT_TRUE(RingEqual(r_s, r_p));                          \
    });                                                          \
  }

#define TEST_UNARY_OP_P(OP)                                    \
  TEST_P(ComputeTest, OP##P) {                                 \
    const auto factory = std::get<0>(GetParam());              \
    const size_t npc = std::get<1>(GetParam());                \
    const FieldType field = std::get<2>(GetParam());           \
                                                               \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
      auto obj = factory(lctx);                                \
      auto compute = obj->getInterface<ICompute>();            \
      auto rnd = obj->getInterface<IRandom>();                 \
                                                               \
      /* GIVEN */                                              \
      auto p0 = rnd->RandP(field, numel(kShape));              \
                                                               \
      /* WHEN */                                               \
      auto r_p = compute->OP##P(p0);                           \
      auto r_pp = compute->OP##P(p0);                          \
                                                               \
      /* THEN */                                               \
      EXPECT_TRUE(RingEqual(r_p, r_pp));                       \
    });                                                        \
  }

#define TEST_UNARY_OP(OP) \
  TEST_UNARY_OP_S(OP)     \
  TEST_UNARY_OP_P(OP)

TEST_UNARY_OP(Neg)
TEST_UNARY_OP(Msb)

#define TEST_UNARY_OP_WITH_BIT_S(OP)                                     \
  TEST_P(ComputeTest, OP##S) {                                           \
    const auto factory = std::get<0>(GetParam());                        \
    const size_t npc = std::get<1>(GetParam());                          \
    const FieldType field = std::get<2>(GetParam());                     \
                                                                         \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {           \
      auto obj = factory(lctx);                                          \
      auto compute = obj->getInterface<ICompute>();                      \
      auto rnd = obj->getInterface<IRandom>();                           \
                                                                         \
      /* GIVEN */                                                        \
      auto p0 = rnd->RandP(field, numel(kShape));                        \
                                                                         \
      for (auto bits : kShiftBits) {                                     \
        /* WHEN */                                                       \
        auto r_s = compute->S2P(compute->OP##S(compute->P2S(p0), bits)); \
        auto r_p = compute->OP##P(p0, bits);                             \
                                                                         \
        /* THEN */                                                       \
        EXPECT_TRUE(RingEqual(r_s, r_p));                                \
      }                                                                  \
    });                                                                  \
  }

#define TEST_UNARY_OP_WITH_BIT_P(OP)                           \
  TEST_P(ComputeTest, OP##P) {                                 \
    const auto factory = std::get<0>(GetParam());              \
    const size_t npc = std::get<1>(GetParam());                \
    const FieldType field = std::get<2>(GetParam());           \
                                                               \
    test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
      auto obj = factory(lctx);                                \
      auto compute = obj->getInterface<ICompute>();            \
      auto rnd = obj->getInterface<IRandom>();                 \
                                                               \
      /* GIVEN */                                              \
      auto p0 = rnd->RandP(field, numel(kShape));              \
                                                               \
      for (auto bits : kShiftBits) {                           \
        /* WHEN */                                             \
        auto r_p = compute->OP##P(p0, bits);                   \
        auto r_pp = compute->OP##P(p0, bits);                  \
                                                               \
        /* THEN */                                             \
        EXPECT_TRUE(RingEqual(r_p, r_pp));                     \
      }                                                        \
    });                                                        \
  }

#define TEST_UNARY_OP_WITH_BIT(OP) \
  TEST_UNARY_OP_WITH_BIT_S(OP)     \
  TEST_UNARY_OP_WITH_BIT_P(OP)

TEST_UNARY_OP_WITH_BIT(LShift)
TEST_UNARY_OP_WITH_BIT(RShift)
TEST_UNARY_OP_WITH_BIT(ARShift)

TEST_P(ComputeTest, TruncPrS) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
    auto obj = factory(lctx);
    auto compute = obj->getInterface<ICompute>();
    auto rnd = obj->getInterface<IRandom>();

    /* GIVEN */
    auto p0 = test::RandP(field, numel(kShape), /*seed*/ 0, /*min*/ 0,
                          /*max*/ 10000);

    const size_t bits = 2;
    auto r_s = compute->S2P(compute->TruncPrS(compute->P2S(p0), bits));
    auto r_p = compute->ARShiftP(p0, bits);

    /* THEN */
    EXPECT_TRUE(RingEqual(r_s, r_p, npc));
  });
}

TEST_P(ComputeTest, MatMulSS) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const std::vector<int64_t> shape_A{M, K};
  const std::vector<int64_t> shape_B{K, N};

  test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
    auto obj = factory(lctx);
    auto compute = obj->getInterface<ICompute>();
    auto rnd = obj->getInterface<IRandom>();

    /* GIVEN */
    auto p0 = rnd->RandP(field, numel(shape_A));
    auto p1 = rnd->RandP(field, numel(shape_B));

    /* WHEN */
    auto tmp = compute->MatMulSS(compute->P2S(p0), compute->P2S(p1), M, N, K);
    auto r_ss = compute->S2P(tmp);
    auto r_pp = compute->MatMulPP(p0, p1, M, N, K);

    /* THEN */
    EXPECT_TRUE(RingEqual(r_ss, r_pp));
  });
}

TEST_P(ComputeTest, MatMulSP) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const std::vector<int64_t> shape_A{M, K};
  const std::vector<int64_t> shape_B{K, N};

  test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
    auto obj = factory(lctx);
    auto compute = obj->getInterface<ICompute>();
    auto rnd = obj->getInterface<IRandom>();

    /* GIVEN */
    auto p0 = rnd->RandP(field, numel(shape_A));
    auto p1 = rnd->RandP(field, numel(shape_B));

    /* WHEN */
    auto tmp = compute->MatMulSP(compute->P2S(p0), p1, M, N, K);
    auto r_ss = compute->S2P(tmp);
    auto r_pp = compute->MatMulPP(p0, p1, M, N, K);

    /* THEN */
    EXPECT_TRUE(RingEqual(r_ss, r_pp));
  });
}

TEST_P(ComputeTest, P2S_S2P) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  test::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
    auto obj = factory(lctx);
    auto compute = obj->getInterface<ICompute>();
    auto rnd = obj->getInterface<IRandom>();

    /* GIVEN */
    auto p0 = rnd->RandP(field, numel(kShape));

    /* WHEN */
    auto s = compute->P2S(p0);
    auto p1 = compute->S2P(s);

    /* THEN */
    EXPECT_TRUE(RingEqual(p0, p1));
  });
}

}  // namespace ppu::mpc::test
