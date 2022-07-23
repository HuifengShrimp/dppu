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


#pragma once

#include <functional>

#include "benchmark/benchmark.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/link/link.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/object.h"
#include "ppu/mpc/util/bench_util.h"
#include "ppu/mpc/util/communicator.h"

namespace ppu::mpc::bench {

namespace {
const std::vector<int64_t> kShape = {3, 1};
size_t kShiftBit = 2;
}  // namespace

class ComputeBench : public benchmark::Fixture {};

/*
 * Benchmark Defines
 */

#define PPU_BENCHMARK_DEFINE_F(BaseClass, Method)                           \
  class BaseClass##_##Method##_Benchmark : public BaseClass {               \
   public:                                                                  \
    BaseClass##_##Method##_Benchmark() {                                    \
      this->SetName(#BaseClass "/" #Method);                                \
    }                                                                       \
                                                                            \
   protected:                                                               \
    virtual void BenchmarkCase(benchmark::State&) BENCHMARK_OVERRIDE;       \
    virtual void BenchmarkCase(benchmark::State&, CreateComputeFn facotry); \
  };                                                                        \
  void BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method)::BenchmarkCase

#define PPU_BENCHMARK_SECTION_START(COMM) \
  auto prev = COMM->getStats();           \
  state.ResumeTiming();

#define PPU_BENCHMARK_SECTION_END(COMM)     \
  state.PauseTiming();                      \
  auto cost = COMM->getStats() - prev;      \
  state.counters["latency"] = cost.latency; \
  state.counters["comm"] = cost.comm;       \
  state.ResumeTiming();

#define PPU_BENCHMARK_SECTION(COMM, CODE) \
  if (lctx->Rank() == 0) {                \
    PPU_BENCHMARK_SECTION_START(COMM)     \
    CODE;                                 \
    PPU_BENCHMARK_SECTION_END(COMM)       \
  } else {                                \
    CODE;                                 \
    state.ResumeTiming();                 \
  }

#define BM_DEFINE_BINARY_OP_SS(OP)                                \
  PPU_BENCHMARK_DEFINE_F(ComputeBench, OP##SS)                    \
  (benchmark::State & state, CreateComputeFn factory) {           \
    for (auto _ : state) {                                        \
      state.PauseTiming();                                        \
      const size_t npc = state.range(0);                          \
      const FieldType field = ppu::FieldType(state.range(1));     \
                                                                  \
      bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
        auto obj = factory(lctx);                                 \
        auto compute = obj->getInterface<ICompute>();             \
        auto rnd = obj->getInterface<IRandom>();                  \
        auto comm = obj->getState<Communicator>();                \
                                                                  \
        /* GIVEN */                                               \
        auto p0 = rnd->RandP(field, numel(kShape));               \
        auto p1 = rnd->RandP(field, numel(kShape));               \
        auto s0 = compute->P2S(p0);                               \
        auto s1 = compute->P2S(p1);                               \
                                                                  \
        /* WHEN */                                                \
        PPU_BENCHMARK_SECTION(comm, compute->OP##SS(s0, s1))      \
      });                                                         \
    }                                                             \
  }

#define BM_DEFINE_BINARY_OP_SP(OP)                                \
  PPU_BENCHMARK_DEFINE_F(ComputeBench, OP##SP)                    \
  (benchmark::State & state, CreateComputeFn factory) {           \
    for (auto _ : state) {                                        \
      state.PauseTiming();                                        \
      const size_t npc = state.range(0);                          \
      const FieldType field = ppu::FieldType(state.range(1));     \
                                                                  \
      bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
        auto obj = factory(lctx);                                 \
        auto compute = obj->getInterface<ICompute>();             \
        auto rnd = obj->getInterface<IRandom>();                  \
        auto comm = obj->getState<Communicator>();                \
                                                                  \
        /* GIVEN */                                               \
        auto p0 = rnd->RandP(field, numel(kShape));               \
        auto p1 = rnd->RandP(field, numel(kShape));               \
        auto s0 = compute->P2S(p0);                               \
                                                                  \
        /* WHEN */                                                \
        PPU_BENCHMARK_SECTION(comm, compute->OP##SP(s0, p1));     \
      });                                                         \
    }                                                             \
  }

#define BM_DEFINE_BINARY_OP(OP) \
  BM_DEFINE_BINARY_OP_SS(OP)    \
  BM_DEFINE_BINARY_OP_SP(OP)

BM_DEFINE_BINARY_OP(Add)
BM_DEFINE_BINARY_OP(Mul)
BM_DEFINE_BINARY_OP(And)
BM_DEFINE_BINARY_OP(Xor)

#define BM_DEFINE_UNARY_OP_S(OP)                                  \
  PPU_BENCHMARK_DEFINE_F(ComputeBench, OP##S)                     \
  (benchmark::State & state, CreateComputeFn factory) {           \
    for (auto _ : state) {                                        \
      state.PauseTiming();                                        \
      const size_t npc = state.range(0);                          \
      const FieldType field = ppu::FieldType(state.range(1));     \
                                                                  \
      bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
        auto obj = factory(lctx);                                 \
        auto compute = obj->getInterface<ICompute>();             \
        auto rnd = obj->getInterface<IRandom>();                  \
        auto comm = obj->getState<Communicator>();                \
                                                                  \
        /* GIVEN */                                               \
        auto p0 = rnd->RandP(field, numel(kShape));               \
        auto s0 = compute->P2S(p0);                               \
                                                                  \
        /* WHEN */                                                \
        PPU_BENCHMARK_SECTION(comm, compute->OP##S(s0));          \
      });                                                         \
    }                                                             \
  }

#define BM_DEFINE_UNARY_OP_P(OP)                                  \
  PPU_BENCHMARK_DEFINE_F(ComputeBench, OP##P)                     \
  (benchmark::State & state, CreateComputeFn factory) {           \
    for (auto _ : state) {                                        \
      state.PauseTiming();                                        \
      const size_t npc = state.range(0);                          \
      const FieldType field = ppu::FieldType(state.range(1));     \
                                                                  \
      bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
        auto obj = factory(lctx);                                 \
        auto compute = obj->getInterface<ICompute>();             \
        auto rnd = obj->getInterface<IRandom>();                  \
        auto comm = obj->getState<Communicator>();                \
                                                                  \
        /* GIVEN */                                               \
        auto p0 = rnd->RandP(field, numel(kShape));               \
                                                                  \
        /* WHEN */                                                \
        PPU_BENCHMARK_SECTION(comm, compute->OP##P(p0);)          \
      });                                                         \
    }                                                             \
  }

#define BM_DEFINE_UNARY_OP(OP) \
  BM_DEFINE_UNARY_OP_S(OP)     \
  BM_DEFINE_UNARY_OP_P(OP)

BM_DEFINE_UNARY_OP(Neg)

#define BM_DEFINE_UNARY_OP_WITH_BIT_S(OP, BIT)                    \
  PPU_BENCHMARK_DEFINE_F(ComputeBench, OP##S)                     \
  (benchmark::State & state, CreateComputeFn factory) {           \
    for (auto _ : state) {                                        \
      state.PauseTiming();                                        \
      const size_t npc = state.range(0);                          \
      const FieldType field = ppu::FieldType(state.range(1));     \
                                                                  \
      bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
        auto obj = factory(lctx);                                 \
        auto compute = obj->getInterface<ICompute>();             \
        auto rnd = obj->getInterface<IRandom>();                  \
        auto comm = obj->getState<Communicator>();                \
                                                                  \
        /* GIVEN */                                               \
        auto p0 = rnd->RandP(field, numel(kShape));               \
        auto s0 = compute->P2S(p0);                               \
                                                                  \
        /* WHEN */                                                \
        PPU_BENCHMARK_SECTION(comm, compute->OP##S(s0, BIT));     \
      });                                                         \
    }                                                             \
  }

#define BM_DEFINE_UNARY_OP_WITH_BIT_P(OP, BIT)                    \
  PPU_BENCHMARK_DEFINE_F(ComputeBench, OP##P)                     \
  (benchmark::State & state, CreateComputeFn factory) {           \
    for (auto _ : state) {                                        \
      state.PauseTiming();                                        \
      const size_t npc = state.range(0);                          \
      const FieldType field = ppu::FieldType(state.range(1));     \
                                                                  \
      bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) { \
        auto obj = factory(lctx);                                 \
        auto compute = obj->getInterface<ICompute>();             \
        auto rnd = obj->getInterface<IRandom>();                  \
        auto comm = obj->getState<Communicator>();                \
                                                                  \
        /* GIVEN */                                               \
        auto p0 = rnd->RandP(field, numel(kShape));               \
                                                                  \
        /* WHEN */                                                \
        PPU_BENCHMARK_SECTION(comm, compute->OP##P(p0, BIT));     \
      });                                                         \
    }                                                             \
  }

#define BM_DEFINE_UNARY_OP_WITH_BIT(OP)        \
  BM_DEFINE_UNARY_OP_WITH_BIT_S(OP, kShiftBit) \
  BM_DEFINE_UNARY_OP_WITH_BIT_P(OP, kShiftBit)

BM_DEFINE_UNARY_OP_WITH_BIT(LShift)
BM_DEFINE_UNARY_OP_WITH_BIT(RShift)
BM_DEFINE_UNARY_OP_WITH_BIT(ARShift)

PPU_BENCHMARK_DEFINE_F(ComputeBench, TruncPrS)
(benchmark::State& state, CreateComputeFn factory) {
  for (auto _ : state) {
    state.PauseTiming();
    const size_t npc = state.range(0);
    const FieldType field = ppu::FieldType(state.range(1));

    bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
      auto obj = factory(lctx);
      auto compute = obj->getInterface<ICompute>();
      auto rnd = obj->getInterface<IRandom>();
      auto comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = bench::RandP(field, numel(kShape), /*seed*/ 0, /*min*/ 0,
                             /*max*/ 10000);

      const size_t bits = 2;
      auto s0 = compute->P2S(p0);

      /* WHEN */
      PPU_BENCHMARK_SECTION(comm, compute->TruncPrS(s0, bits));
    });
  }
}

PPU_BENCHMARK_DEFINE_F(ComputeBench, MatMulSS)
(benchmark::State& state, CreateComputeFn factory) {
  for (auto _ : state) {
    state.PauseTiming();
    const size_t npc = state.range(0);
    const FieldType field = ppu::FieldType(state.range(1));

    const int64_t M = 3;
    const int64_t K = 4;
    const int64_t N = 3;
    const std::vector<int64_t> shape_A{M, K};
    const std::vector<int64_t> shape_B{K, N};

    bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
      auto obj = factory(lctx);
      auto compute = obj->getInterface<ICompute>();
      auto rnd = obj->getInterface<IRandom>();
      auto comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rnd->RandP(field, numel(shape_A));
      auto p1 = rnd->RandP(field, numel(shape_B));
      auto s0 = compute->P2S(p0);
      auto s1 = compute->P2S(p1);

      /* WHEN */
      PPU_BENCHMARK_SECTION(comm, compute->MatMulSS(s0, s1, M, N, K));
    });
  }
}

PPU_BENCHMARK_DEFINE_F(ComputeBench, MatMulSP)
(benchmark::State& state, CreateComputeFn factory) {
  for (auto _ : state) {
    state.PauseTiming();
    const size_t npc = state.range(0);
    const FieldType field = ppu::FieldType(state.range(1));

    const int64_t M = 3;
    const int64_t K = 4;
    const int64_t N = 3;
    const std::vector<int64_t> shape_A{M, K};
    const std::vector<int64_t> shape_B{K, N};

    bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
      auto obj = factory(lctx);
      auto compute = obj->getInterface<ICompute>();
      auto rnd = obj->getInterface<IRandom>();
      auto comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rnd->RandP(field, numel(shape_A));
      auto p1 = rnd->RandP(field, numel(shape_B));
      auto s0 = compute->P2S(p0);

      /* WHEN */
      PPU_BENCHMARK_SECTION(comm, compute->MatMulSP(s0, p1, M, N, K));
    });
  }
}

PPU_BENCHMARK_DEFINE_F(ComputeBench, P2S)
(benchmark::State& state, CreateComputeFn factory) {
  for (auto _ : state) {
    state.PauseTiming();
    const size_t npc = state.range(0);
    const FieldType field = ppu::FieldType(state.range(1));

    bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
      auto obj = factory(lctx);
      auto compute = obj->getInterface<ICompute>();
      auto rnd = obj->getInterface<IRandom>();
      auto comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rnd->RandP(field, numel(kShape));

      /* WHEN */
      PPU_BENCHMARK_SECTION(comm, compute->P2S(p0));
    });
  }
}

PPU_BENCHMARK_DEFINE_F(ComputeBench, S2P)
(benchmark::State& state, CreateComputeFn factory) {
  for (auto _ : state) {
    state.PauseTiming();
    const size_t npc = state.range(0);
    const FieldType field = ppu::FieldType(state.range(1));

    bench::Eval(npc, [&](std::shared_ptr<link::Context> lctx) {
      auto obj = factory(lctx);
      auto compute = obj->getInterface<ICompute>();
      auto rnd = obj->getInterface<IRandom>();
      auto comm = obj->getState<Communicator>();

      /* GIVEN */
      auto p0 = rnd->RandP(field, numel(kShape));
      auto s0 = compute->P2S(p0);

      /* WHEN */
      PPU_BENCHMARK_SECTION(comm, compute->S2P(s0));
    });
  }
}

/*
 * Benchmark Registers
 */
#define PPU_BENCHMARK_REGISTER(TestName, Factory)         \
  void TestName::BenchmarkCase(benchmark::State& state) { \
    TestName::BenchmarkCase(state, Factory);              \
  }                                                       \
  BENCHMARK_PRIVATE_DECLARE(TestName) =                   \
      (::benchmark::internal::RegisterBenchmarkInternal(new TestName()))

#define PPU_BENCHMARK_REGISTER_F(BaseClass, Method, Factory)               \
  PPU_BENCHMARK_REGISTER(BENCHMARK_PRIVATE_CONCAT_NAME(BaseClass, Method), \
                         Factory)

#define BM_REGISTER_BINARY_OP(OP, Factory, Arguments) \
  BM_REGISTER_OP(OP##SS, Factory, Arguments);         \
  BM_REGISTER_OP(OP##SP, Factory, Arguments);

#define BM_REGISTER_UNARY_OP(OP, Factory, Arguments) \
  BM_REGISTER_OP(OP##S, Factory, Arguments);         \
  BM_REGISTER_OP(OP##P, Factory, Arguments);

#define BM_REGISTER_OP(OP, Factory, Arguments) \
  PPU_BENCHMARK_REGISTER_F(ComputeBench, OP, Factory)->Apply(Arguments);

#define BM_PROTOCOL_COMPUTE(Factory, Arguments)     \
  BM_REGISTER_BINARY_OP(Add, Factory, Arguments)    \
  BM_REGISTER_BINARY_OP(Mul, Factory, Arguments)    \
  BM_REGISTER_BINARY_OP(And, Factory, Arguments)    \
  BM_REGISTER_BINARY_OP(Xor, Factory, Arguments)    \
  BM_REGISTER_UNARY_OP(Neg, Factory, Arguments)     \
  BM_REGISTER_UNARY_OP(LShift, Factory, Arguments)  \
  BM_REGISTER_UNARY_OP(RShift, Factory, Arguments)  \
  BM_REGISTER_UNARY_OP(ARShift, Factory, Arguments) \
  BM_REGISTER_OP(TruncPrS, Factory, Arguments)      \
  BM_REGISTER_OP(MatMulSS, Factory, Arguments)      \
  BM_REGISTER_OP(MatMulSP, Factory, Arguments)      \
  BM_REGISTER_OP(P2S, Factory, Arguments)           \
  BM_REGISTER_OP(S2P, Factory, Arguments)

}  // namespace ppu::mpc::bench
