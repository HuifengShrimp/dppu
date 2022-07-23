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


#include "ppu/mpc/aby3/protocol.h"
#include "ppu/mpc/compute_bench.h"

namespace ppu::mpc::bench {

namespace {
static void Aby3BMArguments(benchmark::internal::Benchmark* b) {
  b->Args({3, FieldType::FM32})
      ->Args({3, FieldType::FM64})
      ->Args({3, FieldType::FM128});
}
}  // namespace

BM_PROTOCOL_COMPUTE(makeAby3Protocol, Aby3BMArguments);

}  // namespace ppu::mpc::bench

BENCHMARK_MAIN();