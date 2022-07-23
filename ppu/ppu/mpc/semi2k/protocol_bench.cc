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


#include "ppu/mpc/compute_bench.h"
#include "ppu/mpc/semi2k/protocol.h"

namespace ppu::mpc::bench {

namespace {
static void Semi2kBMArguments(benchmark::internal::Benchmark* b) {
  b->Args({2, FieldType::FM32})
      ->Args({2, FieldType::FM64})
      ->Args({2, FieldType::FM128})
      ->Args({3, FieldType::FM32})
      ->Args({3, FieldType::FM64})
      ->Args({3, FieldType::FM128})
      ->Args({5, FieldType::FM32})
      ->Args({5, FieldType::FM64})
      ->Args({5, FieldType::FM128});
}
}  // namespace

BM_PROTOCOL_COMPUTE(makeSemi2kProtocol, Semi2kBMArguments);

}  // namespace ppu::mpc::bench

BENCHMARK_MAIN();