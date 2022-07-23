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



#include "ppu/link/algorithm/allgather.h"

#include <future>

#include "gtest/gtest.h"

#include "ppu/link/test_util.h"

namespace ppu::link::test {

struct TestParams {
  size_t world_size;
  size_t n_rounds;
};

class AllGatherTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(AllGatherTest, Works) {
  const size_t n_rounds = GetParam().n_rounds;
  const size_t world_size = GetParam().world_size;
  auto contexts = SetupWorld(world_size);

  auto proc = [&](const std::shared_ptr<Context>& ctx) {
    for (size_t round = 0; round < n_rounds; round++) {
      const auto my_data = MakeRoundData(ctx->Rank(), round);
      std::vector<Buffer> result = AllGather(ctx, my_data, "test");

      EXPECT_EQ(result.size(), world_size);
      for (size_t rank = 0; rank < world_size; rank++) {
        EXPECT_EQ(result[rank], MakeRoundData(rank, round));
      }
    }
  };

  std::vector<std::future<void>> jobs(world_size);
  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank] = std::async(proc, contexts[rank]);
  }

  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank].get();
  }
}

TEST_P(AllGatherTest, VectorWorks) {
  const size_t n_rounds = GetParam().n_rounds;
  const size_t world_size = GetParam().world_size;
  auto contexts = SetupWorld(world_size);

  auto proc = [&](const std::shared_ptr<Context>& ctx) {
    for (size_t round = 0; round < n_rounds; round++) {
      for (int size : {0, 1, 5}) {
        std::vector<Buffer> inputs;
        for (int i = 0; i < size; ++i) {
          auto str = std::to_string(i + round + ctx->Rank());
          inputs.emplace_back(str.c_str(), str.size());
        }
        auto result = AllGather(ctx, inputs, "test");

        EXPECT_EQ(result.size(), inputs.size());
        for (int i = 0; i < size; ++i) {
          EXPECT_EQ(result[i].size(), world_size);
          for (size_t rank = 0; rank < world_size; rank++) {
            auto s = std::to_string(i + round + rank);
            EXPECT_EQ(
                std::memcmp(result[i][rank].data<char>(), s.c_str(), s.size()),
                0);
          }
        }
      }
    }
  };

  std::vector<std::future<void>> jobs(world_size);
  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank] = std::async(proc, contexts[rank]);
  }

  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank].get();
  }
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, AllGatherTest,
                         testing::Values(TestParams{2, 20},  //
                                         TestParams{3, 20},  //
                                         TestParams{9, 20}   //
                                         ));

}  // namespace ppu::link::test
