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


#include "ppu/hal/sort.h"

#include <cstddef>

#include "gtest/gtest.h"
#include "io_ops.h"

#include "ppu/hal/test_util.h"

namespace ppu::hal {
namespace {

template <typename T>
std::vector<xt::xarray<T>> EvalSortOp(Visibility in_vtype,
                                      const std::vector<xt::xarray<T>>& in,
                                      size_t dimension, bool is_stable,
                                      bool is_less) {
  HalContext ctx = test::MakeRefHalContext();

  std::vector<Value> in_v;
  in_v.reserve(in.size());
  for (const auto& i : in) {
    in_v.emplace_back(make_value(&ctx, in_vtype, i));
  }

  //   Value a = make_value(&ctx, in_vtype, in);

  std::vector<Value> sorted = sort(&ctx, in_v, dimension, is_stable, is_less);

  std::vector<xt::xarray<T>> res;
  res.reserve(in.size());

  for (auto& i : sorted) {
    if (i.is_secret()) {
      i = _s2p(&ctx, i).as_dtype(i.dtype());
    }
    PPU_ENFORCE(i.is_public());

    res.emplace_back(test::dump_public_as<T>(&ctx, i));
  }

  return res;
}

TEST(SortTest, 1d) {
  // GIVEN
  const std::vector<xt::xarray<float>> x = {
      xt::xtensor<float, 1>({4, 3, 2, 1}),  //
      xt::xtensor<float, 1>({1, 2, 3, 4}),  //
      xt::xtensor<float, 1>({1, 2, 3, 4})};

  // WHAT
  const std::vector<xt::xarray<float>> sorted_x_asc =
      EvalSortOp<float>(VIS_PUBLIC, x, 0, true, true);
  const std::vector<xt::xarray<float>> expected_sorted_x_asc = {
      xt::xtensor<float, 1>({1, 2, 3, 4}),  //
      xt::xtensor<float, 1>({4, 3, 2, 1}),  //
      xt::xtensor<float, 1>({4, 3, 2, 1})};

  const std::vector<xt::xarray<float>> sorted_x_desc =
      EvalSortOp<float>(VIS_PUBLIC, x, 0, true, false);
  const std::vector<xt::xarray<float>> expected_sorted_x_desc = {
      xt::xtensor<float, 1>({4, 3, 2, 1}),  //
      xt::xtensor<float, 1>({1, 2, 3, 4}),  //
      xt::xtensor<float, 1>({1, 2, 3, 4})};

  // THEN
  for (size_t i = 0; i < expected_sorted_x_asc.size(); i++) {
    EXPECT_TRUE(
        xt::allclose(expected_sorted_x_asc[i], sorted_x_asc[i], 0.01, 0.001))
        << expected_sorted_x_asc[i] << std::endl
        << sorted_x_asc[i] << std::endl;
  }
  for (size_t i = 0; i < expected_sorted_x_desc.size(); i++) {
    EXPECT_TRUE(
        xt::allclose(expected_sorted_x_desc[i], sorted_x_desc[i], 0.01, 0.001))
        << expected_sorted_x_desc[i] << std::endl
        << sorted_x_desc[i] << std::endl;
  }
}

TEST(SortTest, 2d) {
  // GIVEN
  const std::vector<xt::xarray<float>> x = {
      xt::xtensor<float, 2>({{4, 3}, {2, 1}}),  //
      xt::xtensor<float, 2>({{1, 2}, {3, 4}}),  //
      xt::xtensor<float, 2>({{1, 2}, {3, 4}})};

  {
    // WHAT
    const std::vector<xt::xarray<float>> sorted_x_asc =
        EvalSortOp<float>(VIS_PUBLIC, x, 1, true, true);
    const std::vector<xt::xarray<float>> expected_sorted_x_asc = {
        xt::xtensor<float, 2>({{3, 4}, {1, 2}}),  //
        xt::xtensor<float, 2>({{2, 1}, {4, 3}}),  //
        xt::xtensor<float, 2>({{2, 1}, {4, 3}})};

    // THEN
    for (size_t i = 0; i < expected_sorted_x_asc.size(); i++) {
      EXPECT_TRUE(
          xt::allclose(expected_sorted_x_asc[i], sorted_x_asc[i], 0.01, 0.001))
          << expected_sorted_x_asc[i] << std::endl
          << sorted_x_asc[i] << std::endl;
    }
  }

  {
    // WHAT
    const std::vector<xt::xarray<float>> sorted_x_asc =
        EvalSortOp<float>(VIS_PUBLIC, x, 0, true, true);
    const std::vector<xt::xarray<float>> expected_sorted_x_asc = {
        xt::xtensor<float, 2>({{2, 1}, {4, 3}}),  //
        xt::xtensor<float, 2>({{3, 4}, {1, 2}}),  //
        xt::xtensor<float, 2>({{3, 4}, {1, 2}})};

    // THEN
    for (size_t i = 0; i < expected_sorted_x_asc.size(); i++) {
      EXPECT_TRUE(
          xt::allclose(expected_sorted_x_asc[i], sorted_x_asc[i], 0.01, 0.001))
          << expected_sorted_x_asc[i] << std::endl
          << sorted_x_asc[i] << std::endl;
    }
  }
}

TEST(SortTest, 3d) {
  // GIVEN
  const std::vector<xt::xarray<float>> x = {
      xt::xtensor<float, 3>{{{1, 2, 3, 4, 5},   //
                             {6, 7, 8, 9, 10},  //
                             {11, 12, 13, 14, 15}},
                            {{10, 20, 30, 40, 50},   //
                             {60, 70, 80, 90, 100},  //
                             {110, 120, 130, 140, 150}}},
      xt::xtensor<float, 3>{{{1, 2, 3, 4, 5},   //
                             {6, 7, 8, 9, 10},  //
                             {11, 12, 13, 14, 15}},
                            {{10, 20, 30, 40, 50},   //
                             {60, 70, 80, 90, 100},  //
                             {110, 120, 130, 140, 150}}}};

  {
    // WHAT
    const std::vector<xt::xarray<float>> sorted_x_asc =
        EvalSortOp<float>(VIS_PUBLIC, x, 0, true, true);
    const std::vector<xt::xarray<float>> expected_sorted_x_asc = {
        xt::xtensor<float, 3>{{{1, 2, 3, 4, 5},   //
                               {6, 7, 8, 9, 10},  //
                               {11, 12, 13, 14, 15}},
                              {{10, 20, 30, 40, 50},   //
                               {60, 70, 80, 90, 100},  //
                               {110, 120, 130, 140, 150}}},
        xt::xtensor<float, 3>{{{1, 2, 3, 4, 5},   //
                               {6, 7, 8, 9, 10},  //
                               {11, 12, 13, 14, 15}},
                              {{10, 20, 30, 40, 50},   //
                               {60, 70, 80, 90, 100},  //
                               {110, 120, 130, 140, 150}}}};

    // THEN
    for (size_t i = 0; i < expected_sorted_x_asc.size(); i++) {
      EXPECT_TRUE(
          xt::allclose(expected_sorted_x_asc[i], sorted_x_asc[i], 0.01, 0.001))
          << expected_sorted_x_asc[i] << std::endl
          << sorted_x_asc[i] << std::endl;
    }
  }
}

}  // namespace
}  // namespace ppu::hal
