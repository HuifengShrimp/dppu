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


#include "ppu/mpc/aby3/ot.h"

#include "gtest/gtest.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/mpc/base2k/public.h"
#include "ppu/mpc/util/test_util.h"

namespace ppu::mpc::aby3 {
class OTTest : public ::testing::TestWithParam<std::tuple<size_t, FieldType>> {
};

template <FieldType _kField>
std::vector<ArrayRef> OTExpectResult(std::vector<ArrayRef> r0_v,
                                     std::vector<ArrayRef> r1_v,
                                     std::vector<ArrayRef> c_v) {
  // EXPECT RESULT
  std::vector<ArrayRef> expect_res_v;
  for (size_t i = 0; i < c_v.size(); ++i) {
    const auto m0 = xt_adapt<ring2k_t>(r0_v[i]);
    const auto m1 = xt_adapt<ring2k_t>(r1_v[i]);
    const auto c = xt_adapt<ring2k_t>(c_v[i]);
    xt::xarray<ring2k_t> m = m0;
    for (size_t j = 0; j < c.shape(0); ++j) {
      m(j) = c(j) ? m1(j) : m0(j);
    }
    expect_res_v.push_back(make_array(m, r0_v[0].eltype()));
  }
  return expect_res_v;
}

TEST_P(OTTest, OT3Party) {
  const Rank kWorldSize = std::get<0>(GetParam());
  const FieldType kField = std::get<1>(GetParam());
  const Shape shape{3, 4};
  const size_t ot_cnt = 5;

  test::Eval(kWorldSize, [&](std::shared_ptr<link::Context> lctx) {
    DISPATCH_ALL_FIELDS(kField, "OTTest.OT", [&]() {
      constexpr size_t num_bits = sizeof(ring2k_t) * 8;

      base2k::Random rnd(lctx);
      base2k::Public pub;

      // GIVEN
      std::vector<ArrayRef> r0_v;
      std::vector<ArrayRef> r1_v;
      std::vector<ArrayRef> c_v;
      std::vector<ArrayRef> res_v;
      for (size_t i = 0; i < ot_cnt; ++i) {
        r0_v.emplace_back(rnd.RandP(kField, shape.numel()));
        r1_v.emplace_back(rnd.RandP(kField, shape.numel()));
        // we assume c_v only have buffer 0 or 1, so we only use the LSB of c_v
        c_v.emplace_back(
            pub.RShiftP(rnd.RandP(kField, shape.numel()), num_bits - 1));
      }

      // OT
      OT3Party ot(lctx);
      if (lctx->Rank() == 0) {
        ot.OTSend(absl::Span<const ArrayRef>(r0_v),
                  absl::Span<const ArrayRef>(r1_v));
      } else if (lctx->Rank() == 1) {
        res_v.resize(c_v.size());
        ot.OTRecv(absl::Span<const ArrayRef>(c_v), absl::MakeSpan(res_v));
      } else if (lctx->Rank() == 2) {
        ot.OTHelp(absl::Span<const ArrayRef>(c_v));
      }

      // CHECK
      std::vector<ArrayRef> expect_res_v =
          OTExpectResult<_kField>(r0_v, r1_v, c_v);

      for (size_t i = 0; i < res_v.size(); ++i) {
        EXPECT_TRUE(test::EqualsPP(kField, res_v[i], expect_res_v[i]));
      }
    });
  });
}

INSTANTIATE_TEST_SUITE_P(
    OTTestInstances, OTTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128)),
    [](const testing::TestParamInfo<OTTest::ParamType>& info) {
      return fmt::format("{}x{}", std::get<0>(info.param),
                         std::get<1>(info.param));
    });

}  // namespace ppu::mpc::aby3
