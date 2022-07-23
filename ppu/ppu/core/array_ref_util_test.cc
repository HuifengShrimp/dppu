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


#include "ppu/core/array_ref_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ppu {

TEST(ArrayRefUtilTest, XtensorSanity) {
  // create a non-empty ndarray
  xt::xarray<double> a = {{0., 1., 2.}, {3., 4., 5.}};
  {
    ASSERT_THAT(a.shape(), testing::ElementsAre(2, 3));
    ASSERT_THAT(a.strides(), testing::ElementsAre(3, 1));
    ASSERT_EQ(a.size(), 6);
  }

  // Assign to scalar, make it a 0-dimension array.
  // see: https://xtensor.readthedocs.io/en/latest/scalar.html
  {
    double s = 1.2;
    a = s;

    ASSERT_TRUE(a.shape().empty());
    ASSERT_TRUE(a.strides().empty());
    ASSERT_EQ(a.size(), 1);
  }

  // create a empty ndarray
  xt::xarray<double> b;
  {
    ASSERT_TRUE(b.shape().empty());
    ASSERT_TRUE(b.strides().empty());
    ASSERT_EQ(b.size(), 1);
  }
}

TEST(ArrayRefUtilTest, MakeNdArray) {
  // make from scalar
  {
    double _ = 1.0f;
    auto a = make_ndarray(_);
    EXPECT_EQ(a.eltype(), F64);
    EXPECT_EQ(a.buf()->size(), a.elsize());
    EXPECT_EQ(a.numel(), 1);
    EXPECT_TRUE(a.shape().empty());
    EXPECT_TRUE(a.strides().empty());
    EXPECT_EQ(a.offset(), 0);
    EXPECT_EQ(a.at<double>({}), 1.0f);

    EXPECT_THROW(xt_adapt<float>(a), std::exception);
  }

  // make from xcontainer.
  {
    xt::xarray<double> _ = {{0., 1., 2.}, {3., 4., 5.}};
    auto a = make_ndarray(_);
    EXPECT_EQ(a.buf()->size(), a.elsize() * _.size());
    EXPECT_EQ(a.eltype(), F64);
    EXPECT_EQ(a.numel(), _.size());
    EXPECT_EQ(a.shape().size(), _.shape().size());
    for (size_t idx = 0; idx < a.shape().size(); ++idx) {
      EXPECT_EQ(a.shape()[idx], _.shape()[idx]);
    }
    EXPECT_EQ(a.strides(),
              std::vector<int64_t>(_.strides().begin(), _.strides().end()));
    EXPECT_EQ(a.offset(), 0);
    EXPECT_EQ(xt_adapt<double>(a), _);

    EXPECT_THROW(xt_adapt<float>(a), std::exception);
  }

  // make from xexpression
  {
    xt::xarray<double> _ = {{0., 1., 2.}, {3., 4., 5.}};
    auto a = make_ndarray(_ * _ + _);
    EXPECT_EQ(a.buf()->size(), a.elsize() * _.size());
    EXPECT_EQ(a.eltype(), F64);
    EXPECT_EQ(a.numel(), _.size());
    EXPECT_EQ(a.shape().size(), _.shape().size());
    for (size_t idx = 0; idx < a.shape().size(); ++idx) {
      EXPECT_EQ(a.shape()[idx], _.shape()[idx]);
    }
    EXPECT_EQ(a.strides(),
              std::vector<int64_t>(_.strides().begin(), _.strides().end()));
    EXPECT_EQ(a.offset(), 0);
    EXPECT_EQ(xt_adapt<double>(a), _ * _ + _);

    EXPECT_THROW(xt_adapt<float>(a), std::exception);
  }
}

template <typename S>
class EncodingTest : public ::testing::Test {};

using EncodingTestTypes = ::testing::Types<
    // <origin, encoded, decoded>
    std::tuple<float, int64_t, float>,       //
    std::tuple<float, int128_t, float>,      //
    std::tuple<double, int128_t, double>,    //
    std::tuple<float, int128_t, double>,     //
    std::tuple<double, int128_t, int64_t>,   //
    std::tuple<int32_t, int64_t, int32_t>,   //
    std::tuple<int32_t, int128_t, int64_t>,  //
    std::tuple<int32_t, int128_t, float>,    //
    std::tuple<float, int128_t, int>         //
    >;

TYPED_TEST_SUITE(EncodingTest, EncodingTestTypes);

TYPED_TEST(EncodingTest, Simple) {
  using OriginT = typename std::tuple_element<0, TypeParam>::type;
  using EncodedT = typename std::tuple_element<1, TypeParam>::type;
  using DecodedT = typename std::tuple_element<2, TypeParam>::type;

  const FieldType kField = PtTypeToField(PtTypeToEnum<EncodedT>::value);

  // GIVEN
  const xt::xarray<OriginT> x = {{1, 2}, {-3, 0}};
  NdArrayRef arr = make_ndarray(x);

  constexpr size_t kFxpBits = 20;
  DataType dtype;
  Type encoded_ty = makeType<RingTy>(kField);
  auto encoded = encodeToRing(arr, encoded_ty, kFxpBits, &dtype);
  {
    EXPECT_EQ(encoded.eltype(), encoded_ty);
    EXPECT_EQ(encoded.offset(), 0);
    EXPECT_THAT(encoded.shape(), testing::ElementsAre(2, 2));
    EXPECT_THAT(encoded.strides(), testing::ElementsAre(2, 1));

    if constexpr (std::is_floating_point_v<OriginT>) {
      EXPECT_EQ(dtype, DT_FXP);
    } else {
      EXPECT_EQ(dtype, DT_INT);
    }
  }

  Type decodec_ty = makePtType(PtTypeToEnum<DecodedT>::value);
  auto decoded = decodeFromRing(encoded, decodec_ty, kFxpBits, dtype);
  {
    EXPECT_EQ(decoded.eltype(), decodec_ty);
    EXPECT_EQ(decoded.offset(), 0);
    EXPECT_THAT(decoded.shape(), testing::ElementsAre(2, 2));
    EXPECT_THAT(decoded.strides(), testing::ElementsAre(2, 1));
  }

  if constexpr (std::is_floating_point_v<OriginT>) {
    EXPECT_TRUE(xt::allclose(xt_adapt<DecodedT>(decoded), x, 0.01, 0.001))
        << "decoded: " << xt_adapt<DecodedT>(decoded) << std::endl
        << "origin: " << x << std::endl;
  } else {
    EXPECT_EQ(xt_adapt<DecodedT>(decoded), x)
        << "decoded: " << xt_adapt<DecodedT>(decoded) << std::endl
        << "origin: " << x << std::endl;
  }
}

}  // namespace ppu
