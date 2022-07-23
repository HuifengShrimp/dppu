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


#include "ppu/hal/value.h"

#include "gtest/gtest.h"

namespace ppu::hal {

TEST(ValueTest, Empty) {
  // default constructor makes a placeholder value.
  Value a;

  EXPECT_EQ(a.vtype(), VIS_INVALID);
  EXPECT_FALSE(a.is_public());
  EXPECT_FALSE(a.is_secret());

  EXPECT_EQ(a.dtype(), DT_INVALID);
  EXPECT_FALSE(a.is_fxp());
  EXPECT_FALSE(a.is_int());
}

// FIXME(jint)
// TEST(ValueTest, Sanity) {
//   // default constructor makes a placeholder value.
//   int ival = 1;
//   double dval = 1.0F;
//   Value a(VIS_PUBLIC, DT_INT, {}, Buffer(&ival, sizeof(int)));
//   {
//     a.GetBuffer()->data<int>()[0] = 1;
//     EXPECT_EQ(a.vtype(), VIS_PUBLIC);
//     EXPECT_TRUE(a.is_public());
//     EXPECT_FALSE(a.is_secret());

//    EXPECT_EQ(a.dtype(), DT_INT);
//    EXPECT_FALSE(a.is_fxp());
//    EXPECT_TRUE(a.is_int());
//    EXPECT_EQ(a.GetBuffer()->size(), sizeof(int));
//    EXPECT_EQ(a.GetBuffer()->data<int>()[0], 1);
//  }

//  Value b(VIS_SECRET, DT_FXP, {}, Buffer(&dval, sizeof(double)));
//  {
//    b.GetBuffer()->data<double>()[0] = 1;

//    EXPECT_EQ(b.vtype(), VIS_SECRET);
//    EXPECT_FALSE(b.is_public());
//    EXPECT_TRUE(b.is_secret());

//    EXPECT_EQ(b.dtype(), DT_FXP);
//    EXPECT_TRUE(b.is_fxp());
//    EXPECT_FALSE(b.is_int());
//    EXPECT_EQ(b.GetBuffer()->data<double>()[0], 1);
//  }

//  Value c = a;
//  {
//    EXPECT_EQ(c.vtype(), VIS_PUBLIC);
//    EXPECT_TRUE(c.is_public());
//    EXPECT_FALSE(c.is_secret());

//    EXPECT_EQ(c.dtype(), DT_INT);
//    EXPECT_FALSE(c.is_fxp());
//    EXPECT_TRUE(c.is_int());
//    EXPECT_EQ(a.GetBuffer()->data<void>(), c.GetBuffer()->data<void>());
//    EXPECT_EQ(c.GetBuffer()->size(), sizeof(int));
//    EXPECT_EQ(c.GetBuffer()->data<int>()[0], 1);
//  }

//  Value d = std::move(b);
//  {
//    EXPECT_EQ(d.vtype(), VIS_SECRET);
//    EXPECT_FALSE(d.is_public());
//    EXPECT_TRUE(d.is_secret());

//    EXPECT_EQ(d.dtype(), DT_FXP);
//    EXPECT_TRUE(d.is_fxp());
//    EXPECT_FALSE(d.is_int());

//    EXPECT_EQ(d.GetBuffer()->data<double>()[0], 1);
//  }
//}

}  // namespace ppu::hal
