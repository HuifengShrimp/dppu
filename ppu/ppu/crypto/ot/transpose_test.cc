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


#include <string.h>

#include <array>
#include <iostream>

#include "gtest/gtest.h"

#include "ppu/crypto/ot/utils.h"

std::array<uint128_t, 128> MakeMatrix128() {
  std::array<uint128_t, 128> ret;
  for (size_t i = 0; i < 128; ++i) {
    ret[i] = rand();
  }
  return ret;
}

namespace ppu {

TEST(MatrixTranspose, NaiveTransposeTest) {
  auto matrix = MakeMatrix128();

  std::array<uint128_t, 128> matrixTranspose;
  memcpy(matrixTranspose.data(), matrix.data(),
         matrix.size() * sizeof(uint128_t));

  NaiveTranspose(&matrixTranspose);

  std::array<uint128_t, 128> matrixT2;

  memcpy(matrixT2.data(), matrixTranspose.data(),
         matrixTranspose.size() * sizeof(uint128_t));
  NaiveTranspose(&matrixT2);

  EXPECT_EQ(matrix, matrixT2);
}

TEST(MatrixTranspose, EklundhTransposeTest) {
  auto matrix = MakeMatrix128();

  std::array<uint128_t, 128> matrixTranspose;
  memcpy(matrixTranspose.data(), matrix.data(),
         matrix.size() * sizeof(uint128_t));

  EklundhTranspose128(&matrixTranspose);

  std::array<uint128_t, 128> matrixT2;

  memcpy(matrixT2.data(), matrixTranspose.data(),
         matrixTranspose.size() * sizeof(uint128_t));
  NaiveTranspose(&matrixT2);

  EXPECT_EQ(matrix, matrixT2);
}

TEST(MatrixTranspose, SseTransposeTest) {
  auto matrix = MakeMatrix128();

  std::array<uint128_t, 128> matrixTranspose;
  memcpy(matrixTranspose.data(), matrix.data(),
         matrix.size() * sizeof(uint128_t));

  SseTranspose128(&matrixTranspose);

  std::array<uint128_t, 128> matrixT2;

  memcpy(matrixT2.data(), matrixTranspose.data(),
         matrixTranspose.size() * sizeof(uint128_t));
  NaiveTranspose(&matrixT2);

  EXPECT_EQ(matrix, matrixT2);
}

}  // end namespace ppu
