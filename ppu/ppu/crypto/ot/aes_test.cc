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


#include "ppu/crypto/ot/aes.h"

#include <future>
#include <random>
#include <thread>

#include "fmt/format.h"
#include "gtest/gtest.h"

#include "ppu/crypto/pseudo_random_generator.h"
#include "ppu/crypto/symmetric_crypto.h"

namespace ppu {
constexpr uint64_t kKeyWidth = 4;
constexpr uint128_t kIv1 = 1;

TEST(BaseAesTest, Test) {
  MultiKeyAES<kKeyWidth> multi_key_aes;
  std::random_device rd;
  PseudoRandomGenerator<uint128_t> prg(rd());

  std::array<uint128_t, kKeyWidth> keys_u128;
  std::array<block, kKeyWidth> keys_block;

  std::array<uint128_t, kKeyWidth> plain_u128, cipher_u128, encrypted_u128;
  std::array<block, kKeyWidth> plain_block, encrypted_block;

  for (uint64_t i = 0; i < kKeyWidth; ++i) {
    keys_u128[i] = prg();
    keys_block[i] = block(keys_u128[i]);
    plain_u128[i] = prg();
    plain_block[i] = block(plain_u128[i]);
  }

  // data from nist website
  // https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core128.pdf
  keys_u128[0] = MakeUint128(0x3C4FCF098815F7AB, 0xA6D2AE2816157E2B);
  plain_u128[0] = MakeUint128(0x2A179373117E3DE9, 0x969F402EE2BEC16B);
  cipher_u128[0] = MakeUint128(0x97EF6624F3CA9EA8, 0x60367A0DB47BD73A);

  keys_block[0] = block(keys_u128[0]);
  plain_block[0] = block(plain_u128[0]);

  multi_key_aes.SetKeys(absl::MakeSpan(keys_block));
  multi_key_aes.EcbEncNBlocks(plain_block.data(), encrypted_block.data());

  auto type = SymmetricCrypto::CryptoType::AES128_ECB;
  SymmetricCrypto crypto(type, keys_u128[0], kIv1);
  crypto.Encrypt(absl::MakeConstSpan(plain_u128),
                 absl::MakeSpan(encrypted_u128));

  AES aes_ni;
  block encrypted_block2;
  aes_ni.SetKey(keys_block[0]);
  aes_ni.EcbEncBlock(plain_block[0], encrypted_block2);

  EXPECT_EQ(plain_u128[0], ((uint128_t)(plain_block[0].mData)));
  EXPECT_EQ(cipher_u128[0], ((uint128_t)(encrypted_block2.mData)));
  EXPECT_EQ(encrypted_u128[0], ((uint128_t)(encrypted_block2.mData)));
  EXPECT_EQ(encrypted_u128[0], ((uint128_t)(encrypted_block[0].mData)));
}
}  // namespace ppu
