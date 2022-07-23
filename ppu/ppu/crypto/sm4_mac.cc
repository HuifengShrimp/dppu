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




#include "ppu/crypto/sm4_mac.h"

#include "ppu/crypto/hmac_sm3.h"
#include "ppu/crypto/ssl_hash.h"
#include "ppu/crypto/symmetric_crypto.h"

namespace ppu::crypto {

namespace {

constexpr size_t HMAC_SIZE = 32;
constexpr size_t HMAC_KEY_SIZE = 16;

}  // namespace

std::vector<uint8_t> Sm4MteEncrypt(utils::ByteContainerView key,
                                   utils::ByteContainerView iv,
                                   utils::ByteContainerView plaintext) {
  // Step 1. get hash of (iv || key)
  std::vector<uint8_t> iv_key_hash =
      Sm3Hash().Update(iv).Update(key).CumulativeHash();

  // Step2. get hmac of plaintext
  PPU_ENFORCE_GE(iv_key_hash.size(), HMAC_KEY_SIZE);
  std::vector<uint8_t> hmac_plaintext =
      HmacSm3(utils::ByteContainerView(iv_key_hash.data(), HMAC_KEY_SIZE))
          .Update(plaintext)
          .CumulativeMac();
  hmac_plaintext.insert(hmac_plaintext.end(), plaintext.begin(),
                        plaintext.end());

  // Step3: do encryption
  std::vector<uint8_t> ciphertext(hmac_plaintext.size());
  SymmetricCrypto(SymmetricCrypto::CryptoType::SM4_CTR, key, iv)
      .Encrypt(hmac_plaintext, absl::MakeSpan(ciphertext));

  return ciphertext;
}

std::vector<uint8_t> Sm4MteDecrypt(utils::ByteContainerView key,
                                   utils::ByteContainerView iv,
                                   utils::ByteContainerView ciphertext) {
  // Step 1. do decryption
  std::vector<uint8_t> hmac_plaintext(ciphertext.size());
  SymmetricCrypto(SymmetricCrypto::CryptoType::SM4_CTR, key, iv)
      .Decrypt(ciphertext, absl::MakeSpan(hmac_plaintext));
  PPU_ENFORCE_GT(hmac_plaintext.size(), HMAC_SIZE);
  utils::ByteContainerView hmac_from_cipher(hmac_plaintext.data(), HMAC_SIZE);
  utils::ByteContainerView plaintext_from_cipher(
      hmac_plaintext.data() + HMAC_SIZE, hmac_plaintext.size() - HMAC_SIZE);

  // Step 2. cal hmac
  std::vector<uint8_t> iv_key_hash =
      Sm3Hash().Update(iv).Update(key).CumulativeHash();
  PPU_ENFORCE_GE(iv_key_hash.size(), HMAC_KEY_SIZE);
  std::vector<uint8_t> hmac_actual =
      HmacSm3(utils::ByteContainerView(iv_key_hash.data(), HMAC_KEY_SIZE))
          .Update(plaintext_from_cipher)
          .CumulativeMac();

  // Step 3. check hmac
  PPU_ENFORCE_EQ(hmac_actual.size(), hmac_from_cipher.size());
  PPU_ENFORCE(std::equal(hmac_actual.begin(), hmac_actual.end(),
                         hmac_from_cipher.begin()));
  return {hmac_plaintext.begin() + HMAC_SIZE, hmac_plaintext.end()};
}

}  // namespace ppu::crypto
