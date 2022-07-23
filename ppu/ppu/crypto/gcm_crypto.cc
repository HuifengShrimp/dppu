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


#include "ppu/crypto/gcm_crypto.h"

#include "openssl/evp.h"

#include "ppu/utils/exception.h"
#include "ppu/utils/scope_guard.h"

namespace ppu::crypto {

namespace {

constexpr size_t kAesMacSize = 16;

const EVP_CIPHER* CreateEvpCipher(GcmCryptoSchema schema) {
  switch (schema) {
    case GcmCryptoSchema::AES128_GCM:
      return EVP_aes_128_gcm();
    case GcmCryptoSchema::AES256_GCM:
      return EVP_aes_256_gcm();
    default:
      PPU_THROW("Unknown crypto schema: {}", static_cast<int>(schema));
  }
}

size_t GetMacSize(GcmCryptoSchema schema) {
  switch (schema) {
    case GcmCryptoSchema::AES128_GCM:
      return kAesMacSize;
    case GcmCryptoSchema::AES256_GCM:
      return kAesMacSize;
    default:
      PPU_THROW("Unknown crypto schema: {}", static_cast<int>(schema));
  }
}

}  // namespace

void GcmCrypto::Encrypt(utils::ByteContainerView plaintext,
                        utils::ByteContainerView aad,
                        absl::Span<uint8_t> ciphertext,
                        absl::Span<uint8_t> mac) const {
  PPU_ENFORCE_EQ(ciphertext.size(), plaintext.size());
  PPU_ENFORCE_EQ(mac.size(), GetMacSize(schema_));

  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  PPU_ENFORCE(ctx, "Failed to new evp cipher context.");
  ON_SCOPE_EXIT([&] { EVP_CIPHER_CTX_free(ctx); });
  const EVP_CIPHER* cipher = CreateEvpCipher(schema_);
  PPU_ENFORCE_EQ(key_.size(), (size_t)EVP_CIPHER_key_length(cipher));
  PPU_ENFORCE_EQ(iv_.size(), (size_t)EVP_CIPHER_iv_length(cipher));
  PPU_ENFORCE_EQ(
      EVP_EncryptInit_ex(ctx, cipher, nullptr, key_.data(), iv_.data()), 1);
  int out_length;
  // Provide AAD data if exist
  const int aad_len = aad.size();
  if (0 != aad_len) {
    PPU_ENFORCE_EQ(
        EVP_EncryptUpdate(ctx, NULL, &out_length, aad.data(), aad_len), 1);
    PPU_ENFORCE_EQ(out_length, (int)aad.size());
  }
  PPU_ENFORCE_EQ(EVP_EncryptUpdate(ctx, ciphertext.data(), &out_length,
                                   plaintext.data(), plaintext.size()),
                 1);
  PPU_ENFORCE_EQ(out_length, (int)plaintext.size(),
                 "Unexpected encrypte out length.");
  // Note that get no output here as the data is always aligned for GCM.
  EVP_EncryptFinal_ex(ctx, nullptr, &out_length);
  PPU_ENFORCE_EQ(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG,
                                     GetMacSize(schema_), mac.data()),
                 1, "Failed to get mac.");
}

void GcmCrypto::Decrypt(utils::ByteContainerView ciphertext,
                        utils::ByteContainerView aad,
                        utils::ByteContainerView mac,
                        absl::Span<uint8_t> plaintext) const {
  PPU_ENFORCE_EQ(ciphertext.size(), plaintext.size());
  PPU_ENFORCE_EQ(mac.size(), GetMacSize(schema_));

  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  PPU_ENFORCE(ctx, "Failed to new evp cipher context.");
  ON_SCOPE_EXIT([&] { EVP_CIPHER_CTX_free(ctx); });
  const EVP_CIPHER* cipher = CreateEvpCipher(schema_);
  PPU_ENFORCE_EQ(key_.size(), (size_t)EVP_CIPHER_key_length(cipher));
  PPU_ENFORCE_EQ(iv_.size(), (size_t)EVP_CIPHER_iv_length(cipher));
  PPU_ENFORCE(
      EVP_DecryptInit_ex(ctx, cipher, nullptr, key_.data(), iv_.data()));

  int out_length;
  // Provide AAD data if exist
  const int aad_len = aad.size();
  if (0 != aad_len) {
    PPU_ENFORCE_EQ(
        EVP_DecryptUpdate(ctx, NULL, &out_length, aad.data(), aad_len), 1);
    PPU_ENFORCE_EQ(out_length, (int)aad.size());
  }
  PPU_ENFORCE_EQ(EVP_DecryptUpdate(ctx, plaintext.data(), &out_length,
                                   ciphertext.data(), ciphertext.size()),
                 1);
  PPU_ENFORCE_EQ(out_length, (int)plaintext.size(),
                 "Unexpcted decryption out length.");
  PPU_ENFORCE_EQ(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG,
                                     GetMacSize(schema_), (void*)mac.data()),
                 1, "Failed to get mac.");
  // Note that get no output here as the data is always aligned for GCM.
  PPU_ENFORCE_EQ(EVP_DecryptFinal_ex(ctx, nullptr, &out_length), 1,
                 "Failed to verfiy mac.");
}

}  // namespace ppu::crypto
