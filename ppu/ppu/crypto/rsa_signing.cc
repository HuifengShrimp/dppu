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




#include "ppu/crypto/rsa_signing.h"

#include "openssl/evp.h"
#include "openssl/pem.h"

#include "ppu/utils/exception.h"
#include "ppu/utils/scope_guard.h"

namespace ppu::crypto {

std::unique_ptr<RsaSigner> RsaSigner::CreateFromPem(
    utils::ByteContainerView pem) {
  BIO* bio_pkey = BIO_new_mem_buf(pem.data(), pem.size());
  PPU_ENFORCE(bio_pkey, "Failed to create BIO");
  ON_SCOPE_EXIT([&bio_pkey] { BIO_free(bio_pkey); });
  RSA* rsa = PEM_read_bio_RSAPrivateKey(bio_pkey, NULL, NULL, NULL);
  PPU_ENFORCE(rsa, "Failed to get rsa from pem.");
  EVP_PKEY* pkey = EVP_PKEY_new();
  PPU_ENFORCE(pkey, "Failed to create evp key.");
  // Use assign here for that rsa will be freed when pkey is freed.
  PPU_ENFORCE_EQ(EVP_PKEY_assign_RSA(pkey, rsa), 1);
  // Using `new` to access a non-public constructor.
  // ref https://abseil.io/tips/134
  return std::unique_ptr<RsaSigner>(
      new RsaSigner(UniquePkey(pkey, &EVP_PKEY_free)));
}

SignatureScheme RsaSigner::GetSignatureSchema() const { return schema_; }

std::vector<uint8_t> RsaSigner::Sign(utils::ByteContainerView message) const {
  EVP_MD_CTX* m_ctx = EVP_MD_CTX_new();
  PPU_ENFORCE(m_ctx != nullptr);
  ON_SCOPE_EXIT([&] { EVP_MD_CTX_free(m_ctx); });

  PPU_ENFORCE_GT(
      EVP_DigestSignInit(m_ctx, nullptr, EVP_sha256(), nullptr, pkey_.get()),
      0);
  PPU_ENFORCE_GT(EVP_DigestSignUpdate(m_ctx, message.data(), message.size()),
                 0);

  // Determine the size of the signature
  // Note that sig_len is the max but not exact size of the output buffer.
  // Ref https://www.openssl.org/docs/man1.1.1/man3/EVP_DigestSignFinal.html
  size_t sig_len;
  PPU_ENFORCE_GT(EVP_DigestSignFinal(m_ctx, nullptr, &sig_len), 0);
  std::vector<uint8_t> signature(sig_len);
  PPU_ENFORCE_GT(EVP_DigestSignFinal(m_ctx, signature.data(), &sig_len), 0);
  // Correct the signature size.
  signature.resize(sig_len);

  return signature;
}

std::unique_ptr<RsaVerifier> RsaVerifier::CreateFromPem(
    utils::ByteContainerView pem) {
  BIO* bio_pkey = BIO_new_mem_buf(pem.data(), pem.size());
  PPU_ENFORCE(bio_pkey, "Failed to create BIO");
  ON_SCOPE_EXIT([&bio_pkey] { BIO_free(bio_pkey); });
  RSA* rsa = PEM_read_bio_RSAPublicKey(bio_pkey, NULL, NULL, NULL);
  PPU_ENFORCE(rsa, "Failed to get rsa from pem.");
  EVP_PKEY* pkey = EVP_PKEY_new();
  PPU_ENFORCE(pkey, "Failed to create evp key.");
  // Use assign here for that rsa will be freed when pkey is freed.
  PPU_ENFORCE_EQ(EVP_PKEY_assign_RSA(pkey, rsa), 1);
  // Using `new` to access a non-public constructor.
  // ref https://abseil.io/tips/134
  return std::unique_ptr<RsaVerifier>(
      new RsaVerifier(UniquePkey(pkey, &EVP_PKEY_free)));
}

SignatureScheme RsaVerifier::GetSignatureSchema() const { return schema_; }

void RsaVerifier::Verify(utils::ByteContainerView message,
                         utils::ByteContainerView signature) const {
  EVP_MD_CTX* m_ctx = EVP_MD_CTX_new();
  PPU_ENFORCE(m_ctx != nullptr);
  ON_SCOPE_EXIT([&] { EVP_MD_CTX_free(m_ctx); });

  PPU_ENFORCE_GT(
      EVP_DigestVerifyInit(m_ctx, nullptr, EVP_sha256(), nullptr, pkey_.get()),
      0);
  PPU_ENFORCE_GT(EVP_DigestVerifyUpdate(m_ctx, message.data(), message.size()),
                 0);
  PPU_ENFORCE_GT(
      EVP_DigestVerifyFinal(m_ctx, signature.data(), signature.size()), 0);
}

}  // namespace ppu::crypto