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




#pragma once

#include "openssl/bio.h"
#include "openssl/evp.h"

#include "ppu/crypto/signing.h"

namespace ppu::crypto {

// RSA sign with sha256
class RsaSigner final : public crypto::AsymmetricSigner {
 public:
  using UniquePkey = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;

  static std::unique_ptr<RsaSigner> CreateFromPem(utils::ByteContainerView pem);

  SignatureScheme GetSignatureSchema() const override;

  std::vector<uint8_t> Sign(utils::ByteContainerView message) const override;

 private:
  RsaSigner(UniquePkey pkey)
      : pkey_(std::move(pkey)),
        schema_(SignatureScheme::RSA_SIGNING_SHA256_HASH) {}

  const UniquePkey pkey_;
  const SignatureScheme schema_;
};

// RSA verify with sha256
class RsaVerifier final : public crypto::AsymmetricVerifier {
 public:
  using UniquePkey = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;

  static std::unique_ptr<RsaVerifier> CreateFromPem(
      utils::ByteContainerView pem);

  SignatureScheme GetSignatureSchema() const override;

  void Verify(utils::ByteContainerView message,
              utils::ByteContainerView signature) const override;

 private:
  RsaVerifier(UniquePkey pkey)
      : pkey_(std::move(pkey)),
        schema_(SignatureScheme::RSA_SIGNING_SHA256_HASH) {}

  const UniquePkey pkey_;
  const SignatureScheme schema_;
};

}  // namespace ppu::crypto