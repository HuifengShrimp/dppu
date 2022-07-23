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

#include <vector>

#include "ppu/utils/byte_container_view.h"

namespace ppu::crypto {

enum class SignatureScheme : int {
  UNKNOWN,
  SM2_SIGNING_SM3_HASH,
  RSA_SIGNING_SHA256_HASH,
};

class AsymmetricSigner {
 public:
  virtual ~AsymmetricSigner() = default;

  virtual SignatureScheme GetSignatureSchema() const = 0;

  virtual std::vector<uint8_t> Sign(utils::ByteContainerView message) const = 0;
};

class AsymmetricVerifier {
 public:
  virtual ~AsymmetricVerifier() = default;

  virtual SignatureScheme GetSignatureSchema() const = 0;

  virtual void Verify(utils::ByteContainerView message,
                      utils::ByteContainerView signature) const = 0;
};

}  // namespace ppu::crypto