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

#include <tuple>
#include <vector>

#include "openssl/bio.h"
#include "openssl/evp.h"

#include "ppu/utils/byte_container_view.h"

namespace ppu::crypto {

namespace internal {

using UniquePkey = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;

UniquePkey CreatePriPkeyFromSm2Pem(utils::ByteContainerView pem);

UniquePkey CreatePubPkeyFromSm2Pem(utils::ByteContainerView pem);

}  // namespace internal

std::tuple<std::string, std::string> CreateSm2KeyPair();

std::tuple<std::string, std::string> CreateRsaKeyPair();

}  // namespace ppu::crypto