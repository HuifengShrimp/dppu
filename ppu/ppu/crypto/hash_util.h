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

#include "ppu/crypto/ssl_hash.h"
#include "ppu/utils/int128.h"

namespace ppu::crypto {

std::vector<uint8_t> Sha256(utils::ByteContainerView data);

std::vector<uint8_t> Sm3(utils::ByteContainerView data);

std::vector<uint8_t> Blake2(utils::ByteContainerView data);
std::vector<uint8_t> Blake3(utils::ByteContainerView data);

uint128_t Blake3_128(utils::ByteContainerView data);

}  // namespace ppu::crypto
