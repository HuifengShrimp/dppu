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




#include "ppu/crypto/hash_util.h"

#include <iostream>

#include "c/blake3.h"

namespace ppu::crypto {

std::vector<uint8_t> Sha256(utils::ByteContainerView data) {
  return SslHash(HashAlgorithm::SHA256).Update(data).CumulativeHash();
}

std::vector<uint8_t> Sm3(utils::ByteContainerView data) {
  return SslHash(HashAlgorithm::SM3).Update(data).CumulativeHash();
}

std::vector<uint8_t> Blake2(utils::ByteContainerView data) {
  return SslHash(HashAlgorithm::BLAKE2B).Update(data).CumulativeHash();
}

std::vector<uint8_t> Blake3(utils::ByteContainerView data) {
  blake3_hasher hasher;

  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, data.data(), data.size());

  std::vector<uint8_t> digest(BLAKE3_OUT_LEN);

  blake3_hasher_finalize(&hasher, digest.data(), BLAKE3_OUT_LEN);

  return digest;
}

uint128_t Blake3_128(utils::ByteContainerView data) {
  std::vector<uint8_t> hash_bytes = Blake3(data);
  uint128_t ret;

  PPU_ENFORCE(hash_bytes.size() >= sizeof(ret));

  std::memcpy((uint8_t*)&ret, hash_bytes.data(), sizeof(ret));
  return ret;
}

}  // namespace ppu::crypto
