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

#include "absl/types/span.h"

#include "ppu/crypto/ot/block.h"
#include "ppu/utils/exception.h"

#ifndef __x86_64__
#define PPU_ENABLE_PORTABLE_AES
#else
#define PPU_ENABLE_AESNI
#endif

namespace ppu {
// aes code from
// https://github.com/ladnir/cryptoTools/blob/master/cryptoTools/Crypto/AES.h
// https://github.com/ladnir/cryptoTools/blob/master/cryptoTools/Crypto/AES.cpp
// performance little better than openssl
// support batch encrypt, encrypt/decrypt 2、4、8、16 block
namespace details {

enum AESTypes { NI, Portable };
template <AESTypes types>
class AES {
 public:
  // Default constructor leave the class in an invalid state
  // until setKey(...) is called.
  AES() = default;
  AES(const AES&) = default;

  // Constructor to initialize the class with the given key
  AES(const block& userKey);

  // Set the key to be used for encryption.
  void SetKey(const block& userKey);

  // Encrypts the plaintext block and stores the result in ciphertext
  void EcbEncBlock(const block& plaintext, block& ciphertext) const;

  // Encrypts the plaintext block and returns the result
  block EcbEncBlock(const block& plaintext) const;

  // Encrypts blockLength starting at the plaintexts pointer and writes the
  // result to the ciphertext pointer
  void EcbEncBlocks(const block* plaintexts, uint64_t blockLength,
                    block* ciphertext) const;

  void EcbEncBlocks(absl::Span<const block> plaintexts,
                    absl::Span<block> ciphertext) const {
    PPU_ENFORCE(plaintexts.size() != ciphertext.size());
    EcbEncBlocks(plaintexts.data(), plaintexts.size(), ciphertext.data());
  }

  // Encrypts 2 blocks pointer to by plaintexts and writes the result to
  // ciphertext
  void EcbEncTwoBlocks(const block* plaintexts, block* ciphertext) const;

  // Encrypts 4 blocks pointer to by plaintexts and writes the result to
  // ciphertext
  void EcbEncFourBlocks(const block* plaintexts, block* ciphertext) const;

  // Encrypts 16 blocks pointer to by plaintexts and writes the result to
  // ciphertext
  void EcbEnc16Blocks(const block* plaintexts, block* ciphertext) const;

  // Encrypts the vector of blocks {baseIdx, baseIdx + 1, ..., baseIdx + length
  // - 1} and writes the result to ciphertext.
  void EcbEncCounterMode(uint64_t base_idx, uint64_t length,
                         block* ciphertext) const {
    EcbEncCounterMode(toBlock(base_idx), length, ciphertext);
  }
  void EcbEncCounterMode(uint64_t baseIdx, absl::Span<block> ciphertext) const {
    EcbEncCounterMode(toBlock(baseIdx), ciphertext.size(), ciphertext.data());
  }
  void EcbEncCounterMode(block baseIdx, absl::Span<block> ciphertext) const {
    EcbEncCounterMode(baseIdx, ciphertext.size(), ciphertext.data());
  }
  void EcbEncCounterMode(block baseIdx, uint64_t length,
                         block* ciphertext) const;

  // Returns the current key.
  const block& GetKey() const { return round_key_[0]; }

  static block RoundEnc(block state, const block& round_key);
  static block FinalEnc(block state, const block& round_key);

  // The expanded key.
  std::array<block, 11> round_key_;
};

// A class to perform AES decryption.
template <AESTypes type>
class AESDec {
 public:
  AESDec() = default;
  AESDec(const AESDec&) = default;
  AESDec(const block& userKey);

  void SetKey(const block& userKey);
  void EcbDecBlock(const block& ciphertext, block& plaintext);
  block EcbDecBlock(const block& ciphertext);

  std::array<block, 11> round_key_;

  static block RoundDec(block state, const block& round_key);
  static block FinalDec(block state, const block& round_key);
};
// void InvCipher(block& state, std::array<block, 11>& RoundKey);

}  // namespace details

#ifdef PPU_ENABLE_AESNI
using AES = details::AES<details::NI>;
using AESDec = details::AESDec<details::NI>;
#else
using AES = details::AES<details::Portable>;
using AESDec = details::AESDec<details::Portable>;
#endif

// Specialization of the AES class to support encryption of N values under N
// different keys
template <int N>
class MultiKeyAES {
 public:
  std::array<AES, N> aes_instances;

  // Default constructor leave the class in an invalid state
  // until setKey(...) is called.
  MultiKeyAES() = default;

  // Constructor to initialize the class with the given key
  MultiKeyAES(absl::Span<block> keys) { SetKeys(keys); }

  // Set the N keys to be used for encryption.
  void SetKeys(absl::Span<block> keys) {
    for (uint64_t i = 0; i < N; ++i) {
      aes_instances[i].SetKey(keys[i]);
    }
  }

  // Computes the encrpytion of N blocks pointed to by plaintext
  // and stores the result at ciphertext.
  void EcbEncNBlocks(const block* plaintext, block* ciphertext) const {
    for (int i = 0; i < N; ++i)
      ciphertext[i] = plaintext[i] ^ aes_instances[i].round_key_[0];
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[1]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[2]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[3]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[4]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[5]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[6]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[7]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[8]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::RoundEnc(ciphertext[i], aes_instances[i].round_key_[9]);
    for (int i = 0; i < N; ++i)
      ciphertext[i] =
          AES::FinalEnc(ciphertext[i], aes_instances[i].round_key_[10]);
  }

  // Utility to compare the keys.
  const MultiKeyAES<N>& operator=(const MultiKeyAES<N>& rhs) {
    for (uint64_t i = 0; i < N; ++i)
      for (uint64_t j = 0; j < 11; ++j)
        aes_instances[i].round_key_[j] = rhs.aes_instances[i].round_key_[j];

    return rhs;
  }
};

//// A class to perform AES decryption.
// class AESDec2
//{
// public:
//    AESDec2() = default;
//    AESDec2(const AESDec2&) = default;
//    AESDec2(const block& userKey);
//
//    void SetKey(const block& userKey);
//    void EcbDecBlock(const block& ciphertext, block& plaintext);
//    block EcbDecBlock(const block& ciphertext);
//
//    block mRoundKey[11];
//
//};

// An AES instance with a fixed and public key.
extern const AES kAesFixedKey;

}  // namespace ppu
