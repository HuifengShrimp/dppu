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

#include <array>
#include <cstring>
#include <numeric>

#include "ppu/crypto/symmetric_crypto.h"

namespace ppu {
namespace internal {
template <typename T, size_t BATCH_SIZE>
struct cipher_data {
  std::array<T, BATCH_SIZE> cipher_budget_;
  T& operator[](size_t idx) { return cipher_budget_[idx]; }
  const T& operator[](size_t idx) const { return cipher_budget_[idx]; }
};

// TODO: `bool` memory usage is not ideal. Figure out a better way.
template <size_t BATCH_SIZE>
struct cipher_data<bool, BATCH_SIZE> {
  std::array<std::uint8_t, BATCH_SIZE> cipher_budget_;
  bool operator[](const size_t& idx) { return !!(cipher_budget_[idx] & 0x01); }
  bool operator[](const size_t& idx) const {
    return !!(cipher_budget_[idx] & 0x01);
  }
};
}  // namespace internal

template <typename T, size_t BATCH_SIZE = 128,
          std::enable_if_t<std::is_standard_layout_v<T>, int> = 0>
class PseudoRandomGenerator {
 public:
  static_assert(BATCH_SIZE % sizeof(uint128_t) == 0);

  explicit PseudoRandomGenerator(uint128_t seed = 0) { SetSeed(seed); }

  uint128_t Seed() const { return seed_; }

  static constexpr size_t BatchSize() { return BATCH_SIZE; }

  void SetSeed(uint128_t seed) {
    seed_ = seed;
    // Reset counter. Make this behave same with STL PRG.
    counter_ = 0;
  }

  T operator()() {
    if (num_consumed_ == cipher_data_.cipher_budget_.size()) {
      // Generate budgets.
      GenerateBudgets();
      // Reset consumed.
      num_consumed_ = 0;
    }
    return cipher_data_[num_consumed_++];
  }

  template <typename Y,
            std::enable_if_t<std::is_trivially_copyable_v<Y>, int> = 0>
  void Fill(absl::Span<Y> out) {
    // `Fill` does not consumes cipher_budgets but do increase the internal
    // counter.
    counter_ = FillAesRandom(SymmetricCrypto::CryptoType::AES128_ECB, seed_,
                             kInitVector, counter_, out);
  }

  inline static constexpr uint128_t kInitVector = 0;

 private:
  void GenerateBudgets() {
    counter_ = FillAesRandom(SymmetricCrypto::CryptoType::AES128_ECB, seed_,
                             kInitVector, counter_,
                             absl::MakeSpan(cipher_data_.cipher_budget_));
  }

  // Seed.
  uint128_t seed_;
  // Counter as encrypt messages.
  uint128_t counter_ = 0;
  // Cipher budget.
  internal::cipher_data<T, BATCH_SIZE> cipher_data_;
  // How many ciphers are consumed.
  size_t num_consumed_ = BATCH_SIZE;
};

}  // namespace ppu
