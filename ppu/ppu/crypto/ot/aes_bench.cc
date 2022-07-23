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


#include <future>
#include <iostream>
#include <random>

#include "benchmark/benchmark.h"

#include "ppu/crypto/ot/aes.h"
#include "ppu/crypto/pseudo_random_generator.h"

constexpr uint128_t kIv1 = 1;

static void BM_OpensslAes(benchmark::State& state) {
  std::array<uint128_t, 1> plain_u128;
  uint128_t key_u128;

  std::random_device rd;
  ppu::PseudoRandomGenerator<uint128_t> prg(rd());

  key_u128 = prg();
  plain_u128[0] = prg();

  auto type = ppu::SymmetricCrypto::CryptoType::AES128_ECB;
  ppu::SymmetricCrypto crypto(type, key_u128, kIv1);
  std::array<uint128_t, 1> encrypted_u128;

  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    state.ResumeTiming();
    for (size_t i = 0; i < n; i++) {
      crypto.Encrypt(absl::MakeConstSpan(plain_u128),
                     absl::MakeSpan(encrypted_u128));
      plain_u128[0] = encrypted_u128[0];
    }
  }
}

static void BM_OcuAes(benchmark::State& state) {
  ppu::block key_block, plain_block;
  uint128_t key_u128, plain_u128;

  std::random_device rd;
  ppu::PseudoRandomGenerator<uint128_t> prg(rd());

  key_u128 = prg();
  key_block = ppu::block(key_u128);
  plain_u128 = prg();
  plain_block = ppu::block(plain_u128);

  ppu::AES aes_ni;
  ppu::block encrypted_block;
  aes_ni.SetKey(key_block);

  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    state.ResumeTiming();
    for (size_t i = 0; i < n; i++) {
      aes_ni.EcbEncBlock(plain_block, encrypted_block);
      plain_block = encrypted_block;
    }
  }
}

constexpr uint64_t kKeyWidth = 4;

static void BM_OcuMultiAes(benchmark::State& state) {
  std::array<ppu::block, kKeyWidth> keys_block;
  std::array<ppu::block, kKeyWidth> plain_block;
  std::array<uint128_t, kKeyWidth> keys_u128;
  std::array<uint128_t, kKeyWidth> plain_u128;

  std::random_device rd;
  ppu::PseudoRandomGenerator<uint128_t> prg(rd());

  for (uint64_t i = 0; i < kKeyWidth; ++i) {
    keys_u128[i] = prg();
    keys_block[i] = ppu::block(keys_u128[i]);
    plain_u128[i] = prg();
    plain_block[i] = ppu::block(plain_u128[i]);
  }

  ppu::MultiKeyAES<kKeyWidth> multi_key_aes;
  std::array<ppu::block, kKeyWidth> encrypted_block;
  multi_key_aes.SetKeys(absl::MakeSpan(keys_block));

  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    state.ResumeTiming();
    for (size_t i = 0; i < n; i++) {
      multi_key_aes.EcbEncNBlocks(plain_block.data(), encrypted_block.data());
      for (uint64_t i = 0; i < kKeyWidth; ++i) {
        plain_block[i] = encrypted_block[i];
      }
    }
  }
}

BENCHMARK(BM_OpensslAes)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1024)
    ->Arg(5120)
    ->Arg(10240)
    ->Arg(20480)
    ->Arg(40960)
    ->Arg(81920)
    ->Arg(1 << 24);

BENCHMARK(BM_OcuAes)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1024)
    ->Arg(5120)
    ->Arg(10240)
    ->Arg(20480)
    ->Arg(40960)
    ->Arg(81920)
    ->Arg(1 << 24);

BENCHMARK(BM_OcuMultiAes)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256)
    ->Arg(1280)
    ->Arg(2560)
    ->Arg(5120)
    ->Arg(10240)
    ->Arg(20480)
    ->Arg(1 << 22);

BENCHMARK_MAIN();
