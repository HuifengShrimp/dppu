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


#include "ppu/crypto/ot/base_ot.h"

#include <chrono>
#include <random>

#ifdef USE_PORTABLE_OT
#include "simplest_ot_portable/ot_receiver.h"
#include "simplest_ot_portable/ot_sender.h"
#else
#include "simplest_ot_x86_asm/ot_receiver.h"
#include "simplest_ot_x86_asm/ot_sender.h"
#endif

#include "ppu/crypto/pseudo_random_generator.h"
#include "ppu/crypto/random_oracle.h"
#include "ppu/crypto/utils.h"
#include "ppu/utils/exception.h"

namespace ppu {
namespace {

#ifdef USE_PORTABLE_OT
// Must be 1 in portable simplest-ot.
constexpr int kBatchSize = 1;
#else
// Must be 4 in x86_asm simplest-ot.
constexpr int kBatchSize = 4;
#endif

template <size_t N>
Buffer StringFromBytes(unsigned char (&bytes)[N]) {
  // Auto detect string size in compile time.
  return {reinterpret_cast<const void*>(bytes), N};
}

}  // namespace

void BaseOtRecv(const std::shared_ptr<link::Context>& ctx,
                const std::vector<bool>& choices,
                absl::Span<Block> recv_blocks) {
  PPU_ENFORCE_EQ(ctx->WorldSize(), 2u);
  PPU_ENFORCE_EQ(choices.size(), recv_blocks.size());
  PPU_ENFORCE(!choices.empty(), "empty choices");

  const int kNumOt = choices.size();
  SIMPLEOT_RECEIVER receiver;

  // Wait for sender S_pack.
  auto buffer = ctx->Recv(ctx->NextRank(), "BASE_OT:S_PACK");
  PPU_ENFORCE_EQ(buffer.size(), static_cast<int64_t>(sizeof(receiver.S_pack)));
  std::memcpy(receiver.S_pack, buffer.data(), buffer.size());

  if (!receiver_procS_check(&receiver)) {
    PPU_THROW("simplest-ot receiver_procS failed");
  }

  receiver_maketable(&receiver);

  for (int i = 0; i < kNumOt; i += kBatchSize) {
    const int batch_size = std::min(kBatchSize, kNumOt - i);

    unsigned char messages[kBatchSize][HASHBYTES];
    unsigned char rs_pack[kBatchSize * PACKBYTES];
    unsigned char batch_choices[kBatchSize] = {0};

    for (int j = 0; j < batch_size; j++) {
      batch_choices[j] = choices[i + j] ? 1 : 0;
    }

    receiver_rsgen(&receiver, rs_pack, batch_choices);
    ctx->Send(ctx->NextRank(), StringFromBytes(rs_pack),
              fmt::format("BASE_OT:{}", i));

    receiver_keygen(&receiver, &messages[0]);
    for (int j = 0; j < batch_size; ++j) {
      static_assert(sizeof(recv_blocks[i]) <= HASHBYTES, "Illegal Block size.");
      std::memcpy(&recv_blocks[i + j], &messages[j][0],
                  sizeof(recv_blocks[i + j]));

      // even though there's already a hash in sender_keygen_check, we need to
      // hash again with the index i to ensure security
      // ref: https://eprint.iacr.org/2021/682

      recv_blocks[i + j] = RandomOracle::GetDefault().Gen(
          recv_blocks[i + j] ^ (i + j));  // output size = 128 bit
    }
  }
}

void BaseOtSend(const std::shared_ptr<link::Context>& ctx,
                absl::Span<std::array<Block, 2>> send_blocks) {
  PPU_ENFORCE(!send_blocks.empty(), "empty inputs");

  const int kNumOt = send_blocks.size();
  SIMPLEOT_SENDER sender;

  // Send S_pack.
  unsigned char S_pack[PACKBYTES];
  sender_genS(&sender, S_pack);
  ctx->Send(ctx->NextRank(), StringFromBytes(S_pack), "BASE_OT:S_PACK");

  for (int i = 0; i < kNumOt; i += kBatchSize) {
    const int batch_size = std::min(kBatchSize, kNumOt - i);

    unsigned char rs_pack[kBatchSize * PACKBYTES];
    unsigned char messages[2][kBatchSize][HASHBYTES];

    auto buffer = ctx->Recv(ctx->NextRank(), fmt::format("BASE_OT:{}", i));
    PPU_ENFORCE_EQ(buffer.size(), static_cast<int64_t>(sizeof(rs_pack)));
    std::memcpy(rs_pack, buffer.data(), buffer.size());
    if (!sender_keygen_check(&sender, rs_pack, messages)) {
      PPU_THROW("simplest-ot: sender_keygen failed");
    }

    for (int j = 0; j < batch_size; ++j) {
      static_assert(sizeof(send_blocks[0][0]) <= HASHBYTES,
                    "Illegal Block size.");

      std::memcpy(&send_blocks[i + j][0], &messages[0][j][0],
                  sizeof(send_blocks[i + j][0]));
      std::memcpy(&send_blocks[i + j][1], &messages[1][j][0],
                  sizeof(send_blocks[i + j][1]));

      // even though there's already a hash in sender_keygen_check, we need to
      // hash again with the index i to ensure security
      // ref: https://eprint.iacr.org/2021/682

      send_blocks[i + j][0] = RandomOracle::GetDefault().Gen(
          send_blocks[i + j][0] ^ (i + j));  // output size = 128 bit
      send_blocks[i + j][1] = RandomOracle::GetDefault().Gen(
          send_blocks[i + j][1] ^ (i + j));  // output size = 128 bit
    }
  }
}

}  // namespace ppu
