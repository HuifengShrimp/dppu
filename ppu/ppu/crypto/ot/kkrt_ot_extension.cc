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


#include "ppu/crypto/ot/kkrt_ot_extension.h"

#include "c/blake3.h"

#include "ppu/crypto/hash_util.h"
#include "ppu/crypto/ot/aes.h"
#include "ppu/crypto/ot/utils.h"
#include "ppu/crypto/pseudo_random_generator.h"
#include "ppu/crypto/random_oracle.h"

namespace ppu {
namespace {

// Security Parameter
constexpr int kKappa = 128;
// IKNP OT Extension Width
constexpr int kIknpWidth = kKkrtWidth * kKappa;
// TODO(shuyan.ycf): switch to 1024 when we have efficient 1024x128 transpose.
constexpr int kBatchSize = 128;
// How many blocks do we have.
constexpr int kNumBlockPerBatch = kBatchSize / kKappa;
static_assert(kBatchSize % kKappa == 0);

constexpr int kBatchSize1024 = 1024;
constexpr int kNumBlockPerBatch1024 = kBatchSize1024 / kKappa;
static_assert(kBatchSize1024 % kKappa == 0);

uint128_t KkrtRandomOracle(const KkrtRow& row) {
  // auto sha_bytes = crypto::Sha256(
  auto sha_bytes = crypto::Blake3(utils::ByteContainerView(
      reinterpret_cast<const char*>(&row), sizeof(row)));
  PPU_ENFORCE_GE(sha_bytes.size(), sizeof(uint128_t));
  uint128_t ret;
  std::memcpy(&ret, sha_bytes.data(), sizeof(ret));
  return ret;
}

inline void KkrtRandomOracle(const KkrtRow& row, uint8_t* buf,
                             uint64_t bufsize) {
  blake3_hasher hasher;

  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, reinterpret_cast<const char*>(&row),
                       sizeof(row));

  blake3_hasher_finalize(&hasher, buf, bufsize);
}

constexpr uint128_t kRandomOracleAesSeed =
    MakeUint128(0x2B7E151628AED2A6, 0xABF7158809CF4F3C);

inline void MultiKeyAesInit(MultiKeyAES<kKkrtWidth>* multi_key_aes) {
  std::array<block, kKkrtWidth> keys_block;

  PseudoRandomGenerator<uint128_t> prg(kRandomOracleAesSeed);

  for (uint64_t i = 0; i < kKkrtWidth; ++i) {
    keys_block[i] = block(prg());
  }

  (*multi_key_aes).SetKeys(absl::MakeSpan(keys_block));
}

inline void MultiKeyAesEncrypt(const MultiKeyAES<kKkrtWidth>& multi_key_aes,
                               uint128_t input, KkrtRow* prc) {
  std::array<block, kKkrtWidth> input_block, cipher_block;
  for (size_t i = 0; i < kKkrtWidth; i++) {
    input_block[i] = block(input);
  }

  multi_key_aes.EcbEncNBlocks(input_block.data(), cipher_block.data());
  for (size_t i = 0; i < kKkrtWidth; i++) {
    (*prc)[i] = (uint128_t)(cipher_block[i].mData);
  }
}

class KkrtGroupPRF : public IGroupPRF {
 public:
  explicit KkrtGroupPRF(size_t n, const KkrtRow& s)
      : size_(n), q_(n, {0}), s_(s) {
    MultiKeyAesInit(&multi_key_aes_);
  }

  size_t Size() const override { return size_; }

  uint128_t Eval(size_t group_idx, uint128_t input) const override {
    // According to KKRT paper, the final PRF output should be:
    //   H(q ^ (c(r) & s))
    PPU_ENFORCE_LT(group_idx, size_);
    // KkrtRow prc = RandomOracle::GetDefault().Gen<kKkrtWidth>(input);
    KkrtRow prc;
    MultiKeyAesEncrypt(multi_key_aes_, input, &prc);
    const auto& q = q_[group_idx];

    for (size_t w = 0; w < kKkrtWidth; ++w) {
      prc[w] &= s_[w];
      prc[w] ^= q[w];
    }
    return KkrtRandomOracle(prc);
  }

  void Eval(size_t group_idx, uint128_t input, uint8_t* outbuf,
            size_t bufsize) const override {
    // According to KKRT paper, the final PRF output should be:
    //   H(q ^ (c(r) & s))
    PPU_ENFORCE_LT(group_idx, size_);
    // KkrtRow prc = RandomOracle::GetDefault().Gen<kKkrtWidth>(input);
    KkrtRow prc;
    MultiKeyAesEncrypt(multi_key_aes_, input, &prc);
    const auto& q = q_[group_idx];

    for (size_t w = 0; w < kKkrtWidth; ++w) {
      prc[w] &= s_[w];
      prc[w] ^= q[w];
    }

    KkrtRandomOracle(prc, outbuf, bufsize);
  }

  template <size_t N>
  void SetQ(const std::array<KkrtRow, N>& q, size_t offset, size_t num_valid) {
    PPU_ENFORCE(num_valid <= q.size() && offset + num_valid <= this->Size());
    for (size_t i = 0; i < num_valid; ++i) {
      q_[offset + i] = q[i];
    }
  }

  template <size_t N>
  void CalcQ(const std::array<KkrtRow, N>& u, size_t offset, size_t num_valid) {
    PPU_ENFORCE(num_valid <= u.size() && offset + num_valid <= this->Size());
    std::array<KkrtRow, N> t;
    for (size_t i = 0; i < num_valid; ++i) {
      for (size_t w = 0; w < kKkrtWidth; ++w) {
        t[i][w] = u[i][w] & s_[w];
        q_[offset + i][w] ^= t[i][w];
      }
    }
  }

  void CalcQ(const std::vector<KkrtRow>& u, size_t offset, size_t num_valid) {
    PPU_ENFORCE(num_valid <= u.size() && offset + num_valid <= this->Size());
    std::vector<KkrtRow> t;
    t.resize(num_valid);
    for (size_t i = 0; i < num_valid; ++i) {
      for (size_t w = 0; w < kKkrtWidth; ++w) {
        t[i][w] = u[i][w] & s_[w];
        q_[offset + i][w] ^= t[i][w];
      }
    }
  }

 private:
  // Group size.
  const size_t size_;
  // Q, received from receiver.
  std::vector<KkrtRow> q_;
  // Sender base ot choice bits: `s`
  KkrtRow s_;

  MultiKeyAES<kKkrtWidth> multi_key_aes_;
};

}  // namespace

std::unique_ptr<IGroupPRF> KkrtOtExtSend(
    const std::shared_ptr<link::Context>& ctx,
    const BaseRecvOptions& base_options, size_t num_ot) {
  PPU_ENFORCE_EQ(base_options.blocks.size(), base_options.choices.size());
  PPU_ENFORCE(kIknpWidth == base_options.choices.size());
  PPU_ENFORCE(num_ot > 0);

  // Build S for sender.
  KkrtRow S{0};
  for (size_t w = 0; w < kKkrtWidth; ++w) {
    for (size_t k = 0; k < kKappa; ++k) {
      S[w] |= uint128_t(base_options.choices[w * kKappa + k] ? 1 : 0) << k;
    }
  }
  // Build PRG from seed Ks.
  std::vector<PseudoRandomGenerator<uint128_t>> prgs;
  for (size_t k = 0; k < kIknpWidth; ++k) {
    prgs.emplace_back(base_options.blocks[k]);
  }

  // Build PRF.
  auto prf = std::make_unique<KkrtGroupPRF>(num_ot, S);

  const size_t num_batch = (num_ot + kBatchSize - 1) / kBatchSize;
  for (size_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    const size_t num_this_batch =
        std::min<size_t>(num_ot - batch_idx * kBatchSize, kBatchSize);
    std::array<KkrtRow, kBatchSize> Q;
    std::array<KkrtRow, kBatchSize> U;
    for (size_t w = 0; w < kKkrtWidth; ++w) {
      std::array<uint128_t, kBatchSize> q;
      for (size_t k = 0; k < kKappa; ++k) {
        const size_t col_idx = w * kKappa + k;
        for (size_t b = 0; b < kNumBlockPerBatch; ++b) {
          q[k * kNumBlockPerBatch + b] = prgs[col_idx]();
        }
      }
      NaiveTranspose(&q);
      // SseTranspose128(&q);
      for (size_t i = 0; i < num_this_batch; ++i) {
        // Q = G(ks)
        Q[i][w] = q[i];
      }
    }

    // Receive U.
    auto buf = ctx->Recv(ctx->NextRank(), fmt::format("KKRT:{}", batch_idx));
    PPU_ENFORCE_EQ(buf.size(), static_cast<int64_t>(sizeof(U)));
    std::memcpy(U.data(), buf.data(), sizeof(U));

    // Build Q = (U & S) ^ G(ks)
    for (size_t i = 0; i < num_this_batch; ++i) {
      for (size_t w = 0; w < kKkrtWidth; ++w) {
        U[i][w] &= S[w];
        Q[i][w] ^= U[i][w];
      }
    }

    // Set to PRF.
    prf->SetQ(Q, batch_idx * kBatchSize, num_this_batch);
  }

  return prf;
}

void KkrtOtExtRecv(const std::shared_ptr<link::Context>& ctx,
                   const BaseSendOptions& base_options,
                   absl::Span<const uint128_t> inputs,
                   absl::Span<uint128_t> recv_blocks) {
  PPU_ENFORCE(base_options.blocks.size() == kIknpWidth);
  PPU_ENFORCE(inputs.size() == recv_blocks.size() && !inputs.empty());

  const size_t num_ot = inputs.size();
  const size_t num_batch = (num_ot + kBatchSize - 1) / kBatchSize;

  std::vector<PseudoRandomGenerator<uint128_t>> prgs0;
  std::vector<PseudoRandomGenerator<uint128_t>> prgs1;
  for (size_t k = 0; k < kIknpWidth; ++k) {
    // Build PRG from seed K0.
    prgs0.emplace_back(base_options.blocks[k][0]);
    // Build PRG from seed K1.
    prgs1.emplace_back(base_options.blocks[k][1]);
  }
  MultiKeyAES<kKkrtWidth> multi_key_aes_;
  MultiKeyAesInit(&multi_key_aes_);

  // Let us do it streaming way.
  for (size_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    const size_t num_this_batch =
        std::min<size_t>(num_ot - batch_idx * kBatchSize, kBatchSize);
    // KKRT can be viewed as a wider IKNP OT EXTENSION.
    std::array<KkrtRow, kBatchSize> T;
    std::array<KkrtRow, kBatchSize> U;
    for (size_t w = 0; w < kKkrtWidth; ++w) {
      std::array<uint128_t, kBatchSize> t;
      std::array<uint128_t, kBatchSize> u;
      for (size_t k = 0; k < kKappa; ++k) {
        const size_t col_idx = w * kKappa + k;
        for (size_t b = 0; b < kNumBlockPerBatch; ++b) {
          t[k * kNumBlockPerBatch + b] = prgs0[col_idx]();
          u[k * kNumBlockPerBatch + b] = prgs1[col_idx]();
        }
      }
      NaiveTranspose(&t);
      NaiveTranspose(&u);
      // SseTranspose128(&t);
      // SseTranspose128(&u);
      for (size_t i = 0; i < num_this_batch; ++i) {
        // T = G(k0)
        T[i][w] = t[i];
        // U = G(k1)
        U[i][w] = u[i];
      }
    }
    // Construct U.
    // U = G(k1) ^ G(k0) ^ PRC(r)
    for (size_t i = 0; i < num_this_batch; ++i) {
      // KkrtRow prc = RandomOracle::GetDefault().Gen<kKkrtWidth>(
      //    inputs[batch_idx * kBatchSize + i]);
      KkrtRow prc;
      MultiKeyAesEncrypt(multi_key_aes_, inputs[batch_idx * kBatchSize + i],
                         &prc);
      for (size_t w = 0; w < kKkrtWidth; ++w) {
        U[i][w] ^= T[i][w];
        U[i][w] ^= prc[w];
      }
    }
    // TODO(shuyan.ycf): link should support byte_container_view
    ctx->SendAsync(ctx->NextRank(), Buffer(U.data(), sizeof(U)),
                   fmt::format("KKRT:{}", batch_idx));
    for (size_t i = 0; i < num_this_batch; ++i) {
      // TODO(shuyan.ycf): make correlation break RO plugable. BTW: libOTe use
      // blake2 and takes 128 bits.
      // It is enough to just take first 128 bits of sha256 results for PSI now.
      recv_blocks[batch_idx * kBatchSize + i] = KkrtRandomOracle(T[i]);
    }
  }
}

void KkrtOtExtSender::Init(const std::shared_ptr<link::Context>& ctx,
                           const BaseRecvOptions& base_options,
                           uint64_t num_ot) {
  PPU_ENFORCE_EQ(base_options.blocks.size(), base_options.choices.size());
  PPU_ENFORCE(kIknpWidth == base_options.choices.size());
  PPU_ENFORCE(num_ot > 0);

  correction_idx_ = 0;

  // Build S for sender.
  KkrtRow S{0};
  for (size_t w = 0; w < kKkrtWidth; ++w) {
    for (size_t k = 0; k < kKappa; ++k) {
      S[w] |= uint128_t(base_options.choices[w * kKappa + k] ? 1 : 0) << k;
    }
  }
  // Build PRG from seed Ks.
  std::vector<AES> aes_gens;
  std::vector<uint64_t> gens_blk_idx;
  aes_gens.resize(base_options.choices.size());
  gens_blk_idx.resize(base_options.choices.size(), 0);
  for (uint64_t i = 0; i < uint64_t(base_options.blocks.size()); i++) {
    aes_gens[i].SetKey(base_options.blocks[i]);
  }

  // Build PRF.
  auto kkrt_oprf = std::make_shared<KkrtGroupPRF>(num_ot, S);
  oprf_ = kkrt_oprf;

  const size_t num_batch = (num_ot + kBatchSize1024 - 1) / kBatchSize1024;
  for (size_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    const size_t num_this_batch =
        std::min<size_t>(num_ot - batch_idx * kBatchSize1024, kBatchSize1024);
    std::array<KkrtRow, kBatchSize1024> Q;
    for (size_t w = 0; w < kKkrtWidth; ++w) {
      std::array<std::array<block, kNumBlockPerBatch1024>, kKappa> q;
      for (size_t k = 0; k < kKappa; ++k) {
        const size_t col_idx = w * kKappa + k;

        aes_gens[col_idx].EcbEncCounterMode(gens_blk_idx[col_idx],
                                            kNumBlockPerBatch1024,
                                            ((block*)q[k].data()));
        gens_blk_idx[col_idx] += kNumBlockPerBatch1024;
      }
      SseTranspose128x1024(q);

      for (size_t i = 0; i < kNumBlockPerBatch1024; ++i) {
        size_t q_idx = i * kKappa;
        size_t q_batch_num = std::min((size_t)kKappa, num_this_batch - q_idx);

        for (size_t j = 0; j < q_batch_num; ++j) {
          Q[q_idx + j][w] = (uint128_t)(q[j][i].mData);
        }
        if (q_batch_num < kKappa) {
          break;
        }
      }
    }

    // Set to PRF.
    kkrt_oprf->SetQ(Q, batch_idx * kBatchSize1024, num_this_batch);
  }
}

void KkrtOtExtSender::RecvCorrection(const std::shared_ptr<link::Context>& ctx,
                                     uint64_t recv_count) {
  std::vector<KkrtRow> U;

  U.resize(recv_count);
  // Receive U.
  auto buf = ctx->Recv(ctx->NextRank(), fmt::format("KKRT:{}", recv_count));

  PPU_ENFORCE_EQ(buf.size(), static_cast<int64_t>(U.size() * sizeof(KkrtRow)));
  std::memcpy(U.data(), buf.data(), U.size() * sizeof(KkrtRow));

  std::shared_ptr<KkrtGroupPRF> kkrtOprf =
      std::dynamic_pointer_cast<KkrtGroupPRF>(oprf_);
  kkrtOprf->CalcQ(U, correction_idx_, recv_count);
  correction_idx_ += recv_count;
}

void KkrtOtExtSender::SetCorrection(const std::string& recvceived_correction,
                                    uint64_t recv_count) {
  std::vector<KkrtRow> U;

  U.resize(recv_count);
  // set U.
  PPU_ENFORCE_EQ(recvceived_correction.size(), U.size() * sizeof(KkrtRow));
  std::memcpy(U.data(), recvceived_correction.data(),
              U.size() * sizeof(KkrtRow));

  std::shared_ptr<KkrtGroupPRF> kkrtOprf =
      std::dynamic_pointer_cast<KkrtGroupPRF>(oprf_);
  kkrtOprf->CalcQ(U, correction_idx_, recv_count);
  correction_idx_ += recv_count;
}

void KkrtOtExtSender::Encode(uint64_t ot_idx, const uint128_t input, void* dest,
                             uint64_t dest_size) {
  oprf_->Eval(ot_idx, input, (uint8_t*)dest, dest_size);
}

void KkrtOtExtReceiver::Init(const std::shared_ptr<link::Context>& ctx,
                             const BaseSendOptions& base_options,
                             uint64_t num_ot) {
  const size_t num_batch = (num_ot + kBatchSize1024 - 1) / kBatchSize1024;

  MultiKeyAesInit(&multi_key_aes_);

  std::vector<std::array<AES, 2>> aes_gens;
  std::vector<uint64_t> gens_blk_idx;
  aes_gens.resize(kIknpWidth);
  for (size_t k = 0; k < kIknpWidth; ++k) {
    aes_gens[k][0].SetKey(base_options.blocks[k][0]);
    aes_gens[k][1].SetKey(base_options.blocks[k][1]);
  }
  gens_blk_idx.resize(base_options.blocks.size(), 0);

  T_.resize(num_ot);
  U_.resize(num_ot);
  correction_idx_ = 0;

  // Let us do it streaming way.
  for (size_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    const size_t num_this_batch =
        std::min<size_t>(num_ot - batch_idx * kBatchSize1024, kBatchSize1024);
    // KKRT can be viewed as a wider IKNP OT EXTENSION.
    for (size_t w = 0; w < kKkrtWidth; ++w) {
      std::array<std::array<block, kNumBlockPerBatch1024>, kKappa> t;
      std::array<std::array<block, kNumBlockPerBatch1024>, kKappa> u;
      for (size_t k = 0; k < kKappa; ++k) {
        const size_t col_idx = w * kKappa + k;
        aes_gens[col_idx][0].EcbEncCounterMode(gens_blk_idx[col_idx],
                                               kNumBlockPerBatch1024,
                                               (block*)(t[k].data()));
        aes_gens[col_idx][1].EcbEncCounterMode(gens_blk_idx[col_idx],
                                               kNumBlockPerBatch1024,
                                               (block*)(u[k].data()));
        gens_blk_idx[col_idx] += kNumBlockPerBatch1024;
      }
      SseTranspose128x1024(t);
      SseTranspose128x1024(u);

      size_t batch_start = batch_idx * kBatchSize1024;
      for (size_t i = 0; i < kNumBlockPerBatch1024; ++i) {
        size_t tu_idx = i * kKappa;
        size_t tu_batch_num = std::min((size_t)kKappa, num_this_batch - tu_idx);

        for (size_t j = 0; j < tu_batch_num; ++j) {
          // T = G(k0)
          T_[batch_start + tu_idx + j][w] = (uint128_t)t[j][i].mData;
          // U = G(k1)
          U_[batch_start + tu_idx + j][w] = (uint128_t)u[j][i].mData;
        }
        if (tu_batch_num < kKappa) {
          break;
        }
      }
    }
  }
}

void KkrtOtExtReceiver::Encode(uint64_t ot_idx,
                               absl::Span<const uint128_t> inputs,
                               absl::Span<uint8_t> dest_encode) {
  PPU_ENFORCE(dest_encode.size() <= sizeof(uint128_t));
  // KkrtRow prc = RandomOracle::GetDefault().Gen<kKkrtWidth>(inputs[ot_idx]);
  KkrtRow prc;
  MultiKeyAesEncrypt(multi_key_aes_, inputs[ot_idx], &prc);

  for (size_t w = 0; w < kKkrtWidth; ++w) {
    U_[ot_idx][w] ^= T_[ot_idx][w];
    U_[ot_idx][w] ^= prc[w];
  }

  KkrtRandomOracle(T_[ot_idx], dest_encode.data(),
                   std::min(dest_encode.size(), sizeof(uint128_t)));
}

void KkrtOtExtReceiver::Encode(uint64_t ot_idx, const uint128_t input,
                               absl::Span<uint8_t> dest_encode) {
  PPU_ENFORCE(dest_encode.size() <= sizeof(uint128_t));
  // KkrtRow prc = RandomOracle::GetDefault().Gen<kKkrtWidth>(input);
  KkrtRow prc;
  MultiKeyAesEncrypt(multi_key_aes_, input, &prc);

  for (size_t w = 0; w < kKkrtWidth; ++w) {
    U_[ot_idx][w] ^= T_[ot_idx][w];
    U_[ot_idx][w] ^= prc[w];
  }

  KkrtRandomOracle(T_[ot_idx], dest_encode.data(),
                   std::min(dest_encode.size(), sizeof(uint128_t)));
}

void KkrtOtExtReceiver::ZeroEncode(uint64_t ot_idx) {
  for (size_t w = 0; w < kKkrtWidth; ++w) {
    U_[ot_idx][w] ^= T_[ot_idx][w];
  }
}

void KkrtOtExtReceiver::SendCorrection(
    const std::shared_ptr<link::Context>& ctx, uint64_t send_count) {
  ctx->SendAsync(ctx->NextRank(),
                 Buffer(reinterpret_cast<const char*>(U_.data()) +
                            (correction_idx_ * sizeof(KkrtRow)),
                        send_count * sizeof(KkrtRow)),
                 fmt::format("KKRT:{}", send_count));
  correction_idx_ += send_count;
}

std::string KkrtOtExtReceiver::ShiftCorrection(uint64_t send_count) {
  std::string correction =
      std::string(reinterpret_cast<const char*>(U_.data()) +
                      (correction_idx_ * sizeof(KkrtRow)),
                  send_count * sizeof(KkrtRow));
  correction_idx_ += send_count;
  return correction;
}

}  // namespace ppu
