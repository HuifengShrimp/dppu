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



#include "ppu/link/context.h"

#include <chrono>
#include <thread>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "ppu/link/algorithm/trace.h"
#include "ppu/utils/exception.h"

namespace ppu::link {

Context::Context(ContextDesc desc, size_t rank,
                 std::vector<std::shared_ptr<IChannel>> channels,
                 std::shared_ptr<IReceiverLoop> msg_loop)
    : desc_(std::move(desc)),
      rank_(rank),
      channels_(std::move(channels)),
      receiver_loop_(std::move(msg_loop)),
      recv_timeout_ms_(desc_.recv_timeout_ms) {
  const size_t world_size = desc_.parties.size();

  PPU_ENFORCE(rank_ < static_cast<size_t>(world_size),
              "rank={} out of range world_size={}", rank, world_size);
  PPU_ENFORCE(channels_.size() == world_size,
              "channels lenth={} does not match world_size={}",
              channels_.size(), world_size);

  for (size_t src = 0; src < world_size; ++src) {
    for (size_t dst = 0; dst < world_size; ++dst) {
      p2p_counter_[std::make_pair(src, dst)] = 0u;
    }
  }

  stats_ = std::make_shared<Statistics>();
}

std::string Context::Id() const { return desc_.id; }

size_t Context::WorldSize() const { return desc_.parties.size(); }

size_t Context::Rank() const { return rank_; }

size_t Context::PrevRank(size_t stride) const {
  return (rank_ - stride + WorldSize()) % WorldSize();
}

size_t Context::NextRank(size_t stride) const {
  return (rank_ + stride) % WorldSize();
}

void Context::ConnectToMesh() {
  const std::string event = fmt::format("connect_{}", Rank());

  SPDLOG_INFO("connecting to mesh, id={}, self={}", Id(), Rank());

  auto try_connect = [&](size_t rank, const std::string& event,
                         size_t /*attempt*/) {
    try {
      SendInternal(rank, event, {});
    } catch (const NetworkError& e) {
      SPDLOG_DEBUG("attempt={} to connect to rank={} error={}", attempt, rank,
                   e.what());
      return false;
    }
    return true;
  };

  // broadcast to all
  for (size_t idx = 0; idx < WorldSize(); idx++) {
    if (idx == Rank()) {
      continue;
    }

    bool succeed = false;
    for (size_t attempt = 0; attempt < desc_.connect_retry_times + 1;
         attempt++) {
      if (attempt != 0) {
        // sleep and retry.
        SPDLOG_INFO(
            "try_connect to rank {} not succeed, sleep_for {}ms and retry.",
            idx, desc_.connect_retry_interval_ms);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(desc_.connect_retry_interval_ms));
      }
      if (try_connect(idx, event, attempt)) {
        succeed = true;
        break;
      }
    }

    if (!succeed) {
      PPU_THROW("connect to mesh failed, failed to setup connection to rank={}",
                idx);
    }
  }
  SPDLOG_DEBUG("connecting to mesh, all partners launched");

  // gather all
  for (size_t idx = 0; idx < WorldSize(); idx++) {
    if (idx == Rank()) {
      continue;
    }

    std::string key = fmt::format("connect_{}", idx);
    (void)RecvInternal(idx, key);  // ignore the received value.
  }
  SPDLOG_INFO("connected to mesh, id={}, self={}", Id(), Rank());
}

// P2P algorithms
void Context::SendAsync(size_t dst_rank, const Buffer& value,
                        std::string_view tag) {
  const auto event = NextP2PId(rank_, dst_rank);

  TraceLog(event, tag, "");

  SendAsyncInternal(dst_rank, event, value);
}

void Context::Send(size_t dst_rank, const Buffer& value, std::string_view tag) {
  const auto event = NextP2PId(rank_, dst_rank);

  TraceLog(event, tag, "");

  SendInternal(dst_rank, event, value);
}

Buffer Context::Recv(size_t src_rank, std::string_view tag) {
  const auto event = NextP2PId(src_rank, rank_);

  TraceLog(event, tag, "");

  return RecvInternal(src_rank, event);
}

void Context::SendAsyncInternal(size_t dst_rank, const std::string& key,
                                const Buffer& value) {
  PPU_ENFORCE(dst_rank < static_cast<size_t>(channels_.size()),
              "rank={} out of range={}", dst_rank, channels_.size());

  channels_[dst_rank]->SendAsync(key, value);

  stats_->sent_actions++;
  stats_->sent_bytes += value.size();
}

void Context::SendInternal(size_t dst_rank, const std::string& key,
                           const Buffer& value) {
  PPU_ENFORCE(dst_rank < static_cast<size_t>(channels_.size()),
              "rank={} out of range={}", dst_rank, channels_.size());

  channels_[dst_rank]->Send(key, value);

  stats_->sent_actions++;
  stats_->sent_bytes += value.size();
}

Buffer Context::RecvInternal(size_t src_rank, const std::string& key) {
  PPU_ENFORCE(src_rank < static_cast<size_t>(channels_.size()),
              "rank={} out of range={}", src_rank, channels_.size());

  auto value = channels_[src_rank]->Recv(key);

  stats_->recv_actions++;
  stats_->recv_bytes += value.size();

  return value;
}

std::unique_ptr<Context> Context::Spawn() {
  ContextDesc sub_desc = desc_;
  sub_desc.id = fmt::format("{}-{}", desc_.id, child_counter_++);

  // sub-context share the same event-loop and statistics with parent.
  auto sub_ctx =
      std::make_unique<Context>(sub_desc, rank_, channels_, receiver_loop_);

  // share statistics with parent.
  sub_ctx->stats_ = this->stats_;

  return sub_ctx;
}

std::string Context::NextId() {
  return fmt::format("{}:{}", desc_.id, ++counter_);
}

std::string Context::NextP2PId(size_t src_rank, size_t dst_rank) {
  return fmt::format("{}:P2P-{}:{}->{}", desc_.id,
                     ++p2p_counter_[std::make_pair(src_rank, dst_rank)],
                     src_rank, dst_rank);
}

std::shared_ptr<IChannel> Context::GetChannel(size_t src_rank) const {
  PPU_ENFORCE(src_rank < WorldSize(), "unexpected rank={} with world_size={}",
              src_rank, WorldSize());
  return channels_[src_rank];
}

void Context::SetRecvTimeout(uint32_t recv_timeout_ms) {
  recv_timeout_ms_ = recv_timeout_ms;
  for (size_t idx = 0; idx < WorldSize(); idx++) {
    if (idx == Rank()) {
      continue;
    }
    channels_[idx]->SetRecvTimeout(recv_timeout_ms_);
  }
  SPDLOG_INFO("set recv timeout, timeout_ms={}", recv_timeout_ms_);
}

uint32_t Context::GetRecvTimeout() const { return recv_timeout_ms_; }

}  // namespace ppu::link
