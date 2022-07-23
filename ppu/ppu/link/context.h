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

#include <atomic>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "ppu/core/buffer.h"
#include "ppu/link/transport/channel.h"

namespace ppu::link {

constexpr size_t kAllRank = std::numeric_limits<size_t>::max();
struct ContextDesc {
  struct Party {
    std::string id;
    std::string host;
  };

  // the UUID of this communication.
  std::string id = "root";

  // party description, describes the world.
  std::vector<Party> parties;

  // connect to mesh retry time.
  uint32_t connect_retry_times = 10;

  // connect to mesh retry interval.
  uint32_t connect_retry_interval_ms = 1000;  // 1 second.

  // recv timeout in milliseconds.
  //
  // 'recv time' is the max time that a party will wait for a given event.
  // for example:
  //
  //      begin recv                 end recv
  // |--------|-------recv-time----------|------------------| alice's timeline
  //
  //                         begin send     end send
  // |-----busy-work-------------|-------------|------------| bob's timeline
  //
  // in above case, when alice begins recv for a specific event, bob is still
  // busy doing its job, when alice's wait time exceed wait_timeout_ms, it raise
  // exception, although bob now is starting to send data.
  //
  // so for long time work(that one party may wait for the others for very long
  // time), this value should be changed accordingly.
  uint32_t recv_timeout_ms = 30 * 1000;  // 30s

  // http max payload size, if a single http request size is greater than this
  // limit, it will be unpacked into small chunks then reassembled.
  uint32_t http_max_payload_size = 32 * 1024;  // 32k byte

  // a single http request timetout.
  uint32_t http_timeout_ms = 20 * 1000;  // 20 seconds.

  // BRPC client channel protocol.
  std::string brpc_channel_protocol = "baidu_std";

  // BRPC client channel connection type.
  std::string brpc_channel_connection_type = "single";
};

struct Statistics {
  // total number of data sent in bytes, excluding key
  std::atomic<size_t> sent_bytes = 0u;

  // total number of sent actions, chuncked mode is treated as a single action.
  std::atomic<size_t> sent_actions = 0u;

  // total number of data received in bytes, excluding key.
  std::atomic<size_t> recv_bytes = 0u;

  // total number of recv actions, chuncked mode is treated as a single action.
  std::atomic<size_t> recv_actions = 0u;
};

// Threading: link context could only be used in one thread, since
// communication rounds are identified by (incremental) counters.
//
// Spawn it if you need to used use in a different thread, the
// channels/event_loop will be shared between parent/child contexts.
class Context {
 public:
  Context(ContextDesc desc, size_t rank,
          std::vector<std::shared_ptr<IChannel>> channels,
          std::shared_ptr<IReceiverLoop> msg_loop);

  std::string Id() const;

  size_t WorldSize() const;

  size_t Rank() const;

  size_t NextRank(size_t stride = 1) const;

  size_t PrevRank(size_t stride = 1) const;

  // P2P algorithms
  void SendAsync(size_t dst_rank, const Buffer& value, std::string_view tag);

  void Send(size_t dst_rank, const Buffer& value, std::string_view tag);

  Buffer Recv(size_t src_rank, std::string_view tag);

  void ConnectToMesh();

  std::unique_ptr<Context> Spawn();

  void SetRecvTimeout(uint32_t recv_timeout_ms);

  uint32_t GetRecvTimeout() const;

 public:
  // for internal algorithms.
  void SendAsyncInternal(size_t dst_rank, const std::string& key,
                         const Buffer& value);
  void SendInternal(size_t dst_rank, const std::string& key,
                    const Buffer& value);
  Buffer RecvInternal(size_t src_rank, const std::string& key);

  // next collective algorithm id.
  std::string NextId();

  // next P2P comm id.
  std::string NextP2PId(size_t src_rank, size_t dst_rank);

  // for external message loop
  std::shared_ptr<IChannel> GetChannel(size_t src_rank) const;

 protected:
  using P2PDirection = std::pair<int, int>;

  const ContextDesc desc_;  // world description.
  const size_t rank_;       // my rank.
  const std::vector<std::shared_ptr<IChannel>> channels_;
  const std::shared_ptr<IReceiverLoop> receiver_loop_;

  // stateful properties.
  size_t counter_ = 0u;  // collective algorithm counter.
  std::map<P2PDirection, int> p2p_counter_;

  size_t child_counter_ = 0u;

  uint32_t recv_timeout_ms_;

  // sub-context will shared statistics with parent
  std::shared_ptr<Statistics> stats_;
};

// a RecvTimeoutGuard is to help set the recv timeout value for the Context.
// for example:
// {
//  RecvTimeoutGuard guard(ctx, timeout);
//  method();
// }
// in above case, the Context's recv_timeout_ms_ is set to timout before the
// method and recovers to its original value automatically after the
// method finishes.
class RecvTimeoutGuard {
 public:
  // set recv timeout and save original value
  RecvTimeoutGuard(const std::shared_ptr<Context>& ctx,
                   uint32_t recv_timeout_ms)
      : ctx_(ctx), recv_timeout_ms_(ctx->GetRecvTimeout()) {
    ctx->SetRecvTimeout(recv_timeout_ms);
  }
  // recover original timeout value
  ~RecvTimeoutGuard() { ctx_->SetRecvTimeout(recv_timeout_ms_); }

  RecvTimeoutGuard(const RecvTimeoutGuard&) = delete;
  RecvTimeoutGuard& operator=(const RecvTimeoutGuard&) = delete;

 private:
  const std::shared_ptr<Context>& ctx_;
  uint32_t recv_timeout_ms_;
};

}  // namespace ppu::link
