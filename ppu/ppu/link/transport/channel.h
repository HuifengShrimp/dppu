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

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "ppu/core/buffer.h"

namespace ppu::link {

// A channel is basic interface for p2p communicator.
class IChannel {
 public:
  virtual ~IChannel() = default;

  // SendAsync asynchronously.
  // return when the message successfully pushed into peer's recv buffer.
  virtual void SendAsync(const std::string& key, const Buffer& value) = 0;

  // SendAsync synchronously.
  // return when the message is successfully pushed into the send buffer.
  // raise when push buffer overflow.
  // Note: if the message failed when transfer, there is no acknowledge.
  virtual void Send(const std::string& key, const Buffer& value) = 0;

  // block waiting message.
  virtual Buffer Recv(const std::string& key) = 0;

  // called by an async dispatcher.
  virtual void OnMessage(const std::string& key, const Buffer& value) = 0;

  // called by an async dispatcher.
  virtual void OnChunkedMessage(const std::string& key, const Buffer& value,
                                size_t chunk_idx, size_t num_chunks) = 0;
  // set receive timeout ms
  virtual void SetRecvTimeout(uint32_t timeout_ms) = 0;

  // get receive timemout ms
  virtual uint32_t GetRecvTimeout() const = 0;
};

// forward declaractions.
class ChunkedMessage;

class ChannelBase : public IChannel {
 public:
  ChannelBase(size_t self_rank, size_t peer_rank)
      : self_rank_(self_rank), peer_rank_(peer_rank) {}

  ChannelBase(size_t self_rank, size_t peer_rank, size_t recv_timeout_ms)
      : self_rank_(self_rank),
        peer_rank_(peer_rank),
        recv_timeout_ms_(recv_timeout_ms) {}

  Buffer Recv(const std::string& key) override;

  void OnMessage(const std::string& key, const Buffer& value) override;

  void OnChunkedMessage(const std::string& key, const Buffer& value,
                        size_t chunk_idx, size_t num_chunks) override;

  void SetRecvTimeout(uint32_t recv_timeout_ms) override;

  uint32_t GetRecvTimeout() const override;

 protected:
  const size_t self_rank_;
  const size_t peer_rank_;

  uint32_t recv_timeout_ms_ = 3 * 60 * 1000;  // 3 minites

  // message database related.
  std::mutex msg_db_mutex_;
  std::condition_variable msg_db_cond_;
  std::map<std::string, Buffer> msg_db_;

  // chunking related.
  std::mutex chunked_values_mutex_;
  std::map<std::string, std::shared_ptr<ChunkedMessage>> chunked_values_;
};

// A receiver loop is a thread loop which receives messages from the world.
// It listens message from all over the world and delivers to listeners.
class IReceiverLoop {
 public:
  virtual ~IReceiverLoop() = default;

  //
  virtual void Stop() = 0;

  // add listener who interested messages from 'rank'
  virtual void AddListener(size_t rank, std::shared_ptr<IChannel> channel) = 0;
};

class ReceiverLoopBase : public IReceiverLoop {
 public:
  void AddListener(size_t rank, std::shared_ptr<IChannel> listener) override;

 protected:
  std::map<size_t, std::shared_ptr<IChannel>> listeners_;
};

}  // namespace ppu::link
