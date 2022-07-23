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



#include "ppu/link/transport/channel.h"

#include "ppu/utils/exception.h"

namespace ppu::link {

class ChunkedMessage {
 public:
  explicit ChunkedMessage(size_t num_chunks)
      : num_chunks_(num_chunks), message_size_(0) {}

  void AddChunk(size_t index, const Buffer& data) {
    std::unique_lock lock(mutex_);
    chunks_.emplace(index, data);
    message_size_ += data.size();
  }

  size_t NumChunks() const { return num_chunks_; }

  size_t NumFilled() const { return chunks_.size(); }

  bool IsFullyFilled() const { return chunks_.size() == num_chunks_; }

  Buffer Reassemble() {
    Buffer out(message_size_);
    size_t bytes_written = 0;
    for (auto& itr : chunks_) {
      std::memcpy(out.data<char>() + bytes_written, itr.second.data(),
                  itr.second.size());
      bytes_written += itr.second.size();
    }
    message_size_ = 0;
    chunks_.clear();
    return out;
  }

 protected:
  const size_t num_chunks_;

  std::mutex mutex_;
  // chunk index to value.
  std::map<size_t, Buffer> chunks_;
  size_t message_size_;
};

Buffer ChannelBase::Recv(const std::string& key) {
  Buffer value;

  std::unique_lock lock(msg_db_mutex_);
  const auto& duration = std::chrono::milliseconds(recv_timeout_ms_);
  if (!msg_db_cond_.wait_for(lock, duration, [&] {
        auto itr = this->msg_db_.find(key);
        if (itr == this->msg_db_.end()) {
          return false;
        }
        value = std::move(itr->second);
        return true;
      })) {
    PPU_THROW_IO_ERROR("Get data timeout, key={}", key);
  }
  msg_db_.erase(key);

  return value;
}

void ChannelBase::OnMessage(const std::string& key, const Buffer& value) {
  std::unique_lock lock(msg_db_mutex_);
  msg_db_.emplace(key, value);
  msg_db_cond_.notify_all();
}

void ChannelBase::OnChunkedMessage(const std::string& key, const Buffer& value,
                                   size_t chunk_idx, size_t num_chunks) {
  if (chunk_idx >= num_chunks) {
    PPU_THROW_LOGIC_ERROR("invalid chunk info, index={}, size={}", chunk_idx,
                          num_chunks);
  }

  std::shared_ptr<ChunkedMessage> data;
  {
    std::unique_lock lock(chunked_values_mutex_);
    auto itr = chunked_values_.find(key);
    if (itr == chunked_values_.end()) {
      itr = chunked_values_
                .emplace(key, std::make_shared<ChunkedMessage>(num_chunks))
                .first;
    }
    data = itr->second;
  }

  data->AddChunk(chunk_idx, value);
  {
    bool should_reassemble = false;
    if (data->IsFullyFilled()) {
      // two threads may arrive here at same time.
      std::unique_lock lock(chunked_values_mutex_);
      auto const& itr = chunked_values_.find(key);
      if (itr == chunked_values_.end()) {
        // this data block is handled by another chunk, just return.
        return;
      }

      chunked_values_.erase(key);

      // only one thread do the reassemble
      should_reassemble = true;
    }

    if (should_reassemble) {
      // notify new value arrived.
      auto reassembled_data = data->Reassemble();
      {
        std::unique_lock lock(msg_db_mutex_);
        msg_db_.emplace(key, std::move(reassembled_data));
        msg_db_cond_.notify_all();
      }
    }
  }
}

void ChannelBase::SetRecvTimeout(uint32_t recv_timeout_ms) {
  recv_timeout_ms_ = recv_timeout_ms;
}

uint32_t ChannelBase::GetRecvTimeout() const { return recv_timeout_ms_; }

void ReceiverLoopBase::AddListener(size_t rank,
                                   std::shared_ptr<IChannel> listener) {
  if (listeners_.find(rank) != listeners_.end()) {
    PPU_THROW_LOGIC_ERROR("duplicated listener for rank={}", rank);
  }
  listeners_[rank] = listener;
}

}  // namespace ppu::link
