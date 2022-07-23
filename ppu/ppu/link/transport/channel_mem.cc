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



#include "ppu/link/transport/channel_mem.h"

#include "ppu/utils/exception.h"

namespace ppu::link {

ChannelMem::ChannelMem(size_t self_rank, size_t peer_rank, size_t timeout_ms)
    : ChannelBase(self_rank, peer_rank, timeout_ms) {}

void ChannelMem::SetPeer(const std::shared_ptr<ChannelMem>& peer_channel) {
  peer_channel_ = peer_channel;
}

void ChannelMem::SendAsync(const std::string& key, const Buffer& value) {
  if (auto ptr = peer_channel_.lock()) {
    ptr->OnMessage(key, value);
  } else {
    PPU_THROW_IO_ERROR("Peer's memory channel released");
  }
}

void ChannelMem::Send(const std::string& key, const Buffer& value) {
  return SendAsync(key, value);
}

}  // namespace ppu::link
