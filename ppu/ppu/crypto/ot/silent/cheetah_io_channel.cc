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


#include "cheetah_io_channel.h"

#include "utils.h"

#include "ppu/core/buffer.h"
#include "ppu/link/link.h"

using std::shared_ptr;
using std::string;

namespace ppu {

CheetahIo::CheetahIo(shared_ptr<link::Context> ctx)
    : ctx_(ctx),
      send_op_(0),
      recv_op_(0),
      send_buffer_used_(0),
      recv_buffer_used_(0) {
  send_buffer_ = new uint8_t[SEND_BUFFER_SIZE];
}

CheetahIo::~CheetahIo() {
  flush();
  delete[] send_buffer_;
}

void CheetahIo::flush() {
  if (send_buffer_used_ == 0) return;

  ctx_->Send(ctx_->NextRank(), Buffer(send_buffer_, send_buffer_used_),
             fmt::format("Cheetah send:{}", send_op_++));

  memset(send_buffer_, 0, SEND_BUFFER_SIZE);
  send_buffer_used_ = 0;
}

void CheetahIo::fill_recv() {
  recv_buffer_ =
      ctx_->Recv(ctx_->NextRank(), fmt::format("Cheetah recv:{}", recv_op_++));
  recv_buffer_used_ = 0;
}

void CheetahIo::send_data_internal(const void *data, int len) {
  size_t send_buffer_left = SEND_BUFFER_SIZE - send_buffer_used_;
  if (send_buffer_left <= (size_t)len) {
    memcpy(send_buffer_ + send_buffer_used_, data, send_buffer_left);
    send_buffer_used_ += send_buffer_left;
    flush();

    send_data_internal(((char *)data) + send_buffer_left,
                       len - send_buffer_left);
  } else {
    memcpy(send_buffer_ + send_buffer_used_, data, len);
    send_buffer_used_ += len;
  }
}

void CheetahIo::recv_data_internal(void *data, int len) {
  if (send_buffer_used_ > 0) flush();

  size_t recv_buffer_left = recv_buffer_.size() - recv_buffer_used_;
  if (recv_buffer_left >= (size_t)len) {
    memcpy(data, recv_buffer_.data<uint8_t>() + recv_buffer_used_, len);
    recv_buffer_used_ += len;
  } else {
    memcpy(data, recv_buffer_.data<uint8_t>() + recv_buffer_used_,
           recv_buffer_left);
    fill_recv();

    recv_data_internal(((char *)data) + recv_buffer_left,
                       len - recv_buffer_left);
  }
}

template <typename T>
void CheetahIo::send_data_partial(const T *data, int len, int bitlength) {
  if (bitlength == sizeof(T) * 8) {
    send_data_internal((const void *)data, len * sizeof(T));
    return;
  }

  int compact_len = (bitlength + 7) / 8;
  uint8_t *bytes = new uint8_t[len];
  for (int i = 0; i < compact_len; i++) {
    for (int j = 0; j < len; j++) {
      bytes[j] = uint8_t(data[j] >> (i * 8));
    }
    send_data_internal(bytes, len);
  }

  delete[] bytes;
}

template <typename T>
void CheetahIo::recv_data_partial(T *data, int len, int bitlength) {
  if (bitlength == sizeof(T) * 8) {
    recv_data_internal((void *)data, len * sizeof(T));
    return;
  }
  memset(data, 0, len * sizeof(T));

  int compact_len = (bitlength + 7) / 8;
  uint8_t *bytes = new uint8_t[len];
  for (int i = 0; i < compact_len; i++) {
    recv_data_internal(bytes, len);
    for (int j = 0; j < len; j++) {
      data[j] |= T(bytes[j]) << (i * 8);
    }
  }
  T mask = (T(1) << bitlength) - 1;
  for (int i = 0; i < len; i++) data[i] &= mask;

  delete[] bytes;
}

template void CheetahIo::send_data_partial<uint32_t>(const uint32_t *data,
                                                     int len, int bitlength);
template void CheetahIo::send_data_partial<uint64_t>(const uint64_t *data,
                                                     int len, int bitlength);
template void CheetahIo::send_data_partial<uint128_t>(const uint128_t *data,
                                                      int len, int bitlength);

template void CheetahIo::recv_data_partial<uint32_t>(uint32_t *data, int len,
                                                     int bitlength);
template void CheetahIo::recv_data_partial<uint64_t>(uint64_t *data, int len,
                                                     int bitlength);
template void CheetahIo::recv_data_partial<uint128_t>(uint128_t *data, int len,
                                                      int bitlength);

}  // namespace ppu
