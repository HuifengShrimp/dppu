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

#include <future>

#include "fmt/format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ppu/link/factory.h"
#include "ppu/link/transport/channel_mem.h"
#include "ppu/utils/exception.h"

namespace ppu::link::test {

class MockChannel : public IChannel {
 public:
  MOCK_METHOD2(SendAsync, void(const std::string &key, const Buffer &value));
  MOCK_METHOD2(Send, void(const std::string &key, const Buffer &value));
  MOCK_METHOD1(Recv, Buffer(const std::string &key));
  MOCK_METHOD2(OnMessage, void(const std::string &key, const Buffer &value));
  MOCK_METHOD4(OnChunkedMessage,
               void(const std::string &key, const Buffer &value,
                    size_t chunk_idx, size_t num_chunks));
  void SetRecvTimeout(uint32_t timeout_ms) override { timeout_ = timeout_ms; }
  uint32_t GetRecvTimeout() const override { return timeout_; }

 private:
  std::uint32_t timeout_;
};

class ContextConnectToMeshTest : public ::testing::Test {
 public:
  void SetUp() override {
    world_size_ = 3;
    self_rank_ = 1;
    channels_.resize(world_size_);
    channels_[0] = std::make_shared<MockChannel>();
    channels_[2] = std::make_shared<MockChannel>();
  }

  std::vector<std::shared_ptr<IChannel>> channels_;
  size_t self_rank_;
  size_t world_size_;
};

TEST_F(ContextConnectToMeshTest, ConnectToMeshShouldOk) {
  // GIVEN
  auto msg_loop = std::make_shared<ReceiverLoopMem>();
  ContextDesc ctx_desc;
  ctx_desc.connect_retry_interval_ms = 100;
  for (size_t rank = 0; rank < world_size_; rank++) {
    const auto id = fmt::format("id-{}", rank);
    const auto host = fmt::format("host-{}", rank);
    ctx_desc.parties.push_back({id, host});
  }
  Context ctx(ctx_desc, self_rank_, channels_, msg_loop);

  // THEN
  std::string event = fmt::format("connect_{}", self_rank_);
  for (size_t i = 0; i < world_size_; ++i) {
    if (i == self_rank_) {
      continue;
    }
    EXPECT_CALL(*std::static_pointer_cast<MockChannel>(channels_[i]),
                Send(event, Buffer()));
    std::string key = fmt::format("connect_{}", i);
    EXPECT_CALL(*std::static_pointer_cast<MockChannel>(channels_[i]),
                Recv(key));
  }
  // WHEN
  ctx.ConnectToMesh();
}

ACTION(ThrowNetworkErrorException) { throw ::ppu::NetworkError(); }

TEST_F(ContextConnectToMeshTest, ThrowExceptionIfNetworkError) {
  // GIVEN
  auto msg_loop = std::make_shared<ReceiverLoopMem>();
  ContextDesc ctx_desc;
  ctx_desc.connect_retry_interval_ms = 100;
  for (size_t rank = 0; rank < world_size_; rank++) {
    const auto id = fmt::format("id-{}", rank);
    const auto host = fmt::format("host-{}", rank);
    ctx_desc.parties.push_back({id, host});
  }
  Context ctx(ctx_desc, self_rank_, channels_, msg_loop);

  std::string event = fmt::format("connect_{}", self_rank_);
  ON_CALL(*std::static_pointer_cast<MockChannel>(channels_[0]),
          Send(event, Buffer()))
      .WillByDefault(ThrowNetworkErrorException());

  // WHEN THEN
  EXPECT_THROW(ctx.ConnectToMesh(), ::ppu::RuntimeError);
}

TEST_F(ContextConnectToMeshTest, SetRecvTimeoutShouldOk) {
  // GIVEN
  auto msg_loop = std::make_shared<ReceiverLoopMem>();
  ContextDesc ctx_desc;
  ctx_desc.recv_timeout_ms = 4000;
  for (size_t rank = 0; rank < world_size_; rank++) {
    const auto id = fmt::format("id-{}", rank);
    const auto host = fmt::format("host-{}", rank);
    ctx_desc.parties.push_back({id, host});
  }
  auto ctx =
      std::make_shared<Context>(ctx_desc, self_rank_, channels_, msg_loop);
  EXPECT_EQ(ctx->GetRecvTimeout(), 4000);
  // WHEN THEN
  {
    RecvTimeoutGuard guard(ctx, 2000);
    EXPECT_EQ(ctx->GetRecvTimeout(), 2000);
  }

  // THEN
  EXPECT_EQ(ctx->GetRecvTimeout(), 4000);
}

class ContextTest : public ::testing::Test {
 public:
  virtual void SetUp() override {
    world_size_ = 3;

    ContextDesc ctx_desc;
    ctx_desc.recv_timeout_ms = 2000;  // 2 second
    for (size_t rank = 0; rank < world_size_; rank++) {
      const auto id = fmt::format("id-{}", rank);
      const auto host = fmt::format("host-{}", rank);
      ctx_desc.parties.push_back({id, host});
    }

    for (size_t rank = 0; rank < world_size_; rank++) {
      ctxs_.push_back(FactoryMem().CreateContext(ctx_desc, rank));
    }

    send_buffer_.resize(world_size_);
    receive_buffer.resize(world_size_);
    futures_.resize(world_size_);
    for (size_t sender = 0; sender < world_size_; ++sender) {
      send_buffer_[sender].resize(world_size_);
      receive_buffer[sender].resize(world_size_);
      futures_[sender].resize(world_size_);
    }
  }

  void join_all() {
    for (size_t i = 0; i < world_size_; ++i) {
      for (size_t j = 0; j < world_size_; ++j) {
        if (futures_[i][j].valid()) {
          futures_[i][j].get();
        }
      }
    }
  }

  std::vector<std::vector<Buffer>> send_buffer_;
  std::vector<std::vector<Buffer>> receive_buffer;
  std::vector<std::vector<std::future<void>>> futures_;
  std::vector<std::shared_ptr<Context>> ctxs_;
  size_t world_size_;
};

TEST_F(ContextTest, SendRecvShouldOk) {
  // GIVEN
  // build sent_values and received buffer
  for (size_t sender = 0; sender < world_size_; ++sender) {
    for (size_t receiver = 0; receiver < world_size_; ++receiver) {
      if (sender == receiver) {
        send_buffer_[sender][receiver].resize(5);
        receive_buffer[sender][receiver].resize(5);
        std::strcpy(send_buffer_[sender][receiver].data<char>(), "null");
        std::strcpy(receive_buffer[sender][receiver].data<char>(), "null");
        continue;
      }
      std::string data = fmt::format("{}->{}", sender, receiver);
      send_buffer_[sender][receiver].resize(data.size() + 1);
      std::strcpy(send_buffer_[sender][receiver].data<char>(), data.c_str());
    }
  }
  // WHEN
  auto recv_fn = [&](size_t receiver, size_t sender) {
    receive_buffer[sender][receiver] = ctxs_[receiver]->Recv(sender, "tag");
  };
  auto send_fn = [&](size_t sender, size_t receiver, const Buffer &value) {
    ctxs_[sender]->SendAsync(receiver, send_buffer_[sender][receiver], "tag");
  };
  for (size_t sender = 0; sender < world_size_; ++sender) {
    for (size_t receiver = 0; receiver < world_size_; ++receiver) {
      if (sender == receiver) {
        continue;
      }
      futures_[sender][receiver] = std::async(recv_fn, receiver, sender);
    }
  }
  for (size_t sender = 0; sender < world_size_; ++sender) {
    for (size_t receiver = 0; receiver < world_size_; ++receiver) {
      if (sender == receiver) {
        continue;
      }
      auto _ =
          std::async(send_fn, sender, receiver, send_buffer_[sender][receiver]);
    }
  }
  join_all();

  // THEN
  EXPECT_EQ(send_buffer_, receive_buffer);
}

}  // namespace ppu::link::test
