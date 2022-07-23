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



#include "ppu/link/transport/channel_brpc.h"

#include "fmt/format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ppu/utils/exception.h"

// disable detect leaks for brpc's "acceptable mem leak"
// https://github.com/apache/incubator-brpc/blob/0.9.6/src/brpc/server.cpp#L1138
extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }

namespace ppu::link::test {

static std::string RandStr(size_t length) {
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

class ChannelBrpcTest : public ::testing::Test {
 protected:
  void SetUp() {
    const size_t send_rank = 0;
    const size_t recv_rank = 0;

    sender_ = std::make_shared<ChannelBrpc>(send_rank, recv_rank, options_);
    receiver_ = std::make_shared<ChannelBrpc>(recv_rank, send_rank, options_);

    // let sender rank as 0, receiver rank as 1.
    // receiver_ listen messages from sender(rank 0).
    receiver_loop_ = std::make_unique<ReceiverLoopBrpc>();
    receiver_loop_->AddListener(0, receiver_);
    receiver_host_ = receiver_loop_->Start("127.0.0.1:0");

    //
    sender_->SetPeerHost(receiver_host_);
  }

 protected:
  ChannelBrpc::Options options_;
  std::shared_ptr<ChannelBrpc> sender_;
  std::shared_ptr<ChannelBrpc> receiver_;
  std::string receiver_host_;
  std::unique_ptr<ReceiverLoopBrpc> receiver_loop_;
};

TEST_F(ChannelBrpcTest, Normal_Empty) {
  const std::string key = "key";
  const std::string sent = "";
  sender_->SendAsync(key, {sent.c_str(), static_cast<int64_t>(sent.size())});
  auto received = receiver_->Recv(key);

  EXPECT_EQ(sent, std::string(received.data<char>(), received.size()));
}

TEST_F(ChannelBrpcTest, Timeout) {
  receiver_->SetRecvTimeout(500u);
  const std::string key = "key";
  std::string received;
  EXPECT_THROW(receiver_->Recv(key), IoError);
}

TEST_F(ChannelBrpcTest, Normal_Len100) {
  const std::string key = "key";
  const std::string sent = RandStr(100u);
  sender_->SendAsync(key, {sent.c_str(), static_cast<int64_t>(sent.size())});
  auto received = receiver_->Recv(key);

  EXPECT_EQ(sent, std::string(received.data<char>(), received.size()));
}

class ChannelBrpcWithLimitTest
    : public ChannelBrpcTest,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {};

TEST_P(ChannelBrpcWithLimitTest, SendAsync) {
  const size_t size_limit_per_call = std::get<0>(GetParam());
  const size_t size_to_send = std::get<1>(GetParam());

  sender_->SetHttpMaxPayloadSize(size_limit_per_call);

  const std::string key = "key";
  const std::string sent = RandStr(size_to_send);
  sender_->SendAsync(key, {sent.c_str(), static_cast<int64_t>(sent.size())});
  auto received = receiver_->Recv(key);

  EXPECT_EQ(sent, std::string(received.data<char>(), received.size()));
}

TEST_P(ChannelBrpcWithLimitTest, Send) {
  const size_t size_limit_per_call = std::get<0>(GetParam());
  const size_t size_to_send = std::get<1>(GetParam());

  sender_->SetHttpMaxPayloadSize(size_limit_per_call);

  const std::string key = "key";
  const std::string sent = RandStr(size_to_send);
  sender_->Send(key, {sent.c_str(), static_cast<int64_t>(sent.size())});
  auto received = receiver_->Recv(key);

  EXPECT_EQ(sent, std::string(received.data<char>(), received.size()));
}

INSTANTIATE_TEST_SUITE_P(
    Normal_Instances, ChannelBrpcWithLimitTest,
    testing::Combine(testing::Values(1, 10),
                     testing::Values(1, 2, 9, 10, 11, 20, 19, 21, 1001)),
    [](const testing::TestParamInfo<ChannelBrpcWithLimitTest::ParamType>&
           info) {
      std::string name = fmt::format("Limit_{}_Len_{}", std::get<0>(info.param),
                                     std::get<1>(info.param));
      return name;
    });

}  // namespace ppu::link::test
