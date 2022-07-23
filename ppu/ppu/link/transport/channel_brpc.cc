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

#include "spdlog/spdlog.h"

#include "ppu/utils/exception.h"

#include "ppu/link/transport/channel_brpc.pb.h"

namespace ppu::link {
namespace internal {

class ReceiverServiceImpl : public pb::ReceiverService {
 public:
  explicit ReceiverServiceImpl(
      std::map<size_t, std::shared_ptr<IChannel>> listener)
      : listeners_(std::move(listener)) {}

  void Push(::google::protobuf::RpcController* /*cntl_base*/,
            const pb::PushRequest* request, pb::PushResponse* response,
            ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);

    try {
      const size_t sender_rank = request->sender_rank();
      const auto& trans_type = request->trans_type();

      // dispatch the message
      if (trans_type == pb::TransType::MONO) {
        OnRpcCall(sender_rank, request->key(), request->value());
      } else if (trans_type == pb::TransType::CHUNKED) {
        const auto& chunk = request->chunk_info();
        OnRpcCall(sender_rank, request->key(), request->value(),
                  chunk.chunk_index(), chunk.num_chunks());
      } else {
        response->set_error_code(pb::ErrorCode::INVALID_REQUEST);
        response->set_error_msg(
            fmt::format("unrecongnized trans type={}, from rank={}", trans_type,
                        sender_rank));
      }
      response->set_error_code(pb::ErrorCode::SUCCESS);
      response->set_error_msg("");
    } catch (const std::exception& e) {
      response->set_error_code(pb::ErrorCode::UNEXPECTED_ERROR);
      response->set_error_msg(fmt::format("dispatch error, key={}, error={}",
                                          request->key(), e.what()));
    }
  }

 protected:
  std::map<size_t, std::shared_ptr<IChannel>> listeners_;

 private:
  void OnRpcCall(size_t src_rank, const std::string& key,
                 const std::string& value) {
    auto itr = listeners_.find(src_rank);
    if (itr == listeners_.end()) {
      PPU_THROW_LOGIC_ERROR("dispatch error, listener rank={} not found",
                            src_rank);
    }
    return itr->second->OnMessage(
        key, {value.c_str(), static_cast<int64_t>(value.size())});
  }

  void OnRpcCall(size_t src_rank, const std::string& key,
                 const std::string& value, size_t chunk_idx,
                 size_t num_chunks) {
    auto itr = listeners_.find(src_rank);
    if (itr == listeners_.end()) {
      PPU_THROW_LOGIC_ERROR("dispatch error, listener rank={} not found",
                            src_rank);
    }
    auto comm_brpc = std::dynamic_pointer_cast<ChannelBrpc>(itr->second);
    comm_brpc->OnChunkedMessage(
        key, {value.c_str(), static_cast<int64_t>(value.size())}, chunk_idx,
        num_chunks);
  }
};

}  // namespace internal

void ReceiverLoopBrpc::StopImpl() {
  server_.Stop(0);
  server_.Join();
}

ReceiverLoopBrpc::~ReceiverLoopBrpc() { StopImpl(); }

void ReceiverLoopBrpc::Stop() { StopImpl(); }

std::string ReceiverLoopBrpc::Start(const std::string& host) {
  if (server_.IsRunning()) {
    PPU_THROW_LOGIC_ERROR("brpc server is already running");
  }

  auto svc = std::make_unique<internal::ReceiverServiceImpl>(listeners_);
  if (server_.AddService(svc.get(), brpc::SERVER_OWNS_SERVICE) == 0) {
    // Once add service succeed, give up ownership
    static_cast<void>(svc.release());
  } else {
    PPU_THROW_IO_ERROR("brpc server failed to add msg service");
  }

  // Start the server.
  brpc::ServerOptions options;
  if (server_.Start(host.c_str(), &options) != 0) {
    PPU_THROW_IO_ERROR("brpc server failed start");
  }

  return butil::endpoint2str(server_.listen_address()).c_str();
}

namespace {

// TODO: move this to somewhere-else.
class BatchDesc {
 protected:
  size_t batch_idx_;
  size_t batch_size_;
  size_t total_size_;

 public:
  BatchDesc(size_t batch_idx, size_t batch_size, size_t total_size)
      : batch_idx_(batch_idx),
        batch_size_(batch_size),
        total_size_(total_size) {}

  // return the index of this batch.
  size_t Index() const { return batch_idx_; }

  // return the offset of the first element in this batch.
  size_t Begin() const { return batch_idx_ * batch_size_; }

  // return the offset after last element in this batch.
  size_t End() const { return std::min(Begin() + batch_size_, total_size_); }

  // return the size of this batch.
  size_t Size() const { return End() - Begin(); }

  std::string ToString() const { return "B:" + std::to_string(batch_idx_); };
};

void OnPushDone(pb::PushResponse* response, brpc::Controller* cntl) {
  std::unique_ptr<pb::PushResponse> response_guard(response);
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    SPDLOG_WARN("send, rpc failed={}, message={}", cntl->ErrorCode(),
                cntl->ErrorText());
  } else if (response->error_code() != pb::ErrorCode::SUCCESS) {
    SPDLOG_WARN("send, peer failed message={}", response->error_msg());
  }
}

}  // namespace

void ChannelBrpc::SetPeerHost(const std::string& peer_host) {
  auto brpc_channel = std::make_unique<brpc::Channel>();
  const auto load_balancer = "";
  brpc::ChannelOptions options;
  {
    options.protocol = options_.channel_protocol;
    options.connection_type = options_.channel_connection_type;
    options.connect_timeout_ms = 20000;
    options.timeout_ms = options_.http_timeout_ms;
    options.max_retry = 3;
    // options.retry_policy = DefaultRpcRetryPolicy();
  }
  int res = brpc_channel->Init(peer_host.c_str(), load_balancer, &options);
  if (res != 0) {
    PPU_THROW_NETWORK_ERROR("Fail to initialize channel, host={}, err_code={}",
                            peer_host, res);
  }

  channel_ = std::move(brpc_channel);
  peer_host_ = peer_host;
}

namespace {

struct SendChunckedBrpcTask {
  std::shared_ptr<ChannelBrpc> channel;
  std::string key;
  Buffer value;

  SendChunckedBrpcTask(std::shared_ptr<ChannelBrpc> _channel, std::string _key,
                       Buffer _value)
      : channel(std::move(_channel)),
        key(std::move(_key)),
        value(std::move(_value)) {}

  static void* Proc(void* args) {
    // take ownership of task.
    std::unique_ptr<SendChunckedBrpcTask> task(
        static_cast<SendChunckedBrpcTask*>(args));

    task->channel->SendChunked(task->key, task->value);
    return nullptr;
  }
};

}  // namespace

void ChannelBrpc::SendAsync(const std::string& key, const Buffer& value) {
  if (value.size() > options_.http_max_payload_size) {
    auto btask = std::make_unique<SendChunckedBrpcTask>(
        this->shared_from_this(), key, value);

    // bthread run in 'detached' mode, we will never wait for it.
    bthread_t tid;
    if (bthread_start_background(&tid, nullptr, SendChunckedBrpcTask::Proc,
                                 btask.get()) == 0) {
      // bthread takes the ownership, release it.
      static_cast<void>(btask.release());
    } else {
      PPU_THROW("failed to push async sending job to bthread");
    }

    return;
  }

  pb::PushRequest request;
  {
    request.set_sender_rank(self_rank_);
    request.set_key(key);
    request.set_value(value.data<char>(), value.size());
    request.set_trans_type(pb::TransType::MONO);
  }

  // allocate |response| & |cntl| on heap, the callback is responsible to
  // release these objects.
  auto* response = new pb::PushResponse();
  auto* cntl = new brpc::Controller();
  pb::ReceiverService::Stub stub(channel_.get());
  stub.Push(cntl, &request, response,
            brpc::NewCallback(OnPushDone, response, cntl));
}

void ChannelBrpc::Send(const std::string& key, const Buffer& value) {
  if (value.size() > options_.http_max_payload_size) {
    SendChunked(key, value);
    return;
  }

  pb::PushRequest request;
  {
    request.set_sender_rank(self_rank_);
    request.set_key(key);
    request.set_value(value.data<char>(), value.size());
    request.set_trans_type(pb::TransType::MONO);
  }

  pb::PushResponse response;
  brpc::Controller cntl;
  pb::ReceiverService::Stub stub(channel_.get());
  stub.Push(&cntl, &request, &response, nullptr);

  // handle failures.
  if (cntl.Failed()) {
    PPU_THROW_NETWORK_ERROR("send, rpc failed={}, message={}", cntl.ErrorCode(),
                            cntl.ErrorText());
  }

  if (response.error_code() != pb::ErrorCode::SUCCESS) {
    PPU_THROW_NETWORK_ERROR("send, peer failed message={}",
                            response.error_msg());
  }
}

// See: chunked streamming
//   https://en.wikipedia.org/wiki/Chunked_transfer_encoding
// See: Brpc does NOT support POST chunked.
//   https://github.com/apache/incubator-brpc/blob/master/docs/en/http_client.md
void ChannelBrpc::SendChunked(const std::string& key, const Buffer& value) {
  const size_t bytes_per_chunk = options_.http_max_payload_size;
  const size_t num_bytes = value.size();
  const size_t num_chunks = (num_bytes + bytes_per_chunk - 1) / bytes_per_chunk;

  constexpr uint32_t kParallelSize = 10;
  const size_t batch_size = kParallelSize;
  const size_t num_batches = (num_chunks + batch_size - 1) / batch_size;

  for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const BatchDesc batch(batch_idx, batch_size, num_chunks);

    // See: "半同步“ from
    // https://github.com/apache/incubator-brpc/blob/master/docs/cn/client.md
    std::vector<brpc::Controller> cntls(batch.Size());
    std::vector<pb::PushResponse> responses(batch.Size());

    // fire batched chunk requests.
    for (size_t idx = 0; idx < batch.Size(); idx++) {
      const size_t chunk_idx = batch.Begin() + idx;
      const size_t chunk_offset = chunk_idx * bytes_per_chunk;

      pb::PushRequest request;
      {
        request.set_sender_rank(self_rank_);
        request.set_key(key);
        request.set_value(
            value.data<char>() + chunk_offset,
            std::min(bytes_per_chunk,
                     static_cast<size_t>(value.size()) - chunk_offset));
        request.set_trans_type(pb::TransType::CHUNKED);
        request.mutable_chunk_info()->set_num_chunks(num_chunks);
        request.mutable_chunk_info()->set_chunk_index(chunk_idx);
      }

      auto& cntl = cntls[idx];
      auto& response = responses[idx];
      pb::ReceiverService::Stub stub(channel_.get());
      stub.Push(&cntl, &request, &response, brpc::DoNothing());
    }

    for (size_t idx = 0; idx < batch.Size(); idx++) {
      brpc::Join(cntls[idx].call_id());
    }

    for (size_t idx = 0; idx < batch.Size(); idx++) {
      const size_t chunk_idx = batch.Begin() + idx;
      const auto& cntl = cntls[idx];
      const auto& response = responses[idx];
      if (cntl.Failed()) {
        PPU_THROW_NETWORK_ERROR(
            "send key={} (chunked {} out of {}) rpc failed: {}, message={}",
            key, chunk_idx + 1, num_chunks, cntl.ErrorCode(), cntl.ErrorText());
      } else if (response.error_code() != pb::ErrorCode::SUCCESS) {
        PPU_THROW_NETWORK_ERROR(
            "send key={} (chunked {} out of {}) response failed, message={}",
            key, chunk_idx + 1, num_chunks, response.error_msg());
      }
    }
  }
}

}  // namespace ppu::link
