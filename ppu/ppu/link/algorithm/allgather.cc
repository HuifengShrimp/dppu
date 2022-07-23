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



#include "ppu/link/algorithm/allgather.h"

#include "fmt/format.h"

#include "ppu/link/algorithm/trace.h"
#include "ppu/utils/exception.h"
#include "ppu/utils/serialize.h"

namespace ppu::link {
namespace {
const char* kType = "ALLGATHER";
}  // namespace

std::vector<Buffer> AllGather(const std::shared_ptr<Context>& ctx,
                              const Buffer& input, std::string_view tag) {
  const auto event = fmt::format("{}:{}", ctx->NextId(), kType);

  TraceLog(event, tag, std::string(input.data<char>(), input.size()));

  // broadcast to all
  for (size_t idx = 0; idx < ctx->WorldSize(); idx++) {
    if (idx == ctx->Rank()) {
      continue;
    }

    ctx->SendAsyncInternal(idx, event, input);
  }

  // gather all
  std::vector<Buffer> outputs(ctx->WorldSize());
  for (size_t idx = 0; idx < ctx->WorldSize(); idx++) {
    if (idx == ctx->Rank()) {
      outputs[idx] = input;
      continue;
    }

    outputs[idx] = ctx->RecvInternal(idx, event);
  }

  return outputs;
}

std::vector<std::vector<Buffer>> AllGather(const std::shared_ptr<Context>& ctx,
                                           const std::vector<Buffer>& inputs,
                                           std::string_view tag) {
  std::vector<std::vector<Buffer>> outputs(inputs.size());
  if (inputs.empty()) {
    return outputs;
  }

  std::vector<Buffer> all_outputs_packed =
      AllGather(ctx, utils::SerializeArrayOfBuffers(inputs), tag);

  PPU_ENFORCE(all_outputs_packed.size() == ctx->WorldSize());

  for (size_t idx = 0; idx < inputs.size(); idx++) {
    outputs[idx].resize(ctx->WorldSize());
  }

  for (size_t rank = 0; rank < all_outputs_packed.size(); ++rank) {
    std::vector<Buffer> outputs_i =
        utils::DeserializeArrayOfBuffers(all_outputs_packed[rank]);
    PPU_ENFORCE(outputs_i.size() == inputs.size());

    for (size_t idx = 0; idx < inputs.size(); idx++) {
      outputs[idx][rank] = std::move(outputs_i[idx]);
    }
  }

  return outputs;
}

}  // namespace ppu::link
