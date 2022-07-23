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



#include "ppu/link/algorithm/gather.h"

#include "fmt/format.h"

#include "ppu/link/algorithm/trace.h"
#include "ppu/utils/exception.h"
#include "ppu/utils/serialize.h"

namespace ppu::link {
namespace {
const char* kType = "GATHER";
}  // namespace

std::vector<Buffer> Gather(const std::shared_ptr<Context>& ctx,
                           const Buffer& input, size_t root,
                           std::string_view tag) {
  const auto event = fmt::format("{}:{}", ctx->NextId(), kType);
  TraceLog(event, tag, "");

  std::vector<Buffer> res;

  if (root == ctx->Rank()) {
    res.resize(ctx->WorldSize());
    for (size_t idx = 0; idx < ctx->WorldSize(); idx++) {
      if (idx == ctx->Rank()) {
        res[idx] = input;
      } else {
        res[idx] = ctx->RecvInternal(idx, event);
      }
    }
  } else {
    ctx->SendAsyncInternal(root, event, input);
  }

  return res;
}

std::vector<std::vector<Buffer>> Gather(const std::shared_ptr<Context>& ctx,
                                        const std::vector<Buffer>& inputs,
                                        size_t root, std::string_view tag) {
  std::vector<std::vector<Buffer>> outputs(inputs.size());
  if (inputs.empty()) {
    return outputs;
  }

  std::vector<Buffer> all_outputs_packed =
      Gather(ctx, utils::SerializeArrayOfBuffers(inputs), root, tag);

  if (root == ctx->Rank()) {
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
  } else {
    PPU_ENFORCE(all_outputs_packed.empty());
  }
  return outputs;
}

}  // namespace ppu::link
