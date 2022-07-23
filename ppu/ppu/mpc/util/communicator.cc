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


#include "ppu/mpc/util/communicator.h"

#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc {

ArrayRef Communicator::allReduce(ReduceOp op, const ArrayRef& in,
                                 std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  std::vector<Buffer> all_str = link::AllGather(lctx_, *buf, tag);

  PPU_ENFORCE(all_str.size() == getWorldSize());
  ArrayRef res = in.clone();
  for (size_t idx = 0; idx < all_str.size(); idx++) {
    if (idx == getRank()) {
      continue;
    }

    if (op == ReduceOp::ADD) {
      ring_add_(res,
                ArrayRef(makeBuffer(std::move(all_str[idx])), in.eltype()));
    } else if (op == ReduceOp::XOR) {
      ring_xor_(res,
                ArrayRef(makeBuffer(std::move(all_str[idx])), in.eltype()));
    } else {
      PPU_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  stats_.latency += 1;
  stats_.comm += buf->size() * (lctx_->WorldSize() - 1);

  return res;
}

ArrayRef Communicator::reduce(ReduceOp op, const ArrayRef& in, size_t root,
                              std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  std::vector<Buffer> all_str = link::Gather(lctx_, *buf, root, tag);

  PPU_ENFORCE(all_str.size() == getWorldSize());
  ArrayRef res = in.clone();
  for (size_t idx = 0; idx < all_str.size(); idx++) {
    if (idx == getRank()) {
      continue;
    }

    if (op == ReduceOp::ADD) {
      ring_add_(res,
                ArrayRef(makeBuffer(std::move(all_str[idx])), in.eltype()));
    } else if (op == ReduceOp::XOR) {
      ring_xor_(res,
                ArrayRef(makeBuffer(std::move(all_str[idx])), in.eltype()));
    } else {
      PPU_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  stats_.latency += 1;
  stats_.comm += buf->size();

  return res;
}

}  // namespace ppu::mpc
