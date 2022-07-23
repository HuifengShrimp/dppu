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


#include "ppu/device/colocated_io.h"

#include "spdlog/spdlog.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/device/io_accessor.h"

#include "ppu/device/colocated_io.pb.h"

namespace ppu::device {

void ColocatedIo::setVar(const std::string &name, PtBufferView bv) {
  pending_.emplace(name, make_ndarray(bv));
}

hal::Value ColocatedIo::getVar(const std::string &name) {
  auto v_str = processor_->getVar(name);

  ValueProto value_pb;
  PPU_ENFORCE(value_pb.ParseFromString(v_str));

  return hal::Value::fromProto(value_pb);
}

void ColocatedIo::sync() {
  const auto &lctx = processor_->lctx();

  IoAccessor io_util(lctx->WorldSize(), processor_->rt_config());

  for (Rank root = 0; root < lctx->WorldSize(); root++) {
    // each party as the root
    std::vector<Buffer> inputs;
    if (lctx->Rank() == root) {
      std::vector<NamedValueList> vars_per_party(lctx->WorldSize());
      for (const auto &[name, arr] : pending_) {
        PPU_ENFORCE(arr.eltype().isa<PtTy>());

        PtBufferView bv(arr.data(), arr.eltype().as<PtTy>()->pt_type(),
                        arr.shape(), arr.strides());

        auto shares = io_util.makeShares(VIS_SECRET, bv);

        PPU_ENFORCE(shares.size() == lctx->WorldSize());
        for (Rank idx = 0; idx < lctx->WorldSize(); idx++) {
          auto *item = vars_per_party[idx].add_items();
          item->set_name(name);
          *item->mutable_value() = std::move(shares[idx]);
        }
      }
      inputs.resize(vars_per_party.size());
      for (size_t idx = 0; idx < vars_per_party.size(); idx++) {
        inputs[idx].resize(vars_per_party[idx].ByteSizeLong());
        PPU_ENFORCE(vars_per_party[idx].SerializeToArray(inputs[idx].data(),
                                                         inputs[idx].size()));
      }
    }

    Buffer data = link::Scatter(lctx, inputs, root,
                                fmt::format("COLOCATED_IO:SYNC:{}", root));

    NamedValueList var_list;
    PPU_ENFORCE(var_list.ParseFromArray(data.data(), data.size()));

    for (const auto &var : var_list.items()) {
      std::string buf;
      PPU_ENFORCE(var.value().SerializeToString(&buf));
      processor_->setVar(var.name(), buf);
    }
  }

  // TODO(jint)
  // 1. Optimize latency, use 1 allgather instead of n scatter.
  // 2. Optimize communication, use zero sharing.
}

} // namespace ppu::device
