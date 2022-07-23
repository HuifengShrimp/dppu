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


#pragma once

#include <future>

#include "ppu/core/array_ref.h"
#include "ppu/device/io_accessor.h"
#include "ppu/device/symbol_table.h"

#include "ppu/ppu.pb.h"

namespace ppu::device {

class LocalIo {
public:
  LocalIo(size_t world_size, const RuntimeConfig &config)
      : symbol_tables_(world_size), io_accessor_(world_size, config) {
    //
  }

  void InFeed(const std::string &name, PtBufferView view,
              Visibility visibility) {
    auto vals = io_accessor_.makeShares(visibility, view);
    PPU_ENFORCE(vals.size() == symbol_tables_.size());
    for (size_t idx = 0; idx < symbol_tables_.size(); ++idx) {
      std::string str;
      PPU_ENFORCE(vals[idx].SerializeToString(&str));
      symbol_tables_[idx].setVar(name, str);
    }
  }

  NdArrayRef OutFeed(const std::string &name, PtType type) {
    std::vector<ValueProto> vals;
    for (auto &lt : symbol_tables_) {
      ValueProto val;
      PPU_ENFORCE(val.ParseFromString(lt.getVar(name)));
      vals.push_back(std::move(val));
    }

    return io_accessor_.combineShares(vals, type);
  }

  SymbolTable *GetSymbolTable(size_t idx) { return &symbol_tables_[idx]; }

private:
  std::vector<device::SymbolTable> symbol_tables_;
  IoAccessor io_accessor_;
};

} // namespace ppu::device
