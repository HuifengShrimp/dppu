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


#include "ppu/device/symbol_table.h"

#include <optional>
#include <utility>

#include "ppu/core/type_util.h"
#include "ppu/utils/exception.h"

namespace ppu::device {

void SymbolTable::setVar(const std::string &name, const std::string &val) {
  sym_table_[name] = val;
}

const std::string &SymbolTable::getVar(const std::string &name) const {
  auto iter = sym_table_.find(name);
  if (iter == sym_table_.end()) {
    PPU_THROW("Variable not found: {}", name);
  }
  return iter->second;
}

bool SymbolTable::hasVar(const std::string &name) const {
  return sym_table_.find(name) != sym_table_.end();
}

void SymbolTable::clear() { sym_table_.clear(); }

} // namespace ppu::device
