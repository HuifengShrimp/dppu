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

#include <string>
#include <unordered_map>

namespace ppu::device {

class SymbolTable {
public:
  SymbolTable() = default;
  ~SymbolTable() = default;

  SymbolTable(const SymbolTable &) = delete;
  SymbolTable &operator=(const SymbolTable &) = delete;

  ///@brief Add a variable into mahine local symbol table
  ///
  ///@param name
  ///@param val
  void setVar(const std::string &name, const std::string &val);

  ///@brief Get a variable from machine local symbol table
  ///
  ///@param name
  ///@return hal::Value
  const std::string &getVar(const std::string &name) const;

  ///@brief Check whether a variable exists in machine local symbol table
  ///
  ///@param name
  ///@return true
  ///@return false
  bool hasVar(const std::string &name) const;

  ///@brief Clear the symbol table
  void clear();

private:
  std::unordered_map<std::string, std::string> sym_table_;
};

} // namespace ppu::device
