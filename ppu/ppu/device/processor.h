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

#include <filesystem>
#include <memory>
#include <random>

#include "mlir/IR/MLIRContext.h"

#include "ppu/device/frame.h"
#include "ppu/device/symbol_table.h"
#include "ppu/hal/context.h"
#include "ppu/link/link.h"

#include "ppu/ppu.pb.h"

namespace ppu::device {

class Processor final {
  const RuntimeConfig rt_config_;

  const std::shared_ptr<link::Context> lctx_;

  std::unique_ptr<HalContext> hal_ctx_;

  SymbolTable sym_table_;

  std::unique_ptr<mlir::MLIRContext> mlir_context_;

public:
  explicit Processor(RuntimeConfig config, std::shared_ptr<link::Context> lctx);
  ~Processor();

  const std::shared_ptr<link::Context> &lctx() { return lctx_; }

  const RuntimeConfig &rt_config() { return rt_config_; }

  //
  HalContext *hctx() { return hal_ctx_.get(); }

  /// Return the environment of this device.
  /// The environment is a key-value symbol table.
  SymbolTable *getEnv() { return &sym_table_; }

  /// Evaluate a ppu executable, with default environment (symbol table).
  void run(const ExecutableProto &exec);

  /// Evaluate a ppu executable with given environment.
  void runWithEnv(const ExecutableProto &exec, SymbolTable *sym_table);

  /// Evaluate a PPHlo code(function) on default environment, with given
  /// input/output name bindings.
  void run(const std::string &pphlo,
           const std::vector<std::string> &input_names,
           const std::vector<std::string> &output_names);

  void setVar(const std::string &name, const std::string &val);

  const std::string &getVar(const std::string &name) const;

  void clearVars();
};

} // namespace ppu::device
