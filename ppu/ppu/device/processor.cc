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


#include "ppu/device/processor.h"

#include <chrono>
#include <fstream>
#include <mutex>
#include <utility>
#include <vector>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "spdlog/spdlog.h"

#include "ppu/core/trace.h"
#include "ppu/core/type_util.h"
#include "ppu/device/frame.h"
#include "ppu/device/pphlo_executor.h"
#include "ppu/device/symbol_table.h"
#include "ppu/dialect/pphlo_dialect.h"
#include "ppu/utils/exception.h"

namespace {

void PPUErrorHandler(void * /*use_data*/, const char *reason,
                     bool /*gen_crash_diag*/) {
  PPU_THROW(reason);
}

} // namespace

namespace ppu::device {

static std::mutex ErrorHandlerMutex;

Processor::Processor(RuntimeConfig config, std::shared_ptr<link::Context> lctx)
    : rt_config_(config), lctx_(lctx) {
  // Set an error handler
  {
    std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
    llvm::remove_fatal_error_handler();
    llvm::install_fatal_error_handler(PPUErrorHandler);
  }

  tr_info.enable = config.enable_action_trace();

  hal_ctx_ = std::make_unique<HalContext>(config, lctx);

  mlir::DialectRegistry registry;
  registry.insert<mlir::pphlo::PPHloDialect, mlir::StandardOpsDialect>();
  mlir_context_ = std::make_unique<mlir::MLIRContext>(registry);
}

Processor::~Processor() {
  std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
  llvm::remove_fatal_error_handler();
}

void Processor::run(const ExecutableProto &exec) { runWithEnv(exec, getEnv()); }

void Processor::runWithEnv(const ExecutableProto &exec,
                           SymbolTable *sym_table) {
  // Profile: start stamp
  auto start = std::chrono::high_resolution_clock::now();
  // Process inputs
  std::vector<hal::Value> inputs;
  inputs.reserve(exec.input_names_size());

  for (int32_t idx = 0; idx < exec.input_names_size(); idx++) {
    const std::string &sym_name = exec.input_names(idx);

    ValueProto value_pb;
    PPU_ENFORCE(value_pb.ParseFromString(sym_table->getVar(sym_name)));
    auto val = hal::Value::fromProto(value_pb);

    inputs.emplace_back(std::move(val));
  }

  if (rt_config_.enable_processor_dump()) {
    // Naming convention for dumped files must align with debug runner.
    std::filesystem::path dump_folder = rt_config_.processor_dump_dir();
    {
      auto fname = fmt::format("{}/exec_{}_{}.txt", dump_folder, exec.name(),
                               lctx_->Rank());
      std::ofstream ir_file(fname, std::ios::binary);
      ir_file << exec.SerializeAsString();
    }

    size_t var_counter = 0;
    for (const auto &val : inputs) {
      std::ofstream inputs_file(
          dump_folder /
              (fmt::format("processor{}{}.txt", lctx_->Rank(), var_counter++)),
          std::ios::binary);
      inputs_file << val.toProto().SerializeAsString();
    }
  }

  auto module = mlir::parseSourceString(exec.code(), mlir_context_.get());
  auto moduleOp = module.get();

  PPHloExecutorConfig config{};
  config.enable_pphlo_trace = rt_config_.enable_pphlo_trace();
  config.enable_type_checker = rt_config_.enable_type_checker();
  config.collect_profiling_data = rt_config_.enable_op_time_profile();

  // Profile: before execution stamp
  auto exec_start = std::chrono::high_resolution_clock::now();
  PPHloExecutor executor(hal_ctx_.get(), config);
  auto outputs = executor.executeModule(moduleOp, inputs);

  // Profile: after execution stamp
  auto exec_end = std::chrono::high_resolution_clock::now();

  // Sync output to symbol table
  for (int32_t idx = 0; idx < exec.output_names_size(); idx++) {
    const std::string &sym_name = exec.output_names(idx);
    const hal::Value &out = outputs[idx];

    std::string out_str;
    PPU_ENFORCE(out.toProto().SerializeToString(&out_str));

    sym_table->setVar(sym_name, out_str);
  }

  auto end = std::chrono::high_resolution_clock::now();

  // Collect time profile data
  auto total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  auto input_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      exec_start - start);
  auto execution_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(exec_end -
                                                                exec_start);
  auto output_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - exec_end);

  SPDLOG_INFO("[Profiling] PPU execution completed, input processing took {}s, "
              "execution took {}s, output processing took {}s, total time {}s.",
              input_time.count(), execution_time.count(), output_time.count(),
              total_time.count());
  if (config.collect_profiling_data) {
    SPDLOG_INFO("Detailed operation profiling data:");
    const auto &data = executor.getOpProfilingData();
    for (const auto &[name, meta] : data) {
      SPDLOG_INFO("Operation {}, executed {} times, duration {}s", name,
                  meta.first, meta.second.count());
    }
  }
}

void Processor::run(const std::string &pphlo,
                    const std::vector<std::string> &input_names,
                    const std::vector<std::string> &output_names) {
  ExecutableProto exec;
  exec.set_name("unnamed");
  *exec.mutable_input_names() = {input_names.begin(), input_names.end()};
  *exec.mutable_output_names() = {output_names.begin(), output_names.end()};
  exec.set_code(pphlo);

  run(exec);
}

void Processor::setVar(const std::string &name, const std::string &val) {
  sym_table_.setVar(name, val);
}

const std::string &Processor::getVar(const std::string &name) const {
  return sym_table_.getVar(name);
}

void Processor::clearVars() { sym_table_.clear(); }

} // namespace ppu::device
