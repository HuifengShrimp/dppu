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

#include <chrono>
#include <deque>
#include <unordered_map>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "ppu/dialect/pphlo_ops.h"
#include "ppu/hal/value.h"

namespace ppu {

class HalContext;

namespace device {

class Frame;

// PPHlo executor can modify this config on the fly, so here we make a partial
// copy of protbuf runtime config
struct PPHloExecutorConfig {
  bool enable_type_checker;
  bool enable_pphlo_trace;
  bool collect_profiling_data;
};

class PPHloExecutor {
public:
  explicit PPHloExecutor(HalContext *ctx, PPHloExecutorConfig config)
      : ctx_(ctx), config_(config) {}

  std::vector<hal::Value> executeModule(mlir::ModuleOp &op,
                                        llvm::ArrayRef<hal::Value> inputs);

  std::vector<hal::Value> executeRegion(mlir::Region &region,
                                        llvm::ArrayRef<hal::Value> inputs);

  HalContext *getContext() const { return ctx_; }

  auto getOpProfilingData() const { return op_profiling_data_; }

private:
  std::vector<hal::Value> executeFunc(mlir::FuncOp &fcn,
                                      llvm::ArrayRef<hal::Value> inputs);
  std::vector<hal::Value> executeBlock(mlir::Block &block);
  std::vector<hal::Value> executeTerminator(mlir::Operation &op);

  void debug_print(mlir::Operation &op, bool before_execution) const;

  template <typename OpT, typename... MoreOpT>
  void dispatchOp(mlir::Operation &op) {
    if (auto casted = llvm::dyn_cast<OpT>(op)) {
      // Pre-execution meta
      if (config_.enable_pphlo_trace) {
        debug_print(op, true);
      }
      std::chrono::high_resolution_clock::time_point s;
      if (config_.collect_profiling_data) {
        s = std::chrono::high_resolution_clock::now();
      }

      // Execute op
      execute(casted);

      // Post execution meta
      if (config_.collect_profiling_data) {
        auto e = std::chrono::high_resolution_clock::now();
        auto opName = op.getName().getIdentifier().str();
        auto duration =
            std::chrono::duration_cast<std::chrono::duration<double>>(e - s);
        auto iter = op_profiling_data_.find(opName);
        if (iter == op_profiling_data_.end()) {
          op_profiling_data_.emplace(opName, std::make_pair(1, duration));
        } else {
          ++iter->second.first;
          iter->second.second += duration;
        }
      }
      if (config_.enable_pphlo_trace) {
        debug_print(op, false);
      }
    } else {
      if constexpr (!sizeof...(MoreOpT)) {
        // If there is no more op types to dispatch, and the previous cast
        // fails..print error message
        errorUnknownOp(op);
      } else {
        dispatchOp<MoreOpT...>(op);
      }
    }
  }

  /// Unary ops
  void execute(mlir::pphlo::ReciprocalOp &op);
  void execute(mlir::pphlo::NegOp &op);
  void execute(mlir::pphlo::ExpOp &op);
  void execute(mlir::pphlo::LogOp &op);
  void execute(mlir::pphlo::Log1pOp &op);
  void execute(mlir::pphlo::CeilOp &op);
  void execute(mlir::pphlo::FloorOp &op);
  void execute(mlir::pphlo::AbsOp &op);
  void execute(mlir::pphlo::TransposeOp &op);
  void execute(mlir::pphlo::LogisticOp &op);
  void execute(mlir::pphlo::NotOp &op);
  void execute(mlir::pphlo::ProtectOp &op);

  /// Binary ops
  void execute(mlir::pphlo::EqualOp &op);
  void execute(mlir::pphlo::LessOp &op);
  void execute(mlir::pphlo::GreaterOp &op);

  void execute(mlir::pphlo::AddOp &op);
  void execute(mlir::pphlo::SubOp &op);
  void execute(mlir::pphlo::MulOp &op);
  void execute(mlir::pphlo::PowOp &op);
  void execute(mlir::pphlo::MaxOp &op);
  void execute(mlir::pphlo::MinOp &op);
  void execute(mlir::pphlo::DotOp &op);
  void execute(mlir::pphlo::ShiftLeftOp &op);
  void execute(mlir::pphlo::ShiftRightLogicalOp &op);

  /// Ternary ops
  void execute(mlir::pphlo::ClampOp &op);

  /// Logical ops
  void execute(mlir::pphlo::AndOp &op);
  void execute(mlir::pphlo::OrOp &op);
  void execute(mlir::pphlo::XorOp &op);

  /// Shape ops
  void execute(mlir::pphlo::BroadcastOp &op);
  void execute(mlir::pphlo::ReshapeOp &op);
  void execute(mlir::pphlo::ConcatenateOp &op);
  void execute(mlir::pphlo::SliceOp &op);
  void execute(mlir::pphlo::GatherOp &op);
  void execute(mlir::pphlo::PadOp &op);
  void execute(mlir::pphlo::ReverseOp &op);

  /// Data generator ops
  void execute(mlir::pphlo::ConstOp &op);
  void execute(mlir::pphlo::IotaOp &op);

  /// Other ops
  void execute(mlir::pphlo::RngUniformOp &op);
  void execute(mlir::pphlo::ConvertOp &op);
  void execute(mlir::pphlo::BitcastConvertOp &op);
  void execute(mlir::pphlo::ConvOp &op);
  void execute(mlir::pphlo::SortOp &op);

  /// Reduce ops
  void execute(mlir::pphlo::ReduceOp &op);
  void execute(mlir::pphlo::ReduceWindowOp &op);

  /// Control flow ops
  void execute(mlir::pphlo::WhileOp &op);
  void execute(mlir::pphlo::IfOp &op);

  /// Debug ops
  void execute(mlir::pphlo::DbgPrintOp &op);

  /// Lowered ops (All these ops will throw at run time)
  void execute(mlir::pphlo::SqrtOp &op);
  void execute(mlir::pphlo::SelectOp &op);
  void execute(mlir::pphlo::RevealOp &op);
  void execute(mlir::pphlo::ReturnOp &op);
  void execute(mlir::pphlo::NotEqualOp &op);
  void execute(mlir::pphlo::LessEqualOp &op);
  void execute(mlir::pphlo::GreaterEqualOp &op);
  void execute(mlir::pphlo::DivOp &op);
  void errorUnknownOp(mlir::Operation &op);

  void executeVReduce(mlir::pphlo::ReduceOp &op);

  Frame *getCurrentFrame() const { return frames_.back(); }

  const hal::Value &lookupValue(::mlir::Value v) const;
  size_t extractShiftBits(const hal::Value &op) const;
  bool getConditionValue(const hal::Value &v) const;

  HalContext *ctx_{nullptr};
  std::deque<Frame *> frames_;
  mlir::pphlo::TypeTools type_tools_;
  PPHloExecutorConfig config_;

  // Profiling thingy
  std::unordered_map<std::string,
                     std::pair<uint64_t, std::chrono::duration<double>>>
      op_profiling_data_;
};

} // namespace device
} // namespace ppu
