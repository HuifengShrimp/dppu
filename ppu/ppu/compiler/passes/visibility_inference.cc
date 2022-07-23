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


#include "ppu/compiler/passes/visibility_inference.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"

#include "ppu/compiler/dialect/scfhlo/IR/scfhlo_ops.h"
#include "ppu/utils/exception.h"

namespace mlir::pphlo {
namespace {

Visibility inferResultVisibility(ArrayRef<Visibility> operands_vis) {
  // If we have any operand that is secret, result type is secret
  if (llvm::any_of(operands_vis,
                   [](Visibility v) { return v == Visibility::VIS_SECRET; })) {
    return Visibility::VIS_SECRET;
  }
  if (operands_vis.empty()) {
    return Visibility::VIS_PUBLIC;
  }
  return operands_vis.front();
}

} // namespace

void VisibilityInference::inferFunc(FuncOp &func) {
  for (auto &blk : func) {
    inferBlock(blk);
  }
}

void VisibilityInference::inferRegion(Region &r) {
  for (auto &blk : r) {
    inferBlock(blk);
  }
}

void VisibilityInference::inferBlock(Block &blk) {
  for (auto &op : blk) {
    inferOperation(op);
  }
}

void VisibilityInference::inferReduce(Operation &op) {
  auto reduceOp = llvm::dyn_cast<mhlo::ReduceOp>(op);

  size_t num_results = op.getNumResults();
  for (size_t idx = 0; idx < reduceOp.getNumResults(); ++idx) {
    auto inputVis = ValueVis_.getValueVisibility(reduceOp.inputs()[idx]);
    auto initVis = ValueVis_.getValueVisibility(reduceOp.init_values()[idx]);

    // Promote input and init to the same visibility
    auto promoted_vis = inferResultVisibility({inputVis, initVis});

    ValueVis_.setValueVisibility(reduceOp.body().getArgument(idx),
                                 promoted_vis);
    ValueVis_.setValueVisibility(reduceOp.body().getArgument(num_results + idx),
                                 promoted_vis);
  }
  // ret0 = reduce(init0, val0)
  // Push inputs to body region
  inferRegion(reduceOp.body());

  // Get body return
  auto *terminator = reduceOp.body().back().getTerminator();
  PPU_ENFORCE(terminator &&
              terminator->getNumOperands() == reduceOp->getNumResults());
  for (size_t idx = 0; idx < reduceOp->getNumResults(); ++idx) {
    auto resultVis = ValueVis_.getValueVisibility(terminator->getOperand(idx));
    ValueVis_.setValueVisibility(reduceOp->getResult(idx), resultVis);
  }
}

void VisibilityInference::inferReduceWindow(Operation &op) {
  auto reduceOp = llvm::dyn_cast<mhlo::ReduceWindowOp>(op);
  PPU_ENFORCE(reduceOp->getNumResults() == 1,
              "Variadic reduce is not supported");
  auto inputVis = ValueVis_.getValueVisibility(reduceOp.inputs().front());
  // ret0 = reduce(init0, val0)
  // Push inputs to body region
  ValueVis_.setValueVisibility(reduceOp.body().getArgument(0), inputVis);
  ValueVis_.setValueVisibility(reduceOp.body().getArgument(1), inputVis);
  inferRegion(reduceOp.body());

  SmallVector<Visibility, 2> operand_vis;
  operand_vis.emplace_back(
      ValueVis_.getValueVisibility(reduceOp.init_values().front()));
  operand_vis.emplace_back(inputVis);
  ValueVis_.setValueVisibility(reduceOp->getResults().front(),
                               inferResultVisibility(operand_vis));
}

void VisibilityInference::inferIf(Operation &op) {
  auto ifOp = llvm::dyn_cast<scfhlo::IfOp>(op);

  // Infer true branch
  for (const auto &blkarg : ifOp.true_branch().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    ifOp.getTrueOperand(blkarg.getArgNumber())));
  }
  inferRegion(ifOp.true_branch());

  // Infer false branch
  for (const auto &blkarg : ifOp.false_branch().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    ifOp.getFalseOperand(blkarg.getArgNumber())));
  }
  inferRegion(ifOp.false_branch());

  // Infer result visibility
  auto &true_return = ifOp.true_branch().back().back();
  auto &false_return = ifOp.false_branch().back().back();
  PPU_ENFORCE(llvm::isa<mhlo::ReturnOp>(true_return));
  PPU_ENFORCE(llvm::isa<mhlo::ReturnOp>(false_return));
  PPU_ENFORCE(true_return.getNumOperands() == false_return.getNumOperands());
  PPU_ENFORCE(true_return.getNumOperands() == ifOp->getNumResults());

  for (const auto &ret : llvm::enumerate(ifOp->getResults())) {
    SmallVector<Visibility, 2> vis;

    // Get true branch result vis
    vis.emplace_back(
        ValueVis_.getValueVisibility(true_return.getOperand(ret.index())));
    // Get false branch result vis
    vis.emplace_back(
        ValueVis_.getValueVisibility(false_return.getOperand(ret.index())));

    ValueVis_.setValueVisibility(ret.value(), inferResultVisibility(vis));
  }
}

void VisibilityInference::inferWhile(Operation &op) {
  auto whileOp = llvm::dyn_cast<mhlo::WhileOp>(op);

  // Initial body visibility
  for (const auto &blkarg : whileOp.body().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    whileOp->getOperand(blkarg.getArgNumber())));
  }
  inferRegion(whileOp.body());

  // body return
  auto &body_return = whileOp.body().back().back();
  PPU_ENFORCE(llvm::isa<mhlo::ReturnOp>(body_return));

  // Update visibility
  for (const auto &blkarg : whileOp.body().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    body_return.getOperand(blkarg.getArgNumber())));
  }

  // Infer again
  inferRegion(whileOp.body());

  // body results visibility is either the same as origin or more strict
  // Now use the return visibility to infer cond region
  PPU_ENFORCE(whileOp.cond().getNumArguments() == body_return.getNumOperands());
  for (const auto &blkarg : whileOp.cond().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    body_return.getOperand(blkarg.getArgNumber())));
  }
  inferRegion(whileOp.cond());

  // Update result visibility
  for (const auto &ret : llvm::enumerate(whileOp->getResults())) {
    ValueVis_.setValueVisibility(
        ret.value(),
        ValueVis_.getValueVisibility(body_return.getOperand(ret.index())));
  }
}

void VisibilityInference::inferSort(Operation &op) {
  auto sortOp = llvm::dyn_cast<mhlo::SortOp>(op);

  // Push inputs to body region
  for (auto &in : llvm::enumerate(op.getOperands())) {
    auto inputVis = ValueVis_.getValueVisibility(in.value());
    ValueVis_.setValueVisibility(
        sortOp.comparator().getArgument(2 * in.index()), inputVis);
    ValueVis_.setValueVisibility(
        sortOp.comparator().getArgument(2 * in.index() + 1), inputVis);

    // Sort does not change result vis
    ValueVis_.setValueVisibility(op.getResult(in.index()), inputVis);
  }
  inferRegion(sortOp.comparator());
}

void VisibilityInference::inferOperation(Operation &op) {
  if (llvm::isa<mhlo::ReduceOp>(op)) {
    inferReduce(op);
  } else if (llvm::isa<mhlo::ReduceWindowOp>(op)) {
    inferReduceWindow(op);
  } else if (llvm::isa<mhlo::WhileOp>(op)) {
    inferWhile(op);
  } else if (llvm::isa<scfhlo::IfOp>(op)) {
    inferIf(op);
  } else if (llvm::isa<mhlo::IfOp>(op)) {
    PPU_THROW("Should not hit mhlo if");
  } else if (llvm::isa<mhlo::ConstOp>(op)) {
    // Constant always returns public
    ValueVis_.setValueVisibility(op.getResult(0), Visibility::VIS_PUBLIC);
  } else if (llvm::isa<mhlo::SortOp>(op)) {
    inferSort(op);
  } else if (llvm::isa<mhlo::GatherOp>(op)) {
    // For gather op, visibility should be the same as first operand
    ValueVis_.setValueVisibility(
        op.getResult(0), ValueVis_.getValueVisibility(op.getOperand(0)));
  } else if (op.getNumResults() == 1) {
    SmallVector<Visibility, 2> operand_vis;
    for (auto operand : op.getOperands()) {
      operand_vis.emplace_back(ValueVis_.getValueVisibility(operand));
    }
    ValueVis_.setValueVisibility(op.getResult(0),
                                 inferResultVisibility(operand_vis));
  } else if (llvm::isa<mlir::ReturnOp>(op) || llvm::isa<mhlo::ReturnOp>(op)) {
    // Do nothing
  } else {
    std::string dump;
    llvm::raw_string_ostream debug_s(dump);
    debug_s << "Unhandled op: ";
    op.print(debug_s);
    llvm_unreachable(debug_s.str().c_str());
  }
}
} // namespace mlir::pphlo