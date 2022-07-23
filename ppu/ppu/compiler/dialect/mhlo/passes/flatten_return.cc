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


#include <cassert>
#include <iostream>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ppu/compiler/dialect/mhlo/passes/pass_details.h"
#include "ppu/compiler/dialect/mhlo/passes/passes.h"

namespace mlir {
namespace mhlo {
namespace {

void FlattenTupleValue(OpBuilder &builder, Location loc, Value value,
                       llvm::SmallVectorImpl<Value> &flatten_values) {
  if (!value.getType().isa<TupleType>()) {
    flatten_values.push_back(value);
    return;
  }

  // This function doesn't handle nested tuple.
  int flattenIdx = 0;
  auto tupleType = value.getType().cast<TupleType>();
  for (auto childType : tupleType.getTypes()) {
    auto getTupleOp = builder.create<mhlo::GetTupleElementOp>(
        loc, childType, value, builder.getI32IntegerAttr(flattenIdx++));
    flatten_values.push_back(getTupleOp);
  }
}

struct ReturnConverter : public OpRewritePattern<ReturnOp> {
  explicit ReturnConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    // Creates a new ReturnOp with flatten values.
    llvm::SmallVector<Value, 4> results;
    for (auto operand : op.getOperands()) {
      FlattenTupleValue(builder, op.getLoc(), operand, results);
    }
    builder.create<mhlo::ReturnOp>(op.getLoc(), results);
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context) {
  OwningRewritePatternList patterns(context);
  patterns.insert<ReturnConverter>(context);
  (void)applyPatternsAndFoldGreedily(target, std::move(patterns));
}

class FlattenReturnPass : public HLOFlattenReturnPassBase<FlattenReturnPass> {
public:
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    ApplyFlatteningTuplePatterns(getFunction(), ctx);
  }
};

std::unique_ptr<FunctionPass> createHLOFlattenReturnPass() {
  return std::make_unique<FlattenReturnPass>();
}

} // end namespace mhlo
} // end namespace mlir
