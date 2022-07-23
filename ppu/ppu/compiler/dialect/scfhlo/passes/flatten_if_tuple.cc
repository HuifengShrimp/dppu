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

#include "ppu/compiler/dialect/scfhlo/passes/pass_details.h"

namespace mlir {
namespace scfhlo {
namespace {

// Calculates the flatten types of a value.
void FlattenTupleType(Value value, llvm::SmallVectorImpl<Type> &types) {
  if (!value.getType().isa<TupleType>()) {
    types.push_back(value.getType());
    return;
  }

  // This function doesn't handle nested tuple.
  auto tupleType = value.getType().cast<TupleType>();
  types.append(tupleType.begin(), tupleType.end());
}

// Flattens value into flatten_values.
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

// FlattenTupleValue and CreateTupleValue is a pair of functions to create and
// flatten tuples in the exact same order. CreateTupleValue returns the result
// of the root TupleOp or given value if the type is not TupleType.
Value CreateTupleValue(OpBuilder &builder, Location loc,
                       const llvm::ArrayRef<Value> &flatten_values,
                       Type tuple_type) {
  if (!tuple_type.isa<TupleType>()) {
    assert(flatten_values.size() == 1);
    return flatten_values[0];
  }

  assert(tuple_type.cast<TupleType>().getTypes().size() ==
         flatten_values.size());
  return builder.create<mhlo::TupleOp>(loc, flatten_values);
}

// Flattens the tuples in the region's arguments and returning values.
void FlattenTupleInRegion(Region &region, PatternRewriter &rewriter) {
  OpBuilder regionOpBuilder(region);

  // Flatten tuples in arguments. The order of arguments must match the order
  // in FlattenTupleType, FlattenTupleValue and CreateTupleValue.
  const int originalNumArgs = region.getNumArguments();
  for (int argIdx : llvm::seq<int>(0, originalNumArgs)) {
    auto argument = region.getArgument(argIdx);

    // Adds new arguments to replace the tuple argument.
    llvm::SmallVector<Type, 4> newTypes;
    llvm::SmallVector<Value, 4> newArguments;
    FlattenTupleType(argument, newTypes);
    for (auto type : newTypes) {
      newArguments.push_back(region.addArgument(type));
    }

    // Replaces uses of the replacing argument.
    auto tupleValue = CreateTupleValue(regionOpBuilder, region.getLoc(),
                                       newArguments, argument.getType());
    argument.replaceAllUsesWith(tupleValue);
  }
  // Removes old tuple arguments.
  for (int argIdx = originalNumArgs - 1; argIdx >= 0; --argIdx) {
    region.eraseArgument(argIdx);
  }

  // Flatten tuples in results.
  for (auto &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    assert(isa<mhlo::ReturnOp>(terminator));
    auto returnOp = llvm::cast<mhlo::ReturnOp>(terminator);

    // Creates a new ReturnOp with flatten values.
    OpBuilder builder(returnOp);
    llvm::SmallVector<Value, 4> results;
    for (auto operand : returnOp.getOperands()) {
      FlattenTupleValue(builder, returnOp.getLoc(), operand, results);
    }
    builder.create<mhlo::ReturnOp>(region.getLoc(), results);
    rewriter.eraseOp(returnOp);
  }
}

// Applies tuple flattening patterns to given target. This helper
// function is used to flatten ops recursively.
template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context);

struct FlattenIfOp : public RewritePattern {
  explicit FlattenIfOp(MLIRContext *context)
      : RewritePattern(mhlo::IfOp::getOperationName(), 1, context,
                       {mhlo::IfOp::getOperationName(),
                        mhlo::TupleOp::getOperationName(),
                        mhlo::GetTupleElementOp::getOperationName()}),
        context(context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ifOp = cast<mhlo::IfOp>(op);
    // HLO WhileOp should have two regions: cond and body.
    if (ifOp->getNumRegions() != 2) {
      return failure();
    }

    OpBuilder builder(ifOp);
    llvm::SmallVector<Value, 4> flattenedTrueOperands;
    llvm::SmallVector<Type, 4> flattenedTrueOperandTypes;
    FlattenTupleType(ifOp.true_arg(), flattenedTrueOperandTypes);
    FlattenTupleValue(builder, ifOp.getLoc(), ifOp.true_arg(),
                      flattenedTrueOperands);

    llvm::SmallVector<Value, 4> flattenedFalseOperands;
    llvm::SmallVector<Type, 4> flattenedFalseOperandTypes;
    FlattenTupleType(ifOp.false_arg(), flattenedFalseOperandTypes);
    FlattenTupleValue(builder, ifOp.getLoc(), ifOp.false_arg(),
                      flattenedFalseOperands);

    auto oldResult = ifOp.getResult();

    llvm::SmallVector<Type, 4> flattenedTypes;
    FlattenTupleType(oldResult, flattenedTypes);

    // The applyPatternsAndFoldGreedily can't be called on child regions, so
    // creates temporary regions to apply flattening rules recursively.
    auto module = ifOp->getParentOfType<ModuleOp>();
    BlockAndValueMapping mapping;
    Region newTrue(module);
    ifOp.true_branch().cloneInto(&newTrue, mapping);
    Region newFalse(module);
    ifOp.false_branch().cloneInto(&newFalse, mapping);

    // Flattens the tuples in child regions.
    FlattenTupleInRegion(newTrue, rewriter);
    FlattenTupleInRegion(newFalse, rewriter);

    // There might be IfOp in child regions, flattens tuple in them too.
    ApplyFlatteningTuplePatterns<MutableArrayRef<Region>>(newTrue, context);
    ApplyFlatteningTuplePatterns<MutableArrayRef<Region>>(newFalse, context);

    // Creates a new scfhlo::IfOp with no tuples.
    auto newIf = builder.create<scfhlo::IfOp>(
        ifOp.getLoc(), flattenedTypes, ifOp.pred(), flattenedTrueOperands,
        flattenedFalseOperands);
    newTrue.cloneInto(&newIf.true_branch(), mapping);
    newFalse.cloneInto(&newIf.false_branch(), mapping);

    // Replaces uses of the old IfOp.
    auto newResultIter = newIf.result_begin();

    llvm::SmallVector<Value, 4> flattenedResults;
    while (flattenedResults.size() < flattenedTypes.size()) {
      assert(newResultIter != newIf->result_end());
      flattenedResults.push_back(*newResultIter++);
    }
    auto tupleValue = CreateTupleValue(builder, ifOp.getLoc(), flattenedResults,
                                       oldResult.getType());
    oldResult.replaceAllUsesWith(tupleValue);

    rewriter.eraseOp(ifOp);
    return success();
  }

private:
  MLIRContext *context;
};

template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context) {
  OwningRewritePatternList patterns(context);
  patterns.insert<FlattenIfOp>(context);
  (void)applyPatternsAndFoldGreedily(target, std::move(patterns));
}

class SCFHLOConvertControlflowPass
    : public SCFHLOConvertControlflowPassBase<SCFHLOConvertControlflowPass> {
public:
  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    ApplyFlatteningTuplePatterns(getFunction(), ctx);
  }
};
} // end namespace

std::unique_ptr<FunctionPass> createSCFHloConvertControlflowPass() {
  return std::make_unique<SCFHLOConvertControlflowPass>();
}

} // namespace scfhlo
} // end namespace mlir
