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


#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ppu/compiler/passes/pass_details.h"
#include "ppu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

// c = a/b; -> x = 1/b; c = a*x;
struct DivideConverter : public OpRewritePattern<DivOp> {
  explicit DivideConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    auto lhs = op.getOperand(0);
    auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs = op.getOperand(1);
    auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();

    // Convert both inputs to fxp
    auto lhs_fxp = builder.create<ConvertOp>(
        op.getLoc(), typetools_.toFxpType(lhs_type), lhs);

    auto rhs_fxp = builder.create<ConvertOp>(
        op.getLoc(), typetools_.toFxpType(rhs_type), rhs);

    // Do reciprocal with fxp
    auto reciprocal_op =
        builder.create<ReciprocalOp>(op.getLoc(), rhs_fxp.getType(), rhs_fxp);

    // Multiply as fxp
    auto mul_op =
        builder.create<MulOp>(op.getLoc(), typetools_.toFxpType(op.getType()),
                              lhs_fxp, reciprocal_op);

    // Convert back to original type
    rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(), mul_op);

    return success();
  }

private:
  TypeTools typetools_;
};

struct DecomposeDivide : public DecomposeDivideBase<DecomposeDivide> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

private:
  void populateOwningPatterns(OwningRewritePatternList *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<DivideConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<FunctionPass> createDecomposeDividePass() {
  return std::make_unique<DecomposeDivide>();
}

} // namespace mlir::pphlo
