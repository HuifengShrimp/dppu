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

// c = sqrt(x) -> c = x^0.5
struct SqrtConverter : public OpRewritePattern<SqrtOp> {
  explicit SqrtConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(SqrtOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    auto shape = op.getType().dyn_cast<RankedTensorType>().getShape();
    auto const_op = builder.create<ConstOp>(
        op.getLoc(), DenseFPElementsAttr::get(
                         RankedTensorType::get(shape, builder.getF32Type()),
                         builder.getF32FloatAttr(0.5).getValue()));

    rewriter.replaceOpWithNewOp<PowOp>(op, op.getType(), op.getOperand(),
                                       const_op);

    return success();
  }
};

struct DecomposeSqrt : public DecomposeSqrtBase<DecomposeSqrt> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

private:
  void populateOwningPatterns(OwningRewritePatternList *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<SqrtConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<FunctionPass> createDecomposeSqrtPass() {
  return std::make_unique<DecomposeSqrt>();
}

} // namespace mlir::pphlo
