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

// If |pred| then |*z| = |x|; else |*z| = |y|;
// Implement by: z = pred * x + (1 - pred) * y = pred * (x - y) + y.
struct SelectOpConverter : public OpRewritePattern<SelectOp> {
  explicit SelectOpConverter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);
    auto pred = op.getOperand(0);
    auto x = op.getOperand(1);
    auto y = op.getOperand(2);

    // x-y
    auto sub = builder.create<SubOp>(op.getLoc(), op.getType(), x, y);

    // pred*(x-y)
    auto mul = builder.create<MulOp>(op.getLoc(), op.getType(), pred, sub);

    // pred*(x-y)+y
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), mul, y);

    return success();
  }
};

struct DecomposeSelect : public DecomposeSelectBase<DecomposeSelect> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

private:
  void populateOwningPatterns(OwningRewritePatternList *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<SelectOpConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<FunctionPass> createDecomposeSelectPass() {
  return std::make_unique<DecomposeSelect>();
}

} // namespace mlir::pphlo
