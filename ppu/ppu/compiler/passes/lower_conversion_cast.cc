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

struct CastConverter : public OpRewritePattern<UnrealizedConversionCastOp> {
  explicit CastConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    auto from_type = op->getOperandTypes()[0];
    auto to_type = op.getResultTypes()[0];

    TypeTools type_tool;

    if (type_tool.isPublicType(to_type) && type_tool.isSecretType(from_type)) {
      // Materialize from secret to public
      rewriter.replaceOpWithNewOp<RevealOp>(op, to_type, op->getOperands());
    } else if (type_tool.isSecretType(to_type) &&
               type_tool.isPublicType(from_type)) {
      // Materialize from public to secret
      rewriter.replaceOpWithNewOp<ProtectOp>(op, to_type, op.getOperands());
    }

    return success();
  }
};

struct LowerConversionCast
    : public LowerConversionCastBase<LowerConversionCast> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

private:
  void populateOwningPatterns(OwningRewritePatternList *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<CastConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<FunctionPass> createLowerConversionCastPass() {
  return std::make_unique<LowerConversionCast>();
}

} // namespace mlir::pphlo
