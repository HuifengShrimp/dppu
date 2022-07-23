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


#include "ppu/compiler/core/core.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "ppu/compiler/common/compilation_context.h"
#include "ppu/compiler/passes/passes.h"
#include "ppu/utils/exception.h"

namespace ppu::compiler {

Core::Core(CompilationContext *ctx) : ctx_(ctx) {}

void Core::doit(mlir::ModuleOp module) {
  mlir::PassManager pm(ctx_->getMLIRContext());
  buildPipeline(&pm);

  ctx_->setupPrettyPrintConfigurations(&pm);

  auto ret = pm.run(module);

  if (ret.failed()) {
    PPU_THROW("Run core pipeline failed");
  }
}

void Core::buildPipeline(mlir::PassManager *pm) {
  {
    // lowering
    auto &optPM = pm->nest<mlir::FuncOp>();
    optPM.addPass(mlir::pphlo::createDecomposeComparisonPass());
    optPM.addPass(mlir::pphlo::createDecomposeSqrtPass());
    optPM.addPass(mlir::pphlo::createDecomposeDividePass());
    optPM.addPass(mlir::pphlo::createDecomposeSelectPass());
  }
  {
    auto &optPM = pm->nest<mlir::FuncOp>();
    // Cleanup
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }
}

} // namespace ppu::compiler
