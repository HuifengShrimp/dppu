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


#include "ppu/compiler/dialect/scfhlo/IR/scfhlo_ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "ppu/compiler/dialect/scfhlo/IR/scfhlo_dialect.cc.inc"

namespace mlir::scfhlo {

void SCFHLODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ppu/compiler/dialect/scfhlo/IR/scfhlo_ops.cc.inc"
      >();
}

} // namespace mlir::scfhlo

#define GET_OP_CLASSES
#include "ppu/compiler/dialect/scfhlo/IR/scfhlo_ops.cc.inc"
