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


#include "ppu/device/frame.h"

#include "mlir/IR/BuiltinTypes.h"

#include "ppu/dialect/pphlo_types.h"
#include "ppu/utils/exception.h"

namespace ppu::device {

namespace {

void checkShape(llvm::ArrayRef<int64_t> mlir_shape,
                const absl::Span<const int64_t> rt_shape) {
  PPU_ENFORCE(mlir_shape.size() == rt_shape.size(),
              "MLIR shape #dims does not match runtime result");

  for (size_t idx = 0; idx < mlir_shape.size(); ++idx) {
    PPU_ENFORCE(mlir_shape[idx] == rt_shape[idx],
                "Shape at dim {} does not match", idx);
  }
}

void checkType(::mlir::RankedTensorType type, const hal::Value &v) {
  // Check shape
  checkShape(type.getShape(), v.shape());

  // dType checker
  mlir::pphlo::TypeTools tool;
  if (tool.isFxpType(type)) {
    PPU_ENFORCE(v.is_fxp());
  } else if (tool.isIntegerType(type)) {
    PPU_ENFORCE(v.is_int());
  } else {
    PPU_ENFORCE("Unknown dtype");
  }

  // vType checker
  if (tool.isPublicType(type)) {
    PPU_ENFORCE(v.is_public());
  } else if (tool.isSecretType(type)) {
    PPU_ENFORCE(v.is_secret());
  } else {
    PPU_ENFORCE("Unknown vtype");
  }
}

} // namespace

void Frame::releaseValue(::mlir::Value operand) { values_.erase(operand); }

const hal::Value &Frame::getValue(::mlir::Value operand) const {
  auto iter = values_.find(operand);
  PPU_ENFORCE(iter != values_.end());

  // If type checker is enabled, do it at getter time
  if (with_type_checker_) {
    checkType(operand.getType().dyn_cast<::mlir::RankedTensorType>(),
              iter->second);
  }

  return iter->second;
}

void Frame::addValue(::mlir::Value operand, hal::Value &&val) {
  values_[operand] = std::move(val);
}

void Frame::addValue(::mlir::Value operand, const hal::Value &val) {
  values_[operand] = val;
}

} // namespace ppu::device
