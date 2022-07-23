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


#pragma once

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ppu/dialect/pphlo_base_enums.h"

#define GET_TYPEDEF_CLASSES
#include "ppu/dialect/pphlo_types.h.inc"
#include "ppu/utils/exception.h"

namespace mlir::pphlo {

class TypeTools {
 private:
  virtual bool isIntegerScalar(const Type &t) const {
    return t.isa<PIntegerType>() || t.isa<SIntegerType>();
  }

  virtual bool isFixedpointScalar(const Type &t) const {
    return t.isa<PFixedpointType>() || t.isa<SFixedpointType>();
  }

 public:
  virtual ~TypeTools();

  bool isIntegerType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return isIntegerType(rt.getElementType());
    }
    return isIntegerScalar(t);
  }

  bool isFxpType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return isFxpType(rt.getElementType());
    }
    return isFixedpointScalar(t);
  }

  bool isPublicType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return isPublicType(rt.getElementType());
    }
    return t.isa<PIntegerType>() || t.isa<PFixedpointType>();
  }

  bool isSecretType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return isSecretType(rt.getElementType());
    }
    return t.isa<SIntegerType>() || t.isa<SFixedpointType>();
  }

  Type toPublicType(const Type &t) const {
    if (isPublicType(t)) {
      return t;
    }
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(rt.getShape(),
                                   toPublicType(rt.getElementType()));
    }
    if (isIntegerType(t)) {
      return PIntegerType::get(t.getContext());
    }
    return PFixedpointType::get(t.getContext());
  }

  Type toSecretType(const Type &t) const {
    if (isSecretType(t)) {
      return t;
    }
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(rt.getShape(),
                                   toSecretType(rt.getElementType()));
    }
    if (isIntegerType(t)) {
      return SIntegerType::get(t.getContext());
    }
    return SFixedpointType::get(t.getContext());
  }

  Type toFxpType(const Type &t) const {
    if (isFxpType(t)) {
      return t;
    }
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(rt.getShape(),
                                   toFxpType(rt.getElementType()));
    }
    if (isSecretType(t)) {
      return SFixedpointType::get(t.getContext());
    }
    return PFixedpointType::get(t.getContext());
  }

  Visibility getTypeVisibility(const Type &t) const {
    if (isPublicType(t)) {
      return Visibility::VIS_PUBLIC;
    }
    PPU_ENFORCE(isSecretType(t));
    return Visibility::VIS_SECRET;
  }
};

}  // namespace mlir::pphlo
