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

#include "ppu/hal/context.h"
#include "ppu/hal/value.h"

namespace ppu::hal {

using UnaryOp = Value(HalContext*, const Value&);
using BinaryOp = Value(HalContext*, const Value&, const Value&);
using TernaryOp = Value(HalContext*, const Value&, const Value&, const Value&);

template <BinaryOp* FnFxp, BinaryOp* FnInt, UnaryOp* FnI2F>
Value DtypeBinaryDispatch(std::string_view op_name, HalContext* ctx,
                          const Value& x, const Value& y) {
  // Promote int to fxp if mismatch.
  if (x.is_int() && y.is_int()) {
    return FnInt(ctx, x, y);
  } else if (x.is_int() && y.is_fxp()) {
    return FnFxp(ctx, FnI2F(ctx, x), y);
  } else if (x.is_fxp() && y.is_int()) {
    return FnFxp(ctx, x, FnI2F(ctx, y));
  } else if (x.is_fxp() && y.is_fxp()) {
    return FnFxp(ctx, x, y);
  } else {
    PPU_THROW("unsupported op {} for x={}, y={}", op_name, x, y);
  }
}

template <UnaryOp* FnFxp, UnaryOp* FnInt>
Value DtypeUnaryDispatch(std::string_view op_name, HalContext* ctx,
                         const Value& x) {
  // Promote int to fxp if mismatch.
  if (x.is_int()) {
    return FnInt(ctx, x);
  } else if (x.is_fxp()) {
    return FnFxp(ctx, x);
  } else {
    PPU_THROW("unsupported op {} for x={}", op_name, x);
  }
}

template <UnaryOp* FnPub, UnaryOp* FnSec>
Value VtypeUnaryDispatch(std::string_view op_name, HalContext* ctx,
                         const Value& in) {
  if (in.is_public()) {
    return FnPub(ctx, in);
  } else if (in.is_secret()) {
    return FnSec(ctx, in);
  } else {
    PPU_THROW("unsupport unary op={} for {}", op_name, in);
  }
}

template <BinaryOp* FnPP, BinaryOp* FnSP, BinaryOp* FnSS>
Value VtypeCommutativeBinaryDispatch(std::string_view op_name, HalContext* ctx,
                                     const Value& x, const Value& y) {
  if (x.is_public() && y.is_public()) {
    return FnPP(ctx, x, y);
  } else if (x.is_secret() && y.is_public()) {
    return FnSP(ctx, x, y);
  } else if (x.is_public() && y.is_secret()) {
    // commutative
    return FnSP(ctx, y, x);
  } else if (x.is_secret() && y.is_secret()) {
    return FnSS(ctx, y, x);
  } else {
    PPU_THROW("unsupported op {} for x={}, y={}", op_name, x, y);
  }
}

}  // namespace ppu::hal
