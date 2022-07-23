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


#include "ppu/hal/type_cast.h"

#include "ppu/hal/const_util.h"
#include "ppu/hal/prot_wrapper.h"  // vtype_cast
#include "ppu/hal/ring.h"
#include "ppu/utils/exception.h"

namespace ppu::hal {

Value p2s(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  return _p2s(ctx, x).as_dtype(x.dtype());
}

Value reveal(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  return _s2p(ctx, x).as_dtype(x.dtype());
}

Value int2fxp(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.dtype() == DataType::DT_INT);

  return _lshift(ctx, x, ctx->FxpBits()).as_fxp();
}

// x >= 0, fxp2int(x) = floor(x)
// e.g. fxp2int(0.5) = 0, fxp2int(1.0) = 1, fxp2int(1.2) = 1
// x <= 0, fxp2int(x) = floor(x + 1 - nonnegligible small value)
// e.g. fxp2int(-0.5) = 0, fxp2int(-1.0) = -1, fxp2int(-1.2) = -1
Value fxp2int(HalContext* ctx, const Value& x) {
  PPU_TRACE_OP(ctx, x);
  PPU_ENFORCE(x.dtype() == DataType::DT_FXP);

  return _arshift(
             ctx,
             _add(ctx, x,
                  _mul(ctx,
                       shaped_const(ctx, 1.0 - (1.0 / (1 << ctx->FxpBits())),
                                    x.shape()),
                       _less(ctx, x, shaped_const(ctx, 0.0f, x.shape())))),
             ctx->FxpBits())
      .as_int();
}

Value cast_dtype(HalContext* ctx, const Value& in, const DataType& to_type) {
  PPU_TRACE_OP(ctx, in, to_type);
  PPU_ENFORCE(in.dtype() != to_type);

  switch (to_type) {
    case DT_FXP:
      return int2fxp(ctx, in);
    case DT_INT:
      return fxp2int(ctx, in);
    default:
      PPU_THROW("Should not hit");
  }
}

}  // namespace ppu::hal
