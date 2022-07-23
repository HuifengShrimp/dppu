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


#include "ppu/hal/io_ops.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/hal/prot_wrapper.h"

namespace ppu::hal {

Value make_secret(HalContext* ctx, PtBufferView bv, Rank owner) {
  PPU_TRACE_OP(ctx, bv);

  if (owner == kInvalidRank) {
    // treat the data provided by rank-0
    return make_secret(ctx, bv, 0);
  }

  auto pd = make_public(ctx, bv);
  return _p2s(ctx, pd).as_dtype(pd.dtype());
}

Value make_public(HalContext* ctx, PtBufferView bv) {
  PPU_TRACE_OP(ctx, bv);

  NdArrayRef raw = make_ndarray(bv);

  const Type encoded_ty = makeType<RingTy>(ctx->GetField());
  DataType dtype;
  NdArrayRef encoded = encodeToRing(raw, encoded_ty, ctx->FxpBits(), &dtype);

  return makeValue(encoded.as(makeType<Ring2kPublTy>(ctx->GetField())), dtype);
}

NdArrayRef dump_public(HalContext* ctx, const Value& v) {
  PPU_TRACE_OP(ctx, v);
  PPU_ENFORCE(v.mpc_type().isa<Ring2kPublTy>());

  auto dtype = v.dtype();

  auto encoded = v.as(makeType<RingTy>(ctx->GetField()));

  Type decode_ty = makePtType(GetDecodeType(dtype));
  return decodeFromRing(encoded, decode_ty, ctx->FxpBits(), dtype);
}

Value make_value(HalContext* ctx, Visibility vtype, PtBufferView bv,
                 Rank rank) {
  switch (vtype) {
    case VIS_PUBLIC:
      return make_public(ctx, bv);
    case VIS_SECRET:
      return make_secret(ctx, bv);
    default:
      PPU_THROW("not support vtype={}", vtype);
  }
}

Value constant(HalContext* ctx, const std::shared_ptr<Buffer>& content,
               const PtType& type, const std::vector<int64_t>& shape) {
  PtBufferView view(content->data(), type, shape, compactStrides(shape));

  return make_public(ctx, view);
}

Value iota(HalContext* ctx, size_t numel) {
  // Build a const vector
  auto buf = makeBuffer(sizeof(int32_t) * numel);
  auto* begin = reinterpret_cast<int32_t*>(buf->data());
  std::iota(begin, begin + numel, 0);

  return constant(ctx, buf, PT_I32, {static_cast<long long>(numel)});
}

}  // namespace ppu::hal
