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


#include "ppu/core/array_ref_util.h"
//
#include "xtensor/xrandom.hpp"

#include "ppu/hal/dispatch.h"  // BinaryOp
#include "ppu/hal/io_ops.h"
#include "ppu/hal/prot_wrapper.h"

namespace ppu::hal::test {

HalContext MakeRefHalContext(RuntimeConfig config);

HalContext MakeRefHalContext();

template <typename T>
auto xt_random(const std::vector<size_t>& shape, double min = -100,
               double max = 100) {
  if constexpr (std::is_integral_v<T>) {
    return xt::random::randint<T>(shape, static_cast<T>(min),
                                  static_cast<T>(max));
  } else if constexpr (std::is_floating_point_v<T>) {
    return xt::random::rand<T>(shape, static_cast<T>(min), static_cast<T>(max));
  } else {
    PPU_THROW("unsupport xt_random type");
  }
}

// Export a value to a typed xarray.
template <typename T>
xt::xarray<T> dump_public_as(HalContext* ctx, const Value& in) {
  auto arr = dump_public(ctx, in);

#define CASE(NAME, TYPE, _)                  \
  case NAME: {                               \
    return xt::cast<T>(xt_adapt<TYPE>(arr)); \
  }

  switch (arr.eltype().as<PtTy>()->pt_type()) {
    FOREACH_PT_TYPES(CASE)

    default:
      PPU_THROW("unexpected type={}", arr.eltype());
  }

#undef CASE
}

template <typename T>
xt::xarray<T> EvalTernaryOp(Visibility in1_vtype, Visibility in2_vtype,
                            Visibility in3_vtype, TernaryOp* op,
                            PtBufferView in1, PtBufferView in2,
                            PtBufferView in3) {
  HalContext ctx = MakeRefHalContext();

  Value a = make_value(&ctx, in1_vtype, in1);
  Value b = make_value(&ctx, in2_vtype, in2);
  Value c = make_value(&ctx, in3_vtype, in3);

  Value d = op(&ctx, a, b, c);

  if (d.is_secret()) {
    d = _s2p(&ctx, d);
  }
  PPU_ENFORCE(d.is_public());

  return dump_public_as<T>(&ctx, d);
}

template <typename T>
xt::xarray<T> EvalBinaryOp(Visibility lhs_vtype, Visibility rhs_vtype,
                           BinaryOp* op, PtBufferView lhs, PtBufferView rhs) {
  HalContext ctx = MakeRefHalContext();

  Value a = make_value(&ctx, lhs_vtype, lhs);
  Value b = make_value(&ctx, rhs_vtype, rhs);

  Value c = op(&ctx, a, b);

  if (c.is_secret()) {
    c = _s2p(&ctx, c).as_dtype(c.dtype());
  }
  PPU_ENFORCE(c.is_public());

  return dump_public_as<T>(&ctx, c);
}

template <typename T>
xt::xarray<T> EvalUnaryOp(Visibility in_vtype, UnaryOp* op, PtBufferView in) {
  HalContext ctx = MakeRefHalContext();

  Value a = make_value(&ctx, in_vtype, in);

  Value b = op(&ctx, a);

  if (b.is_secret()) {
    b = _s2p(&ctx, b).as_dtype(b.dtype());
  }
  PPU_ENFORCE(b.is_public());

  return dump_public_as<T>(&ctx, b);
}

}  // namespace ppu::hal::test
