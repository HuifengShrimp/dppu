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


#include "ppu/mpc/cheetah/arithmetic.h"

#include "ppu/core/array_ref_util.h"
#include "ppu/core/trace.h"
#include "ppu/core/vectorize.h"
#include "ppu/mpc/cheetah/object.h"
#include "ppu/mpc/cheetah/utils.h"
#include "ppu/mpc/interfaces.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/semi2k/type.h"
#include "ppu/mpc/util/communicator.h"
#include "ppu/mpc/util/ring_ops.h"

namespace ppu::mpc::cheetah {

ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        size_t bits) const {
  PPU_TRACE_OP(this, x, bits);
  auto* primitives = ctx->caller()->getState<CheetahState>()->primitives();
  size_t size = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef y(makeType<RingTy>(field), size);

  if (heuristic) {
    // Use heuristic optimization from SecureQ8: Add a large positive to make
    // sure the value is always positive
    ArrayRef adjusted_x =
        ring_add(x, ring_lshift(ring_ones(field, size), x.elsize() * 8 - 5));

    DISPATCH_ALL_FIELDS(field, kName, [&]() {
      using U = typename std::make_unsigned<ring2k_t>::type;
      auto x_ptr = adjusted_x.getOrCreateCompactBuf()->data<U>();
      auto y_ptr = y.getOrCreateCompactBuf()->data<U>();
      primitives->nonlinear()->truncate_msb0(y_ptr, x_ptr, size, bits,
                                             sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
    ring_sub_(y,
              ring_lshift(ring_ones(field, size), x.elsize() * 8 - 5 - bits));
  } else {
    DISPATCH_ALL_FIELDS(field, kName, [&]() {
      using U = typename std::make_unsigned<ring2k_t>::type;
      auto x_ptr = x.getOrCreateCompactBuf()->data<U>();
      auto y_ptr = y.getOrCreateCompactBuf()->data<U>();
      primitives->nonlinear()->truncate(y_ptr, x_ptr, size, bits,
                                        sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
  }
  return y.as(x.eltype());
}

ArrayRef MsbA::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  PPU_TRACE_OP(this, x);
  auto* primitives = ctx->caller()->getState<CheetahState>()->primitives();

  size_t size = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef y(makeType<RingTy>(field), size);

  DISPATCH_ALL_FIELDS(field, kName, [&]() {
    using U = typename std::make_unsigned<ring2k_t>::type;
    auto x_ptr = x.getOrCreateCompactBuf()->data<U>();
    auto y_ptr = y.getOrCreateCompactBuf()->data<U>();
    Buffer msb_buf(size);
    primitives->nonlinear()->msb(msb_buf.data<uint8_t>(), x_ptr, size,
                                 sizeof(U) * 8);
    primitives->nonlinear()->flush();
    cast(y_ptr, msb_buf.data<uint8_t>(), size);
  });
  // Enforce it to be a boolean sharing
  return y.as(makeType<semi2k::BShrTy>(field, 1));
}

}  // namespace ppu::mpc::cheetah
