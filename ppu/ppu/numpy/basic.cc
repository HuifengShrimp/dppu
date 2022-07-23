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


#include "ppu/numpy/basic.h"

#include "ppu/hal/hal.h"

namespace ppu::numpy {

hal::Value identity(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::identity(ctx, in);
}

hal::Value astype(HalContext* ctx, const hal::Value& in, DataType dtype) {
  PPU_TRACE_OP(ctx, in, dtype);
  return hal::cast_dtype(ctx, in, dtype);
}

hal::Value abs(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::abs(ctx, in);
}

hal::Value add(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::add(ctx, x, y);
}

hal::Value bitwise_and(HalContext* ctx, const hal::Value& x,
                       const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::bitwise_and(ctx, x, y);
}

hal::Value bitwise_xor(HalContext* ctx, const hal::Value& x,
                       const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::bitwise_xor(ctx, x, y);
}

hal::Value matmul(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::matmul(ctx, x, y);
}

hal::Value equal(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::equal(ctx, x, y);
}

hal::Value exp(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::exp(ctx, in);
}

hal::Value floor(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::floor(ctx, in);
}

hal::Value greater(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::greater(ctx, x, y);
}

hal::Value greater_equal(HalContext* ctx, const hal::Value& x,
                         const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::greater_equal(ctx, x, y);
}

hal::Value less(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::less(ctx, x, y);
}

hal::Value less_equal(HalContext* ctx, const hal::Value& x,
                      const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::less_equal(ctx, x, y);
}

hal::Value log(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::log(ctx, in);
}

hal::Value log1p(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::log1p(ctx, in);
}

hal::Value logical_not(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::logical_not(ctx, in);
}

hal::Value logistic(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::logistic(ctx, in);
}

hal::Value max(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::max(ctx, x, y);
}

hal::Value mul(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::mul(ctx, x, y);
}

hal::Value negate(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::negate(ctx, in);
}

hal::Value not_equal(HalContext* ctx, const hal::Value& x,
                     const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::not_equal(ctx, x, y);
}

hal::Value power(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::power(ctx, x, y);
}

hal::Value reciprocal(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::reciprocal(ctx, in);
}

hal::Value select(HalContext* ctx, const hal::Value& pred, const hal::Value& a,
                  const hal::Value& b) {
  PPU_TRACE_OP(ctx, pred, a, b);
  return hal::select(ctx, pred, a, b);
}

hal::Value sub(HalContext* ctx, const hal::Value& x, const hal::Value& y) {
  PPU_TRACE_OP(ctx, x, y);
  return hal::sub(ctx, x, y);
}

hal::Value broadcast_to(HalContext* ctx, const hal::Value& in,
                        const std::vector<int64_t>& to_shape) {
  PPU_TRACE_OP(ctx, in, to_shape);
  return hal::broadcast_to(ctx, in, to_shape);
}

hal::Value concatenate(HalContext* ctx, const hal::Value& first,
                       const hal::Value& second, const size_t& axis) {
  PPU_TRACE_OP(ctx, first, second, axis);
  return hal::concatenate(ctx, {first, second}, axis);
}

hal::Value reshape(HalContext* ctx, const hal::Value& in,
                   const std::vector<int64_t>& to_shape) {
  PPU_TRACE_OP(ctx, in, to_shape);
  return hal::reshape(ctx, in, to_shape);
}

hal::Value slice(HalContext* ctx, const hal::Value& input,
                 const std::vector<size_t>& start_indices,
                 const std::vector<size_t>& end_indices,
                 const std::vector<size_t>& strides) {
  PPU_TRACE_OP(ctx, input, start_indices, end_indices, strides);
  return hal::slice(ctx, input, start_indices, end_indices, strides);
}

hal::Value transpose(HalContext* ctx, const hal::Value& in) {
  PPU_TRACE_OP(ctx, in);
  return hal::transpose(ctx, in);
}

}  // namespace ppu::numpy
