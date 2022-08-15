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

namespace ppu::numpy {

/// return the identity representation of in value.
// Useful to normalize stateful value (like lazy value).
hal::Value identity(HalContext* ctx, const hal::Value& in);

// cast input value to dtype.
// @param in, the input value.
hal::Value astype(HalContext* ctx, const hal::Value& in, DataType dtype);

/// the element-wise absolute value function
// @param in, the value to negate
hal::Value abs(HalContext* ctx, const hal::Value& in);

/// general element-wise add operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value add(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// general element-wise bitwise and operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value bitwise_and(HalContext* ctx, const hal::Value& x,
                       const hal::Value& y);

/// general element-wise bitwise and operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value bitwise_xor(HalContext* ctx, const hal::Value& x,
                       const hal::Value& y);

/// matmul operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value matmul(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value equal(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// element-wise natural exponential x -> e^x
// @param in, the input value
hal::Value exp(HalContext* ctx, const hal::Value& in);

/// element-wise floor
// @param in, the input value
hal::Value floor(HalContext* ctx, const hal::Value& in);

/// general element-wise bitwise greater operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value greater(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// general element-wise bitwise greater or equal operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value greater_equal(HalContext* ctx, const hal::Value& x,
                         const hal::Value& y);

/// general element-wise bitwise less operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value less(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// general element-wise bitwise less or equal operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value less_equal(HalContext* ctx, const hal::Value& x,
                      const hal::Value& y);

/// the element-wise natural logarithm
// @param in, the param
hal::Value log(HalContext* ctx, const hal::Value& in);

/// the element-wise natural logarithm of (1 + x)
// @param in, the param
hal::Value log1p(HalContext* ctx, const hal::Value& in);

/// see numpy.logical_not(in)
// @param in, requires integer one or zero
hal::Value logical_not(HalContext* ctx, const hal::Value& in);

/// the element-wise sigmoid function
// @param in, the param
hal::Value logistic(HalContext* ctx, const hal::Value& in);

/// element-wise maximum
// @param x, first input value
// @param y, second input value
hal::Value max(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// general element-wise multiply operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value mul(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// see numpy.negate(in)
// @param in, the value to negate
hal::Value negate(HalContext* ctx, const hal::Value& in);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value not_equal(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// element-wise power x ^ y
// @param x, first input value
// @param y, second input value
hal::Value power(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// the element-wise reciprocal function
// @param in, the param
hal::Value reciprocal(HalContext* ctx, const hal::Value& in);

/// see numpy.select
// @param pred, the predicate, requires integer zero or one
// @param a, the first param
// @param b, the second param
hal::Value select(HalContext* ctx, const hal::Value& pred, const hal::Value& a,
                  const hal::Value& b);

/// general element-wise subtract operator
// @param x, the first parameter
// @param y, the second parameter
hal::Value sub(HalContext* ctx, const hal::Value& x, const hal::Value& y);

/// the broadcast function
// @param in, the input
// @param to_shape, the target shape
hal::Value broadcast_to(HalContext* ctx, const hal::Value& in,
                        const std::vector<int64_t>& to_shape);

/// the concatenate function
// @param first, the first param
// @param second, the second param
// @param axis, the axis
hal::Value concatenate(HalContext* ctx, const hal::Value& first,
                       const hal::Value& second, const size_t& axis);

/// the reshape function
// @param in, the input
// @param to_shape, the target shape
hal::Value reshape(HalContext* ctx, const hal::Value& in,
                   const std::vector<int64_t>& to_shape);

/// the slice function
// @param input, the param
// @param start_indices, the start indices
// @param end_indices, the end indices
// @param strides, the strides
hal::Value slice(HalContext* ctx, const hal::Value& input,
                 const std::vector<size_t>& start_indices,
                 const std::vector<size_t>& end_indices,
                 const std::vector<size_t>& strides);

/// the transpose function
// @param in, the param
hal::Value transpose(HalContext* ctx, const hal::Value& in);

///the logistic regression function
hal::Value logreg(HalContext* ctx, const hal::Value& x, hal::Value& w, hal::Value& y, size_t M, size_t N);

}  // namespace ppu::numpy
