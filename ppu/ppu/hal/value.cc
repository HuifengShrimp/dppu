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


#include "ppu/hal/value.h"

#include <utility>

#include "ppu/core/shape_util.h"

namespace ppu::hal {

Visibility Value::vtype() const {
  if (eltype() == Type()) {
    return VIS_INVALID;
  }

  const auto& mpc_ty = mpc_type();
  if (mpc_ty.isa<Secret>()) {
    return VIS_SECRET;
  } else if (mpc_ty.isa<Public>()) {
    return VIS_PUBLIC;
  } else {
    return VIS_INVALID;
  }
};

Type const& Value::mpc_type() const {
  return eltype().as<ValueTy>()->mpc_type();
}

DataType Value::dtype() const {
  if (eltype() == Type()) {
    return DT_INVALID;
  } else {
    return eltype().as<ValueTy>()->dtype();
  }
}

Value makeValue(const NdArrayRef& arr, DataType dtype) {
  auto eltype = arr.eltype();
  if (eltype.isa<ValueTy>()) {
    eltype = eltype.as<ValueTy>()->mpc_type();
  }

  PPU_ENFORCE(eltype.isa<Ring2k>(), "expect a ring2k type, got={}", eltype);

  const Type ty = makeType<ValueTy>(dtype, eltype);
  return {arr.buf(), ty, arr.shape(), arr.strides(), arr.offset()};
}

void Value::CopyElementFrom(const Value& v, absl::Span<const int64_t> input_idx,
                            absl::Span<const int64_t> output_idx) const {
  PPU_ENFORCE(v.dtype() == dtype());
  PPU_ENFORCE(v.vtype() == vtype());

  auto input_linear_idx =
      flattenIndex(input_idx, absl::MakeSpan(v.shape()), v.strides());
  auto output_linear_idx =
      flattenIndex(output_idx, absl::MakeSpan(shape()), strides());

  memcpy(
      reinterpret_cast<std::byte*>(const_cast<void*>(data())) +
          output_linear_idx * elsize(),
      static_cast<const std::byte*>(v.data()) + input_linear_idx * v.elsize(),
      elsize());
}

Value Value::GetElementAt(absl::Span<const int64_t> index) const {
  PPU_ENFORCE(dtype() != DT_INVALID);
  auto linear_idx = flattenIndex(index, absl::MakeSpan(shape()), strides());
  return makeValue(
      {buf(), mpc_type(), std::vector<size_t>(), std::vector<size_t>(),
       static_cast<int64_t>(linear_idx * elsize())},
      dtype());
}

ValueProto Value::toProto() const {
  ValueProto proto;
  proto.set_type_data(eltype().toString());
  for (const auto& d : shape()) {
    proto.mutable_shape()->add_dims(d);
  }
  if (isCompact()) {
    proto.set_content(data(), buf()->size());
  } else {
    // Make a compact clone
    auto copy = clone();
    PPU_ENFORCE(copy.isCompact(), "Must be a compact copy.");
    proto.set_content(copy.data(), copy.buf()->size());
  }
  return proto;
}

Value Value::fromProto(const ValueProto& proto) {
  auto eltype = Type::fromString(proto.type_data());

  std::vector<int64_t> shape =
      makeShape(proto.shape().dims().begin(), proto.shape().dims().end());

  auto buf =
      makeBuffer(proto.content().c_str(), ppu::numel(shape) * eltype.size());

  return {buf, eltype, shape};
}

}  // namespace ppu::hal
