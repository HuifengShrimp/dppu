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

#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "ppu/core/array_ref.h"
#include "ppu/core/type.h"

namespace ppu::hal {

class Value final : public NdArrayRef {
 public:
  using NdArrayRef::NdArrayRef;

  Value(const Value& other) = default;
  Value& operator=(const Value& other) = default;

  Value(Value&& other) = default;
  Value& operator=(Value&& other) = default;

  // Note: vtype is readonly, client should not change it since the underline
  // Object should be modified accordingly.
  Visibility vtype() const;
  Type const& mpc_type() const;
  bool is_public() const { return vtype() == VIS_PUBLIC; }
  bool is_secret() const { return vtype() == VIS_SECRET; }

  DataType dtype() const;
  bool is_int() const { return dtype() == DT_INT; }
  bool is_fxp() const { return dtype() == DT_FXP; }

  Value& as_dtype(DataType new_dtype) {
    PPU_ENFORCE(dtype() == new_dtype || dtype() == DT_INVALID);
    eltype().as<ValueTy>()->set_dtype(new_dtype);
    return *this;
  }
  Value& as_int() { return as_dtype(DT_INT); }
  Value& as_fxp() { return as_dtype(DT_FXP); }

  ValueProto toProto() const;

  static Value fromProto(const ValueProto& proto);

  void CopyElementFrom(const Value& v, absl::Span<const int64_t> input_idx,
                       absl::Span<const int64_t> output_idx) const;

  Value GetElementAt(absl::Span<const int64_t> index) const;
};

Value makeValue(const NdArrayRef& arr, DataType dtype = DT_INVALID);

inline std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << fmt::format("Value<{}x{}{}>", fmt::join(v.shape(), "x"), v.vtype(),
                     v.dtype());
  return out;
}

}  // namespace ppu::hal
