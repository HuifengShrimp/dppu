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


#include "ppu/device/io_accessor.h"

#include <cstddef>
#include <utility>

#include "ppu/core/array_ref_util.h"
#include "ppu/core/encoding.h"
#include "ppu/hal/value.h"
#include "ppu/mpc/factory.h"
#include "ppu/utils/exception.h"

namespace ppu::device {

IoAccessor::IoAccessor(size_t world_size, RuntimeConfig config)
    : world_size_(world_size), config_(std::move(config)) {
  base_io_ = mpc::Factory::CreateIO(config_.protocol(), world_size_);
}

std::vector<ValueProto> IoAccessor::makeShares(Visibility vis,
                                               PtBufferView bv) {
  const auto field = config_.field();
  // FIXME(jint), use default when fxp_fraction_bits is zero.
  const size_t fxp_bits = FxpFractionalBits(config_);

  // encode to ring.
  NdArrayRef encoded;
  DataType dtype;
  {
    auto raw = make_ndarray(std::move(bv));
    const Type encoded_ty = makeType<RingTy>(field);
    encoded = encodeToRing(raw, encoded_ty, fxp_bits, &dtype);
  }

  // convert to protocol dependent shares.
  std::vector<NdArrayRef> shares;
  {
    if (vis == VIS_PUBLIC) {
      shares = base_io_->makePublic(encoded);
    } else if (vis == VIS_SECRET) {
      shares = base_io_->makeSecret(encoded);
    } else {
      PPU_THROW("Unsupported visibility type {}", vis);
    }
    PPU_ENFORCE(shares.size() == world_size_);
  }

  // serialize to protobufs.
  std::vector<ValueProto> result;
  for (size_t idx = 0; idx < world_size_; idx++) {
    result.push_back(hal::makeValue(shares[idx], dtype).toProto());
  }
  return result;
}

NdArrayRef IoAccessor::combineShares(const std::vector<ValueProto> &protos,
                                     PtType pt_type) {
  PPU_ENFORCE(!protos.empty());

  // deserialize protobuf to values
  std::vector<hal::Value> vals;
  vals.reserve(protos.size());
  for (const auto &proto : protos) {
    vals.push_back(hal::Value::fromProto(proto));
  }

  // TODO: enforce all same vtype|dtype

  const ValueTy *type = vals.front().eltype().as<ValueTy>();
  // reconstruct protocol dependent shares to ring.
  NdArrayRef encoded;
  {
    std::vector<NdArrayRef> shares;
    shares.reserve(vals.size());
    for (const auto &val : vals) {
      shares.push_back(val.as(type->mpc_type()));
    }
    encoded = base_io_->reconstruct(shares);
  }

  const Type to_type = makePtType(pt_type);
  const size_t fxp_bits = FxpFractionalBits(config_);
  const DataType dtype = type->dtype();
  PPU_ENFORCE(fxp_bits != 0, "fxp should never be zero, please check default");
  return decodeFromRing(encoded, to_type, fxp_bits, dtype);
}

} // namespace ppu::device
