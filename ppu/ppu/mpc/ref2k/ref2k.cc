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


#include "ppu/mpc/ref2k/ref2k.h"

#include "ppu/mpc/base2k/public.h"
#include "ppu/mpc/prg_state.h"

namespace ppu::mpc {

template <typename OutType>
class Identical : public UnaryKernel {
 public:
  static constexpr char kName[] = "Identical";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<OutType>(in.eltype().as<Ring2k>()->field()));
  }
};

template <typename OutType, typename KernelType>
class UnaryConvertKernel : public KernelType {
 public:
  static constexpr char kName[] = "UnaryConvert";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    auto field = in.eltype().as<Ring2k>()->field();
    auto ret = KernelType::proc(ctx, in.as(makeType<Ring2kPublTy>(field)));
    return ret.as(makeType<OutType>(field));
  }
};

template <typename OutType, typename KernelType>
class UnaryWithBitsConvertKernel : public KernelType {
 public:
  static constexpr char kName[] = "UnaryConvert";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    auto field = in.eltype().as<Ring2k>()->field();
    auto ret =
        KernelType::proc(ctx, in.as(makeType<Ring2kPublTy>(field)), bits);
    return ret.as(makeType<OutType>(field));
  }
};

class ReverseBitsRefS : public base2k::ReverseBitsP {
 public:
  static constexpr char kName[] = "ReverseBitsS";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t begin,
                size_t end) const override {
    auto field = in.eltype().as<Ring2k>()->field();
    auto ret = base2k::ReverseBitsP::proc(
        ctx, in.as(makeType<Ring2kPublTy>(field)), begin, end);
    return ret.as(makeType<Ref2kSecrTy>(field));
  }
};

template <typename OutType, typename KernelType>
class BinaryConvertKernel : public KernelType {
 public:
  static constexpr char kName[] = "BinaryConvertKernel";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    auto field = lhs.eltype().as<Ring2k>()->field();
    auto ret = KernelType::proc(ctx, lhs.as(makeType<Ring2kPublTy>(field)),
                                rhs.as(makeType<Ring2kPublTy>(field)));
    return ret.as(makeType<OutType>(field));
  }
};

class MatMulSKernel : public base2k::MatMulPP {
 public:
  static constexpr char kName[] = "MatMulSKernel";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, int64_t M, int64_t N,
                int64_t K) const override {
    auto field = lhs.eltype().as<Ring2k>()->field();
    auto ret =
        base2k::MatMulPP::proc(ctx, lhs.as(makeType<Ring2kPublTy>(field)),
                               rhs.as(makeType<Ring2kPublTy>(field)), M, N, K);
    return ret.as(makeType<Ref2kSecrTy>(field));
  }
};

std::unique_ptr<Object> makeRef2kProtocol(
    const std::shared_ptr<link::Context>& lctx) {
  auto obj = std::make_unique<Object>();

  // register random states & kernels.
  obj->addState<PrgState>();
  obj->regKernel<base2k::RandP>();

  // register public kernels.
  obj->regKernel<base2k::NegP>();
  obj->regKernel<base2k::EqzP>();
  obj->regKernel<base2k::AddPP>();
  obj->regKernel<base2k::MulPP>();
  obj->regKernel<base2k::MatMulPP>();
  obj->regKernel<base2k::AndPP>();
  obj->regKernel<base2k::XorPP>();
  obj->regKernel<base2k::LShiftP>();
  obj->regKernel<base2k::RShiftP>();
  obj->regKernel<base2k::ReverseBitsP>();
  obj->regKernel<base2k::ARShiftP>();
  obj->regKernel<base2k::MsbP>();

  // register compute kernels
  obj->regKernel<Identical<Ref2kSecrTy>>("P2S");
  obj->regKernel<Identical<Ring2kPublTy>>("S2P");
  obj->regKernel<UnaryConvertKernel<Ref2kSecrTy, base2k::NegP>>("NegS");
  obj->regKernel<UnaryConvertKernel<Ref2kSecrTy, base2k::EqzP>>("EqzS");
  obj->regKernel<UnaryWithBitsConvertKernel<Ref2kSecrTy, base2k::LShiftP>>(
      "LShiftS");
  obj->regKernel<UnaryWithBitsConvertKernel<Ref2kSecrTy, base2k::RShiftP>>(
      "RShiftS");
  obj->regKernel<UnaryWithBitsConvertKernel<Ref2kSecrTy, base2k::ARShiftP>>(
      "ARShiftS");
  obj->regKernel<UnaryWithBitsConvertKernel<Ref2kSecrTy, base2k::ARShiftP>>(
      "TruncPrS");

  obj->regKernel<ReverseBitsRefS>("ReverseBitsS");

  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::AddPP>>("AddSP");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::AddPP>>("AddSS");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::MulPP>>("MulSP");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::MulPP>>("MulSS");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::AndPP>>("AndSP");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::AndPP>>("AndSS");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::XorPP>>("XorSP");
  obj->regKernel<BinaryConvertKernel<Ref2kSecrTy, base2k::XorPP>>("XorSS");

  obj->regKernel<MatMulSKernel>("MatMulSP");
  obj->regKernel<MatMulSKernel>("MatMulSS");

  obj->regKernel<UnaryConvertKernel<Ref2kSecrTy, base2k::MsbP>>("MsbS");

  return obj;
}

std::vector<NdArrayRef> Ref2kIo::makeSecret(const NdArrayRef& raw) const {
  const auto field = raw.eltype().as<Ring2k>()->field();
  // directly view the data as secret.
  return std::vector<NdArrayRef>(world_size_,
                                 raw.as(makeType<Ref2kSecrTy>(field)));
}

NdArrayRef Ref2kIo::reconstructSecret(
    const std::vector<NdArrayRef>& shares) const {
  const auto field = shares[0].eltype().as<Ring2k>()->field();
  return shares[0].as(makeType<RingTy>(field));
}

std::unique_ptr<Ref2kIo> makeRef2kIo(size_t npc) {
  return std::make_unique<Ref2kIo>(npc);
}

}  // namespace ppu::mpc
