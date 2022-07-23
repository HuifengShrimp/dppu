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


#include "ppu/mpc/fss/protocol.h"

#include "ppu/mpc/abkernels.h"
#include "ppu/mpc/base2k/public.h"
#include "ppu/mpc/object.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/fss/arithmetic.h"
#include "ppu/mpc/fss/boolean.h"
#include "ppu/mpc/fss/conversion.h"
#include "ppu/mpc/fss/object.h"
#include "ppu/mpc/semi2k/object.h"
#include "ppu/mpc/semi2k/type.h"

namespace ppu::mpc {

std::unique_ptr<Object> makeFSSProtocol(
    const std::shared_ptr<link::Context>& lctx) {
  semi2k::registerTypes();

  auto obj = std::make_unique<Object>();

  // add communicator
  obj->addState<Communicator>(lctx);

  // register random states & kernels.
  obj->addState<PrgState>(lctx);
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
  obj->addState<ABState>();
  obj->regKernel<P2S>();
  obj->regKernel<S2P>();
  obj->regKernel<NegS>();
  obj->regKernel<AddSP>();
  obj->regKernel<AddSS>();
  obj->regKernel<MulSP>();
  obj->regKernel<MulSS>();
  obj->regKernel<MatMulSP>();
  obj->regKernel<MatMulSS>();
  obj->regKernel<AndSP>();
  obj->regKernel<AndSS>();
  obj->regKernel<XorSP>();
  obj->regKernel<XorSS>();
  obj->regKernel<EqzS>();
  obj->regKernel<LShiftS>();
  obj->regKernel<RShiftS>();
  obj->regKernel<ARShiftS>();
  obj->regKernel<TruncPrS>();
  obj->regKernel<ReverseBitsS>();
  obj->regKernel<MsbS>();

  // register arithmetic & binary kernels
  obj->addState<Semi2kState>(lctx);
  obj->addState<FSSState>(lctx);

  obj->regKernel<fss::ZeroA>();
  obj->regKernel<fss::P2A>();
  obj->regKernel<fss::A2P>();
  obj->regKernel<fss::NegA>();
  obj->regKernel<fss::AddAP>();
  obj->regKernel<fss::AddAA>();
  obj->regKernel<fss::MulAP>();
  obj->regKernel<fss::MulAA>();
  obj->regKernel<fss::MatMulAP>();
  obj->regKernel<fss::MatMulAA>();
  obj->regKernel<fss::LogReg>();
  obj->regKernel<semi2k::TruncPrA>();

  obj->regKernel<fss::ZeroB>();
  obj->regKernel<fss::B2P>();
  obj->regKernel<fss::P2B>();
  obj->regKernel<fss::AndBP>();
  obj->regKernel<fss::AndBB>();
  obj->regKernel<fss::AddBB>();
  obj->regKernel<fss::A2B>();
  obj->regKernel<fss::B2A>();
  obj->regKernel<fss::XorBP>();
  obj->regKernel<fss::XorBB>();
  obj->regKernel<fss::LShiftB>();
  obj->regKernel<fss::RShiftB>();
  obj->regKernel<fss::ARShiftB>();
  obj->regKernel<fss::ReverseBitsB>();

  return obj;
}

}  // namespace ppu::mpc
