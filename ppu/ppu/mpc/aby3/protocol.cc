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


#include "ppu/mpc/aby3/protocol.h"

#include "ppu/mpc/abkernels.h"
#include "ppu/mpc/aby3/arithmetic.h"
#include "ppu/mpc/aby3/boolean.h"
#include "ppu/mpc/aby3/conversion.h"
#include "ppu/mpc/aby3/defs.h"
#include "ppu/mpc/aby3/type.h"
#include "ppu/mpc/base2k/public.h"
#include "ppu/mpc/object.h"
#include "ppu/mpc/prg_state.h"
#include "ppu/mpc/util/communicator.h"

namespace ppu::mpc {

std::unique_ptr<Object> makeAby3Protocol(
    const std::shared_ptr<link::Context>& lctx) {
  aby3::registerTypes();

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
  obj->addState<aby3::Aby3State>(lctx);
  obj->regKernel<aby3::P2A>();
  obj->regKernel<aby3::A2P>();
  obj->regKernel<aby3::NegA>();
  obj->regKernel<aby3::AddAP>();
  obj->regKernel<aby3::AddAA>();
  obj->regKernel<aby3::MulAP>();
  obj->regKernel<aby3::MulAA>();
  obj->regKernel<aby3::MatMulAP>();
  obj->regKernel<aby3::MatMulAA>();
  obj->regKernel<aby3::TruncPrA>();

  obj->regKernel<aby3::B2P>();
  obj->regKernel<aby3::P2B>();
  obj->regKernel<aby3::AddBB>();
  obj->regKernel<aby3::A2B>();
  obj->regKernel<aby3::B2A>();
  obj->regKernel<aby3::AndBP>();
  obj->regKernel<aby3::AndBB>();
  obj->regKernel<aby3::XorBP>();
  obj->regKernel<aby3::XorBB>();
  obj->regKernel<aby3::LShiftB>();
  obj->regKernel<aby3::RShiftB>();
  obj->regKernel<aby3::ARShiftB>();
  obj->regKernel<aby3::ReverseBitsB>();

  return obj;
}

}  // namespace ppu::mpc
