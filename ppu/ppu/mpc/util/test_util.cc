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


#include "ppu/mpc/util/test_util.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"

#include "ppu/core/array_ref_util.h"
#include "ppu/mpc/util/simulate.h"

namespace ppu::mpc::test {

// Evaluate a function for a given world size.
void Eval(size_t world_size,
          std::function<void(std::shared_ptr<link::Context> lctx)> proc) {
  util::simulate(world_size, proc);
}

bool EqualsPP(FieldType field, const ArrayRef& px, const ArrayRef& py) {
  return DISPATCH_ALL_FIELDS(field, "EqualsPP", [&]() {
    const auto& x = xt_adapt<ring2k_t>(px);
    const auto& y = xt_adapt<ring2k_t>(py);
    return x == y;
  });
}

bool EqualsVP(Rank rank, Rank owner, FieldType field, const ArrayRef& vx,
              const ArrayRef& py) {
  if (vx.numel() != py.numel()) {
    return false;
  }
  return DISPATCH_ALL_FIELDS(field, "EqualsVP", [&]() {
    const auto& x = xt_adapt<ring2k_t>(vx);
    const auto& y = xt_adapt<ring2k_t>(py);
    if (rank == owner) {
      return x == y;
    }
    return true;
  });
}

bool SatisfyTruncateErrorPP(size_t world_size, FieldType field,
                            const ArrayRef& px, const ArrayRef& py) {
  return DISPATCH_ALL_FIELDS(field, "SatisfyTruncateErrorPP", [&]() {
    const auto x = xt_adapt<ring2k_t>(px);
    const auto y = xt_adapt<ring2k_t>(py);

    const int threshold = world_size > 2 ? 2 : 1;

    return xt::all(xt::abs(x - y) <= threshold);
  });
}

ArrayRef RandP(FieldType field, size_t size, std::mt19937::result_type seed,
               int min, int max) {
  return DISPATCH_ALL_FIELDS(field, "test::RandP", [&]() {
    static_assert(std::is_integral_v<ring2k_t>);

    std::mt19937 engine(seed);

    // FIXME(jint)
    auto ty = makeType<Ring2kPublTy>(field);
    return make_array(xt::random::randint<ring2k_t>({size}, min, max, engine),
                      ty);
  });
}

bool RingEqual(const ArrayRef& a, const ArrayRef& b, size_t abs_err) {
  PPU_ENFORCE(a.eltype() == b.eltype(), "type mismatch, a={}, b={}", a.eltype(),
              b.eltype());

  const auto field = a.eltype().as<Ring2k>()->field();

  return DISPATCH_ALL_FIELDS(field, "RingEqual", [&]() {
    using scalar_t = Ring2kTrait<_kField>::scalar_t;

    auto err = xt_adapt<scalar_t>(a) - xt_adapt<scalar_t>(b);
    return xt::all(xt::abs(err) <= abs_err);
  });
}

bool VerifyCost(Kernel* kernel, std::string_view name, FieldType field,
                size_t numel, size_t npc, const Communicator::Stats& cost) {
  if (kernel->kind() == Kernel::Kind::kDynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  if (comm->eval(field, npc) * numel != cost.comm * kBitsPerBytes) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               comm->eval(field, npc) * numel, cost.comm * kBitsPerBytes);
    succeed = false;
  }
  if (latency->eval(field, npc) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(field, npc), cost.latency);
    succeed = false;
  }

  return succeed;
}

}  // namespace ppu::mpc::test
