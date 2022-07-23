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
#include <string>

#include "ppu/mpc/kernel.h"

namespace ppu::mpc {

class Interface {
 protected:
  Object* obj_;

 public:
  explicit Interface(Object* obj) : obj_(obj) {}
  virtual ~Interface() = default;
};

class State {
 public:
  virtual ~State() = default;
};

// A (kernel) dynamic object dispatch a function to a kernel at runtime.
//
// Class that inherit from this class could do `dynamic binding`.
class Object {
  std::map<std::string_view, std::unique_ptr<Kernel>> kernels_;
  std::map<std::string_view, std::unique_ptr<Interface>> interfaces_;
  std::map<std::string_view, std::unique_ptr<State>> states_;

 public:
  virtual ~Object() = default;

  void regKernel(std::string_view name, std::unique_ptr<Kernel> kernel);

  template <typename KernelT>
  void regKernel() {
    regKernel(KernelT::kName, std::make_unique<KernelT>());
  }

  template <typename KernelT>
  void regKernel(std::string_view name) {
    return regKernel(name, std::make_unique<KernelT>());
  }

  Kernel* getKernel(std::string_view name);
  bool hasKernel(std::string_view name) const;

  void addState(std::string_view name, std::unique_ptr<State> state) {
    const auto& itr = states_.find(name);
    PPU_ENFORCE(itr == states_.end(), "state={} already exist", name);
    states_.emplace(name, std::move(state));
  }

  template <typename StateT, typename... Args>
  void addState(Args&&... args) {
    addState(StateT::kName,
             std::make_unique<StateT>(std::forward<Args>(args)...));
  }

  template <typename StateT>
  StateT* getState() {
    const auto& itr = states_.find(StateT::kName);
    PPU_ENFORCE(itr != states_.end(), "state={} not found", StateT::kName);
    return dynamic_cast<StateT*>(itr->second.get());
  }

  template <typename InterfaceT>
  std::unique_ptr<InterfaceT> getInterface() {
    // TODO(jint) interface type restriction.
    return std::make_unique<InterfaceT>(this);
  }

  //
  std::vector<std::string_view> getKernelNames() const {
    std::vector<std::string_view> names;
    for (auto const& itr : kernels_) {
      names.push_back(itr.first);
    }
    return names;
  }

  ArrayRef callImpl(Kernel* kernel, KernelEvalContext* ctx) {
    kernel->evaluate(ctx);
    return ctx->stealOutput();
  }

  template <typename First, typename... Args>
  ArrayRef callImpl(Kernel* kernel, KernelEvalContext* ctx, First&& head,
                    Args&&... tail) {
    ctx->bindParam(std::forward<First>(head));
    if constexpr (sizeof...(Args) == 0) {
      return callImpl(kernel, ctx);
    } else {
      return callImpl(kernel, ctx, std::forward<Args>(tail)...);
    }
  }

  template <typename... Args>
  ArrayRef call(std::string_view name, Args&&... args) {
    Kernel* kernel = getKernel(name);
    KernelEvalContext ctx(this);
    return callImpl(kernel, &ctx, std::forward<Args>(args)...);
  }
};

}  // namespace ppu::mpc
