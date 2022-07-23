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

#include <string_view>

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "spdlog/spdlog.h"

#include "ppu/core/type_util.h"
#include "ppu/utils/scope_guard.h"

#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#include <memory>

inline std::string _demangle(const char* name) {
  int status = -4;  // some arbitrary value to eliminate the compiler warning

  // enable c++11 by passing the flag -std=c++11 to g++
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

  return (status == 0) ? res.get() : name;
}

#else

// does nothing if not g++
inline std::string _demangle(const char* name) { return name; }

#endif

template <class T>
std::string _type_name(const T& t) {
  return _demangle(typeid(t).name());
}

namespace ppu {

struct TraceInfo {
  bool enable = false;

  // FIXME: this is workaround for non-threadlocal tr_info.
  // if not atomic, depth maybe `negative` in multi-thread environment.
  std::atomic<size_t> depth = 0;

  std::string indent() const { return std::string(depth * 2, ' '); }
};

// FIXME(jint): 'thread_local' could not be used with brpc. See:
// https://github.com/apache/incubator-brpc/blob/master/docs/cn/thread_local.md
//
// Now trace only works for single thread application.
//
// extern thread_local TraceInfo tr_info;
extern TraceInfo tr_info;

}  // namespace ppu

#define PPU_TRACE_OP1(ctx, x) \
  SPDLOG_INFO("{}{}::{}({})", tr_info.indent(), _type_name(*ctx), __func__, x);

#define PPU_TRACE_OP2(ctx, x, y)                                      \
  SPDLOG_INFO("{}{}::{}({}, {})", tr_info.indent(), _type_name(*ctx), \
              __func__, x, y);

#define PPU_TRACE_OP3(ctx, x, y, z)                                       \
  SPDLOG_INFO("{}{}::{}({}, {}, {})", tr_info.indent(), _type_name(*ctx), \
              __func__, x, y, z);

#define PPU_TRACE_OP4(ctx, x, y, z, w)                                        \
  SPDLOG_INFO("{}{}::{}({}, {}, {}, {})", tr_info.indent(), _type_name(*ctx), \
              __func__, x, y, z, w);

// https://stackoverflow.com/questions/11761703/overloading-macro-on-number-of-arguments
//
// clang-format off
// PPU_TRACE_OP(ctx, x)        -> SELECT(ctx, x, OP3, OP2, OP1)       -> OP1
// PPU_TRACE_OP(ctx, x, y)     -> SELECT(ctx, x, y, OP3, OP2, OP1)    -> OP2
// PPU_TRACE_OP(ctx, x, y, z)  -> SELECT(ctx, x, y, z, OP3, OP2, OP1) -> OP3
//
// crap!!
// PPU_TRACE_OP(ctx, x, y, z, w)  -> SELECT(ctx, x, y, z, w, OP3, OP2, OP1) -> w
// clang-format on
#define SELECT(_CTX, _1, _2, _3, _4, NAME, ...) NAME
#define PPU_TRACE_OP(...)                                            \
  spdlog::set_pattern("%H:%M:%S TRACE: %v");                         \
  tr_info.depth++;                                                   \
  ON_SCOPE_EXIT([&] { tr_info.depth--; });                           \
  if (tr_info.enable) {                                              \
    SELECT(__VA_ARGS__, PPU_TRACE_OP4, PPU_TRACE_OP3, PPU_TRACE_OP2, \
           PPU_TRACE_OP1)                                            \
    (__VA_ARGS__)                                                    \
  }

namespace std {

// helper function to print indices.
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<size_t>& indices) {
  os << fmt::format("{{{}}}", fmt::join(indices, ","));
  return os;
}

}  // namespace std
