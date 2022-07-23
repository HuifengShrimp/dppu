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


// TODO(jint) move me to somewhere else.

#pragma once

#include <string_view>

namespace ppu::link {

// Link trace is use to track inter-party communication.
//
// Note: MPC programs are communication-intensive, so trace communication will
// cause severe performance degradation and take a huge amount of disk space.
struct TraceOptions {
  bool enable = false;

  // 300M, normally, comm contents are large.
  size_t max_log_file_size = 500 * 1024 * 1024;

  //
  size_t max_log_file_count = 3;
};

void SetupTrace(TraceOptions opts = TraceOptions());

void TraceLog(std::string_view event, std::string_view tag,
              std::string_view value);

}  // namespace ppu::link
