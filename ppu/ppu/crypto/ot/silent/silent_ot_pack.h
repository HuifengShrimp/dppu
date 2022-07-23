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

#include "cheetah_io_channel.h"
#include "silent_ot.h"

#define PRE_OT_DATA_REG_SEND_FILE_ALICE "pre_ot_data_reg_send_alice"
#define PRE_OT_DATA_REG_SEND_FILE_BOB "pre_ot_data_reg_send_bob"
#define PRE_OT_DATA_REG_RECV_FILE_ALICE "pre_ot_data_reg_recv_alice"
#define PRE_OT_DATA_REG_RECV_FILE_BOB "pre_ot_data_reg_recv_bob"

#define KKOT_TYPES 8

namespace ppu {
typedef CheetahIo IO;

class SilentOTPack {
 public:
  int party_;
  IO *io_;
  IO *ios_[1];
  SilentOT *silent_ot_;
  SilentOT *silent_ot_reversed_;

  SilentOTN *kkot_[KKOT_TYPES];

  SilentOTPack(int party, IO *io);
  ~SilentOTPack();
};

}  // namespace ppu
