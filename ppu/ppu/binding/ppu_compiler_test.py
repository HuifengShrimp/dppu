
# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import unittest

import numpy as np
import numpy.testing as npt

import ppu.binding as pyppu


class UnitTests(unittest.TestCase):
    def test_compile_pb(self):
        # Load data from file
        xla = os.path.join(os.getcwd(), "ppu", "binding", "data",
                           "hlo_example.hlo.pb")
        with open(xla, "rb") as f:
            # Build a compile proto
            proto = pyppu.IrProto()
            proto.ir_type = pyppu.IrType.IR_XLA_HLO
            proto.meta.inputs.append(pyppu.Visibility.VIS_SECRET)
            proto.code = f.read()

            # compile
            result = pyppu.compile(proto)

            # inspect compiled result
            self.assertEqual(result.ir_type, pyppu.IrType.IR_MLIR_PPU)

            ir = result.code.decode("utf-8")
            self.assertIn("@main", ir)
            self.assertIn("pphlo", ir)


if __name__ == '__main__':
    unittest.main()
