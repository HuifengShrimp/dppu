
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


import unittest

import numpy as np
import numpy.testing as npt

import ppu.binding as pyppu
from ppu.binding.test_util import Simulator


class UnitTests(unittest.TestCase):
    def test_no_io(self):
        wsize = 3
        config = pyppu.RuntimeConfig(protocol=pyppu.ProtocolKind.SEMI2K,
                                     field=pyppu.FieldType.FM128,
                                     fxp_fraction_bits=18)

        sim = Simulator(wsize, config)

        x = np.random.randint(10, size=(2, 2))

        code = """
func @main(%arg0: tensor<2x2x!pphlo.sint>) -> (tensor<2x2x!pphlo.sint>) {
    %0 = "pphlo.constant"() {value = dense<[[1,2],[3,4]]> : tensor<2x2xi64>} : () -> tensor<2x2x!pphlo.pint>
    %1 = "pphlo.add"(%arg0, %0) : (tensor<2x2x!pphlo.sint>, tensor<2x2x!pphlo.pint>) -> tensor<2x2x!pphlo.sint>
    "pphlo.dbg_print"(%1) : (tensor<2x2x!pphlo.sint>) -> ()
    return %1 : tensor<2x2x!pphlo.sint>
}"""
        sim(code, x)

    def test_raise(self):
        wsize = 3
        config = pyppu.RuntimeConfig(protocol=pyppu.ProtocolKind.SEMI2K,
                                     field=pyppu.FieldType.FM128,
                                     fxp_fraction_bits=18)

        sim = Simulator(wsize, config)

        x = np.random.randint(10, size=(2, 3))
        y = np.random.randint(10, size=(12, 13))

        # Give some insane ir
        code = """
func @main(%arg0: tensor<2x3x!pphlo.sint>, %arg1: tensor<12x13x!pphlo.sint>) -> (tensor<2x2x!pphlo.sint>) {
    %0 = "pphlo.dot"(%arg0, %arg1) : (tensor<2x3x!pphlo.sint>, tensor<12x13x!pphlo.sint>) -> tensor<2x2x!pphlo.sint>
    return %0 : tensor<2x2x!pphlo.sint>
}"""

        with self.assertRaisesRegex(RuntimeError, "stacktrace:.*"):
            sim(code, x, y)


if __name__ == '__main__':
    unittest.main()
