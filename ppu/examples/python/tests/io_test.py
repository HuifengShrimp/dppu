
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

from examples.python.tests.test_utils import TwoPartyTestCase

import examples.python.distributed as distr
from examples.python.distributed import P1, P2, PPU


class TestBasicIo(TwoPartyTestCase):
    def test_py_device(self):
        distr.init(self.zconf)

        x = np.ones((3, 3))
        # put to P1
        x_p1 = distr.put(P1, x)
        self.assertTrue(
            isinstance(x_p1, distr.core.DeviceObject) and x_p1.zrank == P1)
        # let P1 to put it to P2
        x_p2 = distr.put(P2, x_p1)
        self.assertTrue(
            isinstance(x_p2, distr.core.DeviceObject) and x_p2.zrank == P2)
        # get from P2
        self.assertTrue(np.allclose(x, distr.get(x_p2)))

        # make a variable on P1
        @distr.device(P1)
        def r1():
            # pitfall, worker should manually import dependency
            # import numpy as np
            return np.random.rand(1, 1)

        # double a variable on P2
        @distr.device(P2)
        def double(x):
            return x * 2

        # make 10 variables on P1, then double it on P2
        for _ in range(10):
            x = r1()
            y = double(x)

            self.assertTrue(np.allclose(distr.get(x) * 2, distr.get(y)))

    def test_ppu_device(self):
        distr.init(self.zconf)

        def rd():
            # pitfall, worker should manually import dependency
            import tensorflow as tf
            return tf.zeros((3, 4))

        x = distr.device(P1)(rd)()
        x_ = distr.put(PPU, x)
        self.assertTrue(np.allclose(distr.get(x), distr.get(x_)))


if __name__ == '__main__':
    unittest.main()
