
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

import examples.python.distributed as distr
import examples.python.utils.fe_utils as fe

from examples.python.distributed import P1, P2, PPU, config
from examples.python.tests.test_utils import TwoPartyTestCase


class TestTensorFlow(TwoPartyTestCase):
    def test_selu(self):
        distr.init(self.zconf)

        @distr.device(PPU)
        @fe.tf2ppu
        def selu(x, alpha=1.67, lmbda=1.05):
            import sys
            import tensorflow as tf
            return lmbda * tf.where(x > 0, x, alpha * tf.exp(x) - alpha)

        x = np.random.normal(size=(3, 3)).astype(np.float32)
        x_ = distr.put(PPU, x)
        y = selu(x_)
        y2 = selu(x)

        self.assertTrue(
            np.allclose(distr.get(y), distr.get(y2), rtol=0.001, atol=0.00001))


if __name__ == '__main__':
    unittest.main()


class TestJax(TwoPartyTestCase):
    def test_selu(self):
        distr.init(self.zconf)

        @distr.device(PPU)
        @fe.jax2ppu
        def selu(x, alpha=1.67, lmbda=1.05):
            import jax.numpy as jnp
            return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

        x = np.random.normal(size=(3, 3)).astype(np.float32)
        x_ = distr.put(PPU, x)
        y = selu(x_)
        y2 = selu(x)

        self.assertTrue(
            np.allclose(distr.get(y), distr.get(y2), rtol=0.001, atol=0.00001))


if __name__ == '__main__':
    unittest.main()
