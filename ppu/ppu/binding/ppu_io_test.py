
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
from absl.testing import absltest, parameterized

import ppu.binding as pyppu


@parameterized.product(wsize=(2, 3, 5),
                       prot=(pyppu.ProtocolKind.REF2K,
                             pyppu.ProtocolKind.SEMI2K,
                             pyppu.ProtocolKind.ABY3),
                       field=(pyppu.FieldType.FM32, pyppu.FieldType.FM64,
                              pyppu.FieldType.FM128))
class UnitTests(parameterized.TestCase):
    def test_io(self, wsize, prot, field):
        if prot == pyppu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = pyppu.RuntimeConfig(protocol=prot,
                                     field=field,
                                     fxp_fraction_bits=18)
        io = pyppu.Io(wsize, config)

        # SINT
        x = np.random.randint(10, size=(3, 4, 5))

        xs = io.make_shares(x, pyppu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=(3, 4, 5)))
        y = io.reconstruct(xs)

        npt.assert_equal(x, y)

        # SFXP
        x = np.random.rand(3, 4, 5)

        xs = io.make_shares(x, pyppu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=(3, 4, 5)))
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        x = np.random.rand(3, 4, 5)

        xs = io.make_shares(x, pyppu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=(3, 4, 5)))
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_io_strides(self, wsize, prot, field):
        if prot == pyppu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = pyppu.RuntimeConfig(protocol=prot,
                                     field=field,
                                     fxp_fraction_bits=18)
        io = pyppu.Io(wsize, config)

        # SINT
        x = np.random.randint(10, size=(6, 7, 8))
        x = x[0:5:2, 0:7:2, 0:8:2]

        xs = io.make_shares(x, pyppu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=(3, 4, 4)))
        y = io.reconstruct(xs)

        npt.assert_equal(x, y)

        # SFXP
        x = np.random.rand(6, 7, 8)
        x = x[0:5:2, 0:7:2, 0:8:2]

        xs = io.make_shares(x, pyppu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=(3, 4, 4)))
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        x = np.random.rand(6, 7, 8)
        x = x[0:5:2, 0:7:2, 0:8:2]

        xs = io.make_shares(x, pyppu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=(3, 4, 4)))
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_io_scalar(self, wsize, prot, field):
        if prot == pyppu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = pyppu.RuntimeConfig(protocol=prot,
                                     field=field,
                                     fxp_fraction_bits=18)
        io = pyppu.Io(wsize, config)

        # SINT
        x = np.random.randint(10, size=())

        xs = io.make_shares(x, pyppu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=()))
        y = io.reconstruct(xs)

        npt.assert_equal(x, y)

        # SFXP
        x = np.random.random(size=())

        xs = io.make_shares(x, pyppu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=()))
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        x = np.random.random(size=())

        xs = io.make_shares(x, pyppu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(xs[0].shape, pyppu.ShapeProto(dims=()))
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)


if __name__ == '__main__':
    unittest.main()
