
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


# Start two node servers.
# > bazel run //examples/python/distributed:daemon -- up
#
# Run this example script.
# > bazel run //examples/python/e2e:millionare

import numpy as np

import examples.python.distributed as distr
import examples.python.utils.fe_utils as fe
from examples.python.distributed import P1, P2, PPU

import argparse

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = distr.config.parse_json(file.read())
    distr.init(conf)


# make a variable on P1, it's known only for P1.
@distr.device(P1)
def r1():
    # pitfall, worker should manually import dependency
    import numpy as np
    np.random.seed(0)
    return np.random.rand(1, 1)


# make a variable on P2, it's known only for P2.
@distr.device(P2)
def r2():
    import numpy as np
    np.random.seed(1)
    return np.random.rand(1, 1)


x = r1()
assert isinstance(x, distr.core.DeviceObject)
assert x.zrank == P1
y = r2()
assert isinstance(y, distr.core.DeviceObject) and y.zrank == P2


@distr.device(PPU)
@fe.jax2ppu
def jax_max(x, y):
    import jax.numpy as jnp
    # return jnp.maximum(x, y)
    return jnp.maximum(x, y), [x, y], (x, y)


@distr.device(PPU)
@fe.tf2ppu
def tf_max(x, y):
    import tensorflow as tf

    # return jnp.maximum(x, y)
    return tf.math.maximum(x, y), [x, y], (x, y)


# x & y will be automatically send to PPU (as secret shares)
# z will be evaluated as a PPU funciton.
z, list_, tuple_ = tf_max(x, y)

print(distr.get(z))
print(distr.get(list_))
print(distr.get(tuple_))

assert isinstance(z, distr.core.DeviceObject) and z.zrank == PPU

assert np.allclose(distr.get(z),
                   np.maximum(distr.get(x), distr.get(y)),
                   rtol=0.00001)
