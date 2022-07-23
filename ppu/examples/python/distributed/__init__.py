
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


from typing import Any

import jax

from examples.python.distributed import context, core, core_pb2, config


def init(config: core_pb2.WorldDesc):
    global g_ctx
    g_ctx = context.DriverContext(config)


class DeviceFunction(object):
    def __init__(self, zrank: int, ctx=None):
        self.zrank = zrank
        self.ctx = ctx

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            ctx = self.ctx if self.ctx else g_ctx
            dev = ctx.get_dev(self.zrank)
            return dev.call(fn, *args, **kwargs)

        return wrapper

    def get_dev(self):
        ctx = self.ctx if self.ctx else g_ctx
        return ctx.get_dev(self.zrank)


device = DeviceFunction


def put(zrank: int, obj: Any, ctx=None):
    ctx = ctx if ctx else g_ctx
    dev = ctx.get_dev(zrank)

    value_flat, value_tree = jax.tree_util.tree_flatten(obj)
    value_leaves = []
    for _, value in enumerate(value_flat):
        if not isinstance(value, core.DeviceObject):
            value_leaves.append(dev.put(value, ctx.new_name()))
        else:
            if value.zrank == dev.zrank:
                value_leaves.append(value)
            else:
                value_leaves.append(ctx.transfer(obj, zrank))
    return jax.tree_util.tree_unflatten(value_tree, value_leaves)


def get(obj: Any, ctx=None):
    value_flat, value_tree = jax.tree_util.tree_flatten(obj)
    value_leaves = []
    for _, value in enumerate(value_flat):
        assert isinstance(value, core.DeviceObject)
        ctx = ctx if ctx else g_ctx
        dev = ctx.get_dev(value.zrank)
        value_leaves.append(dev.get(value))
    return jax.tree_util.tree_unflatten(value_tree, value_leaves)


# helper constant rank definitions.
PPU = 0
ALICE = 1
P1 = 1
BOB = 2
P2 = 2
CHARLIE = 3
P3 = 3
