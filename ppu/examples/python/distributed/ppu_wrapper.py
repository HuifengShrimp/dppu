
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


from abc import abstractmethod
from collections.abc import Callable
from pkgutil import ImpImporter
from typing import Any, Callable

import dill
import numpy as np
import jax
from jax import numpy as jnp
from ppu import ppu_pb2

import examples.python.distributed as distr
from examples.python.distributed import core_pb2
from examples.python.distributed.core import DeviceObject


def convert_shape_pb_to_int_tuples(x: ppu_pb2.ShapeProto):
    return tuple(list(x.dims))


# NOTE(junfeng): current type_data is like "Value<DT_FXP,Share<FM128,A>>", maybe changed in future.
def convert_data_type_to_numpy_dtype(x: str):
    if x.startswith("Value<DT_FXP,"):
        return np.dtype("float32")
    else:
        return np.dtype("int64")


def convert_to_np_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, jnp.DeviceArray):
        return np.array(x)

    # duck type: tf.Tensor
    try:
        return x.numpy()
    except Exception:
        pass

    raise ValueError(
        "x is not np.ndarray or could't be converted to np.ndarray.")


class PpuFunction:
    @abstractmethod
    def jit(self, *args, **kwargs) -> str:
        """ Just-in-time compile to XLA hlo text
        """
        pass

    def mock_args(self, obj, convert_method: Callable):
        flat_objs, obj_tree = jax.tree_util.tree_flatten(obj)
        flat_res = []

        for flat_obj in flat_objs:
            if isinstance(flat_obj, DeviceObject):
                device_kind = distr.device(flat_obj.zrank).get_dev().kind()
                if device_kind == core_pb2.DeviceKind.PYRT:

                    # what if obj not a array ? like tuple(array, any)?
                    def fetch_meta(obj):
                        return obj.shape, obj.dtype

                    do = distr.device(flat_obj.zrank)(fetch_meta)(flat_obj)
                    shape, dtype = distr.get(do)

                else:
                    # hack
                    ppu_node_id_0 = (distr.device(
                        flat_obj.zrank).get_dev().node_ids())[0]

                    ppu_node_0_py_device_rank = distr.g_ctx.get_py_dev_rank_from_node(
                        ppu_node_id_0)

                    assert ppu_node_0_py_device_rank

                    def wrapper(server, name):
                        from ppu import ppu_pb2

                        shape = convert_shape_pb_to_int_tuples(
                            server.zsymbols[name].shape)

                        dtype = convert_data_type_to_numpy_dtype(
                            server.zsymbols[name].type_data)

                        return shape, dtype

                    routine = dill.dumps((wrapper, [flat_obj.zname], {}),
                                         recurse=True)
                    res = (distr.device(
                        ppu_node_0_py_device_rank).get_dev()._submit(
                            routine, "PY-GET:{}".format(flat_obj.zname)))
                    shape, dtype = dill.loads(res)
                flat_res.append(convert_method(shape, dtype))
            else:
                flat_res.append(obj)

        return jax.tree_util.tree_unflatten(obj_tree, flat_res)
