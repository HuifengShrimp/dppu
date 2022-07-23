
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


from collections.abc import Callable
from typing import Any, Union

import dill
import jax
import grpc

from examples.python.distributed import core_pb2, core_pb2_grpc
from examples.python.distributed.core import DeviceClient, DeviceObject


class PyDeviceClient(DeviceClient):
    def __init__(self, zctx, zrank: int, node_addr: str):
        super().__init__(zctx, zrank)

        self.stub = core_pb2_grpc.NodeServiceStub(
            grpc.insecure_channel(node_addr))

    def kind(self):
        return core_pb2.DeviceKind.PYRT

    def _submit(self, task: str, tag: str) -> None:
        req = core_pb2.EvalRequest(from_node_id=self.zctx.host_node_id(),
                                   tag=tag,
                                   task=task)
        rsp = self.stub.Eval(req)
        if rsp.HasField("exception"):
            raise RuntimeError(dill.loads(rsp.exception))
        return rsp.value

    def call(self, fn: Callable, *args, **kwargs) -> DeviceObject:
        tmp_params = []

        # A notable optimization is, if a parameter is a pyobject, it does not have
        # be transfer to the device via the put method, we can directly carry it with
        # the function.
        def transfer(obj: Any) -> Union[DeviceObject, Any]:
            """
            1. if the object is a python object, do nothing since it will be packed with funciton.
            2. if the object is a DeviceObject
              - if it resides on target device, do nothing
              - if it not resides on the target device, make a tmp variable copy
            """
            flat_objs, obj_tree = jax.tree_util.tree_flatten(obj)
            flat_res = []
            for flat_obj in flat_objs:
                if isinstance(flat_obj,
                              DeviceObject) and flat_obj.zrank != self.zrank:
                    # copy to the device.
                    param = self.zctx.transfer(flat_obj, self.zrank)
                    tmp_params.append(param.zname)
                    flat_res.append(param)
                else:
                    flat_res.append(flat_obj)

            return jax.tree_util.tree_unflatten(obj_tree, flat_res)

        args = [transfer(arg) for arg in args]
        kwargs = {k: transfer(v) for k, v in kwargs.items()}

        res_name = self.zctx.new_name()
        self_rank = self.zrank

        def wrapper(server, *args, **kwargs):
            def deref(obj):
                return server.zsymbols[obj.zname] if isinstance(
                    obj, DeviceObject) else obj

            args = [deref(arg) for arg in args]
            kwargs = {k: deref(v) for k, v in kwargs.items()}

            import jax
            # fire the call
            res = fn(*args, **kwargs)
            output_flat, output_tree = jax.tree_util.tree_flatten(res)
            output_names = []
            # instead of return result to caller, we save it on server's
            # symbol table, and make a reference on the client.
            for i in range(len(output_flat)):
                flat_res_name = '{}-{}'.format(res_name, i)
                server.zsymbols[flat_res_name] = output_flat[i]
                output_names.append(flat_res_name)

            # FIXME(jint), PyRt may have DeviceObject too, now it's a
            # classic reference counting GC problem.
            for zname in tmp_params:
                del server.zsymbols[zname]

            res = [DeviceObject(self_rank, n) for n in output_names]
            return jax.tree_util.tree_unflatten(output_tree, res)

        # serialize the routine.
        routine = dill.dumps((wrapper, args, kwargs), recurse=True)

        # doit
        value = self._submit(routine, 'PY-CALL:{}'.format(res_name))
        return dill.loads(value)

    def put(self, obj: Any, name: str) -> DeviceObject:
        assert not isinstance(obj, DeviceObject)

        def wrapper(server, name: str, value):
            server.zsymbols[name] = value

        routine = dill.dumps((wrapper, [name, obj], {}), recurse=True)
        self._submit(routine, 'PY-PUT:{}'.format(name))
        return DeviceObject(self.zrank, name)

    def get(self, obj: DeviceObject) -> Any:
        assert isinstance(obj, DeviceObject)
        assert obj.zrank == self.zrank

        def wrapper(server, name: str):
            return server.zsymbols[name]

        # doit
        routine = dill.dumps((wrapper, [obj.zname], {}), recurse=True)
        res = self._submit(routine, 'PY-GET:{}'.format(obj.zname))

        return dill.loads(res)
