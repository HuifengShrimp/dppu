
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import dill
import grpc
import jax
import tensorflow as tf

from examples.python.distributed import core_pb2, core_pb2_grpc
from examples.python.distributed.core import DeviceClient, DeviceObject
from examples.python.distributed.ppu_wrapper import convert_to_np_array
from ppu import ppu_pb2
from ppu.binding import Io, compile
from tensorflow.python.ops import resource_variable_ops


class PpuDeviceClient(DeviceClient):
    def __init__(self, zctx, zrank: int, addrs: List[str],
                 ddesc: core_pb2.DeviceDesc):
        super().__init__(zctx, zrank)

        self.ddesc = ddesc
        num_parties = len(addrs)
        self.io = Io(num_parties, self.ddesc.ppu_device.runtime_config)

        self.stubs = [
            core_pb2_grpc.NodeServiceStub(grpc.insecure_channel(addr))
            for addr in addrs
        ]

        self.ppu_node_ids = self.ddesc.ppu_device.node_ids
        self._threadpool = ThreadPoolExecutor(max_workers=os.cpu_count())

    def node_ids(self):
        return list(self.ppu_node_ids)

    def kind(self):
        return core_pb2.DeviceKind.PPU

    def _submit_bcast(self, task: str, tag: str) -> None:
        return self._submit_scatter([task] * len(self.stubs), tag)

    def _submit_thread(self, stub, tag, task):
        req = core_pb2.EvalRequest(from_node_id=self.zctx.host_node_id(),
                                   tag=tag,
                                   task=task)
        rsp = stub.Eval(req)
        if rsp.HasField("exception"):
            raise RuntimeError(rsp.exception)
        return rsp.value

    def _submit_scatter(self, tasks: List[str], tag: str) -> None:
        assert len(tasks) == len(self.stubs)

        rets = []
        futures = []
        for stub, task in zip(self.stubs, tasks):
            futures.append(
                self._threadpool.submit(self._submit_thread, stub, tag, task))

        for f in futures:
            rets.append(f.result())

        return rets

    def call(self, fn: Callable, *args, **kwargs) -> DeviceObject:
        # just-in-time compile fn to device compiable format
        hint, jit_result = fn.jit(*args, **kwargs)

        if hint == "JAX":
            (dfn, pytree) = jit_result
            output_flat, output_tree = jax.tree_util.tree_flatten(pytree)
            output_cnt = len(output_flat)
        elif hint == "TF":
            (dfn, cf) = jit_result
            output_cnt = len(cf.outputs)

        # transfer object to the device worker.
        # TODO(jint) batch me
        args = [self.zctx.transfer(arg, self.zrank) for arg in args]
        kwargs = {
            k: self.zctx.transfer(v, self.zrank)
            for k, v in kwargs.items()
        }

        if hint == "JAX":
            captures = []
        elif hint == "TF":
            captures = []
            for i in range(len(cf.captured_inputs)):
                placeholder_tensor = cf.graph.external_captures[i]
                spec = resource_variable_ops.get_eager_safe_handle_data(
                    placeholder_tensor)
                captures.append(
                    tf.raw_ops.ReadVariableOp(
                        resource=cf.captured_inputs[i],
                        dtype=spec.shape_and_type[0].dtype))
            captures = [
                self.zctx.transfer(capture, self.zrank) for capture in captures
            ]

        in_vis = [ppu_pb2.Visibility.VIS_SECRET] * (len(args) + len(kwargs))

        in_vis.extend([ppu_pb2.Visibility.VIS_SECRET] * len(captures))

        xla_meta = ppu_pb2.XlaMeta(inputs=in_vis)

        input_ir = ppu_pb2.IrProto(ir_type=ppu_pb2.IR_XLA_HLO,
                                   code=dfn,
                                   meta=xla_meta)

        output_ir = compile(input_ir)

        input_names = []

        def flat_input_name(args):
            for a in args:
                flat_a, _ = jax.tree_util.tree_flatten(a)
                input_names.extend([o.zname for o in flat_a])

        flat_input_name(args)
        flat_input_name(kwargs.values())
        flat_input_name(captures)

        executable = ppu_pb2.ExecutableProto()

        output_names = []

        for _ in range(output_cnt):
            name = self.zctx.new_name()
            output_names.append(name)

        executable.code = output_ir.code
        executable.input_names.extend(input_names)
        executable.output_names.extend(output_names)

        call_name = self.zctx.new_name()

        def wrapper(server, executable):
            for input_name in executable.input_names:
                server.rt.set_var(input_name, server.zsymbols[input_name])

            server.rt.run(executable)

            for output_name in executable.output_names:
                server.zsymbols[output_name] = server.rt.get_var(output_name)

            return None

        # serialize the routine.
        routine = dill.dumps((wrapper, [executable], {}), recurse=True)

        # doit

        self._submit_bcast(routine, "PPU-CALL:{}".format(call_name))

        res = [DeviceObject(self.zrank, name) for name in output_names]

        if hint == "JAX":
            return jax.tree_util.tree_unflatten(output_tree, res)
        elif hint == "TF":
            return tf.nest.pack_sequence_as(cf.structured_outputs, res)

    def put(self, obj: Any, name: str) -> DeviceObject:
        assert not isinstance(obj, DeviceObject)

        # TODO(jint) if we are in a DriverContext, we can put it as a public value.
        shares = self.io.make_shares(convert_to_np_array(obj),
                                     ppu_pb2.Visibility.VIS_SECRET)

        def wrapper(server, name: str, value):
            server.zsymbols[name] = value

        tasks = [
            dill.dumps((wrapper, [name, share], {}), recurse=True)
            for share in shares
        ]
        self._submit_scatter(tasks, "PPU-PUT:{}".format(name))

        return DeviceObject(self.zrank, name)

    def get(self, obj: DeviceObject) -> Any:
        assert isinstance(obj, DeviceObject)
        assert obj.zrank == self.zrank

        def wrapper(server, name: str):
            return server.zsymbols[name]

        task = dill.dumps((wrapper, [obj.zname], {}), recurse=True)
        rets = self._submit_bcast(task, "PPU-GET:{}".format(obj.zname))
        shares = list(map(dill.loads, rets))

        return self.io.reconstruct(shares)
