
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


import dill

from examples.python.distributed import config, core, core_pb2, ppu_device, py_device


class Context:
    def __init__(self, zconf: core_pb2.WorldDesc):
        config.validate_config(zconf)
        self.zconf = zconf
        self.devices: [core.DeviceClient] = []
        self._host_node_id = ""

        for ddesc in zconf.devices:
            if ddesc.kind == core_pb2.DeviceKind.PYRT:
                node = config.find_node(zconf, ddesc.py_device.node_id)
                dev = py_device.PyDeviceClient(self, ddesc.rank, node.addr)
            elif ddesc.kind == core_pb2.DeviceKind.PPU:
                assert len(ddesc.ppu_device.node_ids) >= 1
                node_addrs = [
                    config.find_node(zconf, node_id).addr
                    for node_id in ddesc.ppu_device.node_ids
                ]
                dev = ppu_device.PpuDeviceClient(self, ddesc.rank, node_addrs,
                                                 ddesc)
            else:
                raise ValueError("unsupported device kind={}", ddesc.kind)
            self.devices.append(dev)

    def get_dev(self, zrank: int) -> core.DeviceClient:
        matches = [dev for dev in self.devices if dev.zrank == zrank]
        if len(matches) != 1:
            raise ValueError("more than 1 ({}) found".format(len(matches)))
        return matches[0]

    def host_node_id(self):
        return self._host_node_id


class WorkerContext(Context):
    def __init__(self, zconf: core_pb2.WorldDesc, host_node_id):
        super().__init__(zconf)
        self._host_node_id = host_node_id


class DriverContext(Context):
    def __init__(self, zconf: core_pb2.WorldDesc):
        super().__init__(zconf)
        self._uuid = 0

    def new_name(self):
        self._uuid = self._uuid + 1
        return "V{}".format(self._uuid)

    def transfer(self, obj, dst_rank: int) -> core.DeviceObject:
        if not isinstance(obj, core.DeviceObject):
            nname = self.new_name()
            return self.get_dev(dst_rank).put(obj, nname)

        if obj.zrank == dst_rank:
            # already resides on this device.
            return obj

        src_dev = self.get_dev(obj.zrank)
        dst_dev = self.get_dev(dst_rank)

        # FIXME(jint) dont use internal methods like zsymbols & _submit
        if src_dev.kind() == core_pb2.DeviceKind.PYRT:
            # source device is a PYRT, ask it to put.
            def wrapper(server, dname, drank, obj):
                src_obj = server.zsymbols[obj.zname]
                dev = server.zctx.get_dev(drank)
                # fetch the object within server's context
                dev.put(src_obj, dname)

            dname = self.new_name()
            routine = dill.dumps((wrapper, [dname, dst_rank, obj], {}),
                                 recurse=True)
            src_dev._submit(routine, "PY-TRANS:{}".format(dname))
            return core.DeviceObject(dst_rank, dname)
        elif dst_dev.kind() == core_pb2.DeviceKind.PYRT:
            # dest device is PYRT, ask it to get.
            def wrapper(server, dname, obj):
                dev = server.zctx.get_dev(obj.zrank)
                # fetch the object within server's context
                server.zsymbols[dname] = dev.get(obj)

            dname = self.new_name()
            routine = dill.dumps((wrapper, [dname, obj], {}), recurse=True)
            dst_dev._submit(routine, "PY-TRANS:{}".format(dname))
            return core.DeviceObject(dst_rank, dname)
        else:
            raise RuntimeError("transfer from {} to {} not supported".format(
                src_dev.kind(), dst_dev.kind()))

    def get_py_dev_rank_from_node(self, node_id: str):
        for ddesc in self.zconf.devices:
            if (ddesc.kind == core_pb2.DeviceKind.PYRT
                    and ddesc.py_device.node_id == node_id):
                return ddesc.rank
        return None
