
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


import json

import ppu
from google.protobuf import json_format

from examples.python.distributed import core_pb2


def parse_json(config_json: str):
    return json_format.Parse(config_json, core_pb2.WorldDesc())


def find_node(config: core_pb2.WorldDesc, node_id: str):
    nodes = [node for node in config.nodes if node.id == node_id]
    if len(nodes) != 1:
        raise KeyError('Node id={} config not unique, found={}'.format(
            node_id, len(nodes)))
    return nodes[0]


def find_ppu_device_config(config: core_pb2.WorldDesc, node_id: str):
    _devices = filter(
        lambda device: device.kind == core_pb2.DeviceKind.PPU and node_id in
        device.ppu_device.node_ids, config.devices)

    device = next(_devices, None)

    assert next(
        _devices, None
    ) is None, f'node {node_id} belongs to multiple ppu devices, which is unsupported.'

    return device


def validate_config(config: core_pb2.WorldDesc):
    node_set = set()
    addr_set = set()
    for node in config.nodes:
        if node.id in node_set:
            raise KeyError(f'node id: {node.id} is duplicated.')
        node_set.add(node.id)

        if node.addr in addr_set:
            raise KeyError(f'addr : {node.addr} is duplicated.')
        addr_set.add(node.addr)

    device_rank_set = set()
    for device in config.devices:
        if device.rank in device_rank_set:
            raise KeyError(f'device rank: {device.rank} is duplicated,')
        device_rank_set.add(device.rank)

        if device.kind == core_pb2.DeviceKind.PYRT:
            if device.py_device.node_id not in node_set:
                raise KeyError(
                    f'invalid node id: {device.py_device.node_id} for device {device.rank}'
                )
        elif device.kind == core_pb2.DeviceKind.PPU:
            if len(device.ppu_device.node_ids) != len(
                    device.ppu_device.ppu_internal_addrs):
                raise ValueError(
                    f'# of node_ids doesn\'t match # of ppu_internal_addrs for device {device.rank}'
                )
            for id in device.ppu_device.node_ids:
                if id not in node_set:
                    raise KeyError(
                        f'invalid node id: {id} for device {device.rank}')

            for addr in device.ppu_device.ppu_internal_addrs:
                if addr in addr_set:
                    raise KeyError(f'ppu_internal_addr : {addr} is duplicated.')
                addr_set.add(addr)

            if device.ppu_device.runtime_config.protocol == ppu.ppu_pb2.ABY3 and len(
                    device.ppu_device.node_ids) != 3:
                raise KeyError(
                    f'The protocol of ppu device {device.rank} is ABY3 while not 3pc.'
                )
        else:
            raise KeyError(f'invalid device kind: device {device.rank}')
