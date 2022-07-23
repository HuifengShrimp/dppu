
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


import multiprocessing
import socketserver
import time
import unittest
import json

import ppu

import examples.python.distributed as distr
from examples.python.distributed.daemon import run_single_server


def find_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def create_2pc_config():
    port_local_0 = find_port()
    port_local_1 = find_port()
    port_ppu_interanl_0 = find_port()
    port_ppu_interanl_1 = find_port()
    COLOC_2PC = {
        'id':
        'COLOC_2PC',
        'nodes': [
            {
                'id': 'local:0',
                'addr': f'127.0.0.1:{port_local_0}'
            },
            {
                'id': 'local:1',
                'addr': f'127.0.0.1:{port_local_1}'
            },
        ],
        'devices': [
            {
                'kind': 'PPU',
                'rank': 0,
                'ppu_device': {
                    'node_ids': [
                        'local:0',
                        'local:1',
                    ],
                    'ppu_internal_addrs': [
                        f'127.0.0.1:{port_ppu_interanl_0}',
                        f'127.0.0.1:{port_ppu_interanl_1}',
                    ],
                    'runtime_config': {
                        'protocol': ppu.ppu_pb2.SEMI2K,
                        'field': ppu.ppu_pb2.FM128,
                        'sigmoid_mode': ppu.ppu_pb2.REAL,
                    }
                }
            },
            {
                'kind': 'PYRT',
                'rank': 1,
                'py_device': {
                    'node_id': 'local:0'
                }
            },
            {
                'kind': 'PYRT',
                'rank': 2,
                'py_device': {
                    'node_id': 'local:1'
                }
            },
        ]
    }

    return distr.config.parse_json(json.dumps(COLOC_2PC))


def create_3pc_config():
    port_local_0 = find_port()
    port_local_1 = find_port()
    port_local_2 = find_port()
    port_ppu_interanl_0 = find_port()
    port_ppu_interanl_1 = find_port()
    port_ppu_interanl_2 = find_port()
    COLOC_3PC = {
        'id':
        'COLOC_3PC',
        'nodes': [
            {
                'id': 'local:0',
                'addr': f'127.0.0.1:{port_local_0}'
            },
            {
                'id': 'local:1',
                'addr': f'127.0.0.1:{port_local_1}'
            },
            {
                'id': 'local:2',
                'addr': f'127.0.0.1:{port_local_2}'
            },
        ],
        'devices': [
            {
                'kind': 'PPU',
                'rank': 0,
                'ppu_device': {
                    'node_ids': [
                        'local:0',
                        'local:1',
                        'local:2',
                    ],
                    'ppu_internal_addrs': [
                        f'127.0.0.1:{port_ppu_interanl_0}',
                        f'127.0.0.1:{port_ppu_interanl_1}',
                        f'127.0.0.1:{port_ppu_interanl_2}',
                    ],
                    'runtime_config': {
                        'protocol': ppu.ppu_pb2.ABY3,
                        'field': ppu.ppu_pb2.FM128,
                        'sigmoid_mode': ppu.ppu_pb2.REAL,
                    }
                }
            },
            {
                'kind': 'PYRT',
                'rank': 1,
                'py_device': {
                    'node_id': 'local:0'
                }
            },
            {
                'kind': 'PYRT',
                'rank': 2,
                'py_device': {
                    'node_id': 'local:1'
                }
            },
        ]
    }

    return distr.config.parse_json(json.dumps(COLOC_3PC))


class TwoPartyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("TwoPartyTestCase.setUpClass")
        cls.zconf = create_2pc_config()

        cls.workers = []
        for node in cls.zconf.nodes:
            worker = multiprocessing.Process(target=run_single_server,
                                             args=(cls.zconf, node.id))
            worker.daemon = True
            worker.start()
            cls.workers.append(worker)
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        print("TwoPartyTestCase.tearDownClass")


class ThreePartyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("ThreePartyTestCase.setUpClass")
        cls.zconf = create_3pc_config()

        cls.workers = []
        for node in cls.zconf.nodes:
            worker = multiprocessing.Process(target=run_single_server,
                                             args=(cls.zconf, node.id))
            worker.daemon = True
            worker.start()
            cls.workers.append(worker)
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        print("ThreePartyTestCase.tearDownClass")
