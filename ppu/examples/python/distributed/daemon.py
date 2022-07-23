
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


import argparse
import json
import logging
import multiprocessing
import sys
import traceback
from concurrent import futures

import dill
import grpc
import ppu.binding._lib.link as link
from ppu.binding import Runtime

from examples.python.distributed import config, context, core_pb2, core_pb2_grpc

_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT


class NodeServicer(core_pb2_grpc.NodeServiceServicer):
    def __init__(self, zconf: core_pb2.WorldDesc, node_id: str):
        self.my_node_id = node_id

        self.zsymbols = {}
        self.zctx = context.WorkerContext(zconf, node_id)
        self.ppu_device_config = config.find_ppu_device_config(zconf, node_id)
        if self.ppu_device_config:
            self.ppu_internal_addrs = list(
                self.ppu_device_config.ppu_device.ppu_internal_addrs)
            self.ppu_ids = list(self.ppu_device_config.ppu_device.node_ids)
            self.ppu_rank = self.ppu_ids.index(node_id)

            desc = link.Desc()
            for k, v in zip(self.ppu_ids, self.ppu_internal_addrs):
                desc.add_party(k, v)

            self.link = link.create_brpc(desc, self.ppu_rank)
            self.rt = Runtime(self.link,
                              self.ppu_device_config.ppu_device.runtime_config)

    def Eval(self, req, ctx):
        if req.from_node_id:
            logging.info(f'Task: tag={req.tag}, from={req.from_node_id}')
        else:
            logging.info(f'Task: tag={req.tag}')

        try:
            # run with server context
            (fn, args, kwargs) = dill.loads(req.task)
            res = fn(self, *args, **kwargs)
            return core_pb2.EvalResponse(value=dill.dumps(res, recurse=True))
        except Exception as e:
            stack_info = traceback.format_exc()
            logging.info(stack_info)
            return core_pb2.EvalResponse(
                exception=dill.dumps(stack_info, recurse=True))


def run_single_server(zconf, node_id):
    options = [('grpc.max_message_length', 1024 * 1024 * 1024),
               ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
        options=options)
    node = config.find_node(zconf, node_id)

    core_pb2_grpc.add_NodeServiceServicer_to_server(
        NodeServicer(zconf, node_id), server)

    # server.add_insecure_port('[::]:{}'.format(args.port))
    server.add_insecure_port(node.addr)
    server.start()
    server.wait_for_termination()


def serve(args):

    # load the configuration
    with open(args.config, 'r') as f:
        zconf = config.parse_json(f.read())

    if args.command == 'start':
        run_single_server(zconf, args.node_id)
    elif args.command == 'up':
        workers = []
        for node in zconf.nodes:
            worker = multiprocessing.Process(target=run_single_server,
                                             args=(zconf, node.id))
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        raise ValueError('unknown commands.')


parser = argparse.ArgumentParser(description='ppu node server.')

subparsers = parser.add_subparsers(dest='command')
parser_start = subparsers.add_parser('start', help='to start a single node')
parser_start.add_argument("-n",
                          "--node_id",
                          default="local:0",
                          help="the node id")
parser_start.add_argument("-c",
                          "--config",
                          default="examples/python/conf/2pc.json",
                          help="the config")

parser_up = subparsers.add_parser('up', help='to bring up all nodes')
parser_up.add_argument("-c",
                       "--config",
                       default="examples/python/conf/2pc.json",
                       help="the config")


def main():
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    args = parser.parse_args()
    serve(args)


if __name__ == '__main__':
    main()
