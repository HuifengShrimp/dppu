
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


import threading

import ppu.binding._lib.link as link
import ppu.binding as pyppu


# https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread-in-python
class PropagatingThread(threading.Thread):
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.ret


class Simulator(object):
    def __init__(self, wsize: int, rt_config: pyppu.RuntimeConfig):
        self.wsize = wsize
        self.rt_config = rt_config
        self.io = pyppu.Io(wsize, rt_config)

    def __call__(self, code, *args):
        # TODO: support public?
        args = [
            self.io.make_shares(x, pyppu.Visibility.VIS_SECRET) for x in args
        ]

        lctx_desc = link.Desc()
        for rank in range(self.wsize):
            lctx_desc.add_party(f"id_{rank}", f"thread_{rank}")

        def wrapper(rank):
            lctx = link.create_mem(lctx_desc, rank)
            rt = pyppu.Runtime(lctx, self.rt_config)

            input_names = []
            for idx in range(len(args)):
                var_name = f'arg{idx}'
                input_names.append(var_name)
                rt.set_var(var_name, args[idx][rank])

            executable = pyppu.ExecutableProto(
                name='test',
                input_names=input_names,
                output_names=[],  # TODO
                code=code.encode('utf-8'))

            rt.run(executable)
            # TODO: get outputs.

        jobs = [
            PropagatingThread(target=wrapper, args=(rank, ))
            for rank in range(self.wsize)
        ]

        [job.start() for job in jobs]
        [job.join() for job in jobs]
