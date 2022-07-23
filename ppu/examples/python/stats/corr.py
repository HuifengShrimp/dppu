
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
# > bazel run //examples/python/stats:corr

import jax

import examples.python.distributed as distr
import examples.python.utils.dataset_utils as dataset_utils
import examples.python.utils.fe_utils as fe
from examples.python.distributed import P1, P2, PPU

import argparse

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = distr.config.parse_json(file.read())
    distr.init(conf)


@distr.device(P1)
def load_feature_r1():
    import examples.python.utils.dataset_utils as dataset_utils
    # TODO: pre-process for constant column
    return dataset_utils.standardize(dataset_utils.load_feature(0))


@distr.device(P2)
def load_feature_r2():
    import examples.python.utils.dataset_utils as dataset_utils
    # TODO: pre-process for constant column
    return dataset_utils.standardize(dataset_utils.load_feature(1))


def run_dist():
    x1 = load_feature_r1()
    x2 = load_feature_r2()

    @distr.device(PPU)
    @fe.jax2ppu
    def XTX(x1, x2):
        x = jax.numpy.concatenate((x1, x2), axis=1)
        return jax.numpy.matmul(x.transpose(), x), x.shape[0]

    ss_xtx, rows = XTX(x1, x2)
    corr = distr.get(ss_xtx) / distr.get(rows)
    print(corr)


def run_origin():
    x, _ = dataset_utils.load_full_dataset()
    std_x = dataset_utils.standardize(x)
    corr = jax.numpy.matmul(std_x.transpose(), std_x) / (x.shape[0] - 1)
    print(corr)


if __name__ == '__main__':
    # run_origin()
    run_dist()
