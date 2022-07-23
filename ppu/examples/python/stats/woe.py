
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
# > bazel run //examples/python/stats:pvalue

from tkinter.messagebox import NO
import jax

from jax import numpy as np
from numpy import double
from scipy.stats import norm

import pandas as pd

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

BUCKETS_SIZE = 10


@distr.device(P1)
def load_label():
    import examples.python.utils.dataset_utils as dataset_utils

    return dataset_utils.load_label()


def build_select_map(x):
    import numpy

    smap = numpy.zeros((x.shape[1] * BUCKETS_SIZE, x.shape[0]))
    for col in range(x.shape[1]):
        idxs = pd.qcut(x[:, col], BUCKETS_SIZE, labels=False)
        for sample in range(idxs.shape[0]):
            bucket = idxs[sample]
            smap[col * BUCKETS_SIZE + bucket, sample] = 1.0
    return smap, numpy.sum(smap, axis=1)


@distr.device(P2)
def load_select_map_r2():
    import examples.python.utils.dataset_utils as dataset_utils

    x = dataset_utils.load_feature(1)
    return build_select_map(x)


def woe_calc(total_counts, positive_counts, positive_label_cnt,
             negative_label_cnt):
    assert total_counts.shape == positive_counts.shape
    import numpy

    negative_counts = total_counts - positive_counts
    woe = numpy.zeros(total_counts.shape)

    def calc(p, n, pl, nl):
        import math

        if p == 0 or n == 0:
            positive_distrib = (p + 0.5) / pl
            negative_distrib = (n + 0.5) / nl
            return math.log(positive_distrib / negative_distrib)
        else:
            positive_distrib = double(p) / pl
            negative_distrib = double(n) / nl
            return math.log(positive_distrib / negative_distrib)

    for idx in range(total_counts.shape[0]):
        woe[idx] = calc(
            positive_counts[idx],
            negative_counts[idx],
            positive_label_cnt,
            negative_label_cnt,
        )

    return woe


@distr.device(P1)
def woe_calc_for_master():
    import numpy

    x = dataset_utils.load_feature(0)
    y = dataset_utils.load_label()

    total_counts = numpy.zeros(x.shape[1] * BUCKETS_SIZE)
    positive_counts = numpy.zeros(x.shape[1] * BUCKETS_SIZE)
    for col in range(x.shape[1]):
        idxs = pd.qcut(x[:, col], BUCKETS_SIZE, labels=False)
        for sample in range(idxs.shape[0]):
            bucket = idxs[sample]
            total_counts[col * BUCKETS_SIZE + bucket] += 1
            positive_counts[col * BUCKETS_SIZE + bucket] += y[sample]

    positive_label_cnt = numpy.sum(y)
    negative_label_cnt = y.shape[0] - positive_label_cnt

    woe = woe_calc(total_counts, positive_counts, positive_label_cnt,
                   negative_label_cnt)

    return woe


@distr.device(P1)
def woe_calc_for_peer(totals, positives):
    import numpy

    total_counts = numpy.around(totals)
    positive_counts = numpy.around(positives)

    y = dataset_utils.load_label()
    positive_label_cnt = numpy.sum(y)
    negative_label_cnt = y.shape[0] - positive_label_cnt

    woe = woe_calc(total_counts, positive_counts, positive_label_cnt,
                   negative_label_cnt)

    return woe


def run_dist():
    s2, t2 = load_select_map_r2()
    y = load_label()

    @distr.device(PPU)
    @fe.jax2ppu
    def ssbuckercounter(s2, y):
        return np.matmul(s2, y).flatten()

    positive_counts_r2 = ssbuckercounter(s2, y)
    woe_r1 = woe_calc_for_master()
    woe_r2 = woe_calc_for_peer(t2, positive_counts_r2)
    print(distr.get(woe_r1))
    print(distr.get(woe_r2))
    # TODO: woe categories split points.


def run_origin():
    x, _ = dataset_utils.load_full_dataset()
    smap = build_select_map(x[:, 0:1])
    print(smap)


if __name__ == "__main__":
    # run_origin()
    run_dist()
