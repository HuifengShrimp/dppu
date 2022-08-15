
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


import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics
import sys
import os

# curPath = os.path.abspath(os.path.dirname('/home/admin/dev/examples/'))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootpath)

import examples.python.utils.dataset_utils as dataset_utils


# FIXME: For un-normalized data, grad(sigmoid) is likely to overflow, either with exp/tanh or taylor series
# https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
def sigmoid(x):
    # return 0.5 * (jnp.tanh(x / 2) + 1)
    return 1 / (1 + jnp.exp(-x))


def predict(x, w, b):
    return sigmoid(jnp.matmul(x, w) + b)


def loss(x, y, w, b):
    pred = predict(x, w, b)
    label_prob = pred * y + (1 - pred) * (1 - y)
    return -jnp.mean(jnp.log(label_prob))


class LogitRegression:
    def __init__(self, n_epochs=10, n_iters=10, step_size=0.1):
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size

    def fit_auto_grad(self, feature, label):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for (x, y) in zip(xs, ys):
                grad = jax.grad(loss, argnums=(2, 3))(x, y, w_, b_)
                w_ -= grad[0] * self.step_size
                b_ -= grad[1] * self.step_size

            return w_, b_

        return jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

    def fit_manual_grad(self, feature, label):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for (x, y) in zip(xs, ys):
                pred = predict(x, w_, b_)
                err = pred - y
                w_ -= jnp.matmul(jnp.transpose(x),
                                 err) / y.shape[0] * self.step_size
                b_ -= jnp.mean(err) * self.step_size

            return w_, b_

        return jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))


def run_on_cpu():
    x, y = dataset_utils.load_full_dataset()
    print("x")
    print(x.shape)
    print("y")
    print(y.shape)

    lr = LogitRegression()

    w0, b0 = lr.fit_auto_grad(x, y)
    # w0, b0 = jax.jit(lr.fit_auto_grad)(x, y)
    print(w0, b0)
    print("AUC={}".format(metrics.roc_auc_score(y, predict(x, w0, b0))))

    w1, b1 = lr.fit_manual_grad(x, y)
    # w1, b1 = jax.jit(lr.fit_manual_grad)(x, y)
    print(w1, b1)
    print("AUC={}".format(metrics.roc_auc_score(y, predict(x, w1, b1))))


if __name__ == "__main__":
    run_on_cpu()
