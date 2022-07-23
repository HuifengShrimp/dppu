
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
import flax.linen as nn
from typing import Sequence

import examples.python.utils.dataset_utils as dataset_utils


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class SimpleMLP:
    def __init__(self,
                 features=[12, 10, 4],
                 n_batch=10,
                 n_epochs=10,
                 n_iters=10,
                 step_size=0.01):
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size
        self.model = MLP(features)
        self.params = self.model.init(jax.random.PRNGKey(1),
                                      jax.numpy.ones((n_batch, features[0])))

    def log_model(self):
        print("B: {}, Epoch: {}, iter: {}".format(self.n_batch, self.n_epochs,
                                                  self.n_iters))
        print(self.params)

    def predict(self, params, x):
        return self.model.apply(params, x)

    def set_params(self, params):
        self.params = params

    def predict_test(self, x):
        return self.model.apply(self.params, x)

    def loss(self, x, y):
        pred = self.model.apply(self.params, x)

        def mse(y, pred):
            def squared_error(y, y_pred):
                return jnp.multiply(y - y_pred, y - y_pred) / 2.0

            return jnp.mean(squared_error(y, pred))

        return mse(y, pred)

    def fit_auto_grad(self, feature, label):

        # label = jnp.reshape(jnp.array(label), (-1,1))

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        # TODO: this point shall be modified. Since the grad in Jax shall compute the gradients for the input variables, params shall be expilicitly declared
        def loss_func(params, x, y):
            pred = self.model.apply(params, x)

            def mse(y, pred):
                def squared_error(y, y_pred):
                    # TODO: check this
                    return jnp.multiply(y - y_pred, y - y_pred) / 2.0
                    # return jnp.inner(y - y_pred, y - y_pred) / 2.0 # fail, (10, 1) inner (10, 1) -> (10, 10), have to be (10,) inner (10,) -> scalar

                return jnp.mean(squared_error(y, pred))

            return mse(y, pred)

        def body_fun(_, loop_carry):
            params = loop_carry
            for (x, y) in zip(xs, ys):
                losses, grads = jax.value_and_grad(loss_func)(params, x, y)
                params = jax.tree_multimap(lambda p, g: p - self.step_size * g,
                                           params, grads)
            return params

        params = jax.lax.fori_loop(0, self.n_epochs, body_fun, self.params)
        return params

    # Without JIT, we can log the intermediate variables for debug
    def fit_auto_grad_wo_jit(self, feature, label):
        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def loss_func(params, x, y):
            pred = self.model.apply(params, x)

            def mse(y, pred):
                def squared_error(y, y_pred):
                    return jnp.multiply(y - y_pred, y - y_pred) / 2.0

                return jnp.mean(squared_error(y, pred))

            return mse(y, pred)

        def update_func(params, x, y):
            losses, grads = jax.value_and_grad(loss_func)(params, x, y)
            params = jax.tree_multimap(lambda p, g: p - self.step_size * g,
                                       params, grads)
            print(losses)
            return params

        for i in range(self.n_epochs):
            for (x, y) in zip(xs, ys):
                self.params = update_func(self.params, x, y)
        return self.params


def run_on_cpu():
    x, y = dataset_utils.load_full_dataset()
    mlp = SimpleMLP([x.shape[1], 8, 1], n_epochs=10)

    @jax.jit
    def train(x, y):
        return mlp.fit_auto_grad(x, y)

    print("=" * 20)
    params = train(x, y)
    print(params)

    print("=" * 20)
    print(mlp.predict(params, x)[:10])

    print("=" * 20)
    mlp.set_params(params)
    print(mlp.predict_test(x)[:10])

    print("=" * 20, " LOSS")
    print(mlp.loss(x, y))

    @jax.jit
    def test(params, x_, y_):
        return mlp.predict(params, x_)

    # 注意这里的roc_auc_score函数不能放在jit里面，在jit里面涉及到numpy调用的需要改成jnp
    print('AUC={}'.format(metrics.roc_auc_score(y, test(params, x, y))))


if __name__ == '__main__':
    run_on_cpu()
