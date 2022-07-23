
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


"""MNIST example.
Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# Start two node servers.
# > bazel run //examples/python/distributed:daemon -- up
#
# Run this example script.
# > bazel run //examples/python/e2e:cnn

# See issue #620.
# pytype: disable=wrong-keyword-args

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import time
import examples.python.distributed as distr
import examples.python.utils.fe_utils as fe
from examples.python.distributed import P1, P2, PPU
import argparse

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = distr.config.parse_json(file.read())
distr.init(conf)


class CNN(nn.Module):
    """A simple CNN model."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# @jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = CNN().apply({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    # accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    # return grads, loss, accuracy
    return grads, loss


# @jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, images, labels, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(images)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(images))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    # epoch_accuracy = []

    for perm in perms:
        batch_images = images[perm, ...]
        batch_labels = labels[perm, ...]
        # grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        grads, loss = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        # epoch_accuracy.append(accuracy)
    train_loss = jnp.mean(jnp.array(epoch_loss))
    # train_accuracy = jnp.mean(jnp.array(epoch_accuracy))
    # return state, train_loss, train_accuracy
    return state, train_loss


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train',
                                                   batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.float32(train_ds['image']) / 255.
    test_ds['image'] = np.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply,
                                         params=params,
                                         tx=tx)


@distr.device(PPU)
@fe.jax2ppu
def train_and_evaluate(train_images, train_labels, test_images,
                       test_labels) -> train_state.TrainState:
    """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 30000
    num_epochs = 5
    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, learning_rate, momentum)
    prints = []

    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        # state, train_loss, train_accuracy = train_epoch(
        #     state, train_images, train_labels, batch_size, input_rng)
        # _, test_loss, test_accuracy = apply_model(state, test_images,
        #                                           test_labels)
        state, train_loss = train_epoch(state, train_images, train_labels,
                                        batch_size, input_rng)
        _, test_loss = apply_model(state, test_images, test_labels)

        # prints.append({
        #     'epoch': epoch,
        #     'train_loss': train_loss,
        #     'train_accuracy': train_accuracy,
        #     'test_loss': test_loss,
        #     'test_accuracy': test_accuracy
        # })
        prints.append(test_loss)

    return prints


train_ds, test_ds = get_datasets()

start = time.perf_counter()
prints = train_and_evaluate(train_ds['image'], train_ds['label'],
                            test_ds['image'], test_ds['label'])
end = time.perf_counter()
print(f"elapsed time: {end - start:0.4f} seconds")

# for item in prints:
#     print(
#         'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
#         % (item['epoch'], item['train_loss'], item['train_accuracy'] * 100,
#            item['test_loss'], item['test_accuracy'] * 100))

for i in range(len(prints)):
    print('epoch:% 3d,  test_loss: %.4f' % (i, prints[i]))

# jax output:
# epoch:  0,  test_accuracy: 35.44
# epoch:  1,  test_accuracy: 49.27
# epoch:  2,  test_accuracy: 64.95
# epoch:  3,  test_accuracy: 78.33
# epoch:  4,  test_accuracy: 79.20

# epoch:  0,  test_loss: 2.2067
# epoch:  1,  test_loss: 2.0326
# epoch:  2,  test_loss: 1.6355
# epoch:  3,  test_loss: 0.9835
# epoch:  4,  test_loss: 0.6283
