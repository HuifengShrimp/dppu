
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


import unittest
import time

import numpy as np
from examples.python.tests.test_utils import TwoPartyTestCase, ThreePartyTestCase
import examples.python.distributed as distr
import examples.python.utils.fe_utils as fe
from examples.python.distributed import P1, P2, PPU

import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Conv, MaxPool, AvgPool, Flatten, Dense, Relu, LogSoftmax, Softmax
from jax import grad, random

import tensorflow_datasets as tfds
from keras.datasets import cifar10

from examples.python.tests.stax_utils import custom_mse_loss, custom_Dense


def lenet():
    nn_init, nn_apply = stax.serial(
        Conv(out_chan=20, filter_shape=(5, 5), strides=(1, 1), padding='valid'),
        Relu,
        AvgPool(window_shape=(2, 2)),
        Conv(out_chan=50, filter_shape=(5, 5), strides=(1, 1), padding='valid'),
        Relu,
        AvgPool(window_shape=(2, 2)),
        Flatten,
        Dense(500),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


def alexnet():
    nn_init, nn_apply = stax.serial(
        Conv(out_chan=96,
             filter_shape=(11, 11),
             strides=(4, 4),
             padding=((9, 9), (9, 9))),
        Relu,
        AvgPool(window_shape=(3, 3), strides=(2, 2)),
        Conv(out_chan=256,
             filter_shape=(5, 5),
             strides=(1, 1),
             padding=((1, 1), (1, 1))),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(1, 1)),
        Conv(out_chan=384,
             filter_shape=(3, 3),
             strides=(1, 1),
             padding=((1, 1), (1, 1))),
        Relu,
        Conv(out_chan=384,
             filter_shape=(3, 3),
             strides=(1, 1),
             padding=((1, 1), (1, 1))),
        Relu,
        Conv(out_chan=256,
             filter_shape=(3, 3),
             strides=(1, 1),
             padding=((1, 1), (1, 1))),
        Relu,

        # FC
        Flatten,
        Dense(256),
        Relu,
        Dense(256),
        Relu,
        Dense(10))
    return nn_init, nn_apply


def vgg16():
    nn_init, nn_apply = stax.serial(
        Conv(out_chan=64, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=64, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(out_chan=128, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=128, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(out_chan=256, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=256, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=256, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        Conv(out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=(1, 1)),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),

        # FC
        Flatten,
        Dense(256),
        Relu,
        Dense(256),
        Relu,
        Dense(10))
    return nn_init, nn_apply


def custom_model():
    nn_init, nn_apply = stax.serial(
        Conv(2, (1, 1)),
        # stax.MaxPool((2, 2)),
        AvgPool((2, 2), (1, 1)),
        # stax.BatchNorm(),
        Flatten,
        Dense(10),
        # custom_Dense(10),
        # stax.elementwise(stax.relu),
        # stax.Softmax,
    )
    return nn_init, nn_apply


def standard_stax(train_imgs, train_labels):
    ###########################
    ##    Hyper-parameters   ##
    ###########################
    learning_rate = 0.01
    epochs = 5
    batch_size = 10

    ###########################
    ##      Model Config     ##
    ###########################
    # TODO: add customed operators
    nn_init, nn_apply = lenet()
    print("Begin Train!")

    ###########################
    ##  Model Initialization ##
    ###########################
    key = random.PRNGKey(42)
    input_shape = tuple(
        [-1 if idx == 0 else i for idx, i in enumerate(list(train_imgs.shape))])
    output_shape, params_init = nn_init(key, input_shape)

    #################################################
    ##  UNCOMMENT ME TO PEFORM SECRET COMPUTATIONS ##
    #################################################
    # @distr.device(PPU)
    # @fe.jax2ppu
    def train_and_evaluate(train_imgs, train_labels):
        loss_list = []
        predic_list = []
        params_list = []
        print(train_imgs.shape)
        print(train_labels.shape)
        # FIXME: Register in the symbol table.
        # FIXME: This has to be optimized. Currently, the variables have to be np.ndarray and opt_state is not.
        opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
        opt_state = opt_init(params_init)

        def mse_loss(y, label):
            return jnp.sum((y - label)**2)

        def ce_loss(y, label):
            return -jnp.mean(jnp.sum(label * stax.logsoftmax(y), axis=1))

        def apply_model(state, imgs, labels):
            def loss_func(params):
                y = nn_apply(params, imgs)
                # FIXME: One-hot incurs comparison, this shall be moved to data init phase
                one_hot = jax.nn.one_hot(labels, 10)
                return ce_loss(y, one_hot), y

            grad_fn = jax.value_and_grad(loss_func, has_aux=True)
            (loss, y), grads = grad_fn(get_params(state))
            return loss, y, grads

        ###########################
        ##      Train Epoch      ##
        ###########################
        for i in range(1, epochs + 1):

            # TODO: jax.random not supported. Idea: refer to PRF as the key
            # rng, input_rng = jax.random.split(key)

            # train_ds_size = len(train_imgs)
            # steps_per_epoch = train_ds_size // batch_size

            # perms = jax.random.permutation(rng, len(train_imgs))
            # # skip incomplete batch
            # perms = perms[:steps_per_epoch * batch_size]
            # perms = perms.reshape((steps_per_epoch, batch_size))

            # FIXME: currently hack
            ###########################
            ##      Train Batch     ##
            ###########################
            # for perm in perms:
            # batch_images = train_imgs[perm, ...]
            # batch_labels = train_labels[perm, ...]
            imgs_batchs = jnp.array_split(train_imgs,
                                          len(train_imgs) / batch_size,
                                          axis=0)
            labels_batchs = jnp.array_split(train_labels,
                                            len(train_labels) / batch_size,
                                            axis=0)

            for (batch_images, batch_labels) in zip(imgs_batchs, labels_batchs):
                loss, y, grads = apply_model(opt_state, batch_images,
                                             batch_labels)
                opt_state = opt_update(i, grads, opt_state)

                loss_list.append(loss)
                predic_list.append(y)
                params_list.append(get_params(opt_state))
        return loss_list, predic_list, params_list

    loss_list, predic_list, params_list = train_and_evaluate(
        train_imgs, train_labels)
    return loss_list, predic_list


def generate_data(epoch_size=10, dataset='mnist'):
    # generate random data
    key = random.PRNGKey(42)
    class_num = 10

    if dataset == 'mnist':
        input_shape = (epoch_size, 28, 28, 1)
    elif dataset == 'cifar10':
        input_shape = (epoch_size, 32, 32, 3)

    X = np.random.normal(size=input_shape).astype(np.float32)
    y_true = np.random.randint(class_num, size=(epoch_size))
    # y_true = np.asarray(jax.nn.one_hot(y_true, class_num))
    # y_true = y_true.reshape(-1, 10)
    return X, y_true


def get_datasets(name='mnist'):
    """Load MNIST train and test datasets into memory."""
    if name == 'cifar10':
        train_ds, test_ds = cifar10.load_data()
        (train_imgs, train_labels), (test_imgs, test_labels) = train_ds, test_ds
        train_imgs = np.float32(train_imgs) / 255.
        train_labels = np.squeeze(train_labels)
        test_imgs = np.float32(test_imgs) / 255.
        test_labels = np.squeeze(test_labels)

        return (train_imgs, train_labels), (test_imgs, test_labels)
    ds_builder = tfds.builder(name)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train',
                                                   batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.float32(train_ds['image']) / 255.
    test_ds['image'] = np.float32(test_ds['image']) / 255.
    return train_ds, test_ds


class TestBN(ThreePartyTestCase):
    # def test_bn(self):
    #     distr.init(self.zconf)

    #     @distr.device(PPU)
    #     @fe.jax2ppu
    #     def batch_norm(x):
    #         axis = (0,)
    #         mean = jnp.mean(x, axis)
    #         variance = jnp.var(x, axis)
    #         sqrt_var = jnp.sqrt(variance)
    #         # stand_res = jax.nn.normalize(x, axis)
    #         # return stand_res
    #         return (x - mean) / sqrt_var

    #     x = np.random.normal(size=(3, 3)).astype(np.float32)
    #     x_ = distr.put(PPU, x)
    #     y = batch_norm(x)

    #     # pre-load the XLA compile procedure
    #     y1 = batch_norm(x_)

    #     # actual execution time consumption
    #     start = time.perf_counter()
    #     y2 = batch_norm(x_)
    #     end = time.perf_counter()
    #     print(distr.get(y))
    #     print(distr.get(y2))
    #     print(f"batch norm elapsed time: {end - start:0.4f} seconds")

    #     self.assertTrue(np.allclose(
    #         distr.get(y), distr.get(y2), rtol=0.001, atol=0.00001))

    # def test_softmax(self):
    #     distr.init(self.zconf)

    #     @distr.device(PPU)
    #     @fe.jax2ppu
    #     def softmax(x):
    #         return stax.softmax(x)

    #     x = np.random.normal(size=(3, 3)).astype(np.float32)
    #     x_ = distr.put(PPU, x)
    #     y = softmax(x)

    #     # pre-load the XLA compile procedure
    #     y1 = softmax(x_)

    #     # actual execution time consumption
    #     start = time.perf_counter()
    #     y2 = softmax(x_)
    #     end = time.perf_counter()
    #     print(distr.get(y))
    #     print(distr.get(y2))
    #     print(f"softmax elapsed time: {end - start:0.4f} seconds")

    #     self.assertTrue(np.allclose(
    #         distr.get(y), distr.get(y2), rtol=0.001, atol=0.00001))

    def test_stax_nn(self):
        """
        NN functionality test
        """
        distr.init(self.zconf)
        epoch_size = 10

        # Synthetic data
        X, y_true = generate_data(epoch_size=10)

        # MNIST and CIFAR10 data
        # train_ds, test_ds = get_datasets('mnist')
        # train_imgs, train_labels = train_ds['image'][:epoch_size], train_ds['label'][:epoch_size]

        # train_ds, test_ds = get_datasets('cifar10')
        # all_imgs, all_labels = train_ds
        # train_imgs, train_labels = all_imgs[:epoch_size], all_labels[:epoch_size]

        # X, y_true = train_imgs, train_labels
        """
        Below two lines are actually performed implicitly using @zjax.device(PPU) annotation:
            X_ = distr.put(PPU, X)
            y_true_ = distr.put(PPU, y_true)
            <====>
            @zjax.device(PPU)
            def func(X, y_true):
                # impl
        """

        # losses = standard_stax(X, y_true)

        start = time.perf_counter()
        losses, predicts = standard_stax(X, y_true)
        end = time.perf_counter()

        print(f"standard stax nn elapsed time: {end - start:0.4f} seconds")

        for i in range(len(losses)):
            # Compute on PPU with fe.jax2ppu enabled
            if isinstance(losses[i], distr.core.DeviceObject):
                print(f'{i}: {distr.get(losses[i])}')
            else:
                print(f'{i}: {losses[i]}')

            if isinstance(predicts[i], distr.core.DeviceObject):
                print(f'{i}: {np.argmax(distr.get(predicts[i]), axis=1)}')
            else:
                print(f'{i}: {np.argmax(predicts[i], axis=1)}')
        print(f'Target: {y_true}')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
