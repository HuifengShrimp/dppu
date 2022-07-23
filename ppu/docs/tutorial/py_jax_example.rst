JAX on PPU
==========

The most *easy and stable* way to use PPU is using :ref:`frontend API <development/basic_concepts:API level>`, and let PPU compiler/runtime to translate and evaluate. This guide demonstrates how to run JAX program on PPU.

`JAX <https://github.com/google/jax>`_ is a AI framework from Google. Users can write program in numpy syntax, and let JAX to translate it to GPU/TPU for acceleration.

Coding with JAX
---------------

First, let write a simple `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ in JAX.

.. code-block:: python
   :linenos:
   :emphasize-lines: 2,15

    import jax
    import jax.numpy as jnp

    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    def loss(x, y, w):
        pred = sigmoid(jnp.dot(x, w))
        label_prob = pred * y + (1 - pred) * (1 - y)
        return -jnp.sum(jnp.log(label_prob))

    def logit_regression(x, y, epochs=3, step_size=0.1):
        w = jnp.zeros(x.shape[1])
        for _ in range(epochs):
            grad = jax.grad(loss, 2)(x, y, w)
            w -= grad * step_size
        return w

    def load_dataset():
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer()
        return normalize(ds['data']), ds['target'].astype(dtype=np.float64)

    x, y = load_dataset()
    w = fit(x, y)


Note, L2 and L15 shows how JAX support numpy-like API and how to do `auto differentation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_.

Move to PPU
-----------

Now, let's run the above program on PPU, before that, please ensure that sf-ppu is installed.

.. code-block:: python
   :name: jax-lr-ppu
   :linenos:

    import examples.python.distributed as distr
    import examples.python.utils.fe_utils as fe

    with open('conf/2pc.conf', 'r') as file:
        conf = distr.config.parse_json(file.read())
    distr.init(conf)

    @distr.device(distr.P1)
    def load_feature():
        return load_dataset()[0]

    @distr.device(distr.P2)
    def load_label():
        return load_dataset()[1]

    @distr.device(distr.PPU)
    @fe.jax2ppu
    def fit(x, y):
        return logit_regression(x, y)

    def run_on_ppu():
        x = load_feature()
        y = load_label()

        w = train(x, y)
        print(distr.get(w))

`distr` is a simple distributed framework shipped with PPU examples, it's like `Ray <https://github.com/ray-project/ray>`_ which could schedule python function in a distributed environment. In the above example.

.. code-block:: python
   :linenos:
   :emphasize-lines: 1

    @distr.device(distr.P1)
    def load_feature():
        return load_dataset()[0]

* `distr.device(distr.P1)` schedule the decorated function `load_feature` to the node named `P1`.

.. code-block:: python
   :linenos:
   :emphasize-lines: 1,2

    @distr.device(distr.PPU)
    @fe.jax2ppu
    def fit(x, y):
        return logit_regression(x, y)

* `distr.device(distr.PPU)` schedule the decorated function `fit` to the `PPU` virtual device.
* `fe.jax2ppu` translates the JAX program to PPU assembly.

.. note::
  PPU **compile** the JAX program without changing any user-level code, with this approach, users can reuse all features (like autodiff and tensor ops) with their familiar AI front-end.

The above example is located at :ppu_code_host:`here </ppu/blob/master/examples%2Fpython%2Fe2e%2Fjax_lr_ppu.py>`. To run the example, start two terminals, in the first terminal.

.. code-block:: bash

  bazel run //examples/python/distributed:daemon -- up

In the second terminal.

.. code-block:: bash

  bazel run //examples/python/e2e:sslr

More examples
-------------
* :ppu_code_host:`This example </ppu/blob/master/examples%2Fpython%2Fe2e%2Ftf_lr_ppu.py>` demonstrates how to use tensorflow as PPU frontend.

* :ppu_code_host:`This example </ppu/blob/master/examples%2Fpython%2Fe2e%2Fflax_dnn_ppu.py>` demonstrats ehow to use flax to write DNN.

