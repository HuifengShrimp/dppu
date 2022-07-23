PPHlo example
=============

.. note::
   The most easy and stable way to use PPU is using frontend API, and let PPU compiler/runtime to translate and evaluate.

PPHlo is short for PPU High-Level-Ops, it's an assembly language inspired by `XLA <https://www.tensorflow.org/xla/operation_semantics>`_, and defined in `MLIR <https://mlir.llvm.org/>`_ format. 

Please see :doc:`PPHlo reference page <../reference/pphlo_doc>` for details.

Intro to PPHlo
--------------

PPHlo program looks like this:

.. code-block::
   :linenos:

    func @main(%arg0: tensor<!pphlo.sint>, %arg1: tensor<!pphlo.sint>) -> () {
      %0 = "pphlo.multiply"(%arg0, %arg1) : (tensor<!pphlo.sint>, tensor<!pphlo.sint>) -> tensor<!pphlo.sint>
      "pphlo.dbg_print"(%0) : (tensor<!pphlo.sint>) -> ()
      return
    }

The above `main` function takes two parameter `%arg0` and `%arg1`, multiply them and print the result.

.. note::
   If we take a closer look, the type of parameter is `tensor<!pphlo.sint>`, which means a **secret** integer tensor (in contrast, there is also a **public** integer), the **secret/public** is a PPU-only keyword, which gives PPU the ability for secure computation.

Run PPHlo example
-----------------

Here is a simple PPHlo example located in PPU repo, to run it, we have to first :doc:`Build from source<../development/build>`.

.. literalinclude:: ../../examples/cpp/simple_pphlo.cc
  :language: cpp

To run the example, start two terminals.

In the first terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_pphlo -- --rank 0

In the second terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_pphlo -- --rank 1
