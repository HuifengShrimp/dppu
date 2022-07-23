Logistic regression example
===========================

To use PPU C++ API, we have to first :doc:`Build from source<../development/build>`, this document shows how to write a pravicy preserving logistic regression use PPU C++ api.


Logistic Regression
-------------------

.. literalinclude:: ../../examples/cpp/simple_lr.cc
  :language: cpp

.. todo:: could you help to add comments for *simple_lr.cc* @shantang


Run it
------

Start two terminals.

In the first terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_lr -- -rank 0 -dataset examples/cpp/data/perfect_logit_a.csv -has_label 1

In the second terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_lr -- -rank 1 -dataset examples/cpp/data/perfect_logit_b.csv

