.. _tutorial:

Getting started
===============

Before diving into the detail, it worth a while to understand the basic concepts, since PPU use quite use different model than other platforms.

.. toctree::
   :maxdepth: 1

   ../development/basic_concepts


PPU is designed to be easily integrated into other distributed system, it's **recommanded to use** `Secretflow <TODO: doc url>`_ to write PPU program.

The following examples demonstrate how to work with PPU.

.. toctree::
   :maxdepth: 1

   py_jax_example
   PPU IR<pphlo_example>
   C++ Logit regression<cpp_lr_example>
   C++ PSI<cpp_dhpsi_example>


.. note::
  For low-level access, you can also use PPU's *C++ API*, but C++ API is **not guaranteed to be stable**.
  For more stable API, consider using :ref:`frontend API <development/basic_concepts:API level>`.

