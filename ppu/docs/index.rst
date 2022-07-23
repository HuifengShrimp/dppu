.. PPU documentation master file, created by
   sphinx-quickstart on Thu Dec 31 23:12:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PPU documentation
=================

Introduction
------------

PPU (Privacy-Preserving Processing Unit) is a (secure computation) domain specific compiler and runtime suite, which provides **provable secure computation** service.

PPU compiler use `XLA <https://www.tensorflow.org/xla/operation_semantics>`_ as its front-end IR, which supports diverse AI framework (like Tensorflow, JAX and PyTorch). PPU compiler translate XLA to an IR which could be interpreted by PPU runtime.

PPU runtime use `MPC <https://en.wikipedia.org/wiki/Secure_multi-party_computation>`_ as its back-end to protect privacy information. PPU runtime is designed to be **extensible**, so we can add MPC protocols with minimum effort and let PPU compiler/runtime to translate and interpret complicated AI models on it.

-------------------------

.. image:: imgs/ppu_arch.png
   :scale: 80 %

Content
-------

.. toctree::
   :maxdepth: 2

   tutorial/index
   development/index
   reference/index

