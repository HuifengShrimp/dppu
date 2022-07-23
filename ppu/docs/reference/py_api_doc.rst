Python API Reference
====================

Python API is used to control & access PPU, for example, to do data infeed/outfeed, to compile a XLA program to PPHlo, or to fire a PPHlo on a PPU runtime. It's not to write logic, to write program on PPU, see :doc:`program PPU <../tutorial/index>` for details.


Runtime setup
-------------

.. autoclass:: ppu.Runtime
    :members:


Runtime IO
----------

.. autoclass:: ppu.Io
    :members:

Compiler
--------

.. autofunction:: ppu.compile
