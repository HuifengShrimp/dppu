MPC Protocol status
===================

Currently we provide two protocols in ppu.

.. include:: complexity.md
   :parser: myst_parser.sphinx_

`ABY3 <https://eprint.iacr.org/2018/403.pdf>`_: A **honest majority** 3PC-protocol. PPU provides
**semi-honest** implementation.

`Semi2k-SPDZ* <https://eprint.iacr.org/2018/482.pdf>`_ : A **semi-honest** NPC-protocol similar to
SPDZ but requires a trusted third party to generate offline randoms. It provides **dishonest majority**
if the additional trusted third party is not considered as one of the participants.

-------------------------

Currently, we mainly focused on bridging existing AI frameworks to PPU via XLA, an intermediate
representation where we can hook Tensorflow, Torch and Jax.

We welcome security experts to help contribute more security protocols.
