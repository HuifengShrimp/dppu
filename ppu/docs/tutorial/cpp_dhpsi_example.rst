PSI Example
===========

A PSI example based on curve25519.

Which PSI I should use
----------------------

The :ppu_code_host:`ECDH-PSI </ppu/blob/master/ppu%2Fpsi%2Fecdh_psi.h#L131>` is favored if the bandwidth is the bottleneck.
If the computing is the bottleneck, you should try the BaRK-OPRF based
PSI :ppu_code_host:`KKRT-PSI API </ppu/blob/master/ppu%2Fpsi%2Fkkrt_psi.h>`.

.. todo:: Add more informations for other state-of-the-art PSI protocols.

C++ Example
------------

A simple in-memory psi example. Currently PPU is using `curve25519-donna <https://github.com/floodyberry/curve25519-donna>`_,
which is recommended by Alibaba Gemini Lab.

.. literalinclude:: ../../examples/cpp/simple_dh_psi.cc
  :language: cpp

.. todo:: Add the streaming example where user could perform PSI for billion items.

.. todo:: Add python wrapper for PSI.


How To Run
------------

Start two terminals.

In the first terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_dh_psi -- -rank 0 -data_size 1000

In the second terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_dh_psi -- -rank 1 -data_size 1000


Benchmark
----------

benchmark result without data load time 

dh-psi Benchmark
>>>>>>>>>>>>>>>>

:ppu_code_host:`DH-PSI benchmark code </ppu/blob/master/ppu%2Fpsi%2ecdh_psi_bench.cc>`

cpu limited by docker(--cpu)

+---------------------------+-----+--------+--------+---------+---------+---------+
| Intel(R) Xeon(R) Platinum | cpu | 2^20   | 2^21   | 2^22    | 2^23    | 2^24    |
+===========================+=====+========+========+=========+=========+=========+
|                           | 4c  | 40.181 | 81.227 | 163.509 | 330.466 | 666.807 |
|  8269CY CPU @ 2.50GHz     +-----+--------+--------+---------+---------+---------+
|                           | 8c  | 20.682 | 42.054 | 85.272  | 173.836 | 354.842 |
|  with curve25519-donna    +-----+--------+--------+---------+---------+---------+
|                           | 16c | 11.639 | 23.670 | 48.965  | 100.903 | 208.156 |
+---------------------------+-----+--------+--------+---------+---------+---------+

`ipp-crypto Multi-buffer Functions <https://www.intel.com/content/www/us/en/develop/documentation/ipp-crypto-reference/top/multi-buffer-cryptography-functions/montgomery-curve25519-elliptic-curve-functions.html>`_


+---------------------------+-----+--------+--------+---------+---------+---------+
| Intel(R) Xeon(R) Platinum | cpu | 2^20   | 2^21   | 2^22    | 2^23    | 2^24    |
+===========================+=====+========+========+=========+=========+=========+
|                           | 4c  | 7.37   | 15.32  | 31.932  | 66.802  | 139.994 |
|  8369B CPU @ 2.70GHz      +-----+--------+--------+---------+---------+---------+
|                           | 8c  | 4.3    | 9.095  | 18.919  | 40.828  | 87.649  |
|  curve25519(ipp-crypto)   +-----+--------+--------+---------+---------+---------+
|                           | 16c | 2.921  | 6.081  | 13.186  | 29.614  | 65.186  |
+---------------------------+-----+--------+--------+---------+---------+---------+

kkrt-psi Benchmark
>>>>>>>>>>>>>>>>>>>

All of our experiments use a single thread for each party. 

If the bandwidth is enough, the upstream could try to perform multi-threading optimizations

bandwitdh limited by `wondershaper <https://github.com/magnific0/wondershaper/>`_.

10Mbps = 10240Kbps, 100Mbps = 102400Kbps, 1000Mbps = 1024000Kbps

.. code-block:: bash

  wondershaper -a lo -u 10240

Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz

+-----------+---------+--------+--------+--------+---------+
| bandwidth |  phase  |  2^18  |  2^20  |  2^22  |  2^24   |
+===========+=========+========+========+========+=========+
|           | offline | 0.012  | 0.012  | 0.012  | 0.014   |
|    LAN    +---------+--------+--------+--------+---------+
|           | online  | 0.495  | 2.474  | 10.765 | 44.368  |
+-----------+---------+--------+--------+--------+---------+
|           | offline | 0.012  | 0.012  | 0.024  | 0.014   |
|  100Mbps  +---------+--------+--------+--------+---------+
|           | online  | 2.694  | 11.048 | 46.983 | 192.37  |
+-----------+---------+--------+--------+--------+---------+
|           | offline | 0.016  | 0.019  | 0.0312 | 0.018   |
|  10Mbps   +---------+--------+--------+--------+---------+
|           | online  | 25.434 | 100.68 | 415.94 | 1672.21 |
+-----------+---------+--------+--------+--------+---------+
