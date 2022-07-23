# Welcome to end2end test

Please follow the naming convention.

  ${FE}_${ALGO}[_ppu].py

where:
- FE: is short for FrontEnd, like tf, jax, flax, stax, pytorch.
- ALGO: is short for Algorithm, like cnn, dnn, lr, xgb.
- `_ppu` demonstrate how to run FE/ALGO on ppu, without `_ppu` is the cpu/gpu version.


