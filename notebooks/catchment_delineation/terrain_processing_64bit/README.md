# Terrain processing 64-bit

Terrain processing using PySheds involves filling depressions, which calls scikit-image's `morphology.reconstruction` function in the background. However, the original scikit-image implementation was not capable of handling very large grids, which caused problems processing some vassdragsområder at 10 m resolution. See the issues [here](https://github.com/mdbartos/pysheds/issues/187) and [here](https://github.com/scikit-image/scikit-image/issues/6277) for full details.

The code in this folder replaces the original scikit-image functions with some modified Cython capable of handling larger grids.