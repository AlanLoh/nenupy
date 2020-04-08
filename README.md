# **nenupy**

[![nenupy](https://img.shields.io/pypi/v/nenupy.svg)](
    https://pypi.python.org/pypi/nenupy)
[![PyPI download month](https://img.shields.io/pypi/dm/nenupy.svg)](
    https://pypi.python.org/pypi/nenupy/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/nenupy.svg)](
    https://pypi.python.org/pypi/nenupy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3667816.svg)](https://doi.org/10.5281/zenodo.3667816)
[![Documentation Status](https://readthedocs.org/projects/nenupy/badge/?version=latest)](https://nenupy.readthedocs.io/en/latest/?badge=latest)

<!-- ![Alt text](./Logo-NenuFAR-noir.svg) -->
<p align="center">
<img src="./Logo-NenuFAR-noir.svg" width="20%">
</p>

*nenupy* is a Python3 ([install](https://www.anaconda.com/download/) via Anaconda) package, written by A. Loh (LESIA, Obs. Paris), in order to handle *NenuFAR* observations.
[*NenuFAR*](https://nenufar.obs-nancay.fr) is a low-frequency radiotelescope located in Nancay, France.

## Installation
### pip
To install *nenupy* with pip/pip3:
```
pip install nenupy
```
<!-- or
```
python3 -m pip install --index-url https://test.pypi.org/simple/ nenupy
``` -->

If `nenupy` is already installed, the newer version can be installed:
```
pip install nenupy --upgrade
```
<!-- ```
python3 -m pip install --index-url https://test.pypi.org/simple/ nenupy --upgrade
``` -->

### Package requirement
* [*astropy*](http://www.astropy.org)
* [*pygsm*](https://github.com/telegraphic/PyGSM): please follow the *pygsm* package instructions to properly install it, it cannot be done via pip.
* *healpy* (install with `conda install -c conda-forge healpy`)
