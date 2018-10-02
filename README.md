# **nenupy**
*nenupy* is a Python package, written by A. Loh (LESIA, Obs. Paris), in order to handle *NenuFAR* observations.
[*NenuFAR*](https://nenufar.obs-nancay.fr) is a low-frequency radiotelescope located in Nancay, France.

## Installation
### pip
To install *nenupy* with pip:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ nenupy
```

### Package requirement
* [*astropy*](http://www.astropy.org)
* ...

## Example
Loading the environment, within python3:
```python
from nenupy3.read import SST, BST
```
`SST` and `BST` are two separate modules to read **Sub-band Statistics** and **Beamlet Statistics** data respectively.

Once a reading module is loaded, a *NenuFAR* observation can be read:
```python
bst_obs = BST('some_observation_BST.fits')
```

Data can be retrieved, using keywords (such as `freq`, `polar` and `time`):
```python
bst_obs.getData( freq=[20, 60], time='2018-09-01 10:00:00.0', polar='nw' )
```

Once the function `getData()` has been called, the data are stored in the `d` attribute, (`t`, `f` for time and frequency as well). You can then plot the data:
```python
from matplotlib import pyplot as plt
plt.plot( bst_obs.t.mjd, bst_obs.d )
plt.show()
```
