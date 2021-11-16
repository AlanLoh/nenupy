Installation
============

Installing with pip
-------------------

These instructions cover installation with the Python package
management tool `PyPI <https://pypi.org/project/nenupytf/>`_.
``Python3.7`` or higher is required and no support will be provided for ``Python2``.

.. code-block:: bash

   $ pip install nenupy

Updates need to be regularly check for while this package is still in developpment:

.. code-block:: bash

   $ pip install nenupy --upgrade

.. note:: 

    There is no support for a ``conda`` insatallation yet.


Installation on nancep
----------------------

.. code-block:: bash

   $ python3.8 -m pip install nenupy

or the current 'beta' version hosted on `GitHub <https://github.com/AlanLoh/nenupy>`_:

.. code-block:: bash

   $ pip3 install --user --upgrade https://github.com/AlanLoh/nenupy/tarball/master

.. note::
    
    You would probably have to update some packages:
        
        .. code-block:: bash

            $ python3.8 -m pip install astropy --upgrade
            $ python3.8 -m pip install healpy --upgrade
            $ python3.8 -m pip install h5py --upgrade
            $ python3.8 -m pip install ephem --upgrade



Dependencies
------------

* `numpy <https://docs.scipy.org/doc/numpy/reference/>`_
* `matplotlib <https://matplotlib.org/3.1.1/contents.html>`_
* `astropy <https://docs.astropy.org/en/stable/>`_
* `scipy <https://www.scipy.org/>`_
* `healpy <https://healpy.readthedocs.io/en/latest/>`_
* `reproject <https://reproject.readthedocs.io/en/stable/>`_
* `numba <http://numba.pydata.org/>`_
* `numexpr <https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/index.html>`_
* `pyproj <https://pyproj4.github.io/pyproj/stable/index.html>`_
* `pygsm <https://github.com/telegraphic/PyGSM>`_
* `dask <https://dask.org/>`_
* `astroplan <https://astroplan.readthedocs.io/en/latest/>`_
* `mocpy <https://cds-astro.github.io/mocpy/>`_
* `pyvo <https://pyvo.readthedocs.io/en/latest/>`_


.. note::
    To install PyGSM, follow the project instruction on their `github page <https://github.com/telegraphic/PyGSM>`_.

