Installation
============

The *NenuFAR* Python3 package to read and analyze *UnDySPuTeD* data, ``nenupytf``, is designed 
to be very modular and portable. Hence the simplicity of its installation
and the limited package dependencies.


Installing with pip
-------------------

These instructions cover installation with the Python package
management tool `PyPI <https://pypi.org/project/nenupytf/>`_.
Python3 is required and no support will be provided for Python2.

``nenupytf`` can simply be installed by running:

.. code-block:: bash

   $ pip install nenupytf

Updates need to be regularly check for while this package is still in developpment:

.. code-block:: bash

   $ pip install nenupytf --upgrade

.. note:: 

    There is no support for a ``conda`` insatallation yet.


Dependencies
------------

``nenupytf`` requires the following dependencies:

* `numpy <https://docs.scipy.org/doc/numpy/reference/>`_
* `matplotlib <https://matplotlib.org/3.1.1/contents.html>`_
* `astropy <https://docs.astropy.org/en/stable/>`_
* `psutil <https://psutil.readthedocs.io/en/latest/>`_

If they are not previously installed, they will be during the ``nenupytf`` installation.


Working on nancep servers
-------------------------

``nenupytf`` is already installed and ready to work on the nancep servers
at the `Station de Radioastronomie de Nancay <https://www.obs-nancay.fr/?lang=en>`_.
An account registered user may use ``nenupytf`` functionalities once environment
paths are properly set:

.. code-block:: bash

   $ use Nenupy3


Checking installation
---------------------

Once local installation is complete, or to check that ``nenupytf`` is correctly set (on a computer or on nancep servers), running the following command must show the correct path to the script ``nenupytf-info``:

.. code-block:: bash

    $ which nenupytf-info




