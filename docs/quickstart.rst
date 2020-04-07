Quickstart
==========

This quick tutorial assumes that the ``nenupytf`` package is properly installed (see :ref:`Installation`).

Observation directory
---------------------

*NenuFAR/UnDySPuTeD* high-rate time-frequency data files are usually gathered in one directory per observation.
These files are of several different types although the only required ones are those labelled ``'*.spectra'``, each one corresponding to the one of the ouptut lanes of *UnDySPuTeD* (to a maximum of four although there may be less depending on the observation configuration).

For the rest of this tutorial, ``observation_XX`` is the observation name and the data are stored in ``'/path/to/observation'``: 

.. code-block:: bash

   $ ls /path/to/observation

   observation_XX_0.spectra
   observation_XX_1.spectra
   observation_XX_2.spectra
   observation_XX_3.spectra

Observation properties
----------------------

Before running into the analysis, it may be usefull to understand what is inside these different lane files.
The command-line ``nenupytf-info`` is here for that. It takes the path to the observation directory as argument:

.. code-block:: bash

    $ nenupytf-info -o /path/to/observation

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_0.spectra
    Lane: 0
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 11.9140625 -- 49.4140625 MHz
    Beams: [0]
    ----------------------------------------

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_1.spectra
    Lane: 1
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 49.4140625 -- 86.9140625 MHz
    Beams: [0]
    ----------------------------------------

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_2.spectra
    Lane: 2
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 11.9140625 -- 49.4140625 MHz
    Beams: [1]
    ----------------------------------------

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_3.spectra
    Lane: 3
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 49.4140625 -- 86.9140625 MHz
    Beams: [1]
    ----------------------------------------

This command prints all the relevant informations about each lane file:

* the path
* the lane index
* the minimal and maximal UT times
* the minimal and maximal frequencies
* the beam index/indices 

This can also be retrieved within the Python interpreter (see :ref:`Initialization`).

Initialization
--------------

The :class:`.Spectrum` object is the most relevant to handle a given observation. In order to instanciate such object, it needs to be provided with a path to an observation directory (by default the current directory is used):

.. code-block:: python

    >>> from nenupytf.read import Spectrum
    >>> spectrum = Spectrum('/path/to/observation') 

Observation properties can also be printed directly:

.. code-block:: python

    >>> spectrum.info()

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_0.spectra
    Lane: 0
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 11.9140625 -- 49.4140625 MHz
    Beams: [0]
    ----------------------------------------

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_1.spectra
    Lane: 1
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 49.4140625 -- 86.9140625 MHz
    Beams: [0]
    ----------------------------------------

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_2.spectra
    Lane: 2
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 11.9140625 -- 49.4140625 MHz
    Beams: [1]
    ----------------------------------------

    --------------- nenupytf ---------------
    Info on /path/to/observation/observation_XX_3.spectra
    Lane: 3
    Time: 2019-11-28T19:01:45.0000000 -- 2019-11-29T02:59:34.7455616
    Frequency: 49.4140625 -- 86.9140625 MHz
    Beams: [1]
    ----------------------------------------


Data selection
--------------

Data selection can then be easily achieved with :func:`Spectrum.select`.
The following example illustrates the selection of the first second of data from the beam 0 between 30 and 55 MHz (i.e. frequencies that are spead over the lanes 0 and 1. The result is stored in the  ``spec`` variable.

.. code-block:: python

    >>> spec = spectrum.select(
            stokes='I'
            time=['2019-11-28T19:01:45.0000000', '2019-11-28T19:01:46.0000000'],
            freq=[30, 55],
            beam=0
        )

Averaging
---------

Data can be averaged in time and frequency with :func:`Spectrum.average`. The following example shows how to average 5 min of data with a 0.1 MHz frequency resolution and a 0.1 sec time resolution:

.. code-block:: python

    >>> spec = spectrum.average(
            stokes='I'
            time=['2019-11-28T19:02:00.0000000', '2019-11-28T19:07:00.0000000'],
            freq=[30, 55],
            beam=0,
            dt=0.1,
            df=0.1
        )

Displaying the data
-------------------

To quickly display the data:

.. code-block:: python

    >>> from nenupytf.display import plotdb
    >>> plotdb(spec)



