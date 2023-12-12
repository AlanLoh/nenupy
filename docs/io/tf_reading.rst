.. _tf_reading_doc:

UnDySPuTeD Time-Frequency Data
==============================

blabala see tests :class:`~nenupy.io.tf.Spectra` or :meth:`~nenupy.astro.astro_tools.altaz_to_radec`


Reading a spectra file
-----------------------

:class:`~nenupy.io.tf.Spectra`

.. code-block:: python

    >>> from nenupy.io import Spectra
    >>> sp = Spectra(filename="my_file_0.spectra")


:meth:`~nenupy.io.tf.Spectra.info`

.. code-block:: python

    >>> sp.info()
    filename: my_file_0.spectra
    time_min: 2023-05-27T08:39:02.0000050
    time_max: 2023-05-27T08:59:34.2445748
    dt: 20.97152 ms
    frequency_min: 19.921875 MHz
    frequency_max: 57.421875 MHz
    df: 3.0517578125 kHz
    Available beam indices: ['0']


Pipeline configuration
----------------------

Predefined pipeline steps
^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`~nenupy.io.tf.Spectra.pipeline` :class:`~nenupy.io.tf.TFPipeline` :class:`~nenupy.io.tf.TFTask`

.. code-block:: python

    >>> sp.pipeline.info()
    Pipeline configuration:
        0 - Correct bandpass
        (1 - Remove subband channels)
        (2 - Rebin in time)
        (3 - Rebin in frequency)
        4 - Compute Stokes parameters

.. code-block:: python

    >>> sp.pipeline_parameters.info()
    channels: 64
    dt: 0.02097152 s
    df: 3051.7578125 Hz
    tmin: 2023-05-27T08:39:02.0000050
    tmax: 2023-05-27T08:59:34.2445748
    fmin: 19.921875 MHz
    fmax: 57.421875 MHz
    beam: 0
    dispersion_measure: None
    rotation_measure: None
    rebin_dt: None
    rebin_df: None
    remove_channels: None
    dreambeam_skycoord: None
    dreambeam_dt: None
    dreambeam_parallactic: True
    stokes: I
    ignore_volume_warning: False


Pipeline parameter modification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> from astropy.time import Time
    >>> import astropy.units as u

    >>> sp.pipeline.parameters["tmin"] = "2023-05-27T08:40:00"
    >>> sp.pipeline.parameters["tmax"] = Time("2023-05-27 08:42:00", format="iso")
    >>> sp.pipeline.parameters["fmin"] = 50
    >>> sp.pipeline.parameters["fmax"] = 55*u.MHz

Managing pipeline tasks
^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~nenupy.io.tf.TFPipeline.remove`

.. code-block:: python

    >>> sp.pipeline.remove(2)
    >>> sp.pipeline.info()
    Pipeline configuration:
        0 - Correct bandpass
        (1 - Remove subband channels)
        (2 - Rebin in frequency)
        3 - Compute Stokes parameters

:meth:`~nenupy.io.tf.TFPipeline.insert` :meth:`~nenupy.io.tf.TFPipeline.append`

.. code-block:: python

    >>> from nenupy.io.tf import TFTask
    >>> sp.pipeline.insert(TFTask.time_rebin(), 1)
    >>> sp.pipeline.info()
    Pipeline configuration:
        0 - Correct bandpass
        (1 - Rebin in time)
        (2 - Remove subband channels)
        (3 - Rebin in frequency)
        4 - Compute Stokes parameters

Adding custom steps
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> from nenupy.io.tf import TFTask
    >>> 
    >>> custom_task = TFTask(
            name="my task - multiply the data by n_channels",
            func=lambda data, channels: data*channels,
            args_to_update=["channels"]
        )
    >>> sp.pipeline.insert(custom_task, 3)
    >>> sp.pipeline.info()
    Pipeline configuration:
        0 - Correct bandpass
        (1 - Rebin in time)
        (2 - Remove subband channels)
        3 - my task - multiply the data by n_channels
        (4 - Rebin in frequency)
        5 - Compute Stokes parameters


Getting the data
----------------

:meth:`~nenupy.io.tf.Spectra.get`

.. code-block:: python

    >>> data = sp.get(stokes="I")


.. code-block:: python
    
    >>> data = sp.get(stokes="I", tmin="2023-05-27T08:41:30")

.. note::

    There is a hardcoded size limit to the data output (i.e. after rebinning and all other pipeline operations) fixed at 2 GB, to prevent memory issues.
    Users willing to bypass this limit may explicitely ask for it using the `ignore_data_size` argument of :meth:`~nenupy.io.tf.Spectra.get`:

    .. code-block:: python

        >>> sp.get(tmin="2023-05-27T08:40:00", tmax="2023-05-27T18:00:00", ignore_data_size=True)


