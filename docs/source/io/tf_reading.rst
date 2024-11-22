.. _tf_reading_doc:

UnDySPuTeD Time-Frequency Data
==============================

*UnDySPuTeD* (stands for Unified Dynamic Spectrum Pulsar and
Time Domain receiver) is the receiver of the NenuFAR
beamformer mode, fed by the (up-to-)96 core Mini-Arrays (2
polarizations) from the *LANewBa* backend. 

The raw data flow from *LANewBa* consists of 195312.5 pairs
of complex X and Y values per second per beamlet. These data
are downsampled in channels per subband (of 195.3125 kHz)
numbered from 16 to 2048 channels, ``fftlen``, (to achieve a
frequency resolution of :math:`\delta \nu = 12 - 0.1\, \rm{kHz}`
respectively). After computation of cross and auto-correlations,
the data are downsampled again in time, integrating from 4 to 1024
spectra, ``nfft2int`` (implying a time resolution :math:`195312.5^{-1} \times \rm{fftlen} \times \rm{fft2int}`,
:math:`\delta t = 0.3 - 83.9\, \rm{ms}`).

Each NenuFAR/*UnDySPuTeD*/DynSpec observation results in the
production of several proprietary formatted files (``'*.spectra'``),
each corresponding to an individual lane of the *UnDySPuTeD* receiver.
Depending on the observation configuration, the bandwidth and/or
the different observed beams (i.e., beamforming in different sky
directions) can be spread accross these files.

.. seealso::
    `DynSpec data product <https://nenufar.obs-nancay.fr/en/astronomer/#data-products>`_

Reading a spectra file
----------------------

:mod:`~nenupy.io.tf` is the module designed to
read and analyze *UnDySPuTeD* DynSpec high-rate data. It
benefits from `Dask <https://docs.dask.org/en/latest/>`_, with
the possibility of reading and applying complex pipelines
to larger-than-memory data sets.

The class :class:`~nenupy.io.tf.Spectra` offers
the possibility to read and analyze these observation files:

.. code-block:: python

    >>> from nenupy.io import Spectra
    >>> sp = Spectra(filename="my_file_0.spectra")

Once a *DynSpec* file is 'lazy'-read/loaded (i.e., without
being directly stored in memory), and before applying any processing,
it might be handy to check the data properties.
Basic information may be displayed by the :meth:`~nenupy.io.tf.Spectra.info` method:

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

Data access and processing are both achieved via a single method :meth:`~nenupy.io.tf.Spectra.get`.
The definition of such process is made by configuring the *pipeline* that will be
applied every time :meth:`~nenupy.io.tf.Spectra.get` is called.

Predefined pipeline steps
^^^^^^^^^^^^^^^^^^^^^^^^^

Each :class:`~nenupy.io.tf.Spectra` object contains an associated
:class:`~nenupy.io.tf.TFPipeline` object (stored in its 
:attr:`~nenupy.io.tf.Spectra.pipeline` attribute).
Depending on the pipeline configuration, some :class:`~nenupy.io.tf.TFTask`
may be successively called to process the data.

The :class:`~nenupy.io.tf.TFPipeline` consists of several steps, or tasks,
which can be displayed using :meth:`~nenupy.io.tf.TFPipeline.info`:

.. code-block:: python

    >>> sp.pipeline.info()
    Pipeline configuration:
        0 - Correct bandpass
        (1 - Remove subband channels)
        (2 - Rebin in time)
        (3 - Rebin in frequency)
        4 - Compute Stokes parameters

.. note::
    Some tasks are displayed in parentheses, which means that even though they
    are included in the pipeline, the current configuration does not make them
    do anything to the data. For instance, no channels to be flagged are listed,
    or no time/frequency rebin values have been specified.

.. seealso::
    :class:`~nenupy.io.tf.TFTask` lists all the pre-defined tasks available.

The pipeline tasks are using the parameters listed in the 
:attr:`~nenupy.io.tf.TFPipeline.parameters` attribute (returning a 
:class:`~nenupy.io.tf_utils.TFPipelineParameters` object) as their configuration.
One can access the current state of these parameters by calling 
:meth:`~nenupy.io.tf_utils.TFPipelineParameters.info`:

.. code-block:: python

    >>> print( sp.pipeline.parameters.info() )
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

.. _custom_pipeline_param_doc:

Pipeline parameter modification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters may be modified like a dictionnary would.
See their list and description in the documentation of :meth:`~nenupy.io.tf.Spectra.get`.
Some checks are made to assert the correct formatting of the values and/or their
relevance regarding the loaded dataset.
For instance, the time and frequency range of the data selection to be applied 
(the "step-0" of the pipeline), can be defined using `astropy` or not:

.. code-block:: python

    >>> from astropy.time import Time
    >>> import astropy.units as u

    >>> sp.pipeline.parameters["tmin"] = "2023-05-27T08:40:00"
    >>> sp.pipeline.parameters["tmax"] = Time("2023-05-27 08:42:00", format="iso")
    >>> sp.pipeline.parameters["fmin"] = 50
    >>> sp.pipeline.parameters["fmax"] = 55*u.MHz

.. note::
    The user may also update the pipeline parameters as arguments while calling
    :meth:`~nenupy.io.tf.Spectra.get`. This may be convenient for fast modification
    and won't affect future settings as they are forgotten after their usage
    (contrary to :attr:`~nenupy.io.tf.Spectra.parameters`).

Managing pipeline tasks
^^^^^^^^^^^^^^^^^^^^^^^

Tasks may be removed from the pipeline using the :meth:`~nenupy.io.tf.TFPipeline.remove`
method, taking as input the index of the task in the pipeline list:

.. code-block:: python

    >>> sp.pipeline.remove(2) # remove the (2 - Rebin in time) task
    >>> sp.pipeline.info()
    Pipeline configuration:
        0 - Correct bandpass
        (1 - Remove subband channels)
        (2 - Rebin in frequency)
        3 - Compute Stokes parameters

Alternatively, :class:`~nenupy.io.tf.TFTask` may be added using 
:meth:`~nenupy.io.tf.TFPipeline.insert` or :meth:`~nenupy.io.tf.TFPipeline.append`
methods:

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

.. warning::
    The order in which :class:`~nenupy.io.tf.TFTask` are listed represents
    their calling sequence in the pipeline. It is then crucial to assert that
    a given task can ingest the data processed through the previous tasks.
    For instance, it would make no sense to configure the channel removal task
    after the rebinning in frequency. It would even result in an error since the
    data shape would not match what the channel removal task is expecting.

.. _custom_task_doc:

Adding custom steps
^^^^^^^^^^^^^^^^^^^

The :class:`~nenupy.io.tf.TFTask` class is flexible enough to allow the user
defining their own data processing steps.
This is a more advanced operation as it requires to dive a bit within the `nenupy`
code, but it is also very convenient to test new methods without (or rather before)
having to update the source code.
Here is a basic example:

.. code-block:: python

    >>> from nenupy.io.tf import TFTask

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

Running the pipeline
--------------------

After the tasks have been listed, the parameters have been set, the pipeline
can be run at once by calling :meth:`~nenupy.io.tf.Spectra.get`.
The minimal operation on this call is to select the data based on the time and
franquency range defined in :attr:`~nenupy.io.tf.Spectra.parameters`, as well
as the numerical beam index.
Although `Dask <https://docs.dask.org/en/latest/>`_ allows for operations on
large datasets, it is wise to consider the output volume and/or computing
ressource that would be required for a given pipeline configuration. 

.. code-block:: python

    >>> data = sp.get(stokes="I")

It is also possible to modify the pipeline parameters on the fly while calling
:meth:`~nenupy.io.tf.Spectra.get`. Their values will however be forgotten once
after the method resolution:

.. code-block:: python
    
    >>> data = sp.get(stokes="I", tmin="2023-05-27T08:41:30")

.. note::

    There is a hardcoded size limit to the data output (i.e. after rebinning and all other pipeline operations) fixed at 2 GB, to prevent memory issues.
    Users willing to bypass this limit may explicitely ask for it using the ``ignore_volume_warning`` properties of :meth:`~nenupy.io.tf.Spectra.pipeline`.
    This property can easily be updated directly by the :meth:`~nenupy.io.tf.Spectra.get` method:

    .. code-block:: python

        >>> sp.get(
                tmin="2023-05-27T08:40:00", tmax="2023-05-27T18:00:00",
                ignore_volume_warning=True
            )


Saving the data
---------------

The result of the pipeline operation may also be saved in a HDF5 file if the
argument ``file_name`` is provided.
The saved data volume may be larger than the available memory.

.. code-block:: python
    :emphasize-lines: 2

    >>> sp.get(
            file_name="/my/path/filename.hdf5"
            stokes="I",
            tmin="2023-05-27T08:41:30"
        )
