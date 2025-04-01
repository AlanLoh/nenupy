"""
    *******************
    Time-frequency data
    *******************

    .. inheritance-diagram:: nenupy.io.tf.Spectra
        :parts: 2

"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import astropy.units as u
from astropy.time import Time
from typing import Union, Tuple, List, Any, Callable
from functools import partial
import inspect
import logging

log = logging.getLogger(__name__)

import nenupy.io.tf_utils as utils
from nenupy.beamlet import SData


__all__ = ["TFTask", "TFPipeline", "Spectra"]


BLOCK_HEADER = [
    ("idx", "uint64"),  # first effective sample index used for this block
    ("TIMESTAMP", "uint64"),  # TIMESTAMP of first sample used for this block
    ("BLOCKSEQNUMBER", "uint64"),  # BLOCKSEQNUMBER of first sample used for this block
    ("fftlen", "int32"),  # fftlen : FFT length -> freq resolution= 0.1953125/fftlen MHz
    (
        "nfft2int",
        "int32",
    ),  # nfft2int : Nber of integrated FFTs -> time resolution= fftlen*nfft2int*5.12 us
    ("fftovlp", "int32"),  # fftovlp : FFToverlap : 0 (for none) or fftlen/2
    (
        "apodisation",
        "int32",
    ),  # apodisation with 0 : none (01 to 99 : a generic K=1 cosine-sum window with coefficient 0.xx, 101 : hamming window, 102 : hann window)
    (
        "nffte",
        "int32",
    ),  # nffte : nber of FFTs to be read per beamlet within this block (so we have 'nffte' spectra/timesample per block)
    (
        "nbchan",
        "int32",
    ),  # nbchan : nber of subsequent beamlets/channels or nber of NenuFAR/LOFAR 195kHz beamlets, set to a NEGATIVE value when data packets lost ! (so we have 'fftlen*nbchan' frequencies for each spectra)
]

SUBBAND_WIDTH = 195312.5 * u.Hz

DATA_VOLUME_SECURITY_THRESHOLD = 2 * u.Gibyte


# ============================================================= #
# -------------------------- TFTask --------------------------- #
class TFTask:
    """Class to handle a single task/operation designed to be applied to time-frequency data.

    Example
    -------
    Here is a complete example of how a :class:`~nenupy.io.tf.TFTask`
    works on its own.

    .. code-block:: python
        :emphasize-lines: 26,27,28,29,30,36,39,40,41,42,43

        >>> from nenupy.io.tf import TFTask
        >>> from nenupy.io.tf_utils import TFPipelineParameters, _ValueParameter
        >>> import numpy as np

        # Generate a dataset
        >>> data = np.ones((5, 2, 1)) # time, frequency, polarization

        # Define a function that modifies the data
        >>> def multiply_by_first_axis_index(data: np.ndarray, scale: float) -> np.ndarray:
        >>>    '''A completely made-up function'''
        >>>    first_axis = np.arange(data.shape[0])
        >>>    return scale * data * first_axis[:, None, None]

        # Generate an object listing parameters available to the function(s)
        >>> custom_parameters = TFPipelineParameters(
                _ValueParameter(
                    name="scale",
                    default=1.,
                    param_type=float,
                    min_val=0.1,
                    max_val=2.0
                )
            )

        # Generate a TFTask object
        >>> custom_task = TFTask(
                name="My task - custom",
                func=multiply_by_first_axis_index,
                args_to_update=["scale"]
            )

        # Modify the value of the pipeline parameter
        >>> custom_parameters["scale"] = 1.5

        # Update the TFTask, the new parameter value is taken into account
        >>> custom_task.update(parameters=custom_parameters)

        # Call the TFTask and run it
        >>> time_unix, frequency_hz, result = custom_task(
                time_unix=None,
                frequency_hz=None,
                data=data
            )

        >>> print(result[:, 0, :].ravel())
        [0.  1.5 3.  4.5 6. ]

    """

    def __init__(
        self,
        name: str,
        func: Callable,
        args_to_update: List[str] = [],
        repeatable: bool = False,
    ):
        """Generate a :class:`~nenupy.io.tf.TFTask` instance.

        Parameters
        ----------
        name : `str`
            Name of the task (it can be anything)
        func : `Callable`
            Function that applies the operation to the data
        args_to_update : List[`str`], optional
            List of parameter names that are extracted from :class:`~nenupy.io.tf_utils.TFPipelineParameters` and are required by ``func``.
            These parameters will be updated by their current values (stored in :class:`~nenupy.io.tf_utils.TFPipelineParameters`) prior to running the task., by default []
        repeatable : `bool`, optional
            Allow or not the task to be called several times by a given pipeline, by default `False`

        See Also
        --------
        :ref:`custom_task_doc`

        """
        self.name = name
        self.is_active = False
        self._func = func
        self.args_to_update = args_to_update
        self.repeatable = repeatable
        self._extra_params = {}

    def __repr__(self) -> str:
        module = self.__class__.__module__
        class_name = self.__class__.__name__
        return f"<class '{module}.{class_name}({self.name})'>"

    @property
    def _func_call(self) -> Callable:
        return partial(self._func, **self._extra_params)

    @property
    def is_active(self) -> bool:
        """Attribute modified by :meth:`~nenupy.io.tf.TFTask.update`
        which tells whether the configured parameters enable the embedded function usage.

        Returns
        -------
        `bool`
            Whether the task will be run by the pipeline or not
        """
        return self._is_active

    @is_active.setter
    def is_active(self, isactive: bool) -> None:
        self._is_active = isactive

    @property
    def args_to_update(self) -> List[str]:
        """List of embedded function's arguments to update before running the function.
        If ``func`` contains keyword arguments, their values will be updated
        by the matching key value during the call of :meth:`~nenupy.io.tf.TFTask.update`.

        Returns
        -------
        List[`str`]
            List of function's arguments to :meth:`~nenupy.io.tf.TFTask.update`

        Raises
        ------
        KeyError
            Raised if any listed argument does not correspond with one of the function's.
        """
        return self._args_to_update

    @args_to_update.setter
    def args_to_update(self, arg_list: List[str]) -> None:
        # Check that they are relevant with respect to the function
        func_args = self._func.__code__.co_varnames[: self._func.__code__.co_argcount]
        for arg in arg_list:
            if arg not in func_args:
                raise KeyError(
                    f"args_to_update - '{arg}' does not correspond with any of the embedded function keyword arguments: {func_args}."
                )
        self._args_to_update = arg_list

    @classmethod
    def correct_bandpass(cls):
        """:class:`~nenupy.io.tf.TFTask` to correct for the sub-band bandpass response.

        A Poly-Phase Filter is involved in the NenuFAR data acquisition pipeline to split the data stream into sub-bands.
        The combination of the filter shape and a Fourier transform results in a non-flat response across each sub-band.
        This :class:`~nenupy.io.tf.TFTask` calls the :func:`~nenupy.io.tf_utils.correct_bandpass` function.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 4

            >>> from nenupy.io.tf import Spectra, TFTask, TFPipeline

            >>> sp = Spectra("/my/file.spectra")
            >>> sp.pipeline = TFPipeline(sp, TFTask.correct_bandpass())
            >>> data = sp.get(...)

        .. figure:: ../_images/io_images/tf_bandpass_correction.png
            :width: 650
            :align: center

        """

        def wrapper_task(data, channels):
            return utils.correct_bandpass(data=data, n_channels=channels)

        return cls("Correct bandpass", wrapper_task, ["channels"], repeatable=False)

    @classmethod
    def flatten_subband(cls):
        """:class:`~nenupy.io.tf.TFTask` to flatten each sub-band bandpass.
        Based on the temporal median over each suband, a linear correction is applied to flatten the signal.
        This :class:`~nenupy.io.tf.TFTask` calls the :func:`~nenupy.io.tf_utils.flatten_subband` function.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 4

            >>> from nenupy.io.tf import Spectra, TFTask, TFPipeline

            >>> sp = Spectra("/my/file.spectra")
            >>> sp.pipeline = TFPipeline(sp, TFTask.flatten_subband())
            >>> data = sp.get(...)

        .. figure:: ../_images/io_images/tf_sb_flatten.png
            :width: 650
            :align: center

        Warning
        -------
        This correction assumes that the signal's spectrum could be considered flat at the sub-band resolution.
        The method is not recommended for data other than Stokes I.

        """

        def wrapper_task(data, channels, smooth_frequency_profile):
            return utils.flatten_subband(data=data, channels=channels, smooth_frequency_profile=smooth_frequency_profile)

        return cls("Flatten subband", wrapper_task, ["channels", "smooth_frequency_profile"], repeatable=False)

    @classmethod
    def remove_channels(cls):
        """:class:`~nenupy.io.tf.TFTask` calling :func:`~nenupy.io.tf_utils.remove_channels_per_subband` to set a list of sub-band channels to `NaN` values.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 4,6

            >>> from nenupy.io.tf import Spectra, TFTask, TFPipeline

            >>> sp = Spectra("/my/file.spectra")
            >>> sp.pipeline = TFPipeline(sp, TFTask.remove_channels())
            >>> result = sp.get(
                    remove_channels=[0, 2, 4, 10, -1]
                )

        .. figure:: ../_images/io_images/tf_remove_channels.png
            :width: 650
            :align: center

        """

        def wrapper_task(data, channels, remove_channels):
            if (remove_channels is None) or (len(remove_channels) == 0):
                return data
            return utils.remove_channels_per_subband(
                data=data, n_channels=channels, channels_to_remove=remove_channels
            )

        return cls(
            "Remove subband channels",
            wrapper_task,
            ["channels", "remove_channels"],
            repeatable=False,
        )

    @classmethod
    def correct_polarization(cls):
        """:class:`~nenupy.io.tf.TFTask` calling :func:`~nenupy.io.tf_utils.apply_dreambeam_corrections`.
        This allows for beam correction (including leakage and parallactic angle correction).

        Warning
        -------
        This task must be computed early in the pipeline process as it involves full Jones operations.
        As of now, NenuFAR cross-polarization terms are not fully taken into account by DreamBeam.
        Results may vary and the tests we have made are not yet conclusive.
        We are working on an improved version.

        """

        def wrapper_task(
            time_unix,
            frequency_hz,
            data,
            dt,
            calib_dt,
            channels,
            skycoord,
            dreambeam_parallactic,
        ):
            # DreamBeam correction (beam gain + parallactic angle)
            if (
                (calib_dt is None)
                or (skycoord is None)
                or (dreambeam_parallactic is None)
            ):
                return time_unix, frequency_hz, data
            data = utils.apply_dreambeam_corrections(
                time_unix=time_unix,
                frequency_hz=frequency_hz,
                data=data,
                dt_sec=dt.to_value(u.s),
                time_step_sec=calib_dt.to_value(u.s),
                n_channels=channels,
                skycoord=skycoord,
                parallactic=dreambeam_parallactic,
            )
            return time_unix, frequency_hz, data

        return cls(
            "Polarization corection with DreamBeam",
            wrapper_task,
            [
                "channels",
                "dt",
                "skycoord",
                "calib_dt",
                "dreambeam_parallactic",
            ],
            repeatable=False,
        )

    @classmethod
    def correct_parallactic_rotation(cls):
        """:class:`~nenupy.io.tf.TFTask` calling :func:`~nenupy.io.tf_utils.correct_parallactic`.
        This allows for parallactic angle correction by aplying the inverse rotation.

        Warning
        -------
        This task must be computed early in the pipeline process as it involves full Jones operations.

        """

        def wrapper_task(
            time_unix,
            frequency_hz,
            data,
            dt,
            channels,
            calib_dt,
            skycoord,
        ):
            # DreamBeam correction (beam gain + parallactic angle)
            if (
                (calib_dt is None)
                or (skycoord is None)
            ):
                return time_unix, frequency_hz, data
            data = utils.correct_parallactic(
                time_unix=time_unix,
                frequency_hz=frequency_hz,
                data=data,
                dt_sec=dt.to_value(u.s),
                time_step_sec=calib_dt.to_value(u.s),
                n_channels=channels,
                skycoord=skycoord
            )
            return time_unix, frequency_hz, data

        return cls(
            "Parallactic corection",
            wrapper_task,
            [
                "channels",
                "dt",
                "skycoord",
                "calib_dt",
            ],
            repeatable=False,
        )

    @classmethod
    def correct_faraday_rotation(cls):
        """:class:`~nenupy.io.tf.TFTask` calling :func:`~nenupy.io.tf_utils.de_faraday_data` to correct for Faraday rotation for a given ``'rotation_measure'`` set in :attr:`~nenupy.io.tf.TFPipeline.parameters`.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 6,7,17,18

            >>> from nenupy.io.tf import Spectra, TFTask
            >>> import astropy.units as u

            >>> sp_pulsar = Spectra(".../pulsar.spectra")
            >>> sp_pulsar.pipeline.set_default()
            >>> sp_pulsar.pipeline.insert(TFTask.de_disperse(), 1)
            >>> sp_pulsar.pipeline.insert(TFTask.correct_faraday_rotation(), 2)
            >>> sp_pulsar.pipeline.info()
            >>> defaraday_data = sp_pulsar.get(
                    tmin="2023-10-02T08:40:40.8500000",
                    tmax="2023-10-02T08:40:43.16000000",
                    fmin=50,
                    fmax=65,
                    stokes=["U/I"],
                    remove_channels=[0, 1, -1],
                    rebin_df = 12207.03125 * 4 * u.Hz,
                    dispersion_measure=2.972719 * u.pc / u.cm**3,
                    rotation_measure=5.9 * u.rad / u.m**2
                )
            >>> defaraday_data[:600, :, :].plot(polarization="U/I", db=False)

        .. figure:: ../_images/io_images/tf_pulsar_defaraday.png
            :width: 650
            :align: center

        Notes
        -----
        In the above example, we first had to de-disperse the data with :meth:`~nenupy.io.tf.TFTask.de_disperse`.
        """

        def apply_faraday(frequency_hz, data, rotation_measure):
            if rotation_measure is None:
                return frequency_hz, data
            return frequency_hz, utils.de_faraday_data(
                data=data,
                frequency=frequency_hz * u.Hz,
                rotation_measure=rotation_measure,
            )

        return cls(
            "Correct faraday rotation",
            apply_faraday,
            ["rotation_measure"],
            repeatable=False,
        )

    @classmethod
    def de_disperse(cls):
        """:class:`~nenupy.io.tf.TFTask` calling :func:`~nenupy.io.tf_utils.de_disperse_array` to de-disperse the data using the ``'dispersion_measure'`` set in :attr:`~nenupy.io.tf.TFPipeline.parameters`.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 6,16

            >>> from nenupy.io.tf import Spectra, TFTask
            >>> import astropy.units as u

            >>> sp_pulsar = Spectra(".../pulsar.spectra")
            >>> sp_pulsar.pipeline.set_default()
            >>> sp_pulsar.pipeline.insert(TFTask.de_disperse(), 1)
            >>> sp_pulsar.pipeline.info()
            >>> dedisperse_data = sp_pulsar.get(
                    tmin="2023-10-02T08:40:40.8500000",
                    tmax="2023-10-02T08:40:43.16000000",
                    fmin=50,
                    fmax=65,
                    stokes=["I"],
                    remove_channels=[0, 1, -1],
                    rebin_df = 12207.03125 * 4 *u.Hz,
                    dispersion_measure=2.972719 *u.pc / (u.cm**3)
                )
            >>> dedisperse_data.plot(polarization="I", db=True)

        .. figure:: ../_images/io_images/tf_pulsar_dedispersion.png
            :width: 650
            :align: center

        Warning
        -------
        Due to the configuration of the underlying :class:`~dask.array.core.Array`, its :meth:`dask.array.Array.compute` method has to be applied priori to de-dispersing the data.
        Therefore, a potential huge data volume may be computed at once.
        By default, a security exception is raised to prevent computing a too large data set.
        To bypass this limit, set ``'ignore_volume_warning'`` of :attr:`~nenupy.io.tf.TFPipeline.parameters` to `True`.

        """

        def wrapper_task(
            frequency_hz, data, dt, dispersion_measure, ignore_volume_warning
        ):
            if dispersion_measure is None:
                return frequency_hz, data
            # Make sure the data volume is not too big!
            projected_data_volume = data.nbytes * u.byte
            if (projected_data_volume >= DATA_VOLUME_SECURITY_THRESHOLD) and (
                not ignore_volume_warning
            ):
                log.warning(
                    f"Data processing will produce {projected_data_volume.to(u.Gibyte)}."
                    f"The pipeline is interrupted because the volume threshold is {DATA_VOLUME_SECURITY_THRESHOLD.to(u.Gibyte)}."
                )
                return frequency_hz, data

            tmp_chuncks = data.chunks
            data = utils.de_disperse_array(
                data=data.compute(),  # forced to leave Dask
                frequencies=frequency_hz * u.Hz,
                time_step=dt,
                dispersion_measure=dispersion_measure,
            )
            return frequency_hz, da.from_array(data, chunks=tmp_chuncks)

        return cls(
            "De-disperse",
            wrapper_task,
            ["dt", "dispersion_measure", "ignore_volume_warning"],
            repeatable=False,
        )

    @classmethod
    def time_rebin(cls):
        """:class:`~nenupy.io.tf.TFTask` to re-bin the data in time.
        The targetted time resolution is defined by the ``'rebin_dt'`` argument, set in :attr:`~nenupy.io.tf.TFPipeline.parameters`.
        This :class:`~nenupy.io.tf.TFTask` calls the :func:`~nenupy.io.tf_utils.rebin_along_dimension` function.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra, TFTask, TFPipeline
            >>> import astropy.units as u
            >>> sp = Spectra("/my/file.spectra")
            >>> sp.pipeline = TFPipeline(sp, TFTask.time_rebin())

        Then, either perform a one_time application of the `rebin_dt` parameter (that is forgotten after the :meth:`~nenupy.io.tf.Spectra.get` call):

        .. code-block:: python

            >>> data = sp.get(..., rebin_dt=0.2*u.s,...)

        Or, set it for further usage:

        .. code-block:: python

            >>> sp.pipeline.parameters["rebin_dt"] = 0.2*u.s
            >>> data = sp.get(...)


        .. figure:: ../_images/io_images/tf_time_rebin.png
            :width: 650
            :align: center

        """

        def rebin_time(time_unix, data, dt, rebin_dt):
            if rebin_dt is None:
                return time_unix, data
            log.info("Rebinning in time...")
            return utils.rebin_along_dimension(
                data=data,
                axis_array=time_unix,
                axis=0,
                dx=dt.to_value(u.s),
                new_dx=rebin_dt.to_value(u.s),
            )

        return cls("Rebin in time", rebin_time, ["dt", "rebin_dt"], repeatable=True)

    @classmethod
    def frequency_rebin(cls):
        """:class:`~nenupy.io.tf.TFTask` to re-bin the data in frequency.
        The targetted frequency resolution is defined by the ``'rebin_df'`` argument, set in :attr:`~nenupy.io.tf.TFPipeline.parameters`.
        The minimum value is the frequency resolution of the data, i.e.
        :attr:`~nenupy.io.tf.Spectra.df`.
        This :class:`~nenupy.io.tf.TFTask` calls the :func:`~nenupy.io.tf_utils.rebin_along_dimension` function.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 5,7

            >>> from nenupy.io.tf import Spectra, TFTask, TFPipeline
            >>> import astropy.units as u

            >>> sp = Spectra("/my/file.spectra")
            >>> sp.pipeline = TFPipeline(sp, TFTask.frequency_rebin())
            >>> result = sp.get(
                    rebin_df=50 * u.kHz
                )

        .. figure:: ../_images/io_images/tf_frequency_rebin.png
            :width: 650
            :align: center

        """

        def rebin_freq(frequency_hz, data, df, rebin_df):
            if rebin_df is None:
                return frequency_hz, data
            log.info("Rebinning in frequency...")
            return utils.rebin_along_dimension(
                data=data,
                axis_array=frequency_hz,
                axis=1,
                dx=df.to_value(u.Hz),
                new_dx=rebin_df.to_value(u.Hz),
            )

        return cls(
            "Rebin in frequency", rebin_freq, ["df", "rebin_df"], repeatable=True
        )

    @classmethod
    def get_stokes(cls):
        """:class:`~nenupy.io.tf.TFTask` to compute the Stokes parameters by calling :func:`~nenupy.io.tf_utils.compute_stokes_parameters`.
        The computed Stokes parameters are defined by the  ``'stokes'`` argument, set in :attr:`~nenupy.io.tf.TFPipeline.parameters`.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra, TFTask, TFPipeline

            >>> sp = Spectra("/my/file.spectra")
            >>> sp.pipeline = TFPipeline(sp, TFTask.get_stokes())
            >>> result = sp.get(stokes=["I", "U/I", "Q/I", "V/I"])

            >>> result.plot(polarization="U/I") # to display

        .. figure:: ../_images/io_images/tf_stokes.png
            :width: 650
            :align: center

        """

        def compute_stokes(data, stokes):
            if (stokes is None) or (stokes == "") or (stokes == []):
                return data
            return utils.compute_stokes_parameters(data_array=data, stokes=stokes)

        return cls(
            "Compute Stokes parameters", compute_stokes, ["stokes"], repeatable=False
        )

    def update(self, parameters: utils.TFPipelineParameters) -> None:
        """Update the embedded function's arguments to those listed in ``parameters``
        that match :attr:`~nenupy.io.tf.TFTask.args_to_update`.
        Determine also if the :class:`~nenupy.io.tf.TFTask` is active or not, i.e.,
        if the configured parameters enable it to be run.

        Parameters
        ----------
        parameters : :class:`~nenupy.io.tf_utils.TFPipelineParameters`
            Pipeline parameters object, it must contain the parameter requested by :attr:`~nenupy.io.tf.TFTask.args_to_update`.

        Raises
        ------
        KeyError
            Raised if the parameters listed in :attr:`~nenupy.io.tf.TFTask.args_to_update` do not correspond with those listed in ``parameters``.
        """

        log.debug(f"Updating TFTask {self.name} parameters before running it.")
        for key in self.args_to_update:
            try:
                value = parameters[key]
            except KeyError:
                raise
            log.debug(f"TFTask {self.name}: setting {key} to {value}")

            self._extra_params[key] = value

        # Check if func_call returns the correct function
        if not np.any([val is None for _, val in self._func_call.keywords.items()]):
            # If all keyword argument are None set the function as inactive
            self.is_active = True

    def __call__(
        self,
        time_unix: np.ndarray,
        frequency_hz: np.ndarray,
        data: da.Array,
        **kwds: Any,
    ) -> Any:
        """Run the :class:`~nenupy.io.tf.TFTask`.
        Before running the tasks, it is recommended to call
        :meth:`~nenupy.io.tf.TFTask.update` first in order to take
        into account the most up-to-date values of the parmaeters that
        may be used by the task.

        Parameters
        ----------
        time_unix : :class:`~numpy.ndarray`
            Time description of the data array, in unix format (array of `float`)
        frequency_hz : :class:`~numpy.ndarray`
            Frequency description of the data array, in Hz (array of `float`)
        data : :class:`~numpy.ndarray` or :class:`~dask.array.Array`
            Data to be processed
        **kwds : Any
            Extra keyword arguments directly passed to the embedded function

        Returns
        -------
        Tuple[:class:`~numpy.ndarray`, :class:`~numpy.ndarray`, :class:`~numpy.ndarray` or :class:`~dask.array.Array`]
            (time_unix, frequency_hz, processed data)
        """
        func_args = inspect.getfullargspec(self._func).args

        if ("time_unix" not in func_args) and ("frequency_hz" not in func_args):
            # Only data
            data = self._func_call(data=data, **kwds)
        elif "time_unix" not in func_args:
            # Missing time_unix
            frequency_hz, data = self._func_call(
                data=data, frequency_hz=frequency_hz, **kwds
            )
        elif "frequency_hz" not in func_args:
            # Missing frequency_hz
            time_unix, data = self._func_call(data=data, time_unix=time_unix, **kwds)
        else:
            # Not missing anything
            time_unix, frequency_hz, data = self._func_call(
                time_unix=time_unix, frequency_hz=frequency_hz, data=data, **kwds
            )
        return time_unix, frequency_hz, data


# ============================================================= #
# ------------------------ TFPipeline ------------------------- #
class TFPipeline:
    """Class to manage the series of tasks designed to be applied to
    time-frequency data.
    """

    def __init__(self, data_obj: Any, *tasks: TFTask):
        """Generate an instance of :class:`~nenupy.io.tf.TFPipeline`.

        Parameters
        ----------
        data_obj : :class:`~nenupy.io.tf.Spectra`
            An instance of :class:`~nenupy.io.tf.Spectra` needed to determine
            the properties of the corresponding data file.
        *tasks : :class:`~nenupy.io.tf.TFTask`
            A sequence of tasks that will be applied when the :meth:`~nenupy.io.tf.TFPipeline.run` method is called.
        """
        self.data_obj = data_obj

        # Set the predefined parameter list and initialize the
        # boundaries from the data_obj
        self.parameters = utils.TFPipelineParameters.set_default(
            time_min=self.data_obj.time_min,
            time_max=self.data_obj.time_max,
            freq_min=self.data_obj.frequency_min,
            freq_max=self.data_obj.frequency_max,
            beams=np.array(list(self.data_obj._beam_indices_dict.keys())).astype(int),
            channels=self.data_obj.n_channels,
            dt=self.data_obj.dt,
            df=self.data_obj.df,
        )

        self.tasks = list(tasks)

    def __repr__(self) -> str:
        return self.info()

    def __iter__(self):
        yield from self.tasks

    @property
    def parameters(self) -> utils.TFPipelineParameters:
        """Attribute listing all the available parameters.
        Each time a :class:`~nenupy.io.tf.TFTask` is run, its :meth:`~nenupy.io.tf.TFTask.update`
        method is called and this parameter list is passed.

        Returns
        -------
        :class:`~nenupy.io.tf_utils.TFPipelineParameters`
            Pipeline parameters.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, params: utils.TFPipelineParameters) -> None:
        self._parameters = params

    def info(self, return_str: bool = False) -> str:
        """Display the current pipeline configuration.
        The tasks are ordered as they will be applied.
        Tasks in parenthesis are not considered 'activated'.
        This can happen because the task is looking for an argument in :attr:`~nenupy.io.tf.TFPipeline.parameters` whose value does not fulfill the requirements.

        Parameters
        ----------
        return_str : `bool`, optional
            Return the information message as a string variable, by default `False`.

        Returns
        -------
        `str`
            If `return_str` is set to `True`, the message is returned as a string variable.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 4

            >>> from nenupy.io.tf import Spectra, TFPipeline, TFTask
            >>> sp = Spectra("/my/file.spectra")
            >>> pipeline = TFPipeline(sp, TFTask.correct_bandpass(), TFTask.correct_faraday_rotation())
            >>> pipeline.info()
            Pipeline configuration:
                0 - Correct bandpass
                (1 - Correct faraday rotation)

        """
        message = "Pipeline configuration:"
        for i, task in enumerate(self.tasks):
            task.update(self.parameters)
            if task.is_active:
                message += f"\n\t{i} - {task.name}"
            else:
                message += f"\n\t({i} - {task.name})"
        if return_str:
            return message
        else:
            print(message)

    def set_default(self) -> None:
        """Set the default pipeline. This will reset any existing list of :class:`~nenupy.io.tf.TFTask`.
        This method also reset the parameters configuration (listed in :attr:`~nenupy.io.tf.TFPipeline.parameters`).
        The list of tasks is:

        - Bandpass correction (:meth:`~nenupy.io.tf.TFTask.correct_bandpass`)
        - Remove channels at the subband edges (:meth:`~nenupy.io.tf.TFTask.remove_channels`)
        - Rebin the data in time (:meth:`~nenupy.io.tf.TFTask.time_rebin`)
        - Rebin the data in frequency (:meth:`~nenupy.io.tf.TFTask.frequency_rebin`)
        - Convert the data to Stokes parameters (:meth:`~nenupy.io.tf.TFTask.get_stokes`)

        """

        self.parameters.reset()
        self.tasks = [
            TFTask.correct_bandpass(),
            TFTask.remove_channels(),
            TFTask.time_rebin(),
            TFTask.frequency_rebin(),
            TFTask.get_stokes(),
        ]

    def insert(self, operation: TFTask, index: int) -> None:
        """Insert a pipeline task at a given position among the list of planned tasks.

        Parameters
        ----------
        operation : :class:`~nenupy.io.tf.TFTask`
            The task to perform.
        index : `int`
            Index at which the `operation` should be inserted (see :meth:`list.insert`).

        Raises
        ------
        TypeError
            If `operation` is not a :class:`~nenupy.io.tf.TFTask`.
        """
        if operation.__class__.__name__ != TFTask.__name__:
            raise TypeError(f"Tried to append {type(operation)} instead of {TFTask}.")
        if self.contains(operation.name) and (not operation.repeatable):
            log.warning(
                f"{operation} is already registered in the pipeline and is not repeatable."
            )
            return
        self.tasks.insert(index, operation)

    def append(self, operation: TFTask) -> None:
        """Append a pipeline task at the end of the list of planned tasks.

        Parameters
        ----------
        operation : :class:`~nenupy.io.tf.TFTask`
            The task to perform.

        Raises
        ------
        TypeError
            If `operation` is not a :class:`~nenupy.io.tf.TFTask`.
        """
        if operation.__class__.__name__ != TFTask.__name__:
            raise TypeError(f"Tried to append {type(operation)} instead of {TFTask}.")
        if self.contains(operation.name) and (not operation.repeatable):
            log.warning(
                f"{operation} is already registered in the pipeline and is not repeatable."
            )
            return
        self.tasks.append(operation)

    def remove(self, *args: Union[str, int]) -> None:
        """Remove a task from the list.

        Parameters
        ----------
        *args : `str` or `int`
            If an integer is found, the task indexed at the corresponding value will be removed.
            If a string is found, the first instance of the :class:`~nenupy.io.tf.TFTask` whose :attr:`~nenupy.io.tf.TFTask.name` corresponds to `arg` will be removed.

        Raises
        ------
        TypeError
            If `args` contains any item that is neither `int` nor `str`.

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 10

            >>> from nenupy.io.tf import Spectra, TFPipeline, TFTask
            >>> pipeline = TFPipeline( Spectra("/my/file.spectra") )
            >>> pipeline.set_default()
            >>> pipeline.info()
            Pipeline configuration:
                0 - Correct bandpass
                (1 - Remove subband channels)
                (2 - Rebin in time)
                (3 - Rebin in frequency)
                4 - Compute Stokes parameters
            >>> pipeline.remove("Rebin in frequency", 1)
            >>> pipeline.info()
            Pipeline configuration:
                0 - Correct bandpass
                (1 - Rebin in time)
                2 - Compute Stokes parameters

        """
        for arg in args:
            if isinstance(arg, str):
                try:
                    index = [op.name for op in self.tasks].index(arg)
                except ValueError as error:
                    print(f"{error}, ignoring...")
                    continue
            elif isinstance(arg, int):
                index = arg
            else:
                raise TypeError()
            log.info(f"Removing task '{self.tasks[index].name}'.")
            del self.tasks[index]

    def contains(self, task_name: str) -> bool:
        """Check if the pipeline contains a :class:`~nenupy.io.tf.TFTask` named ``task_name``.

        Parameters
        ----------
        task_name : `str`
            Name of the :class:`~nenupy.io.tf.TFTask`.

        Returns
        -------
        `bool`
            Whether the task is present or not.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra, TFPipeline
            >>> pipeline = TFPipeline( Spectra("/my/file.spectra") )
            >>> pipeline.contains("Correct bandpass")
            True

        """
        return np.any([op.name == task_name for op in self.tasks])

    def run(
        self, time_unix: np.ndarray, frequency_hz: np.ndarray, data: da.Array
    ) -> Tuple[np.ndarray, np.ndarray, da.Array]:
        """Run all the tasks registered in the pipeline.

        Parameters
        ----------
        time_unix : :class:`~numpy.ndarray`
            Original array of time samples in Unix format (should match the first dimension of `data`).
        frequency_hz : :class:`~numpy.ndarray`
            Original array of frequency samples in Hz (should match the second dimension of `data`)
        data : :class:`~dask.array.Array`
            Original data array, its shape should be `(time, frequency, 2, 2)`.

        Returns
        -------
        (:class:`~numpy.ndarray`, :class:`~numpy.ndarray`, :class:`~dask.array.Array`)
            Returns quantities equivalent to the inputs.
            The sizes of `time_unix` and `frequency_hz` may have change (if rebinning steps have been included in the pipeline for instance).
            The first two dimensions of `data` should still match those of `time_unix` and `frequency_hz`.
            The last dimensions depend on the polarization computation.
        """
        for task in self.tasks:
            task.update(self.parameters)
            time_unix, frequency_hz, data = task(time_unix, frequency_hz, data)
        return time_unix, frequency_hz, data


# ============================================================= #
# -------------------------- Spectra -------------------------- #
class Spectra:
    """Class to read UnDySPuTeD Time-Frequency files (.spectra extension).

    Notes
    -----
    .. versionadded:: 2.6.0

    """

    def __init__(self, filename: str, check_missing_data: bool = True):
        """Generate an instance of :class:`~nenupy.io.tf.Spectra`.

        Parameters
        ----------
        filename : `str`
            Path to the *UnDySPuTeD* file to read.
        check_missing_data : `bool`, optional
            Check for missing data, not checking may speed up the reading time, by default `False`

        Raises
        ------
        ValueError
            If the `filename` extension differs from '.spectra'.
        """

        self.filename = filename

        # Decode the main header and lazy load the data
        self._n_time_per_block = 0
        self.n_channels = 0
        self.n_subbands = 0
        self.dt = None
        self.df = None
        data = self._lazy_load_data()

        # Compute the boolean mask of bad blocks
        bad_block_mask = self._get_bad_data_mask(
            data, bypass_verification=not check_missing_data
        )

        self._missing_data_time_indices = np.where(bad_block_mask)[0]

        # Compute the main data block descriptors (time / frequency / beam)
        log.info("Computing time-frequency axes...")
        subband_width_hz = SUBBAND_WIDTH.to_value(u.Hz)
        # self._block_start_unix = (
        #     data["TIMESTAMP"][~bad_block_mask]
        #     + data["BLOCKSEQNUMBER"][~bad_block_mask] / subband_width_hz
        # )
        self._block_start_unix = (
            data["TIMESTAMP"]
            + data["BLOCKSEQNUMBER"] / subband_width_hz
        )
        # Reconstruct missing times
        self._block_start_unix[bad_block_mask] = np.interp(
            np.where(bad_block_mask)[0],
            np.arange(self._block_start_unix.size)[~bad_block_mask],
            self._block_start_unix[~bad_block_mask]
        )

        self._subband_start_hz = (
            data["data"]["channel"][0, :] * subband_width_hz
        )  # Assumed constant over time
        self._beam_indices_dict = utils.sort_beam_edges(
            beam_array=data["data"]["beam"][0],  # Asummed same for all time step
            n_channels=self.n_channels,
        )

        # Transform the data in Dask Array, once correctly reshaped
        self.data = self._to_dask_tf(data=data, mask=bad_block_mask)

        self.pipeline = TFPipeline(self)
        self.pipeline.set_default()

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def filename(self) -> str:
        """`str` : Path to the .spectra file."""

        return self._filename

    @filename.setter
    def filename(self, name: str):
        log.info(f"Reading {name}...")
        if not name.endswith(".spectra"):
            raise ValueError("A file whose extension is '.spectra' is expected.")
        self._filename = name

    @property
    def time_min(self) -> Time:
        """:class:`~astropy.time.Time` : Starting time of the data content.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra
            >>> sp = Spectra("/my/file.spectra")
            >>> sp.time_min.isot
            '2023-05-27T08:39:02.0000050'

        """
        time = Time(self._block_start_unix[0], format="unix", precision=7)
        time.format = "isot"
        return time

    @property
    def time_max(self) -> Time:
        """:class:`~astropy.time.Time` : Final time of the data content.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra
            >>> sp = Spectra("/my/file.spectra")
            >>> sp.time_max.isot
            '2023-05-27T08:59:34.2445748'

        """
        block_dt_sec = self._n_time_per_block * self.dt.to_value(u.s)
        time = Time(
            self._block_start_unix[-1] + block_dt_sec, format="unix", precision=7
        )
        time.format = "isot"
        return time

    @property
    def frequency_min(self) -> u.Quantity:
        """:class:`~astropy.units.Quantity` : Lowest recorded frequency.
        This value is defined at the channel granularity, i.e. it corresponds to the first channel of the lowest sub-band.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra
            >>> sp = Spectra("/my/file.spectra")
            >>> sp.frequency_min
            19.921875 MHz

        """
        freq_mins = []
        for _, boundaries in self._beam_indices_dict.items():
            sb_index = int(boundaries[0] / self.n_channels)
            freq_mins.append(self._subband_start_hz[sb_index])
        freq = np.min(freq_mins) * u.Hz
        return freq.to(u.MHz)

    @property
    def frequency_max(self) -> u.Quantity:
        """:class:`~astropy.units.Quantity` : Highest recorded frequency.
        This value is defined at the channel granularity, i.e. it corresponds to the last channel of the highest sub-band.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra
            >>> sp = Spectra("/my/file.spectra")
            >>> sp.frequency_max
            57.421875 MHz

        """
        freq_maxs = []
        for _, boundaries in self._beam_indices_dict.items():
            sb_index = int((boundaries[1] + 1) / self.n_channels - 1)
            freq_maxs.append(self._subband_start_hz[sb_index])
        freq = np.max(freq_maxs) * u.Hz + SUBBAND_WIDTH
        return freq.to(u.MHz)

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def info(self) -> None:
        """Display informations about the file.

        Example
        -------
         .. code-block:: python

            >>> from nenupy.io.tf import Spectra

            >>> sp = Spectra("/my/file.spectra")
            >>> sp.info()
            filename: /my/file.spectra
            time_min: 2023-05-27T08:39:02.0000050
            time_max: 2023-05-27T08:59:34.2445748
            dt: 20.97152 ms
            frequency_min: 19.921875 MHz
            frequency_max: 57.421875 MHz
            df: 3.0517578125 kHz
            Available beam indices: ['0']

        """
        message = "\n".join(
            [
                f"filename: {self.filename}",
                f"time_min: {self.time_min.isot}",
                f"time_max: {self.time_max.isot}",
                f"dt: {self.dt.to(u.ms)}",
                f"frequency_min: {self.frequency_min.to(u.MHz)}",
                f"frequency_max: {self.frequency_max.to(u.MHz)}",
                f"df: {self.df.to(u.kHz)}",
                f"Available beam indices: {list(self._beam_indices_dict.keys())}",
            ]
        )
        print(message)

    def get(self, file_name: str = None, **pipeline_kwargs) -> SData:
        r"""Perform data selection and pipeline computation.

        Parameters
        ----------
        file_name : `str`, default: ``None``
            If different than ``None`` (default value), name of the HDF5 file (extension '.hdf5') to create and store the result.
        **pipeline_kwargs
            Any :attr:`~nenupy.io.tf.Spectra.pipeline` parameter passed as keyword argument from the list below.
            Changes applied here are not kept once the method has resolved.

        Other Parameters
        ----------------
        tmin : `str` or :class:`~astropy.time.Time`, default: :math:`{\rm min}(t)`
            Lower edge of time selection, can either be given as a :class:`~astropy.time.Time` object or an ISOT/ISO string.
        tmax : `str` or :class:`~astropy.time.Time`, default: :math:`{\rm max}(t)`
            Upper edge of time selection, can either be given as an :class:`~astropy.time.Time` object or an ISOT/ISO string.
        fmin : `float` or :class:`~astropy.units.Quantity`, default: :math:`{\rm min}(\nu)`
            Lower frequency boundary selection, can either be given as a :class:`~astropy.units.Quantity` object or float (assumed to be in MHz in that case).
        fmax : `float` or :class:`~astropy.units.Quantity`, default: :math:`{\rm max}(\nu)`
            Higher frequency boundary selection, can either be given as a :class:`~astropy.units.Quantity` object or float (assumed to be in MHz in that case).
        beam : `int`, default: first recorded beam
            Beam selection, a single integer corresponding to the index of a recorded numerical beam is expected.
        dispersion_measure : `float` or :class:`~astropy.units.Quantity`, default: ``None``
            Enable de-dispersion of the data by this Dispersion Measure. Note that the :meth:`~nenupy.io.tf.TFTask.de_disperse` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`). It can either be provided as a :class:`~astropy.Quantity` object or a float (assumed to be in :math:`{\rm pc}\,{\rm cm}^{-3}` in that case).
        rotation_measure : `float` or :class:`~astropy.unit.Quantity`, default: ``None``
            Enable the correction of the Faraday rotation using this Rotation Measure. Note that the :meth:`~nenupy.io.tf.TFTask.correct_faraday_rotation` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`). It can either be provided as a :class:`~astropy.Quantity` object or a float (assumed to be in :math:`{\rm rad}\,{\rm m}^{-2}` in that case).
        rebin_dt : `float` or :class:`~astropy.units.Quantity`, default: ``None``
            Desired rebinning time resolution, can either be given as a :class:`~astropy.units.Quantity` object or a float (assumed to be in sec in that case). Note that the :meth:`~nenupy.io.tf.TFTask.time_rebin` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        rebin_df : `float` or :class:`~astropy.units.Quantity`, default: ``None``
            Desired rebinning frequency resolution, can either be given as a :class:`~astropy.units.Quantity` object or float (assumed to be in kHz in that case). Note that the :meth:`~nenupy.io.tf.TFTask.frequency_rebin` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        smooth_frequency_profile : `bool`, default `False`
            Keyword used if :meth:`~nenupy.io.tf.TFTask.flatten_subband` is in the pipeline: smooth the frequency profile, this option mau result in unexpected behavior if the subbands are not contiguous or the selected beamlets do not belong to the same digital beam. 
        remove_channels : `list` or :class:`~numpy.ndarray`, default: ``None``
            List of subband channels to remove, e.g. `remove_channels=[0,1,-1]` would remove the first, second (low-freq) and last channels from each subband. Note that the :meth:`~nenupy.io.tf.TFTask.remove_channels` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        skycoord : :class:`~astropy.coordinates.SkyCoord`, default: ``None``
            Tracked celestial coordinates used for beam and polarization corrections, a :class:`~astropy.coordinates.SkyCoord` object is expected. Note that the :meth:`~nenupy.io.tf.TFTask.correct_polarization` or :meth:`~nenupy.io.tf.TFTask.correct_parallactic_rotation` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        calib_dt : `float` or :class:`~astropy.units.Quantity`, default: ``None``
            Time resolution used for beam and polarization corrections, a :class:`~astropy.Quantity` object or a float (assumed in seconds) are expected. Note that the :meth:`~nenupy.io.tf.TFTask.correct_polarization` or :meth:`~nenupy.io.tf.TFTask.correct_parallactic_rotation` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        dreambeam_parallactic : `bool`, default: `True`
            `DreamBeam <https://dreambeam.readthedocs.io/en/latest/>`_ parallactic angle correction (along with ``'skycoord'`` and ``'calib_dt'``), a boolean is expected. Note that the :meth:`~nenupy.io.tf.TFTask.correct_polarization` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        stokes : `str` or `list[str]`, default: ``"I"``
            Stokes parameter selection, can either be given as a string or a list of strings, e.g. ``['I', 'Q', 'V/I']``. Note that the :meth:`~nenupy.io.tf.TFTask.get_stokes` task should be present in the planned pipeline (:attr:`~nenupy.io.tf.Spectra.pipeline`).
        ignore_volume_warning : `bool`, default: `False`
            Ignore or not (default value) the limit regarding output data volume.

        Returns
        -------
        :class:`~nenupy.beamlet.sdata.SData`
            Processed data selection.

        Examples
        --------
        .. code-block:: python

            >>> from nenupy.io.tf import Spectra
            >>> sp = Spectra("/my/file.spectra")
            >>> result = sp.get(tmin="2024-01-01 12:00:00", tmax="2024-01-01 13:00:00")

        """

        # Make a copy of the previous parmaeters
        parameters_copy = self.pipeline.parameters.copy()
        # Update the pipeline parameters to user's last requests
        for param, value in pipeline_kwargs.items():
            self.pipeline.parameters[param] = value

        try:
            # Select the data in time and frequency (the beam selection is implicit on the frequency idexing)
            frequency_hz, time_unix, data = self.select_raw_data(
                tmin_unix=self.pipeline.parameters["tmin"].unix,
                tmax_unix=self.pipeline.parameters["tmax"].unix,
                fmin_hz=self.pipeline.parameters["fmin"].to_value(u.Hz),
                fmax_hz=self.pipeline.parameters["fmax"].to_value(u.Hz),
                beam=self.pipeline.parameters["beam"],
            )

            # Run the pipeline - within Dask
            log.info(self.pipeline.info(return_str=True))
            time_unix, frequency_hz, data = self.pipeline.run(
                frequency_hz=frequency_hz, time_unix=time_unix, data=data
            )

            time = Time(time_unix, format="unix", precision=7)
            frequency = frequency_hz * u.Hz

            if (file_name is None) or (file_name == ""):
                # Simply compute the data
                # The data volume security is ON
                data = self._data_to_numpy_array(data)  # compute() the Dask array

                if data is None:
                    # The hardcoded volume threshold is reached.
                    log.warning("No data returned.")
                    return

                result = self._to_sdata(data=data, time=time, frequency=frequency)

                self.pipeline.parameters = parameters_copy  # Reset the parameters
                return result

            else:
                # Save the result of the pipeline in a file
                # No security on the resulting data volume
                log.info(
                    f"Estimated data volume to store: {(data.nbytes * u.byte).to(u.Gibyte):.3f}..."
                )
                utils.store_dask_tf_data(
                    file_name=file_name,
                    data=data,
                    time=time,
                    frequency=frequency,
                    polarization=["XX", "XY", "YX", "YY"]
                    if not self.pipeline.contains("Compute Stokes parameters")
                    else self.pipeline.parameters["stokes"],
                    mode="w" if self.pipeline.parameters["overwrite"] else "auto",
                    beam=self.pipeline.parameters["beam"],
                )
                self.pipeline.parameters = parameters_copy  # Reset the parameters
                return

        except Exception:
            # Restore the parameters to their original default values
            self.pipeline.parameters = parameters_copy
            raise

    def select_raw_data(
        self,
        tmin_unix: float,
        tmax_unix: float,
        fmin_hz: float,
        fmax_hz: float,
        beam: int,
    ) -> Tuple[np.ndarray, np.ndarray, da.Array]:
        """Select a subset from a time-frequency NenuFAR dataset.

        Parameters
        ----------
        tmin_unix : float
            Start time of the selection in UNIX format.
        tmax_unix : float
            Stop time of the selection in UNIX format.
        fmin_hz : float
            Minimal frequency of the selection in Hz.
        fmax_hz : float
            Maximal frequency of the selection in Hz.
        beam : int
            Beam index of the selection.

        Raises
        ------
        KeyError
            If `beam` does not correspond to any recorded value.

        Returns
        -------
        (:class:`~numpy.ndarray`, :class:`~numpy.ndarray`, :class:`~dask.array.Array`)
            Length-3 tuple, respectively containing the frequency in Hz and the time in unix as `numpy` arrays, then the selected dataset.
            The latter should be shaped as (time, frequency, 2, 2) where the last two dimensions are the Jones matrix of the electric field cross correlations.
            If the chosen time arguments lead to an empty selection, a length-3 tuple of `None` is returned.
            If the frequency selection is off, the closest subband is returned.

        See Also
        --------
        :func:`~nenupy.io.tf_utils.compute_spectra_time`, :func:`~nenupy.io.tf_utils.compute_spectra_frequencies`

        Notes
        -----
        .. versionadded:: 2.6.0

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 3

            >>> import nenupy
            >>> sp = Spectra("/my/file.spectra")
            >>> data = sp.select_raw_data(
                    tmin_unix=1685176742.000005,
                    tmax_unix=1685176802.000005,
                    fmin_hz=21e6,
                    fmax_hz=22e6,
                    beam=0
                )
            >>> data[0].shape, data[1].shape, data[2].shape
            ((384,), (2856,), (2856, 384, 2, 2))

        """

        log.info(
            f"Selecting times (between {Time(tmin_unix, format='unix').isot} and {Time(tmax_unix, format='unix').isot})..."
        )

        # Find out which block indices are at the edges of the desired time range
        block_idx_min = int(
            np.argmin(np.abs(np.ceil(self._block_start_unix - tmin_unix)))
        )  # n_blocks - np.argmax(((self._block_start_unix - tmin) <= 0)[::-1]) - 1
        block_idx_max = int(
            np.argmin(np.abs(np.ceil(self._block_start_unix - tmax_unix)))
        )  # n_blocks - np.argmax(((self._block_start_unix - tmax) <= 0)[::-1]) - 1
        log.debug(
            f"\tClosest time block from requested range are #{block_idx_min} and #{block_idx_max}."
        )

        # Get the closest time index within each of the two bounding blocks
        dt_sec = self.dt.to_value(u.s)
        # Compute the time index within the block and bound it between 0 and the number of spectra in each block
        time_idx_min_in_block = int(
            np.round((tmin_unix - self._block_start_unix[block_idx_min]) / dt_sec)
        )
        time_idx_min_in_block = max(
            0, min(self._n_time_per_block - 1, time_idx_min_in_block)
        )  # bound the value between in between channels indices
        time_idx_min = block_idx_min * self._n_time_per_block + time_idx_min_in_block
        # Do the same for the higher edge of the desired time range
        time_idx_max_in_block = int(
            np.round((tmax_unix - self._block_start_unix[block_idx_max]) / dt_sec)
        )
        time_idx_max_in_block = max(
            0, min(self._n_time_per_block - 1, time_idx_max_in_block)
        )
        time_idx_max = block_idx_max * self._n_time_per_block + time_idx_max_in_block
        log.info(f"\t{time_idx_max - time_idx_min + 1} time samples selected.")

        # Raise warnings if the time selection results in no data selected
        if time_idx_min == time_idx_max:
            if (time_idx_min > 0) and (
                time_idx_min < self._block_start_unix.size * self._n_time_per_block - 1
            ):
                log.warning("Desired time selection encompasses missing data.")
                if tmin_unix < self._block_start_unix[block_idx_min]:
                    # The found block is just after the missing data
                    closest_tmin = Time(
                        self._block_start_unix[block_idx_min - 1]
                        + self._n_time_per_block * dt_sec,
                        format="unix",
                    ).isot
                    closest_tmax = Time(
                        self._block_start_unix[block_idx_min], format="unix"
                    ).isot
                else:
                    # The found block is just before the missing data
                    closest_tmin = Time(
                        self._block_start_unix[block_idx_min]
                        + self._n_time_per_block * dt_sec,
                        format="unix",
                    ).isot
                    closest_tmax = Time(
                        self._block_start_unix[block_idx_min + 1], format="unix"
                    ).isot
                log.info(
                    f"Time selection lies in the data gap between {closest_tmin} and {closest_tmax}."
                )
            log.warning("Time selection leads to empty dataset.")
            return (None, None, None)

        # Compute the time ramp between those blocks
        time_unix = utils.compute_spectra_time(
            block_start_time_unix=self._block_start_unix[
                block_idx_min : block_idx_max + 1
            ],
            ntime_per_block=self._n_time_per_block,
            time_step_s=self.dt.to_value(u.s),
        )
        # Cut down the first and last time blocks
        time_unix = time_unix[
            time_idx_min_in_block : time_unix.size
            - (self._n_time_per_block - time_idx_max_in_block)
            + 1
        ]

        log.info(
            f"Selecting frequencies (between {(fmin_hz*u.Hz).to(u.MHz)} and {(fmax_hz*u.Hz).to(u.MHz)})..."
        )
        try:
            beam_idx_start, beam_idx_stop = self._beam_indices_dict[str(beam)]
        except KeyError as e:
            log.error(
                f"Beam index selection '{beam}' does not correspond to one of the recorded values: {list(self._beam_indices_dict.keys())}."
            )
            raise

        # Find out the subband edges covering the selected frequency range
        subbands_in_beam = self._subband_start_hz[
            int(beam_idx_start / self.n_channels) : int(
                (beam_idx_stop + 1) / self.n_channels
            )
        ]
        sb_idx_min = int(np.argmin(np.abs(np.ceil(subbands_in_beam - fmin_hz))))
        sb_idx_max = int(np.argmin(np.abs(np.ceil(subbands_in_beam - fmax_hz))))
        log.debug(
            f"\tClosest beamlet indices from requested range are #{sb_idx_min} and #{sb_idx_max}."
        )

        # Select frequencies at the subband granularity at minimum
        # Later, we want to correct for bandpass, edge channels and so on...
        frequency_idx_min = sb_idx_min * self.n_channels
        frequency_idx_max = (sb_idx_max + 1) * self.n_channels
        frequency_hz = utils.compute_spectra_frequencies(
            subband_start_hz=subbands_in_beam[sb_idx_min : sb_idx_max + 1],
            n_channels=self.n_channels,
            frequency_step_hz=self.df.to_value(u.Hz),
        )
        log.info(
            f"\t{frequency_idx_max - frequency_idx_min} frequency samples selected."
        )

        selected_data = self.data[:, beam_idx_start : beam_idx_stop + 1, ...][
            time_idx_min : time_idx_max + 1, frequency_idx_min:frequency_idx_max, ...
        ]
        log.debug(f"Data of shape {selected_data.shape} selected.")

        return frequency_hz.compute(), time_unix.compute(), selected_data

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _lazy_load_data(self) -> np.ndarray:
        # Read the header of the first block
        with open(self.filename, "rb") as rf:
            header_dtype = np.dtype(BLOCK_HEADER)
            header = np.frombuffer(
                rf.read(header_dtype.itemsize),
                count=1,
                dtype=header_dtype,
            )[0]
        self._n_time_per_block = header["nffte"]
        self.n_channels = header["fftlen"]
        self.n_subbands = np.abs(header["nbchan"])  # this could be negative

        # Fill in global attributes
        self.dt = (self.n_channels * header["nfft2int"] / SUBBAND_WIDTH).to(
            u.s
        )  # * 5.12e-6 * u.s
        self.df = SUBBAND_WIDTH / self.n_channels  # 0.1953125 / self.n_channels * u.MHz

        # Deduce the structure of the full file
        beamlet_data_structure = (self._n_time_per_block, self.n_channels, 2)
        beamlet_dtype = np.dtype(
            [
                ("lane", "int32"),
                ("beam", "int32"),
                ("channel", "int32"),
                ("fft0", "float32", beamlet_data_structure),
                ("fft1", "float32", beamlet_data_structure),
            ]
        )
        global_struct = BLOCK_HEADER + [("data", beamlet_dtype, (self.n_subbands))]

        # Open the file as a memory-mapped object
        with open(self.filename, "rb") as rf:
            tmp = np.memmap(rf, dtype="int8", mode="r")

        log.info(f"\t{self.filename} has been correctly parsed.")

        return tmp.view(np.dtype(global_struct))

    @staticmethod
    def _get_bad_data_mask(
        data: np.ndarray, bypass_verification: bool = False
    ) -> np.ndarray:
        """ """

        if bypass_verification:
            log.info("Skipping missing data verification...")
            return np.zeros(data.size, dtype=bool)

        log.info("Checking for missing data (can take up to 1 min)...")

        # Either the TIMESTAMP is set to 0, the first idx, or the SB number is negative
        # which indicates missing data. In all those cases we will ignore the associated data
        # since we cannot properly reconstruct the time ramp or the data themselves.
        block_timestamp_mask = (
            data["TIMESTAMP"] < 0.1
        )  # it should be 0, but to be sure...
        block_start_idx_mask = data["idx"] == 0
        block_nsubbands_mask = data["nbchan"] < 0

        # Computing the mask, setting the first index at non-zero since it should be the normal value.
        block_start_idx_mask[0] = False  # Fake value, just to trick the mask
        bad_block_mask = (
            block_timestamp_mask + block_start_idx_mask + block_nsubbands_mask
        )

        log.info(
            f"\tThere are {np.sum(bad_block_mask)}/{block_timestamp_mask.size} blocks containing missing data and/or wrong time information."
        )

        return bad_block_mask

    def _to_dask_tf(self, data: np.ndarray, mask: np.ndarray) -> da.Array:
        """ """
        log.info("Re-organize data into Jones matrices...")

        # Transform the array in a Dask array, one chunk per block
        # Filter out the bad blocks
        # data = da.from_array(data, chunks=(1,))[~mask]
        data = da.from_array(data, chunks=(1,))
        data["data"]["fft0"][mask] = np.nan
        data["data"]["fft1"][mask] = np.nan

        # Convert the data to cross correlation electric field matrix
        # The data will be shaped like (n_block, n_subband, n_time_per_block, n_channels, 2, 2)
        data = utils.spectra_data_to_matrix(data["data"]["fft0"], data["data"]["fft1"])

        # Reshape the data into (time, frequency, 2, 2)
        data = utils.blocks_to_tf_data(
            data=data, n_block_times=self._n_time_per_block, n_channels=self.n_channels
        )

        log.info("Data loaded.")

        return data

    def _data_to_numpy_array(self, data: da.Array) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        data : da.Array
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """

        # Abort the process if the projected data volume is larger than the threshold
        projected_data_volume = data.nbytes * u.byte
        if (projected_data_volume >= DATA_VOLUME_SECURITY_THRESHOLD) and (
            not self.pipeline.parameters["ignore_volume_warning"]
        ):
            log.warning(
                f"Data processing will produce {projected_data_volume.to(u.Gibyte)}."
                f"The pipeline is interrupted because the volume threshold is {DATA_VOLUME_SECURITY_THRESHOLD.to(u.Gibyte)}."
                "Consider reducing the amount of selected data."
                "Otherwise, you may use the ignore_volume_warning argument (with caution!) or save the data output to HDF5 format."
            )
            return

        log.info(
            f"Computing the data (estimated volume: {projected_data_volume.to(u.Mibyte):.2})..."
        )

        with ProgressBar():
            data = data.compute()

        log.info(
            f"\tData of shape (time, frequency, (polarization)) = {data.shape} produced."
        )

        return data

    def _to_sdata(self, data: np.ndarray, time: Time, frequency: u.Quantity) -> SData:
        """_summary_

        Parameters
        ----------
        data : np.ndarray
            _description_
        time : Time
            _description_
        frequency : u.Quantity
            _description_

        Returns
        -------
        SData
            _description_
        """

        # If other data product than Stokes are made
        if not self.pipeline.contains("Compute Stokes parameters"):
            # Make sure there are 3 dimensions (time, frequency, polarization)
            # If there are more, the dimensions > 2 are all merged together.
            # If there are 2 dimensions, an empty third is added
            data = data.reshape(*data.shape[:2], -1)
            return SData(
                data=data,
                time=time,
                freq=frequency,
                polar=["XX", "XY", "YX", "YY"],
            )
        else:
            # If the data are regular Stokes parameters
            return SData(
                data=data,
                time=time,
                freq=frequency,
                polar=self.pipeline.parameters["stokes"],
            )
