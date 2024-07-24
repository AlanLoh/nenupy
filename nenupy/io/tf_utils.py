"""
    ******************************
    Support to Time-Frequency data
    ******************************

    This module contains all the relevant functions to read
    and process the NenuFAR beamformed time-frequency data.
    The functions called by the various :class:`~nenupy.io.tf.TFTask`
    are listed here.
    This module's content is not meant to be used outside of this
    scope unless the user knows what they are doing.

    .. seealso::

        :ref:`tf_reading_doc`

"""

import numpy as np
import os
import dask.array as da
from dask.diagnostics import ProgressBar
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from typing import Union, List, Tuple, Any
from functools import partial
from abc import ABC, abstractmethod
import copy
import h5py
import logging

log = logging.getLogger(__name__)

from nenupy.astro import dispersion_delay, faraday_angle
from nenupy.astro.beam_correction import compute_jones_matrices

__all__ = [
    "blocks_to_tf_data",
    "compute_spectra_frequencies",
    "compute_spectra_time",
    "compute_stokes_parameters",
    "correct_bandpass",
    "crop_subband_edges",
    "de_disperse_array",
    "de_faraday_data",
    "flatten_subband",
    "get_bandpass",
    "plot_dynamic_spectrum",
    "polarization_angle",
    "rebin_along_dimension",
    "remove_channels_per_subband",
    "reshape_to_subbands",
    "sort_beam_edges",
    "spectra_data_to_matrix",
    "store_dask_tf_data",
    "TFPipelineParameters",
    "ReducedSpectra",
]


# ============================================================= #
# ---------------- apply_dreambeam_corrections ---------------- #
def apply_dreambeam_corrections(
    time_unix: np.ndarray,
    frequency_hz: np.ndarray,
    data: np.ndarray,
    dt_sec: float,
    time_step_sec: float,
    n_channels: int,
    skycoord: SkyCoord,
    parallactic: bool = True,
) -> np.ndarray:
    """ """

    log.info("Applying DreamBeam corrections...")

    # Basic checks to make sure the dimensions are correct
    freq_size = frequency_hz.size
    time_size = time_unix.size
    if time_size != data.shape[0]:
        raise ValueError("There is a problem in the time dimension!")
    if (freq_size != data.shape[1]) or (freq_size % n_channels != 0):
        raise ValueError("There is a problem in the frequency dimension!")
    n_subbands = int(freq_size / n_channels)

    # Compute the number of time samples that will be corrected together
    time_group_size = int(np.round(time_step_sec / dt_sec))
    log.debug(
        f"\tGroups of {time_group_size} time blocks will be corrected altogether ({dt_sec*time_group_size} sec resolution)."
    )
    n_time_groups = time_size // time_group_size
    leftover_time_samples = time_size % time_group_size

    # Computing DreamBeam matrices
    db_time, db_frequency, db_jones = compute_jones_matrices(
        start_time=Time(time_unix[0], format="unix", precision=7),
        time_step=TimeDelta(time_group_size * dt_sec, format="sec"),
        duration=TimeDelta(time_unix[-1] - time_unix[0], format="sec"),
        skycoord=skycoord,
        parallactic=parallactic,
    )
    db_time = db_time.unix
    db_frequency = db_frequency.to_value(u.Hz)
    db_jones = np.swapaxes(db_jones, 0, 1)

    # Reshape the data at the time and frequency resolutions
    # Take into account leftover times
    data_leftover = data[-leftover_time_samples:, ...].reshape(
        (leftover_time_samples, n_subbands, n_channels, 2, 2)
    )
    data = data[: time_size - leftover_time_samples, ...].reshape(
        (n_time_groups, time_group_size, n_subbands, n_channels, 2, 2)
    )

    # Compute the frequency indices to select the corresponding Jones matrices
    subband_start_frequencies = frequency_hz.reshape((n_subbands, n_channels))[:, 0]
    freq_start_idx = np.argmax(db_frequency >= subband_start_frequencies[0])
    freq_stop_idx = db_frequency.size - np.argmax(
        db_frequency[::-1] < subband_start_frequencies[-1]
    )

    # Do the same with the time
    group_start_time = time_unix[: time_size - leftover_time_samples].reshape(
        (n_time_groups, time_group_size)
    )[:, 0]
    time_start_idx = np.argmax(db_time >= group_start_time[0])
    time_stop_idx = db_time.size - np.argmax(db_time[::-1] < group_start_time[-1])

    jones = db_jones[
        time_start_idx : time_stop_idx + 1, freq_start_idx : freq_stop_idx + 1, :, :
    ][:, None, :, None, :, :]
    jones_leftover = db_jones[-1, freq_start_idx : freq_stop_idx + 1, :, :][
        None, :, None, :, :
    ]

    # Invert the matrices that will be used to correct the observed signals
    # Jones matrices are at the subband resolution and an arbitrary time resolution
    jones = np.linalg.inv(jones)
    jones_leftover = np.linalg.inv(jones_leftover)

    # Compute the Hermitian matrices
    jones_transpose = np.swapaxes(jones, -2, -1)
    jones_leftover_transpose = np.swapaxes(jones_leftover, -2, -1)
    jones_hermitian = np.conjugate(jones_transpose)
    jones_leftover_hermitian = np.conjugate(jones_leftover_transpose)

    # This would raise an indexerror if jones_values are at smaller t/f range than data
    return np.concatenate(
        (
            np.matmul(jones, np.matmul(data, jones_hermitian)).reshape(
                (time_size - leftover_time_samples, freq_size, 2, 2)
            ),
            np.matmul(
                jones_leftover, np.matmul(data_leftover, jones_leftover_hermitian)
            ).reshape((leftover_time_samples, freq_size, 2, 2)),
        ),
        axis=0,
    )


# ============================================================= #
# --------------------- blocks_to_tf_data --------------------- #
def blocks_to_tf_data(data: da.Array, n_block_times: int, n_channels: int) -> da.Array:
    """Parse time-frequency data at the reading of a .spectra file.
    Invert the halves of each beamlet and reshape the
    array in 2D (time, frequency) or 4D (time, frequency, 2, 2).

    Parameters
    ----------
    data : :class:`~dask.array.Array`
        Raw data. Number of dimensions should either be 4 (n_block, n_subband, n_time_per_block, n_channels) or 6 (n_block, n_subband, n_time_per_block, n_channels, 2, 2)
    n_block_times : `int`
        Number of time blocks
    n_channels : `int`
        Number of frequency channels in each beamlet

    Returns
    -------
    :class:`~dask.array.Array`
        Data reshaped

    Raises
    ------
    ValueError
        Raised if n_channels is odd
    IndexError
        Raised if the number of dimensions of data is different than 4 or 6

    Warning
    -------
    Usage only within :class:`~nenupy.io.tf.Spectra`.
            
    Example
    -------
    .. code-block:: python

        >>> from nenupy.io.tf_utils import blocks_to_tf_data
        >>> import numpy as np

        >>> n_block = 5
        >>> n_subband = 32
        >>> n_time_per_block = 24
        >>> n_channels = 8
        >>> data = np.ones(((n_block, n_subband, n_time_per_block, n_channels, 2, 2)))
        >>> data_reshaped = blocks_to_tf_data(
                data=data,
                n_block_times=n_time_per_block,
                n_channels=n_channels
            )
        >>> data_reshaped.shape
        (120, 256, 2, 2)

    """
    ntb, nfb = data.shape[:2]
    n_times = ntb * n_block_times
    n_freqs = nfb * n_channels
    if n_channels % 2.0 != 0.0:
        raise ValueError(f"Problem with n_channels value: {n_channels}!")

    # Prepare the various shapes
    temp_shape = (n_times, int(n_freqs / n_channels), 2, int(n_channels / 2))
    final_shape = (n_times, n_freqs)

    # Add dimension if this comes in eJones matrix
    if data.ndim == 4:
        pass
    elif data.ndim == 6:
        final_shape += (2, 2)
        temp_shape += (2, 2)
    else:
        raise IndexError(f"The data has an unexpected shape of {data.shape}.")

    # Swap the subband (1) and nffte (2) dimensions to group
    # frequency and times dimensions together.
    # Reshape in order to isolate the halves of every beamlet.
    data = np.swapaxes(data, 1, 2).reshape(temp_shape)

    # Invert the halves and reshape to the final form.
    return data[:, :, ::-1, ...].reshape(final_shape)


# ============================================================= #
# ---------------- compute_spectra_frequencies ---------------- #
def compute_spectra_frequencies(
    subband_start_hz: np.ndarray, n_channels: int, frequency_step_hz: float
) -> da.Array:
    """Compute the frequency axis of a time-frequency file.
    Re-construct the whole frequency range, knowing the starting frequency of
    each sub-band, the number of channels per sub-band and the frequency
    resolution.

    Parameters
    ----------
    subband_start_hz : :class:`~numpy.ndarray`
        Array of sub-band starting frequencies
    n_channels : `int`
        Number of channels per subband
    frequency_step_hz : `float`
        Frequency resolution in Hz

    Returns
    -------
    :class:`~dask.array.Array`
        The frequency array (`Dask <https://docs.dask.org/en/stable/>`_ format), in Hz

    Example
    -------
    .. code-block:: python
        :emphasize-lines: 7,8,9,10,11

        >>> from nenupy.io.tf_utils import compute_spectra_frequencies
        >>> import astropy.units as u

        >>> sb_start_freq = [50.1953125, 50.390625, 50.5859375, 50.781250] * u.MHz
        >>> n_channels = 16
        >>> df = 12.20703125 * u.kHz
        >>> freq_axis = compute_spectra_frequencies(
                subband_start_hz=sb_start_freq.to_value(u.Hz),
                n_channels=n_channels,
                frequency_step_hz=df.to_value(u.Hz)
            )
        >>> freq_axis.compute()
        array([50097656.25, 50109863.28125,....50854492.1875, 50866699.21875])

    """

    # Construct the frequency array
    frequencies = da.tile(np.arange(n_channels) - n_channels / 2, subband_start_hz.size)
    frequencies = frequencies.reshape((subband_start_hz.size, n_channels))
    frequencies *= frequency_step_hz
    frequencies += subband_start_hz[:, None]
    frequencies = frequencies.ravel()

    log.debug(f"\tFrequency axis computed (size={frequencies.size}).")

    return frequencies


# ============================================================= #
# ------------------- compute_spectra_time -------------------- #
def compute_spectra_time(
    block_start_time_unix: np.ndarray, ntime_per_block: int, time_step_s: float
) -> da.Array:
    """Compute the time axis of a time-frequency file.
    Re-construct the whole time range, knowing the starting unix time of
    each time block, the number of time samples per block and the time
    resolution.

    Parameters
    ----------
    block_start_time_unix : :class:`~numpy.ndarray`
        Array of start times of each block, in UNIX format
    ntime_per_block : `int`
        Number of time samples per block
    time_step_s : `float`
        Time resolution in seconds

    Returns
    -------
    :class:`~dask.array.Array`
        The time array (`Dask <https://docs.dask.org/en/stable/>`_ format), in unix

    Example
    -------
    .. code-block:: python
        :emphasize-lines: 14,15,16,17,18

        >>> from nenupy.io.tf_utils import compute_spectra_time
        >>> from astropy.time import Time
        >>> import astropy.units as u

        >>> nffte = 42
        >>> dt = 0.02097152 * u.s
        >>> start_times = Time([
                '2024-07-15T08:31:12.000', '2024-07-15T08:31:12.881',
                '2024-07-15T08:31:13.762', '2024-07-15T08:31:14.642',
                '2024-07-15T08:31:15.523', '2024-07-15T08:31:16.404',
                '2024-07-15T08:31:17.285', '2024-07-15T08:31:18.166',
                '2024-07-15T08:31:19.046', '2024-07-15T08:31:19.927'
            ])
        >>> time_axis = compute_spectra_time(
                block_start_time_unix=start_times.unix,
                ntime_per_block=nffte,
                time_step_s=dt.to_value(u.s)
            )
        >>> time_axis.compute()
        array([1721032272.0, 1721032272.0209715, ...])

    """

    # Construct the elapsed time per block (1D array)
    time_seconds_per_block = da.arange(ntime_per_block, dtype="float64") * time_step_s

    # Create the time ramp with broadcasting
    unix_time = time_seconds_per_block[None, :] + block_start_time_unix[:, None]

    # Return the flatten array
    unix_time = unix_time.ravel()

    log.debug(f"\tTime axis computed (size={unix_time.size}).")

    return unix_time


# ============================================================= #
# ----------------- compute_stokes_parameters ----------------- #
def compute_stokes_parameters(
    data_array: np.ndarray, stokes: Union[List[str], str]
) -> np.ndarray:
    r"""Compute the Stokes parameters from data organized in Jones matrices.

    ``data_array``'s last two dimensions are assumed to be:

    .. math::

        d = \begin{pmatrix}
        X\overline{X} & X\overline{Y} \\
        Y\overline{X} & Y\overline{Y}
        \end{pmatrix},

    so that the Stokes parameters are computed such as:

    .. math::

        \begin{align}
        I &= \Re(X\overline{X}) + \Re(Y\overline{Y})\\
        Q &= \Re(X\overline{X}) - \Re(Y\overline{Y})\\
        U &= 2\Re(X\overline{Y})\\
        V &= 2\Im(X\overline{Y})
        \end{align}
        
    Parameters
    ----------
    data_array : :class:`~numpy.ndarray`
        The data array (a Dask :class:`~dask.array.Array` is also accepted),
        the first dimensions may corresponds with anything (for e.g., time,
        frequency, beam...) and are kept through the computation, the last
        two dimensions must be the :math:`2 \times 2` Jones matrices.
    stokes : List[`str`] or `str`
        Stokes parameters to compute, if a list is given, the result will
        store them in the same order in the last result dimension, available
        values are "I", "Q", "U", "V", "Q/I", "U/I", "V/I".

    Returns
    -------
    :class:`~numpy.ndarray`
        New data array transformed in Stokes parameters,
        the last two dimensions are replaced by a single dimension
        listing in order the requested stokes parameters.
        If the input data_array is a Dask :class:`~dask.array.Array`,
        the returned result is of the same type.

    Raises
    ------
    Exception
        Raised if the last two dimensions of data_array are different than (2, 2)
    NotImplementedError
        Raised if the requested Stokes parameter is not known
    
    Example
    -------
    .. code-block:: python

        >>> from nenupy.io.tf import Spectra
        >>> from nenupy.io.tf_utils import compute_stokes_parameters
        
        >>> sp = Spectra(".../my_file.spectra")
        >>> result = compute_stokes_parameters(
                data_array=sp.data[:1000, :100, :, :], # shape: (time, frequency, 2, 2)
                stokes=["I", "U/I", "V/I"]
            )
        >>> result.shape
        (1000, 100, 3)

    """

    log.info(f"Computing Stokes parameters {stokes}...")

    # Assert that the last dimensions are shaped like a cross correlation electric field matrix
    if data_array.shape[-2:] != (2, 2):
        raise Exception("The data_array last 2 dimensions are not of shape (2, 2).")

    result = None

    if isinstance(stokes, str):
        # Make sure the Stokes iterable is a list and not just the string.
        stokes = [stokes]
    for stokes_i in stokes:
        stokes_i = stokes_i.replace(" ", "").upper()
        # Compute the correct Stokes value
        if stokes_i == "I":
            data_i = data_array[..., 0, 0].real + data_array[..., 1, 1].real
        elif stokes_i == "Q":
            data_i = data_array[..., 0, 0].real - data_array[..., 1, 1].real
        elif stokes_i == "U":
            data_i = data_array[..., 0, 1].real * 2
        elif stokes_i == "V":
            data_i = data_array[..., 0, 1].imag * 2
        elif stokes_i == "Q/I":
            data_i = (data_array[..., 0, 0].real - data_array[..., 1, 1].real) / (
                data_array[..., 0, 0].real + data_array[..., 1, 1].real
            )
        elif stokes_i == "U/I":
            data_i = (
                data_array[..., 0, 1].real
                * 2
                / (data_array[..., 0, 0].real + data_array[..., 1, 1].real)
            )
        elif stokes_i == "V/I":
            data_i = (
                data_array[..., 0, 1].imag
                * 2
                / (data_array[..., 0, 0].real + data_array[..., 1, 1].real)
            )
        else:
            raise NotImplementedError(f"Stokes parameter {stokes_i} unknown.")

        log.info(f"\tStokes {stokes_i} computed.")

        # Stack everything
        if result is None:
            result = np.expand_dims(data_i, axis=-1)
        else:
            result = np.concatenate([result, data_i[..., None]], axis=-1)

    return result


# ============================================================= #
# --------------------- correct_bandpass ---------------------- #
def correct_bandpass(data: np.ndarray, n_channels: int) -> np.ndarray:
    """Correct the Polyphase-filter band-pass response at each sub-band.
    This methods computes the bandpass theoretical response of sub-bands
    made of ``n_channels`` and multiply the ``data`` by this reponse. 
    
    Returns
    -------
    :class:`~numpy.ndarray`
        Bandpass response corrected data.

    Raises
    ------
    ValueError
        Raised if the shape of data does not match the number of channels.
    
    See Also
    --------
    :func:`~nenupy.io.tf_utils.get_bandpass`
    """

    log.info("Correcting for bandpass...")

    # Compute the bandpass
    bandpass = get_bandpass(n_channels=n_channels)

    # Reshape the data array to isolate individual subbands
    n_times, n_freqs, _, _ = data.shape
    if n_freqs % n_channels != 0:
        raise ValueError(
            "The frequency dimension of `data` doesn't match the argument `n_channels`."
        )
    data = data.reshape(
        (n_times, int(n_freqs / n_channels), n_channels, 2, 2)  # subband  # channels
    )

    # Multiply the channels by the bandpass to correct them
    data *= bandpass[None, None, :, None, None]

    log.debug(f"\tEach subband corrected by the bandpass of size {bandpass.size}.")

    # Re-reshape the data into time, frequency, (2, 2) array
    return data.reshape((n_times, n_freqs, 2, 2))


# ============================================================= #
# -------------------- crop_subband_edges --------------------- #
def crop_subband_edges(
    data: np.ndarray,
    n_channels: int,
    lower_edge_channels: int = 0,
    higher_edge_channels: int = 0,
) -> np.ndarray:
    """Set edge channels of each subband to `NaN`.
    Each subband of ``data`` is determined thanks to ``n_channels``
    (and the function :func:`~nenupy.io.tf_utils.reshape_to_subbands`).
    This method is a bit faster than :func:`~nenupy.io.tf_utils.remove_channels_per_subband`
    but it is restricted to edge channels. The other method is prefered for its polyvalence.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Data to be corrected, must be at least two-dimensional, the first two dimensions being respectively the time and the frequency
    n_channels : `int`
        Number of channels per subband
    lower_edge_channels : `int`
        Number of channels to set to `NaN` for the lowest part of each subband, by default 0
    higher_edge_channels : `int`
        Number of channels to set to `NaN` for the highest part of each subband, by default 0

    Returns
    -------
    :class:`~numpy.ndarray`
        Corrected data, same shape as input array

    Raises
    ------
    ValueError
        Raised if the cropped channels are greater than ``n_channels``.

    Examples
    --------
    .. code-block:: python

        >>> from nenupy.io.tf_utils import crop_subband_edges
        >>> import numpy as np

        >>> result = crop_subband_edges(
                data=np.ones((2, 10)),
                n_channels=5,
                lower_edge_channels=1, # set to NaN the first channel of each subband
                higher_edge_channels=0
            )
        >>> print(result)
        [[nan  1.  1.  1.  1. nan  1.  1.  1.  1.]
        [nan  1.  1.  1.  1. nan  1.  1.  1.  1.]]

        >>> result = crop_subband_edges(
                data=np.ones((2, 10)),
                n_channels=5,
                lower_edge_channels=0,
                higher_edge_channels=2 # set to NaN the last 2 channels of each subband
            )
        >>> print(result)
        [[ 1.  1.  1. nan nan  1.  1.  1. nan nan]
        [ 1.  1.  1. nan nan  1.  1.  1. nan nan]]

    See Also
    --------
    :func:`~nenupy.io.tf_utils.remove_channels_per_subband`
    """

    log.info("Removing edge channels...")

    if lower_edge_channels + higher_edge_channels >= n_channels:
        raise ValueError(
            f"{lower_edge_channels + higher_edge_channels} channels to crop out of {n_channels} channels subbands."
        )

    original_shape = data.shape
    n_times = original_shape[0]
    n_freqs = original_shape[1]
    data = reshape_to_subbands(data=data, n_channels=n_channels)

    # Set to NaN edge channels
    data[:, :, :lower_edge_channels, ...] = np.nan  # lower edge
    data[:, :, n_channels - higher_edge_channels :, ...] = np.nan  # upper edge
    data = data.reshape((n_times, n_freqs) + original_shape[2:])

    log.info(
        f"\t{lower_edge_channels} lower and {higher_edge_channels} higher "
        "band channels have been set to NaN at the subband edges."
    )

    return data


# ============================================================= #
# --------------------- de_disperse_array --------------------- #
def de_disperse_array(
    data: np.ndarray,
    frequencies: u.Quantity,
    time_step: u.Quantity,
    dispersion_measure: u.Quantity,
) -> np.ndarray:
    """De-disperse in time an array ``data`` whose first two
    dimensions are time and frequency respectively. The array
    must be regularly sampled in time. The de-dispersion is made
    relatively to the highest frequency. De-dedispersed array
    is filled with ``NaN`` in time-frequency places where the
    shifted values were.

    :param data:
        Data array to de-disperse.
    :type data:
        :class:`~numpy.ndarray`
    :param frequencies:
        1D array of frequencies corresponding to the second
        dimension of ``data``.
    :type frequencies:
        :class:`~astropy.units.Quantity`
    :param time_step:
        Time step between two spectra.
    :type time_step:
        :class:`~astropy.units.Quantity`
    :param dispersion_measure:
        Dispersion Measure (in pc/cm3).
    :type dispersion_measure:
        :class:`~astropy.units.Quantity`
    """

    log.info(f"De-dispersing data by DM={dispersion_measure.to(u.pc/u.cm**3)}...")

    if data.ndim < 2:
        raise Exception(
            f"Input data is {data.shape}. >2D array is required "
            "(time, frequency, ...)."
        )
    if data.shape[1] != frequencies.size:
        raise ValueError(
            f"The size of frequencies ({frequencies.size}) does "
            f"not match dimension 1 of data ({data.shape[1]})."
        )

    # Compute the relative delays
    delays = dispersion_delay(
        frequency=frequencies, dispersion_measure=dispersion_measure
    )
    delays -= dispersion_delay(
        frequency=frequencies.max(), dispersion_measure=dispersion_measure
    )

    # Convert the delays into indices
    cell_delays = np.round((delays / time_step).decompose().to_value()).astype(int)

    # Shift the array in time
    time_size = data.shape[0]
    for i in range(frequencies.size):
        data[:, i, ...] = np.roll(data[:, i, ...], -cell_delays[i], 0)
        # Mask right edge of dynspec
        data[time_size - cell_delays[i] :, i, :, :] = np.nan

    return data


# ============================================================= #
# ---------------------- de_faraday_data ---------------------- #
def de_faraday_data(
    data: np.ndarray, frequency: u.Quantity, rotation_measure: u.Quantity
) -> np.ndarray:
    r"""Correct the data from Faraday Rotation effect.

    Linearly polarized light travelling through a magnetized plasma is subject to
    a rotation of its polarization direction due to charged particles
    (mostly dominated by electrons) reacting to and influencing 
    differentially the magnetic field components of the electromagnetic
    radiation.

    This function computes the chromatic Faraday rotation angle to correct for
    :math:`\theta (\nu)`
    with :func:`~nenupy.astro.astro_tools.faraday_angle`, given the
    ``rotation_measure`` of the intervening plasma. It is equivalent as
    computing :math:`\Delta \theta = | \theta (\nu_{\rm ref}) - \theta (\nu) |`, i.e., the difference of rotation angle
    where every ``frequency`` is compared to an 'infinite' frequency (since
    :math:`\theta \propto \nu^{-2}`).

    .. math::

        \mathbf{d}_{\rm corrected}(t, \nu) = 
        \mathbf{R}(\nu)
        \mathbf{d}(t, \nu)
        \mathbf{R}(\nu)^\top,

    where

    .. math::

        \mathbf{R}(\nu) = \begin{pmatrix}
            \cos \theta (\nu) & - \sin \theta (\nu) \\
            \sin \theta (\nu) & \cos \theta (\nu)
        \end{pmatrix}
    
    is the Faraday rotation matrix and

    .. math::

        \mathbf{d}(t, \nu) = \begin{pmatrix}
            e_{\rm x}\overline{e_{\rm x}} & e_{\rm x}\overline{e_{\rm y}} \\
            e_{\rm y}\overline{e_{\rm x}} & e_{\rm y}\overline{e_{\rm y}}
        \end{pmatrix}(t, \nu),

    and similarly :math:`\mathbf{d}_{\rm corrected}` are
    the ``data`` and corrected data returned by this function, shaped as
    Jones matrices of the signal. :math:`t` and :math:`\nu` are the time
    and frequency.

    Parameters
    ----------
    frequency : :class:`~astropy.units.Quantity`
        Light frequency (in units equivalent to MHz).
    data : :class:`~numpy.ndarray`
        Data in Jones format to be corrected. It needs to be shaped like (time, frequency, 2, 2)
    rotation_measure : :class:`~astropy.units.Quantity`
        Rotation measure (in units equivalent to rad/m2).

    Returns
    -------
    :class:`~numpy.ndarray`
        Faraday rotation corrected data.

    Raises
    ------
    Exception
        Raised if the data dimensions do not meet the requirements.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import astropy.units as u
        >>> import matplotlib.pyplot as plt
        >>> from nenupy.io.tf_utils import de_faraday_data

        >>> times = np.arange(10)
        >>> frequencies = np.linspace(52, 62, 100) * u.MHz
        >>> linearly_polarized_light = 0.5 * np.array([[1, 1], [1, 1]]) # 45 deg
        >>> linearly_polarized_light = np.tile(linearly_polarized_light, (times.size, frequencies.size, 1, 1))
        >>> stokes_u = linearly_polarized_light[..., 0, 1] * 2
        >>> faraday_rotated_light = de_faraday_data(
                data=linearly_polarized_light,
                frequency=frequencies,
                rotation_measure=5 * u.rad / u.m**2
            )
        >>> faraday_stokes_u = faraday_rotated_light[..., 0, 1] * 2

        >>> fig = plt.figure(figsize=(10, 4))
        >>> axes = fig.subplots(nrows=1, ncols=2)
        >>> im_0 = axes[0].pcolormesh(times, frequencies.to_value(u.MHz), stokes_u.T)
        >>> im_1 = axes[1].pcolormesh(times, frequencies.to_value(u.MHz), faraday_stokes_u.T)
        >>> cbar_0 = plt.colorbar(im_0)
        >>> cbar_1 = plt.colorbar(im_1)
        >>> cbar_1.set_label(r"Stokes U")
        >>> axes[0].set_ylabel("Frequency (MHz)")
        >>> axes[0].set_xlabel("Time (arbitrary units)")
        >>> axes[1].set_xlabel("Time (arbitrary units)")

    .. figure:: ../_images/io_images/defarday_example.png
        :width: 650
        :align: center

    See Also
    --------
    :func:`~nenupy.astro.astro_tools.faraday_angle`

    """
    log.info("Correcting for Faraday rotation...")

    # Check the dimensions
    if (data.ndim != 4) or (data.shape[1:] != (frequency.size, 2, 2)):
        raise Exception("Wrong data dimensions!")

    # Computing the Faraday angles compared to infinite frequency
    log.info(
        f"\tComputing {frequency.size} Faraday rotation angles at the RM={rotation_measure}..."
    )
    rotation_angle = faraday_angle(
        frequency=frequency, rotation_measure=rotation_measure, inverse=True
    ).to_value(u.rad)

    log.info("\tApplying Faraday rotation Jones matrices...")
    cosa = np.cos(rotation_angle)
    sina = np.sin(rotation_angle)
    jones = np.transpose(np.array([[cosa, -sina], [sina, cosa]]), (2, 0, 1))
    jones_transpose = np.transpose(jones, (0, 2, 1))

    return np.matmul(np.matmul(jones, data), jones_transpose)


# ============================================================= #
# ----------------------- get_bandpass ------------------------ #
def get_bandpass(n_channels: int) -> np.ndarray:
    """Compute the theoretical bandpass response of a sub-band.
    Function and coefficient developped/computed by C. Viou.

    Parameters
    ----------
    n_channels : int
        Number of channels per sub-band

    Returns
    -------
    :class:`~numpy.ndarray`
        The bandpass response of a subband
    
    Raises
    ------
    ValueError
        Raised if ``n_channels`` is lower than 16 or is odd.
    
    Example
    -------
    .. code-block:: python

        >>> from nenupy.io.tf_utils import get_bandpass

        >>> bp = get_bandpass(n_channels=32)

    """
    kaiser_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bandpass_coeffs.dat"
    )
    kaiser = np.loadtxt(kaiser_file)

    if (n_channels < 16) or (n_channels%2 != 0):
        raise ValueError("n_channels cannot be lower than 16 and must be even.")

    n_tap = 16
    over_sampling = n_channels // n_tap
    n_fft = over_sampling * kaiser.size

    g_high_res = np.fft.fft(kaiser, n_fft)
    mid = n_channels // 2
    middle = np.r_[g_high_res[-mid:], g_high_res[:mid]]
    right = g_high_res[mid : mid + n_channels]
    left = g_high_res[-mid - n_channels : -mid]

    midsq = np.abs(middle) ** 2
    leftsq = np.abs(left) ** 2
    rightsq = np.abs(right) ** 2
    g = 2**25 / np.sqrt(midsq + leftsq + rightsq)

    return g**2.0


# def smooth_subbands(data: np.ndarray, channels: int) -> np.ndarray:
#     from scipy.signal import savgol_filter

#     n_subbands = int(data.shape[1] / channels)
#     subband_shape = (n_subbands, channels) + data.shape[2:]

#     median_frequency_profile = np.nanmedian(data, axis=0)

#     # Compute the Savgol filter of the frequency profile on a low order polynomial
#     # The window is also of the order of the subband as we don't expect much change from one to another
#     smoothed_fprofile = savgol_filter(
#         median_frequency_profile,
#         window_length=2 * channels + 1,
#         polyorder=1
#     )
#     # smoothed_fprofile_sb = smoothed_fprofile.reshape(subband_shape)

#     # Compute the shift between the median profile and the smoothed one
#     relative_profile_difference = smoothed_fprofile / median_frequency_profile

#     # Reshape per subband
#     nan_values = np.isnan(median_frequency_profile)
#     nan_values_sb = nan_values.reshape(subband_shape)
#     difference_sb = relative_profile_difference.reshape(subband_shape)

#     # Linear fit of the relative difference per subband
#     non_nan_channels = np.sum(nan_values_sb, axis=1)
#     linear_approx = np.arange(n_channels)[:, None] * (diff_sb[:, -1] - diff_sb[:, 0]) / n_channels + diff_sb[:, 0]




# ============================================================= #
# ---------------------- flatten_subband ---------------------- #
def flatten_subband(data: np.ndarray, channels: int, smooth_frequency_profile: bool = False) -> np.ndarray:
    # Check that data has not been altered, i.e. dimension 1 should be a multiple of channels
    if data.shape[1] % channels != 0:
        raise ValueError(
            f"data's frequency dimension (of size {data.shape[1]}) is "
            f"not a multiple of {channels=}. data's second "
            "dimension should be of size number_of_subbands*number_of_channels."
        )
    n_subbands = int(data.shape[1] / channels)
    pol_dims = data.ndim - 2  # the first two dimensions should be time and frequency

    # Compute the median spectral profile (along the time axis)
    median_frequency_profile = np.nanmedian(data, axis=0)
    subband_shape = (n_subbands, channels) + data.shape[2:]
    # Reshape to have the spectral profile as (subbands, channels, (polarizations...))
    median_subband_profile = median_frequency_profile.reshape(subband_shape)

    # # Select two data points (away from subband edges) that will be
    # # used to compute the affine function that approximates each subband.
    # ind1, ind2 = int(np.round(channels*1/3)), int(np.round(channels*2/3))
    # # Get the y-values corresponding to these two indices, each y is of shape (subbands, (polarizations...))
    # y1, y2 = median_subband_profile[:, ind1, ...], median_subband_profile[:, ind2, ...]

    # Split the subband in two and compute the 2 medians that will be
    # used to compute the affine function that approximates each subband.
    ind1, ind2 = (
        int(np.floor(channels / 2)) / 2,
        channels - int(np.floor(channels / 2)) / 2,
    )
    y1 = np.nanmedian(
        median_subband_profile[:, : int(np.floor(channels / 2)), ...], axis=1
    )
    y2 = np.nanmedian(
        median_subband_profile[:, int(np.ceil(channels / 2)) :, ...], axis=1
    )

    # Smooth the future slopes by shifting upward or downards y1 and y2 with respect to the difference
    # between them and the next point of the next subband
    # This step may result in non desirable results if subbands are not contiguous or not belonging to the same beam
    if smooth_frequency_profile:
        smooth_sb_diff = y1[1:, ...] - y2[:-1, ...]
        y1[1:, ...] += smooth_sb_diff / 2
        y2[:-1, ...] -= smooth_sb_diff / 2

    # Compute the linear approximations of each subbands, linear_subbands's shape is (channels, subbands, (polarizations...))
    x_values = np.arange(channels)[
        (...,) + (np.newaxis,) * (pol_dims + 1)
    ]  # +1 --> subbands
    slope = (y2 - y1) / (ind2 - ind1)
    linear_subbands = (x_values - ind1) * slope + y1  # linear equation

    # Compute the subband mean value and the normalised linear subbands
    subband_mean_values = np.nanmedian(
        linear_subbands, axis=0
    )  # shape (subbands, (polarizations))
    normalised_linear_subbands = np.swapaxes(
        linear_subbands / subband_mean_values[None, ...], 0, 1
    ).reshape(data.shape[1:])

    # Correct the data by the normalised linear subbands to flatten them
    return data / normalised_linear_subbands[None, ...]


# ============================================================= #
# ------------------- plot_dynamic_spectrum ------------------- #
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter


def plot_dynamic_spectrum(
    data: np.ndarray,
    time: Time,
    frequency: u.Quantity,
    fig: mpl.figure.Figure = None,
    ax: mpl.axes.Axes = None,
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 200,
    xlabel: str = None,
    ylabel: str = None,
    clabel: str = None,
    title: str = None,
    cmap: str = "YlGnBu_r",
    norm: str = "linear",
    vmin: float = None,
    vmax: float = None
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot a dynamic spectrum.
    This function uses :func:`~matplotlib.pyplot.pcolormesh` in the background.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Two-dimensional array, shaped like (time, frequency).
    time : :class:`~astropy.time.Time`
        One-dimensional time array. 
    frequency : :class:`~astropy.units.Quantity`
        One-dimensional frequency array, if ``ylabel`` is `None`, the :attr:`~astropy.units.Quantity.unit` is automatically used to describe the axis.
    fig : :class:`~matplotlib.figure.Figure`, optional
        Matplotlib figure if already existing, by default `None`
    ax : :class:`~matplotlib.axes.Axes`, optional
        Matplotlib ax if already existing, by default `None`
    figsize : Tuple[`int`, `int`], optional
        Size of the figure in inches, by default (10, 5)
    dpi : `int`, optional
        Dots per inch (best quality is around 300), by default 200
    xlabel : `str`, optional
        Label of the x-axis (time), by default `None` (i.e., generic label)
    ylabel : `str`, optional
        Label of the y-axis (time), by default `None` (i.e., generic label)
    clabel : `str`, optional
        Label of the colorbar, by default `None` (i.e., empty)
    title : `str`, optional
        Title of the graph, by default `None` (i.e., empty)
    cmap : `str`, optional
        Colormap (see `matplotlib colormaps <link https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_), by default "YlGnBu_r"
    norm : `str`, optional
        Normalization of the colorbar ('linear' or 'log'), by default "linear"
    vmin : `float`, optional
        Minimal data value to plot, by default `None`
    vmax : `float`, optional
        Maximal data value to plot, by default `None`

    Returns
    -------
    Tuple[:class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`]
        The figure and ax objects

    Raises
    ------
    ValueError
        Raised if ``norm`` does not match supported value.
    
    Examples
    --------
    .. code-block:: python

        >>> from nenupy.io.tf_utils import plot_dynamic_spectrum
        >>> from astropy.time import Time, TimeDelta
        >>> import astropy.units as u
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> n_times = 20
        >>> dt = TimeDelta(1800, format="sec")
        >>> n_freqs = 10
        >>> df = 1 * u.MHz
        >>> fig, ax = plot_dynamic_spectrum(
                data=np.arange(n_times * n_freqs).reshape((n_times, n_freqs)),
                time=Time("2024-01-01 00:00:00") + np.arange(n_times) * dt,
                frequency=50 * u.MHz + np.arange(n_freqs) * df,
                clabel="my color bar",
                norm="linear",
            )
    
    .. figure:: ../_images/io_images/plot_dynamic_spectrum.png
        :width: 650
        :align: center

    Warning
    -------
    Do not forget to release the matplotlib cache (using
    :func:`~matplotlib.pyplot.close`), either after calling several times
    this method, and/or once you are done with your desired plot.
    Otherwise you may suffer from significant performance issues
    as the plots stacks in the memory...

        >>> plt.close(fig)

    or 

        >>> plt.close("all")

    """

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    if ax is None:
        ax = fig.add_subplot()

    # Normalization
    if norm == "linear":
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
    elif norm == "log":
        if vmin is None:
            vmin = data[data > 0.].min()
        if vmax is None:
            vmax = data.max()
    else:
        raise ValueError("Invald norm, the following are supported: 'linear', 'log'.")

    im = ax.pcolormesh(
        time.datetime,
        frequency.value,
        data.T,
        shading="nearest",
        norm=norm,
        cmap=cmap,
        vmin=data.min() if vmin is None else vmin,
        vmax=data.max() if vmax is None else vmax,
    )

    # Colorbar
    cbar = plt.colorbar(im, pad=0.03)
    cbar.set_label(clabel)

    # Global
    ax.minorticks_on()
    ax.set_title(title)

    # X axis
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.set_xlabel(f"Time ({time.scale.upper()})" if xlabel is None else xlabel)

    # Y axis
    ax.set_ylabel(f"Frequency ({frequency.unit})" if ylabel is None else ylabel)

    return fig, ax


# ============================================================= #
# -------------------- polarization_angle --------------------- #
def polarization_angle(stokes_u: np.ndarray, stokes_q: np.ndarray) -> np.ndarray:
    """ """
    return 0.5 * np.arctan(stokes_u / stokes_q)


# ============================================================= #
# ------------------- rebin_along_dimension ------------------- #
def rebin_along_dimension(
    data: np.ndarray, axis_array: np.ndarray, axis: int, dx: float, new_dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ """

    # Basic checks to make sure that dimensions are OK
    if axis_array.ndim != 1:
        raise IndexError("axis_array should be 1D.")
    elif data.shape[axis] != axis_array.size:
        raise IndexError(
            f"Axis selected ({axis}) dimension {data.shape[axis]} does not match axis_array's shape {axis_array.shape}."
        )
    elif dx > new_dx:
        raise ValueError("Rebin expect a `new_dx` value larger than original `dx`.")

    initial_size = axis_array.size
    bin_size = int(np.floor(new_dx / dx))
    final_size = int(np.floor(initial_size / bin_size))
    leftovers = initial_size % final_size

    d_shape = data.shape

    log.info(f"\tdx: {dx} | new_dx: {new_dx} -> rebin factor: {bin_size}.")

    # Reshape the data and the axis to ease the averaging
    data = data[
        tuple(
            [
                slice(None) if i != axis else slice(None, initial_size - leftovers)
                for i in range(len(d_shape))
            ]
        )
    ].reshape(
        d_shape[:axis]
        + (final_size, int((initial_size - leftovers) / final_size))
        + d_shape[axis + 1 :]
    )
    axis_array = axis_array[: initial_size - leftovers].reshape(
        (final_size, int((initial_size - leftovers) / final_size))
    )

    # Average the data and the axis along the right dimension
    data = np.nanmean(data, axis=axis + 1)
    axis_array = np.nanmean(axis_array, axis=1)

    log.info(f"\tData rebinned, last {leftovers} samples were not considered.")

    return axis_array, data


# ============================================================= #
# ---------------- remove_channels_per_subband ---------------- #
def remove_channels_per_subband(
    data: np.ndarray, n_channels: int, channels_to_remove: Union[list, np.ndarray]
) -> np.ndarray:
    """Set channel indices of a time-frequency dataset to `NaN` values.
    Each subband of ``data`` is determined thanks to ``n_channels``
    (and the function :func:`~nenupy.io.tf_utils.reshape_to_subbands`).

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Data to be corrected, must be at least two-dimensional, the first two dimensions being respectively the time and the frequency
    n_channels : `int`
        Number of channels per subband
    channels_to_remove : Union[`list`, :class:`~numpy.ndarray`]
        Array of channel indices to set at `NaN` values, if `None` nothing is done and ``data`` is returned

    Returns
    -------
    :class:`~numpy.ndarray`
        Time-frequency correlations array, shaped as the original input, except that some channels are set to `NaN`.

    Raises
    ------
    TypeError
        Raised if ``channels_to_remove`` is not of the correct type or cannot be converted to a :class:`~numpy.ndarray`.
    IndexError
        Raised if any of the indices listed in ``channels_to_remove`` does not correspond to the ``n_channels`` argument.

    Examples
    --------
    .. code-block:: python

        >>> from nenupy.io.tf_utils import remove_channels_per_subband
        >>> import numpy as np

        >>> result = remove_channels_per_subband(
                data=np.ones((2, 10)),
                n_channels=5,
                channels_to_remove=[1, 3]
            )
        >>> print(result)
        [[ 1. nan  1. nan  1.  1. nan  1. nan  1.]
        [ 1. nan  1. nan  1.  1. nan  1. nan  1.]]

    See Also
    --------
    :func:`~nenupy.io.tf_utils.crop_subband_edges`
    """

    if channels_to_remove is None:
        # Don't do anything
        return data
    elif not isinstance(channels_to_remove, np.ndarray):
        try:
            channels_to_remove = np.array(channels_to_remove)
        except:
            raise TypeError("channels_to_remove must be a numpy array.")

    if len(channels_to_remove) == 0:
        # Empty list, no channels to remove
        return data
    elif not np.all(np.isin(channels_to_remove, np.arange(-n_channels, n_channels))):
        raise IndexError(
            f"Some channel indices are outside the available range (n_channels={n_channels})."
        )

    log.info("Removing channels...")

    original_shape = data.shape
    data = reshape_to_subbands(data=data, n_channels=n_channels)

    data[:, :, channels_to_remove, ...] = np.nan

    data = data.reshape(original_shape)

    log.info(f"\tChannels {channels_to_remove} set to NaN.")

    return data


# ============================================================= #
# -------------------- reshape_to_subbands -------------------- #
def reshape_to_subbands(data: np.ndarray, n_channels: int) -> np.ndarray:
    """Reshape a time-frequency data array by the sub-band dimension.
    Given a ``data`` array with one frequency axis of size `n_frequencies`, this functions split this axis in two axes of size `n_subbands` and ``n_channels``.

    :param data: Time-frequency correlations array, its second dimension must be the frequencies.
    :type data: :class:`~numpy.ndarray`
    :param n_channels: _description_
    :type n_channels: int
    :raises ValueError: _description_
    :return: _description_
    :rtype: :class:`~numpy.ndarray`

        :Example:

        .. code-block:: python

            >>> from nenupy.io.tf_utils import reshape_to_subbands
            >>> import numpy as np
            >>>
            >>> data = np.arange(3*10).reshape((3, 10))
            >>> result = reshape_to_subbands(data, 5)
            >>> result.shape
            (3, 2, 5)

    """

    # Reshape the data array to isolate individual subbands
    shape = data.shape
    n_freqs = shape[1]
    if n_freqs % n_channels != 0:
        raise ValueError(
            "The frequency dimension of `data` doesn't match the argument `n_channels`."
        )

    data = data.reshape((shape[0], int(n_freqs / n_channels), n_channels) + shape[2:])

    return data


# ============================================================= #
# ---------------------- sort_beam_edges ---------------------- #
def sort_beam_edges(beam_array: np.ndarray, n_channels: int) -> dict:
    """Find out where in the frequency axis a beam starts and end.
    This outputs a dictionnary such as:
    {
        "<beam_index>": (freq_axis_start_idx, freq_axis_end_idx,),
        ...
    }
    """
    indices = {}
    unique_beams = np.unique(beam_array)
    for b in unique_beams:
        b_idx = np.where(beam_array == b)[0]
        indices[str(b)] = (
            b_idx[0] * n_channels,  # freq start of beam
            (b_idx[-1] + 1) * n_channels - 1,  # freq stop of beam
        )
    return indices


# ============================================================= #
# ------------------ spectra_data_to_matrix ------------------- #
def spectra_data_to_matrix(fft0: da.Array, fft1: da.Array) -> da.Array:
    """fft0[..., :] = [XX, YY] = [XrXr+XiXi : YrYr+YiYi] = [ExEx*, EyEy*]
    fft1[..., :] = [Re(X*Y), Im(X*Y)] = [XrYr+XiYi : XrYi-XiYr] = [Re(ExEy*), Im(ExEy*)]
    Returns a (..., 2, 2) matrix (Dask)

    ExEy* = (XrYr+XiYi) + i(XiYr - XrYi)
    EyEx* = (YrXr+YiXi) + i(YiXr - YrXi)
    """
    row1 = da.stack(
        [fft0[..., 0], fft1[..., 0] - 1j * fft1[..., 1]], axis=-1  # XX  # XY*
    )
    row2 = da.stack(
        [fft1[..., 0] + 1j * fft1[..., 1], fft0[..., 1]], axis=-1  # YX*  # YY
    )
    return da.stack([row1, row2], axis=-1)


# ============================================================= #
# -------------------- store_dask_tf_data --------------------- #
def _time_to_keywords(prefix: str, time: Time) -> dict:
    """Returns a dictionnary of keywords in the HDF5 format."""
    return {
        f"{prefix.upper()}_MJD": time.mjd,
        f"{prefix.upper()}_TAI": time.tai.isot,
        f"{prefix.upper()}_UTC": time.isot + "Z",
    }


def store_dask_tf_data(
    file_name: str,
    data: da.Array,
    time: Time,
    frequency: u.Quantity,
    polarization: np.ndarray,
    beam: int = 0,
    stored_frequency_unit: str = "MHz",
    mode="auto",
    **metadata,
) -> None:
    log.info(f"Storing the data in '{file_name}'")

    # Check that the file_name has the correct extension
    if not file_name.lower().endswith(".hdf5"):
        raise ValueError(f"HDF5 files must ends with '.hdf5', got {file_name} instead.")
    elif mode.lower() == "auto":
        if os.path.isfile(file_name):
            mode = "a"
        else:
            mode = "w"

    stored_freq_quantity = u.Unit(stored_frequency_unit)
    frequency_min = frequency.min()
    frequency_max = frequency.max()

    with h5py.File(file_name, mode) as wf:
        beam_group_name = f"BEAM_{beam:03}"

        if mode == "w":
            log.info("\tCreating a brand new file...")
            # Update main attributes
            wf.attrs.update(metadata)
            wf.attrs["SOFTWARE_NAME"] = "nenupy"
            # wf.attrs["SOFTWARE_VERSION"] = nenupy.__version__
            wf.attrs["SOFTWARE_MAINTAINER"] = "alan.loh@obspm.fr"
            wf.attrs["FILENAME"] = file_name
            wf.attrs.update(_time_to_keywords("OBSERVATION_START", time[0]))
            wf.attrs.update(_time_to_keywords("OBSERVATION_END", time[-1]))
            wf.attrs["TOTAL_INTEGRATION_TIME"] = (time[-1] - time[0]).sec
            wf.attrs["OBSERVATION_FREQUENCY_MIN"] = frequency_min.to_value(
                stored_freq_quantity
            )
            wf.attrs["OBSERVATION_FREQUENCY_MAX"] = frequency_max.to_value(
                stored_freq_quantity
            )
            wf.attrs["OBSERVATION_FREQUENCY_CENTER"] = (
                (frequency_max + frequency_min) / 2
            ).to_value(stored_freq_quantity)
            wf.attrs["OBSERVATION_FREQUENCY_UNIT"] = stored_frequency_unit

            sub_array_group = wf.create_group(
                "SUB_ARRAY_POINTING_000"
            )  # TODO modify if a Spectra file can be generated from more than 1 analog beam

        elif mode == "a":
            log.info("\tTrying to append data to existing file...")
            sub_array_group = wf["SUB_ARRAY_POINTING_000"]
            if beam_group_name in sub_array_group.keys():
                raise Exception(
                    f"File '{file_name}' already contains '{beam_group_name}'."
                )

        else:
            raise KeyError(f"Invalid mode '{mode}'. Select 'w' or 'a' or 'auto'.")

        beam_group = sub_array_group.create_group(beam_group_name)
        beam_group.attrs.update(_time_to_keywords("TIME_START", time[0]))
        beam_group.attrs.update(_time_to_keywords("TIME_END", time[-1]))
        beam_group.attrs["FREQUENCY_MIN"] = frequency_min.to_value(stored_freq_quantity)
        beam_group.attrs["FREQUENCY_MAX"] = frequency_max.to_value(stored_freq_quantity)
        beam_group.attrs["FREQUENCY_UNIT"] = stored_frequency_unit

        coordinates_group = beam_group.create_group("COORDINATES")

        # Set time and frequency axes
        coordinates_group["time"] = time.jd
        coordinates_group["time"].make_scale("Time (JD)")
        coordinates_group["frequency"] = frequency.to_value(stored_freq_quantity)
        coordinates_group["frequency"].make_scale(
            f"Frequency ({stored_frequency_unit})"
        )
        coordinates_group.attrs["units"] = ["jd", stored_frequency_unit]

        log.info("\tTime and frequency axes written.")

        # Ravel the last polarization dimensions (above dim=2 -> freq)
        data = np.reshape(data, data.shape[:2] + (-1,))

        for pi in range(data.shape[-1]):
            current_polar = polarization[pi].upper()
            log.info(f"\tDealing with polarization '{current_polar}'...")
            data_i = data[:, :, pi]

            # data_group = beam_group.create_group(f"{current_polar.upper()}")

            dataset = beam_group.create_dataset(
                name=f"{current_polar}", shape=data_i.shape, dtype=data_i.dtype
            )

            dataset.dims[0].label = "time"
            dataset.dims[0].attach_scale(coordinates_group["time"])
            dataset.dims[1].label = "frequency"
            dataset.dims[1].attach_scale(coordinates_group["frequency"])

            with ProgressBar():
                da.store(data_i, dataset, compute=True, return_stored=False)

    log.info(f"\t'{file_name}' written.")


# ============================================================= #
# ------------------------ _Parameter ------------------------- #
class _TFParameter(ABC):
    def __init__(
        self,
        name: str,
        param_type: Any = None,
        partial_type_kw: dict = None,
        help_msg: str = "",
    ):
        self.name = name
        self.param_type = param_type
        self.partial_type_kw = partial_type_kw
        self.help_msg = help_msg
        self._value = None

    def __str__(self) -> str:
        return f"{self.name}={self.value}"

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, v: Any):
        if v is None:
            # Fill value, don't check anything.
            self._value = None
            return

        # Check that the value is of correct type, if not try to convert
        v = self._type_check(v)

        # Check that the value is within the expected boundaries
        if not self.is_expected_value(v):
            raise ValueError(f"Unexpected value of {self.name}. {self.help_msg}")

        log.debug(f"Parameter '{self.name}' set to {v}.")
        self._value = v

    def info(self):
        print(self.help_msg)

    def _type_check(self, value: Any) -> Any:
        if self.param_type is None:
            # No specific type has been defined
            return value
        elif isinstance(value, self.param_type):
            # Best scenario, the value is of expected type
            return value

        log.debug(
            f"{self.name} is of type {type(value)}. "
            f"Trying to convert it to {self.param_type}..."
        )
        try:
            if not (self.partial_type_kw is None):
                converted_value = partial(self.param_type, **self.partial_type_kw)(
                    value
                )
                log.debug(f"{self.name} set to {converted_value}.")
                return converted_value
            return self.param_type(value)
        except:
            raise TypeError(
                f"Type of '{self.name}' should be {self.param_type}, got {type(value)} instead! {self.help_msg}"
            )

    @abstractmethod
    def is_expected_value(self, value: Any) -> bool:
        raise NotImplementedError(
            f"Need to implement 'is_expected_value' in child class {self.__class__.__name__}."
        )


class _FixedParameter(_TFParameter):
    def __init__(self, name: str, value: Any = None):
        super().__init__(name=name)

        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, v: Any):
        raise Exception(f"_FixedParameter {self.name}'s value attribute cannot be set.")

    def is_expected_value(self, value: Any) -> bool:
        raise Exception("This should not have been called!")


class _ValueParameter(_TFParameter):
    def __init__(
        self,
        name: str,
        default: Any = None,
        param_type: Any = None,
        min_val: Any = None,
        max_val: Any = None,
        resolution: Any = 0,
        partial_type_kw: dict = None,
        help_msg: str = "",
    ):
        super().__init__(
            name=name,
            param_type=param_type,
            partial_type_kw=partial_type_kw,
            help_msg=help_msg,
        )

        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution

        self.value = default

    def is_expected_value(self, value: Any) -> bool:
        if not (self.min_val is None):
            if value < self.min_val - self.resolution / 2:
                log.error(
                    f"{self.name}'s value ({value}) is lower than the min_val {self.min_val}!"
                )
                return False
        if not (self.max_val is None):
            if value > self.max_val + self.resolution / 2:
                log.error(
                    f"{self.name}'s value ({value}) is greater than the max_val {self.max_val}!"
                )
                return False
        return True


class _RestrictedParameter(_TFParameter):
    def __init__(
        self,
        name: str,
        default: Any = None,
        param_type: Any = None,
        available_values: list = [],
        help_msg: str = "",
    ):
        super().__init__(name=name, param_type=param_type, help_msg=help_msg)

        self.available_values = available_values

        self.value = default

    def is_expected_value(self, value: Any) -> bool:
        if not hasattr(value, "__iter__"):
            value = [value]
        for val in value:
            if val not in self.available_values:
                log.error(f"{self.name} value must be one of {self.available_values}.")
                return False
        return True


class _BooleanParameter(_TFParameter):
    def __init__(self, name: str, default: Any = None, help_msg: str = ""):
        super().__init__(name=name, param_type=None, help_msg=help_msg)

        self.value = default

    def is_expected_value(self, value: bool) -> bool:
        if not isinstance(value, bool):
            log.error(f"{self.name}'s value should be a Boolean, got {value} instead.")
            return False
        return True


# ============================================================= #
# ------------------- TFPipelineParameters -------------------- #
class TFPipelineParameters:
    def __init__(self, *parameters):
        self.parameters = parameters
        self._original_parameters = parameters

    def __setitem__(self, name: str, value: Any):
        """_summary_

        Parameters
        ----------
        name : str
            _description_
        value : Any
            _description_
        """
        param = self._get_parameter(name)
        param.value = value

    def __getitem__(self, name: str) -> _TFParameter:
        return self._get_parameter(name).value

    def __repr__(self) -> str:
        message = "\n".join([str(param) for param in self.parameters])
        return message

    def info(self) -> None:
        for param in self.parameters:
            print(f"{param.name}: {param.value}")

    def _get_parameter(self, name: str):
        for param in self.parameters:
            if param.name == name.lower():
                return param
        else:
            param_names = [param.name for param in self.parameters]
            raise KeyError(
                f"Unknown parameter '{name}'! Available parameters are {param_names}."
            )

    def reset(self) -> None:
        """Reset all parameters to their original values.

        Returns
        -------
        _type_
            _description_
        """
        self.parameters = self._original_parameters

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def set_default(
        cls,
        time_min: Time,
        time_max: Time,
        freq_min: u.Quantity,
        freq_max: u.Quantity,
        beams: list,
        channels: int,
        dt: u.Quantity,
        df: u.Quantity,
    ):
        return cls(
            _FixedParameter(name="channels", value=channels),
            _FixedParameter(name="dt", value=dt),
            _FixedParameter(name="df", value=df),
            _ValueParameter(
                name="tmin",
                default=time_min,
                param_type=Time,
                min_val=time_min,
                max_val=time_max,
                resolution=dt,
                partial_type_kw={"precision": 7},
                help_msg="Lower edge of time selection, can either be given as an astropy.Time object or an ISOT/ISO string.",
            ),
            _ValueParameter(
                name="tmax",
                default=time_max,
                param_type=Time,
                min_val=time_min,
                max_val=time_max,
                resolution=dt,
                partial_type_kw={"precision": 7},
                help_msg="Upper edge of time selection, can either be given as an astropy.Time object or an ISOT/ISO string.",
            ),
            _ValueParameter(
                name="fmin",
                default=freq_min,
                param_type=u.Quantity,
                min_val=freq_min,
                max_val=freq_max,
                resolution=df,
                partial_type_kw={"unit": "MHz"},
                help_msg="Lower frequency boundary selection, can either be given as an astropy.Quantity object or float (assumed to be in MHz).",
            ),
            _ValueParameter(
                name="fmax",
                default=freq_max,
                param_type=u.Quantity,
                min_val=freq_min,
                max_val=freq_max,
                resolution=df,
                partial_type_kw={"unit": "MHz"},
                help_msg="Higher frequency boundary selection, can either be given as an astropy.Quantity object or float (assumed to be in MHz).",
            ),
            _RestrictedParameter(
                name="beam",
                default=beams[0],
                param_type=int,
                available_values=beams,
                help_msg="Beam selection, a single integer corresponding to the index of a recorded numerical beam is expected.",
            ),
            _ValueParameter(
                name="dispersion_measure",
                default=None,
                param_type=u.Quantity,
                partial_type_kw={"unit": "pc cm-3"},
                help_msg="Enable the correction by this Dispersion Measure, can either be given as an astropy.Quantity object or float (assumed to be in pc/cm^3).",
            ),
            _ValueParameter(
                name="rotation_measure",
                default=None,
                param_type=u.Quantity,
                partial_type_kw={"unit": "rad m-2"},
                help_msg="Enable the correction by this Rotation Measure, can either be given as an astropy.Quantity object or float (assumed to be in rad/m^2).",
            ),
            _ValueParameter(
                name="rebin_dt",
                default=None,
                param_type=u.Quantity,
                min_val=dt,
                partial_type_kw={"unit": "s"},
                help_msg="Desired rebinning time resolution, can either be given as an astropy.Quantity object or float (assumed to be in sec).",
            ),
            _ValueParameter(
                name="rebin_df",
                default=None,
                param_type=u.Quantity,
                min_val=df,
                partial_type_kw={"unit": "kHz"},
                help_msg="Desired rebinning frequency resolution, can either be given as an astropy.Quantity object or float (assumed to be in kHz).",
            ),
            # _BooleanParameter(
            #     name="correct_bandpass", default=True,
            #     help_msg="Enable/disable the correction of each subband bandpass."
            # ),
            # _BooleanParameter(
            #     name="correct_jump", default=False,
            #     help_msg="Enable/disable the auto-correction of 6-min jumps, the results of this process are highly data dependent."
            # ),
            _RestrictedParameter(
                name="remove_channels",
                default=None,
                param_type=list,
                available_values=np.arange(-channels, channels),
                help_msg="List of subband channels to remove, e.g. `remove_channels=[0,1,-1]` would remove the first, second (low-freq) and last channels from each subband.",
            ),
            _ValueParameter(
                name="dreambeam_skycoord",
                default=None,
                param_type=SkyCoord,
                help_msg="Tracked celestial coordinates used during DreamBeam correction (along with 'dreambeam_dt' and 'dreambeam_parallactic'), an astropy.SkyCoord object is expected.",
            ),
            _ValueParameter(
                name="dreambeam_dt",
                default=None,
                param_type=u.Quantity,
                partial_type_kw={"unit": "s"},
                help_msg="DreamBeam correction time resolution (along with 'dreambeam_skycoord' and 'dreambeam_parallactic'), an astropy.Quantity or a float (assumed in seconds) are expected.",
            ),
            _BooleanParameter(
                name="dreambeam_parallactic",
                default=True,
                help_msg="DreamBeam parallactic angle correction (along with 'dreambeam_skycoord' and 'dreambeam_dt'), a boolean is expected.",
            ),
            _ValueParameter(
                name="stokes",
                default="I",
                param_type=None,
                help_msg="Stokes parameter selection, can either be given as a string or a list of strings, e.g. ['I', 'Q', 'V/I'].",
            ),
            _BooleanParameter(
                name="ignore_volume_warning",
                default=False,
                help_msg="Ignore or not (default value) the limit regarding output data volume.",
            ),
            _BooleanParameter(
                name="overwrite",
                default=False,
                help_msg="Overwrite or not (default value) the resulting HDF5 file.",
            ),
            _BooleanParameter(
                name="smooth_frequency_profile",
                default=False,
                help_msg="Smooth the adjacent subbands (option of task `flatten_subband`).",
            ),
        )


# ============================================================= #
# ---------------------- ReducedSpectra ----------------------- #
class ReducedSpectra:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self._rfile = h5py.File(file_name, "r")
        log.info(f"'{self.file_name}' opened.")

    def infos(self):
        return "hello"

    def get(
        self, subarray_pointing_id: int = 0, beam_id: int = 0, data_key: str = None
    ) -> Tuple[Time, u.Quantity, np.ndarray]:
        """_summary_

        Parameters
        ----------
        data_key : str, optional
            _description_, by default None

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        KeyError
            _description_
        """
        data_ext = self._rfile[
            f"SUB_ARRAY_POINTING_{subarray_pointing_id:03}/BEAM_{beam_id:03}"
        ]
        available_keys = list(data_ext.keys())
        available_keys.remove("COORDINATES")
        if data_key is None:
            # If no key is selected, take the first one by default
            data_key = available_keys[0]
        elif data_key not in available_keys:
            self.close()
            raise KeyError(
                f"Invalid data_key '{data_key}', available values: {available_keys}."
            )
        log.info(f"Selected data extension '{data_key}'.")

        times_axis, frequency_axis = self._build_axes(data_ext["COORDINATES"])

        return times_axis, frequency_axis, data_ext[data_key][:]

    def plot(
        self,
        subarray_pointing_id: int = 0,
        beam_id: int = 0,
        data_key: str = None,
        **kwargs,
    ):
        time, frequency, data = self.get(subarray_pointing_id, beam_id, data_key)

        try:
            fig, ax = plot_dynamic_spectrum(
                data=data, time=time, frequency=frequency, **kwargs
            )
            plt.show()

        except:
            self.close()
            raise

    def close(self):
        self._rfile.close()
        log.info(f"'{self.file_name}' closed.")

    @staticmethod
    def _build_axes(hdf5_coordinates_ext) -> Tuple[Time, u.Quantity]:
        """_summary_

        Returns
        -------
        Tuple[Time, u.Quantity]
            _description_
        """

        units = hdf5_coordinates_ext.attrs["units"]

        time = Time(hdf5_coordinates_ext["time"][:], format=units[0])
        frequency = hdf5_coordinates_ext["frequency"][:] * u.Unit(units[1])

        return time, frequency
