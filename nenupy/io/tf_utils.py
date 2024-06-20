"""
    ******************************
    Support to Time-Frequency data
    ******************************

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
    "ReducedSpectra"
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
    """Inverts the halves of each beamlet and reshape the
    array in 2D (time, frequency) or 4D (time, frequency, 2, 2)
    if this is an 'ejones' matrix.
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
    """ """

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
    """ """

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
    """data_array: >2 D, last 2 dimensions are ((XX, XY), (YX, YY))"""

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

    .. image:: ../_images/bandpass_corr.png
        :width: 800

    :param data: _description_
    :type data: np.ndarray
    :param n_channels: _description_
    :type n_channels: int
    :raises ValueError: _description_
    :return: _description_
    :rtype: np.ndarray
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
    lower_edge_channels: int,
    higher_edge_channels: int,
) -> np.ndarray:
    """ """

    log.info("Removing edge channels...")

    if lower_edge_channels + higher_edge_channels >= n_channels:
        raise ValueError(
            f"{lower_edge_channels + higher_edge_channels} channels to crop out of {n_channels} channels subbands."
        )

    n_times, n_freqs, _, _ = data.shape
    data = reshape_to_subbands(data=data, n_channels=n_channels)

    # Set to NaN edge channels
    data[:, :, :lower_edge_channels, :, :] = np.nan  # lower edge
    data[:, :, n_channels - higher_edge_channels :, :] = np.nan  # upper edge
    data = data.reshape((n_times, n_freqs, 2, 2))

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
    frequency: u.Quantity, data: np.ndarray, rotation_measure: u.Quantity
) -> np.ndarray:
    """ """
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
    """ """
    kaiser_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bandpass_coeffs.dat"
    )
    kaiser = np.loadtxt(kaiser_file)

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


# ============================================================= #
# ---------------------- flatten_subband ---------------------- #
def flatten_subband(data: np.ndarray, channels: int) -> np.ndarray:

    # Check that data has not been altered, i.e. dimension 1 should be a multiple of channels
    if data.shape[1] % channels != 0:
        raise ValueError(
            f"data's frequency dimension (of size {data.shape[1]}) is "
            f"not a multiple of channels={channels}. data's second "
            "dimension should be of size number_of_subbands*number_of_channels."
        )
    n_subbands = int(data.shape[1]/channels)
    pol_dims = data.ndim - 2 # the first two dimensions should be time and frequency

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
    ind1, ind2 = int(np.floor(channels/2))/2, channels - int(np.floor(channels/2))/2
    y1 = np.nanmedian(median_subband_profile[:, :int(np.floor(channels/2)), ...], axis=1)
    y2 = np.nanmedian(median_subband_profile[:, int(np.ceil(channels/2)):, ...], axis=1)

    # Compute the linear approximations of each subbands, linear_subbands's shape is (channels, subbands, (polarizations...))
    x_values = np.arange(channels)[(...,) + (np.newaxis,) * (pol_dims + 1)] # +1 --> subbands
    linear_subbands = (x_values - ind1) * (y2 - y1) / (ind2 - ind1) + y1 # linear equation

    # Compute the subband mean value and the normalised linear subbands
    subband_mean_values = np.nanmedian(linear_subbands, axis=0) # shape (subbands, (polarizations))
    normalised_linear_subbands = np.swapaxes(linear_subbands / subband_mean_values[None, ...], 0, 1).reshape(data.shape[1:])

    # Correct the data by the normalised linear subbands to flatten them
    return data / normalised_linear_subbands[None, ...]

# ============================================================= #
# -------------------- plot_dynamic_spectrum --------------------- #
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

def plot_dynamic_spectrum(data: np.ndarray, time: Time, frequency: u.Quantity, fig: mpl.figure.Figure = None, ax: mpl.axes.Axes = None, **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """_summary_

    Parameters
    ----------
    data : np.ndarray
        _description_
    time : Time
        _description_
    frequency : u.Quantity
        _description_
    fig : mpl.figure.Figure, optional
        _description_, by default None
    ax : mpl.axes.Axes, optional
        _description_, by default None

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axes]
        _description_
    """
    
    if fig is None:
        fig = plt.figure(
            figsize=kwargs.get("figsize", (10, 5)),
            dpi=kwargs.get("dpi", 200)
        )
    
    if ax is None:
        ax = fig.add_subplot()
    
    im = ax.pcolormesh(
        time.datetime,
        frequency.value,
        data.T,
        shading="nearest",
        norm=kwargs.get("norm", "linear"),
        cmap=kwargs.get("cmap", "YlGnBu_r"),
        vmin=kwargs.get("vmin", data.min()),
        vmax=kwargs.get("vmax", data.max())
    )

    # Colorbar
    cbar = plt.colorbar(im, pad=0.03)
    cbar.set_label(kwargs.get("clabel", ""))

    # Global    
    ax.minorticks_on()
    ax.set_title(kwargs.get("title", ""))

    # X axis
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.set_xlabel(kwargs.get("xlabel", f"Time ({time.scale.upper()})"))

    # Y axis
    ax.set_ylabel(kwargs.get("ylabel", f"Frequency ({frequency.unit})"))

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
    The ``data`` array is first re-shaped as `(n_times, n_subbands, n_channels, 2, 2)` using :func:`~nenupy.io.tf_utils.reshape_to_subbands`.

    :param data: Time-frequency correlations array, its dimensions must be `(n_times, n_frequencies, 2, 2)`. The data type must be `float`.
    :type data: :class:`~numpy.ndarray`
    :param n_channels: Number of channels per sub-band.
    :type n_channels: `int`
    :param channels_to_remove: Array of channel indices to set at `NaN` values.
    :type channels_to_remove: :class:`~numpy.ndarray` or `list`
    :raises TypeError: If ``channels_to_remove`` is not of the correct type.
    :raises IndexError: If any of the indices listed in ``channels_to_remove`` does not correspond to the ``n_channels`` argument.
    :return: Time-frequency correlations array, shaped as the original input, except that some channels are set to `NaN`.
    :rtype: :class:`~numpy.ndarray`

    :Example:

        .. code-block:: python

            >>> from nenupy.io.tf_utils import remove_channels_per_subband
            >>> import numpy as np
            >>>
            >>> data = np.arange(2*4*2*2, dtype=float).reshape((2, 4, 2, 2))
            >>> result = remove_channels_per_subband(data, 2, [0])
            >>> result[:, :, 0, 0]
            array([[nan,  4., nan, 12.],
                   [nan, 20., nan, 28.]])

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

    n_times, n_freqs, _, _ = data.shape
    data = reshape_to_subbands(data=data, n_channels=n_channels)

    data[:, :, channels_to_remove, :, :] = np.nan

    data = data.reshape((n_times, n_freqs, 2, 2))

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

def store_dask_tf_data(file_name: str, data: da.Array, time: Time, frequency: u.Quantity, polarization: np.ndarray, beam: int = 0, stored_frequency_unit: str = "MHz", mode="auto", **metadata) -> None:

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
            wf.attrs["OBSERVATION_FREQUENCY_MIN"] = frequency_min.to_value(stored_freq_quantity)
            wf.attrs["OBSERVATION_FREQUENCY_MAX"] = frequency_max.to_value(stored_freq_quantity)
            wf.attrs["OBSERVATION_FREQUENCY_CENTER"] = (
                ((frequency_max + frequency_min) / 2).to_value(stored_freq_quantity)
            )
            wf.attrs["OBSERVATION_FREQUENCY_UNIT"] = stored_frequency_unit

            sub_array_group = wf.create_group("SUB_ARRAY_POINTING_000") # TODO modify if a Spectra file can be generated from more than 1 analog beam
        
        elif mode == "a":
            log.info("\tTrying to append data to existing file...")
            sub_array_group = wf["SUB_ARRAY_POINTING_000"]
            if beam_group_name in sub_array_group.keys():
                raise Exception(f"File '{file_name}' already contains '{beam_group_name}'.")

        else:
            raise KeyError(f"Invalid mode '{mode}'. Select 'w' or 'a' or 'auto'.")

        beam_group = sub_array_group.create_group(beam_group_name)
        beam_group.attrs.update(_time_to_keywords("TIME_START", time[0]))
        beam_group.attrs.update(_time_to_keywords("TIME_END", time[-1]))
        beam_group.attrs["FREQUENCY_MIN"] = (frequency_min.to_value(stored_freq_quantity))
        beam_group.attrs["FREQUENCY_MAX"] = (frequency_max.to_value(stored_freq_quantity))
        beam_group.attrs["FREQUENCY_UNIT"] = stored_frequency_unit

        coordinates_group = beam_group.create_group("COORDINATES")

        # Set time and frequency axes
        coordinates_group["time"] = time.jd
        coordinates_group["time"].make_scale("Time (JD)")
        coordinates_group["frequency"] = frequency.to_value(stored_freq_quantity)
        coordinates_group["frequency"].make_scale(f"Frequency ({stored_frequency_unit})")
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
                name=f"{current_polar}",
                shape=data_i.shape,
                dtype=data_i.dtype
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
        raise NotImplementedError(f"Need to implement 'is_expected_value' in child class {self.__class__.__name__}.")


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
            if value < self.min_val - self.resolution/2:
                log.error(
                    f"{self.name}'s value ({value}) is lower than the min_val {self.min_val}!"
                )
                return False
        if not (self.max_val is None):
            if value > self.max_val + self.resolution/2:
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
                help_msg="Ignore or not (default value) the limit regarding output data volume."
            ),
            _BooleanParameter(
                name="overwrite",
                default=False,
                help_msg="Overwrite or not (default value) the resulting HDF5 file."
            )
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
    
    def get(self, subarray_pointing_id: int = 0, beam_id: int = 0, data_key: str = None) -> Tuple[Time, u.Quantity, np.ndarray]:
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
        data_ext = self._rfile[f"SUB_ARRAY_POINTING_{subarray_pointing_id:03}/BEAM_{beam_id:03}"]
        available_keys = list(data_ext.keys())
        available_keys.remove("COORDINATES")
        if data_key is None:
            # If no key is selected, take the first one by default
            data_key = available_keys[0]
        elif data_key not in available_keys:
            self.close()
            raise KeyError(f"Invalid data_key '{data_key}', available values: {available_keys}.")
        log.info(f"Selected data extension '{data_key}'.")

        times_axis, frequency_axis = self._build_axes(data_ext["COORDINATES"])

        return times_axis, frequency_axis, data_ext[data_key][:]

    def plot(self, subarray_pointing_id: int = 0, beam_id: int = 0, data_key: str = None, **kwargs):

        time, frequency, data = self.get(subarray_pointing_id, beam_id, data_key)

        try:
            fig, ax = plot_dynamic_spectrum(
                data=data,
                time=time,
                frequency=frequency,
                **kwargs
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