"""

"""

import numpy as np
import os
import dask.array as da
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from typing import Union, List, Tuple
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
    "get_bandpass",
    "polarization_angle",
    "rebin_along_dimension",
    "sort_beam_edges",
    "spectra_data_to_matrix"
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
        parallactic: bool = True
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
    n_subbands = int(freq_size/n_channels)

    # Compute the number of time samples that will be corrected together
    time_group_size = int(np.round(time_step_sec/dt_sec))
    log.debug(f"\tGroups of {time_group_size} time blocks will be corrected altogether ({dt_sec*time_group_size} sec resolution).")
    n_time_groups = time_size // time_group_size
    leftover_time_samples = time_size % time_group_size

    # Computing DreamBeam matrices
    db_time, db_frequency, db_jones = compute_jones_matrices(
        start_time=Time(time_unix[0], format="unix", precision=7),
        time_step=TimeDelta(time_group_size * dt_sec, format="sec"),
        duration=TimeDelta(time_unix[-1] - time_unix[0], format="sec"),
        skycoord=skycoord,
        parallactic=parallactic
    )
    db_time = db_time.unix
    db_frequency = db_frequency.to_value(u.Hz)
    db_jones = np.swapaxes(db_jones, 0, 1)

    # Reshape the data at the time and frequency resolutions
    # Take into account leftover times
    data_leftover = data[-leftover_time_samples:, ...].reshape(
        (
            leftover_time_samples,
            n_subbands,
            n_channels,
            2, 2
        )
    )
    data = data[: time_size - leftover_time_samples, ...].reshape(
        (
            n_time_groups,
            time_group_size,
            n_subbands,
            n_channels,
            2, 2
        )
    )

    # Compute the frequency indices to select the corresponding Jones matrices
    subband_start_frequencies = frequency_hz.reshape((n_subbands, n_channels))[:, 0]
    freq_start_idx = np.argmax(db_frequency >= subband_start_frequencies[0])
    freq_stop_idx = db_frequency.size - np.argmax(db_frequency[::-1] < subband_start_frequencies[-1])

    # Do the same with the time
    group_start_time = time_unix[: time_size - leftover_time_samples].reshape((n_time_groups, time_group_size))[:, 0]
    time_start_idx = np.argmax(db_time >= group_start_time[0])
    time_stop_idx = db_time.size - np.argmax(db_time[::-1] < group_start_time[-1])

    jones = db_jones[time_start_idx:time_stop_idx + 1, freq_start_idx:freq_stop_idx + 1, :, :][:, None, :, None, :, :]
    jones_leftover = db_jones[-1, freq_start_idx:freq_stop_idx + 1, :, :][None, :, None, :, :]

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
            np.matmul(
                jones,
                np.matmul(data, jones_hermitian)
            ).reshape((time_size - leftover_time_samples, freq_size, 2, 2)),
            np.matmul(
                jones_leftover,
                np.matmul(data_leftover, jones_leftover_hermitian)
            ).reshape((leftover_time_samples, freq_size, 2, 2))
        ),
        axis=0
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
def compute_spectra_frequencies(subband_start_hz: np.ndarray, n_channels: int, frequency_step_hz: float) -> da.Array:
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
def compute_spectra_time(block_start_time_unix: np.ndarray, ntime_per_block: int, time_step_s: float) -> da.Array:
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
def compute_stokes_parameters(data_array: np.ndarray, stokes: Union[List[str], str]) -> np.ndarray:
    """ data_array: >2 D, last 2 dimensions are ((XX, XY), (YX, YY))
    """

    log.info("Computing Stokes parameters...")

    # Assert that the last dimensions are shaped like a cross correlation electric field matrix
    if data_array.shape[-2:] != (2, 2):
        raise Exception("The data_array last 2 dimensions are not of shape (2, 2).")

    result = None

    for stokes_i in stokes:
        # Compute the correct Stokes value
        if stokes_i.upper() == "I":
            data_i = data_array[..., 0, 0].real + data_array[..., 1, 1].real
        elif stokes_i.upper() == "Q":
            data_i = data_array[..., 0, 0].real - data_array[..., 1, 1].real
        elif stokes_i.upper() == "U":
            data_i = data_array[..., 0, 1].real * 2
        elif stokes_i.upper() == "V":
            data_i = data_array[..., 0, 1].imag * 2
        elif stokes_i.upper() == "Q/I":
            data_i = (data_array[..., 0, 0].real - data_array[..., 1, 1].real)/(data_array[..., 0, 0].real + data_array[..., 1, 1].real)
        elif stokes_i.upper() == "U/I":
            data_i = data_array[..., 0, 1].real * 2 / (data_array[..., 0, 0].real + data_array[..., 1, 1].real)
        elif stokes_i.upper() == "V/I":
            data_i = data_array[..., 0, 1].imag * 2 / (data_array[..., 0, 0].real + data_array[..., 1, 1].real)
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
    """ """

    log.info("Correcting for bandpass...")

    # Compute the bandpass
    bandpass = get_bandpass(n_channels=n_channels)

    # Reshape the data array to isolate individual subbands
    n_times, n_freqs, _, _ = data.shape
    if n_freqs % n_channels != 0:
        raise ValueError("The frequency dimension of `data` doesn't match the argument `n_channels`.")
    data = data.reshape(
        (
            n_times,
            int(n_freqs / n_channels), # subband
            n_channels, # channels
            2, 2
        )
    )

    # Multiply the channels by the bandpass to correct them
    data *= bandpass[None, None, :, None, None]

    log.debug(f"\tEach subband corrected by the bandpass of size {bandpass.size}.")

    # Re-reshape the data into time, frequency, (2, 2) array
    return data.reshape((n_times, n_freqs, 2, 2))

# ============================================================= #
# -------------------- crop_subband_edges --------------------- #
def crop_subband_edges(data: np.ndarray, n_channels: int, lower_edge_channels: int, higher_edge_channels: int) -> np.ndarray:
    """ """

    log.info("Removing edge channels...")

    if lower_edge_channels + higher_edge_channels >= n_channels:
        raise ValueError(f"{lower_edge_channels + higher_edge_channels} channels to crop out of {n_channels} channels subbands.")

    # Reshape the data array to isolate individual subbands
    n_times, n_freqs, _, _ = data.shape
    if n_freqs % n_channels != 0:
        raise ValueError("The frequency dimension of `data` doesn't match the argument `n_channels`.")

    data = data.reshape(
        (
            n_times,
            int(n_freqs / n_channels), # subband
            n_channels, # channels
            2, 2
        )
    )

    # Set to NaN edge channels
    data[:, :, : lower_edge_channels, :, :] = np.nan # lower edge
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

    log.info("De-dispersing data...")

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
    for i in range(frequencies.size):
        data[:, i, ...] = np.roll(data[:, i, ...], -cell_delays[i], 0)
        # # Mask right edge of dynspec
        data[-cell_delays[i] :, i, ...] = np.nan

    return data

# ============================================================= #
# ---------------------- de_faraday_data ---------------------- #
def de_faraday_data(data: np.ndarray, frequency: u.Quantity, rotation_measure: u.Quantity) -> np.ndarray:
    """ """

    log.info("Correcting for Faraday rotation...")

    # Check the dimensions
    if (data.ndim != 4) or (data.shape[1:] != (frequency.size, 2, 2)):
        raise Exception("Wrong data dimensions!")

    # Computing the Faraday angles compared to infinite frequency
    log.info(f"\tComputing {frequency.size} Faraday rotation angles at the RM={rotation_measure}...")
    rotation_angle = faraday_angle(
        frequency=frequency,
        rotation_measure=rotation_measure,
        inverse=True
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
    kaiser_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bandpass_coeffs.dat")
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
# -------------------- polarization_angle --------------------- #
def polarization_angle(stokes_u: np.ndarray, stokes_q: np.ndarray) -> np.ndarray:
    """ """
    return 0.5 * np.arctan(stokes_u / stokes_q)

# ============================================================= #
# ------------------- rebin_along_dimension ------------------- #
def rebin_along_dimension(data: np.ndarray, axis_array: np.ndarray, axis: int, dx: float, new_dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """ """

    # Basic checks to make sure that dimensions are OK
    if axis_array.ndim != 1:
        raise IndexError("axis_array should be 1D.")
    elif data.shape[axis] != axis_array.size:
        raise IndexError(f"Axis selected ({axis}) dimension {data.shape[axis]} does not match axis_array's shape {axis_array.shape}.")
    elif dx > new_dx:
        raise ValueError("Rebin expect a `new_dx` value larger than original `dx`.")

    initial_size = axis_array.size
    bin_size = int(np.floor(new_dx / dx))
    final_size = int(np.floor(initial_size / bin_size))
    leftovers = initial_size % final_size
    
    d_shape = data.shape

    log.info(f"\tdx: {dx} | new_dx: {new_dx} -> rebin factor: {bin_size}.")

    # Reshape the data and the axis to ease the averaging
    data = data[tuple([slice(None) if i != axis else slice(None, initial_size - leftovers) for i in range(len(d_shape))])].reshape(
        d_shape[:axis] + (final_size, int((initial_size - leftovers) / final_size)) + d_shape[axis + 1:]
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
