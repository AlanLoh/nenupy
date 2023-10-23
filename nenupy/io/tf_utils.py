"""

"""

import numpy as np
import os
import dask.array as da
import astropy.units as u
from astropy.time import Time
from typing import Union, List, Tuple
import logging
log = logging.getLogger(__name__)


__all__ = [
    "blocks_to_tf_data",
    "compute_spectra_frequencies",
    "compute_spectra_time",
    "compute_stokes_parameters",
    "rebin_along_dimension",
    "sort_beam_edges",
    "spectra_data_to_matrix"
]

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
