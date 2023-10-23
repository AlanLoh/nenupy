"""
    ***************
    Dynspec ToolBox
    ***************
"""

__author__ = "Alan Loh"
__copyright__ = "Copyright 2023, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = ["de_disperse_array"]

import numpy as np
import astropy.units as u

from nenupy.astro import dispersion_delay


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
        # Mask right edge of dynspec
        data[-cell_delays[i] :, i, ...] = np.nan

    return data
