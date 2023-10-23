__author__ = "Alan Loh"
__copyright__ = "Copyright 2023, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"

import numpy as np
from nenupy.astro import dispersion_delay
import astropy.units as u

from nenupy.undysputed.utils import de_disperse_array


# ============================================================= #
# ------------------ test_de_disperse_array ------------------- #
def test_de_disperse_array():
    # Create a pulsar profile
    frequencies = np.linspace(20, 60, 50)*u.MHz
    dm = 2.97*u.pc/(u.cm**3)
    dt = 0.5*u.s
    amp = 10
    data = np.zeros((80, frequencies.size, 5))
    delays = dispersion_delay(frequencies, dm)
    indices = np.round(delays.to_value(u.s) / dt.to_value(u.s))
    data[indices.astype(int), np.arange(frequencies.size), :] = amp
    # -> plt.imshow(data.T, origin="lower")

    # Dedisperse it
    dedispersed_data = de_disperse_array(
        data=data,
        frequencies=frequencies,
        time_step=dt,
        dispersion_measure=dm
    )

    # The mean of the summed dynamic spectrum should be equal to amp
    assert np.sum(np.nanmean(dedispersed_data, axis=(1, 2))[:10]) == amp

    # The de-dispersed dynamic spetrum should have all values gathered in 2 time indices max
    assert np.where(np.nanmean(dedispersed_data, axis=(1, 2))[:10])[0].size <= 2
