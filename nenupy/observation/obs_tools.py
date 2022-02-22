#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *********************
    Interferometric Array
    *********************
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2022, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "in_analog_beam_max_frequency"
]


from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy.astro.astro_tools import SolarSystemSource
from nenupy.astro.target import FixedTarget, SolarSystemTarget
from nenupy.instru import MiniArray


# ============================================================= #
# ---------------- Interferometer class errors ---------------- #
# ============================================================= #
def in_analog_beam_max_frequency(
        source1: str,
        source2: str,
        time: Time = Time.now(),
    ) -> u.Quantity:
    """ Given two sources at any time(s), computes the maximal
        frequency(ies) in order to observe them simultaneously within
        the same NenuFAR analog beam.

        :Example:
            .. code-block:: python

                in_analog_beam_max_frequency(
                    source1="Sun",
                    source2="Moon",
                    time=Time("2022-02-21T12:00:00")
                )

                in_analog_beam_max_frequency(
                    source1="Sun",
                    source2="PSR J2330-2005",
                    time=Time(["2022-02-21T12:00:00", "2022-02-21T15:00:00"])
                )

                in_analog_beam_max_frequency(
                    source1="Saturn",
                    source2="Sun",
                    time=Time("2022-02-04T06:00:00") + np.arange(10)*TimeDelta(2, format="jd")
                )

    """

    def _select_target_type(source_name):
        # Check whether the source name matches a solar system object
        if source_name.upper() in SolarSystemSource._member_names_:
            return SolarSystemTarget.from_name(source_name, time=time)
        else:
            return FixedTarget.from_name(source_name, time=time)

    # Initialize the two sources
    src1 = _select_target_type(source1)
    src2 = _select_target_type(source2)
    # Compute their angular separations
    src_separations = src1.separation(src2)
    if src_separations.isscalar:
        src_separations = src_separations.reshape((1,))

    # Evaluate the analog beam Half Width at Half Maximum over the frequency range
    frequencies = np.linspace(15, 85, 200)*u.MHz
    ma = MiniArray()
    analog_beam_fwhm = ma.angular_resolution(frequency=frequencies)
    analog_beam_hwhm = analog_beam_fwhm/2

    # Compute the maximal frequencies at which the two sources
    # are still within the HWHM of the analog beam
    max_frequencies = u.Quantity(np.zeros(time.size), unit="MHz")
    within_analog_beam = analog_beam_hwhm[None, :] >= src_separations[:, None]
    for i in range(time.size):
        try:
            max_frequencies[i] = frequencies[within_analog_beam[i, :]].max()
        except ValueError:
            # The separation is greater than the analog beam at any freq
            pass
    return max_frequencies
# ============================================================= #
# ============================================================= #

