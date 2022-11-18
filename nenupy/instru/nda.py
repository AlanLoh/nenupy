#! /usr/bin/python3
# -*- coding: utf-8 -*-


""" 
    ******************************
    Nancay Decameter Array Classes
    ******************************

"""

__author__ = "Alan Loh"
__copyright__ = "Copyright 2022, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = []


import logging
log = logging.getLogger(__name__)

from nenupy.instru.interferometer import Interferometer
from nenupy.astro.sky import Sky
from nenupy.astro.pointing import Pointing

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
import dask.array as da
from enum import Enum


# ============================================================= #
# ---------------- Polarization / Antenna Gain ---------------- #
# ============================================================= #
class _NDAAntennaGain:
    """ """

    def __init__(self, polarization: str = 'RH'):
        self.polarization = polarization


    def __getitem__(self, sky: Sky) -> np.ndarray:
        ...


class NDAPolarization(Enum):
    """ """

    LH = _NDAAntennaGain('LH')
    RH = _NDAAntennaGain('RH')
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------------- NDA ---------------------------- #
# ============================================================= #
class NDA(Interferometer):

    def __init__(self):
        position = EarthLocation(...)
        antenna_names = ...
        antenna_positions = ...
        antenna_gains = np.array([
            self._antenna_gain for _ in range(antenna_names.size)
        ])

        super().__init__(
            position=position,
            antenna_names=antenna_names,
            antenna_positions=antenna_positions,
            antenna_gains=antenna_gains
        )


    def effective_area(self,
            frequency: u.Quantity = 50*u.MHz,
            elevation: u.Quantity = 90*u.deg
        ) -> u.Quantity:
        """ """
        raise NotImplementedError()

    
    def instrument_temperature(self,
            frequency: u.Quantity = 50*u.MHz,
            lna_filter: int = 0
        ) -> u.Quantity:
        """ """
        raise NotImplementedError()


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    # @lru_cache(maxsize=1)
    def _antenna_gain(self, sky: Sky, pointing: Pointing) -> np.ndarray:
        """
        """
        gain = da.ones(
            (
                sky.time.size,
                sky.frequency.size,
                sky.polarization.size,
                sky.coordinates.size
            ),
            dtype=np.float64
        )
        for i, pol in enumerate(sky.polarization):
            if not isinstance(pol, NDAPolarization):
                log.warning(
                    f"Invalid value encountered in <attr 'Sky.polarization'>: '{pol}'. "
                    f"Polarization has been set to '{NDAPolarization.RH}' by default."
                )
                pol = NDAPolarization.RH
            gain[:, :, i, :] = pol.value[sky]
        return gain

# ============================================================= #
# ============================================================= #

