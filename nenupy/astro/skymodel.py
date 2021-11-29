#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Skymodel
    ********
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "HpxGSM"
]


import logging
log = logging.getLogger(__name__)
import numpy as np

try:
    from pygsm import GlobalSkyModel
except ImportError:
    log.warning("Unable to load 'pygsm', some functionalities may not be working.")
    GlobalSkyModel = None
try:
    import healpy as hp
except ImportError:
    log.warning("Unable to load 'healpy', some functionalities may not be working.")
    hp = None

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import dask.array as da

from nenupy import nenufar_position, HiddenPrints
from nenupy.astro.sky import HpxSky


# ============================================================= #
# -------------------------- HpxGSM --------------------------- #
# ============================================================= #
class HpxGSM(HpxSky):
    """ """

    def __init__(self,
            resolution: u.Quantity = 1*u.deg,
            time: Time = Time.now(),
            frequency: u.Quantity = 50*u.MHz,
            observer: EarthLocation = nenufar_position
        ):
        super().__init__(
            resolution=resolution,
            time=time,
            frequency=frequency,
            observer=observer
        )

        self.value = self._generate_gsm_map()


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def shaped_like(cls, other):
        """ """
        if not isinstance(other, HpxSky):
            raise TypeError(
                f"{HpxSky.__class__} instance expected."
            )
        return cls(
            resolution=other.resolution,
            time=other.time,
            frequency=other.frequency,
            observer=other.observer
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _generate_gsm_map(self) -> da.Array:
        """ """

        # Generate the GSM map at the given frequency
        gsm = GlobalSkyModel(freq_unit="MHz")
        gsm_map = gsm.generate(self.frequency)

        # Resize the GSM HEALPix map to the required dimensions
        gsm_map_nside = hp.pixelfunc.npix2nside(gsm_map.shape[-1])
        if gsm_map_nside != self.nside:
            gsm_map = hp.pixelfunc.ud_grade(
                map_in=gsm_map,
                nside_out=self.nside
            )
        
        
        # gsm_map = gal_to_eq.rotate_map_alms(
        #     gsm_map
        # )

        # Add frequency if size=1
        if self.frequency.size == 1:
            gsm_map = np.expand_dims(gsm_map, axis=0)

        # Convert the map, currently in Galactic coordinates to equatorial
        gal_to_eq = hp.rotator.Rotator(
            deg=True,
            rot=[0, 0],
            coord=['G', 'C']
        )
        for i in range(self.frequency.size):
            with HiddenPrints():
                gsm_map[i, :] = gal_to_eq.rotate_map_pixel(
                    gsm_map[i, :]
                )
    
        # Transform into dask array
        gsm_map = da.from_array(gsm_map)

        # Add time/polarization dimensions
        gsm_map = np.tile(gsm_map, (self.time.size, 1, 1, 1))
        gsm_map = np.moveaxis(gsm_map, source=2, destination=1)

        return gsm_map
# ============================================================= #
# ============================================================= #
