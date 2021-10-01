#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "Sky",
    "HpxGSM"
]

import logging
log = logging.getLogger(__name__)

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

from nenupy import nenufar_position
from nenupy.astro2.sky import HpxSky


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
    # ----------------------- Internal ------------------------ #
    def _generate_gsm_map(self):
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
        
        # Convert the map, currently in Galactic cooridnates to equatorial
        gal_to_eq = hp.rotator.Rotator(
            deg=True,
            rot=[0, 0],
            coord=['G', 'C']
        )
        gsm_map = gal_to_eq.rotate_map_alms(
            gsm_map
        )

        return gsm_map
# ============================================================= #
# ============================================================= #
