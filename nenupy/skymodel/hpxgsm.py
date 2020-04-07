#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***********
    HEALPix GSM
    ***********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HpxGSM'
]


from pygsm import GlobalSkyModel
from healpy.rotator import Rotator
from healpy.pixelfunc import ud_grade

from nenupy.astro import HpxSky
from nenupy.instru import HiddenPrints


# ============================================================= #
# -------------------------- HpxGSM --------------------------- #
# ============================================================= #
class HpxGSM(HpxSky):
    """
    """

    def __init__(self, freq=50, resolution=1):
        super().__init__(
            resolution=resolution
        )
        self.freq = freq 
        
        gsm_map = self._load_gsm(self.freq)
        gsm_map = self._resize(gsm_map, self.nside)
        gsm_map = self._to_celestial(gsm_map)
        self.skymap = gsm_map


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _load_gsm(freq):
        """
        """
        gsm = GlobalSkyModel(freq_unit='MHz')
        gsmap = gsm.generate(freq)
        return gsmap


    @staticmethod
    def _to_celestial(sky):
        """ Convert the GSM from naitve Galactic to Equatorial
            coordinates.
        """
        rot = Rotator(
            deg=True,
            rot=[0, 0],
            coord=['G', 'C']
        )
        with HiddenPrints():
            sky = rot.rotate_map_alms(sky)
        return sky


    @staticmethod
    def _resize(sky, nside):
        """ Resize the GSM to match tjhe desired resolution
        """
        sky = ud_grade(
            map_in=sky,
            nside_out=nside
        )
        return sky
# ============================================================= #

