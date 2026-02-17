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
__all__ = [
    "NDA"
]


import logging
log = logging.getLogger(__name__)

from nenupy.instru.interferometer import Interferometer
from nenupy.astro.sky import Sky
from nenupy.astro.pointing import Pointing
from nenupy import nda_position

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, Latitude, Longitude
import dask.array as da
from enum import Enum
import os
import json
import healpy as hp

with open(os.path.join(os.path.dirname(__file__), "nda_array.json")) as nda:
    NDA_SUBBARRAYS = json.load(nda)

NDA_ANTENNAS_RELATIVE_POSITIONS = np.array([
    [0.00000, 0.00000, 0.00000],
    [0.00000, 7.50000, 0.00000],
    [0.00000, 15.0000, 0.00000],
    [0.00000, 22.5000, 0.00000],
    [6.50000, 0.00000, 0.00000],
    [6.50000, 7.50000, 0.00000],
    [6.50000, 15.0000, 0.00000],
    [6.50000, 22.5000, 0.00000]
]) * u.m


# ============================================================= #
# ---------------- Polarization / Antenna Gain ---------------- #
# ============================================================= #
class _NDAAntennaGain:
    """ """

    def __init__(self, polarization: str = "RH"):
        self.polarization = polarization
        
        nside = 10
        az, alt = hp.pix2ang(
            nside=nside,
            ipix=np.arange(
                hp.nside2npix(nside),
                dtype=np.int64
            ),
            lonlat=True,
            nest=False
        )
        self.values = np.sin(np.radians(alt))**2

        # Get theta, phi for non-rotated map
        t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi
        # Define a rotator
        r = hp.Rotator(deg=False, rot=[0, np.radians(20)])
        # Get theta, phi under rotated co-ordinates
        trot, prot = r(t,p)
        # Interpolate map onto these co-ordinates
        rot_map = hp.get_interp_val(self.values, trot, prot)
        self.values = rot_map / rot_map.max()

    def __getitem__(self, sky: Sky) -> np.ndarray:
        
        horizontal_coordinates = sky.horizontal_coordinates

        gain = hp.pixelfunc.get_interp_val(
            m=self.values,
            theta=horizontal_coordinates.az.deg,
            phi=horizontal_coordinates.alt.deg,
            nest=False,
            lonlat=True
        )

        return gain


class NDAPolarization(Enum):
    """ """

    LH = _NDAAntennaGain("LH")
    RH = _NDAAntennaGain("RH")
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------------- MiniNDA -------------------------- #
# ============================================================= #
class NDA(Interferometer):

    def __init__(self, index: int = 0, polar: str = "R"):
        self.index = index
        self.polar = polar
        self.block_id = f"{self.polar}{self.index}"
        
        # Position of the central block
        # position_json = NDA_SUBBARRAYS[self.block_id]
        # position = EarthLocation(
        #     lat=Latitude(position_json["lat"], unit=u.deg).deg,
        #     lon=Longitude(position_json["lon"], unit=u.deg).deg,
        #     height=position_json["height"] * u.m
        # )

        antenna_positions = np.loadtxt("/Users/aloh/Documents/Work/NDA/Antenna_positions/adam.txt")

        # antenna_names = np.array([f"{self.block_id}_{i}" for i in range(NDA_ANTENNAS_RELATIVE_POSITIONS.shape[0])])
        antenna_names = np.array([f"{self.block_id}_{i}" for i in range(antenna_positions.shape[0])])

        antenna_gains = np.array([
            self._antenna_gain for _ in range(antenna_names.size)
        ])

        super().__init__(
            position=nda_position,
            antenna_names=antenna_names,
            antenna_positions=antenna_positions,#NDA_ANTENNAS_RELATIVE_POSITIONS,
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

    # @lru_cache(maxsize=1)
    def _antenna_gain(self, sky: Sky, pointing: Pointing):
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


# # ============================================================= #
# # ---------------------------- NDA ---------------------------- #
# # ============================================================= #
# class NDA(Interferometer):

#     def __init__(self, array_selected: str = "R"):
#         position = nda_position
#         antenna_names = np.arange(72)
#         antenna_positions = np.loadtxt("/Users/aloh/Downloads/adam.txt")
#         antenna_gains = np.array([
#             self._antenna_gain for _ in range(antenna_names.size)
#         ])

#         super().__init__(
#             position=position,
#             antenna_names=antenna_names,
#             antenna_positions=antenna_positions,
#             antenna_gains=antenna_gains
#         )


#     def effective_area(self,
#             frequency: u.Quantity = 50*u.MHz,
#             elevation: u.Quantity = 90*u.deg
#         ) -> u.Quantity:
#         """ """
#         raise NotImplementedError()

    
#     def instrument_temperature(self,
#             frequency: u.Quantity = 50*u.MHz,
#             lna_filter: int = 0
#         ) -> u.Quantity:
#         """ """
#         raise NotImplementedError()


#     # --------------------------------------------------------- #
#     # ----------------------- Internal ------------------------ #
#     # @lru_cache(maxsize=1)
#     def _antenna_gain(self, sky: Sky, pointing: Pointing) -> np.ndarray:
#         """
#         """
#         gain = da.ones(
#             (
#                 sky.time.size,
#                 sky.frequency.size,
#                 sky.polarization.size,
#                 sky.coordinates.size
#             ),
#             dtype=np.float64
#         )
#         for i, pol in enumerate(sky.polarization):
#             if not isinstance(pol, NDAPolarization):
#                 log.warning(
#                     f"Invalid value encountered in <attr 'Sky.polarization'>: '{pol}'. "
#                     f"Polarization has been set to '{NDAPolarization.RH}' by default."
#                 )
#                 pol = NDAPolarization.RH
#             gain[:, :, i, :] = pol.value[sky]
#         return gain

# # ============================================================= #
# # ============================================================= #

