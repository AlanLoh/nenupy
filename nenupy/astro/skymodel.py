#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Skymodel
    ********
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2025, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "HpxGSM",
    "Skymodel"
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
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import dask.array as da

from nenupy import nenufar_position, HiddenPrints
from nenupy.astro.sky import HpxSky, Sky
from nenupy.astro import altaz_to_radec


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


# ============================================================= #
# ------------------------- Skymodel -------------------------- #
# ============================================================= #
class Skymodel:

    def __init__(self, frequency: u.Quantity = 50 * u.MHz):
        self.frequency = frequency
        self.data = self._load_gsm(frequency=self.frequency)

    def radec_project(self, skycoord: SkyCoord) -> np.ndarray:
        return hp.pixelfunc.get_interp_val(
            m=self.data,
            theta=skycoord.ra.deg,
            phi=skycoord.dec.deg,
            nest=False,
            lonlat=True
        )

    def altaz_map_at(self, time: Time, n_azimuths: int = 500, n_elevations: int = 300, return_coords: bool = False) -> np.ndarray:
        azimuths = np.linspace(0, 360, n_azimuths)
        elevations = np.linspace(0, 90, n_elevations)
        az_grid, alt_grid = np.meshgrid(azimuths, elevations)
        radec = altaz_to_radec(
            SkyCoord(
                az_grid, alt_grid, unit="deg",
                frame=AltAz(
                    obstime=time,
                    location=nenufar_position
                )
            )
        )
        if return_coords:
            return az_grid, alt_grid, radec, self.radec_project(skycoord=radec)
        else:
            return self.radec_project(skycoord=radec)

    def to_sky(self, skycoord: SkyCoord, time: Time) -> Sky:
        return Sky(
            coordinates=skycoord.ravel(),
            time=time,
            frequency=self.frequency,
            value=self.radec_project(skycoord).reshape((time.size, self.frequency.size, 1, skycoord.size))
        )

    def to_hpxsky(self, time: Time) -> HpxSky:
        raise NotImplementedError

    @staticmethod
    def _load_gsm(frequency: u.Quantity) -> np.ndarray:
        gsm = GlobalSkyModel(freq_unit="MHz")
        gsm_map = gsm.generate(frequency.to_value(u.MHz))
        gal_to_eq = hp.rotator.Rotator(
            deg=True,
            rot=[0, 0],
            coord=["G", "C"]
        )
        return np.array(gal_to_eq.rotate_map_pixel(gsm_map))
# ============================================================= #
# ============================================================= #
