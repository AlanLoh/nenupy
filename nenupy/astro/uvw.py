#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ************
    UVW Coverage
    ************
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2022, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "compute_uvw"
]


import numpy as np

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time

from nenupy.instru.interferometer import Interferometer
from nenupy.astro import hour_angle, altaz_to_radec
from nenupy import nenufar_position

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------ compute_uvw ------------------------ #
# ============================================================= #
def compute_uvw(
        interferometer: Interferometer,
        phase_center: SkyCoord = None,
        time: Time = Time.now(),
        observer: EarthLocation = nenufar_position
    ):
    """ """

    # Get the baselines in ITRF coordinates
    baselines_itrf = interferometer.baselines.bsl
    xyz = baselines_itrf[np.tril_indices(interferometer.size)].T
    #xyz = np.array(baselines_itrf).T

    # Select zenith phase center if nothing is provided
    if phase_center is None:
        log.debug("Default zenith phase center selected.")
        zenith = SkyCoord(
            np.zeros(time.size),
            np.ones(time.size)*90,
            unit="deg",
            frame=AltAz(
                obstime=time,
                location=observer
            )
        )
        phase_center = altaz_to_radec(zenith)
    center_dec_rad = phase_center.dec.rad
    if np.isscalar(center_dec_rad):
        center_dec_rad = np.repeat(center_dec_rad, time.size)

    # Compute the hour angle of the phase center
    lha = hour_angle(
        radec=phase_center,
        time=time,
        observer=observer,
        fast_compute=True
    )
    lha_rad = lha.rad
    
    # Compute UVW projection
    sr = np.sin(lha_rad)
    cr = np.cos(lha_rad)
    sd = np.sin(center_dec_rad)
    cd = np.cos(center_dec_rad)
    rot_uvw = np.array([
        [    sr,     cr,  np.zeros(time.size)],
        [-sd*cr,  sd*sr,                   cd],
        [ cd*cr, -cd*sr,                   sd]
    ])

    # Project the baselines in the UVW frame
    uvw = - np.dot(np.moveaxis(rot_uvw, -1, 0), xyz)

    return np.moveaxis(uvw, -1, 1)
# ============================================================= #
# ============================================================= #