#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
"""

import os, sys
import numpy as np

import healpy as hp
from astropy.time import Time

from pygsm import GlobalSkyModel

from nenupy.astro import toRadec, getSrc


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['Skymodel']


# ============================================================= #
# ------------------------- Skymodel -------------------------- #
# ============================================================= #
class Skymodel(object):
    """
    """

    def __init__(self, **kwargs):
        self._kwargs(kwargs)

    # ========================================================= #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, t):
        if not isinstance(t, Time):
            if t.lower() == 'now':
                t = Time.now()
            else:
                t = Time(t)
        self._time = t
        return


    # ========================================================= #
    # ------------------------ Methods ------------------------ #
    def get_skymodel(self, time='now', model='gsm', **kwargs):
        """ Get the GMS sky at a given time and frequency
            above NenuFAR
        """
        self.time = time

        if model.lower() == 'gsm': 
            if not hasattr(self, 'map'):
                self._load_gsm_model()
        elif model.lower() == 'pointsource':
            if not hasattr(self, 'map'):
                try:
                    ra = kwargs['ra']
                    dec = kwargs['dec']
                except:
                    raise ValueErrror('ra and dec should be specified.')
                self._load_point_model(ra=ra, dec=dec)
        else:
            raise ValueErrror('Only gsm or pointsource')

        radec = toRadec((0., 0.),
            time=self.time,
            loc='NenuFAR',
            unit='rad')

        rot_dec = radec.dec.rad
        rot_ra = (radec.ra.rad)%(2.*np.pi)

        rot = hp.Rotator(deg=False,
            rot=[rot_ra, rot_dec],
            coord=['G', 'C'])

        sys.stdout = open(os.devnull, 'w')
        skymap = rot.rotate_map_alms(self.map)
        # skymap = rot.rotate_map_pixel(self.map)
        sys.stdout = sys.__stdout__

        return skymap

    # ========================================================= #
    # ----------------------- Internal ------------------------ #
    def _load_gsm_model(self):
        """
        """
        gsm   = GlobalSkyModel(freq_unit='MHz')
        gsmap = gsm.generate(self.freq)
        self.map = hp.pixelfunc.ud_grade(gsmap,
            nside_out=self.nside)
        return
    # --------------------------------------------------------- #
    def _load_point_model(self, ra, dec):
        """ ra, dec in radians
        """
        src = getSrc(source=(ra, dec))
        src_gal = src.galactic
        self.map = np.repeat(1.e-1, hp.nside2npix(self.nside))
        pixel = hp.ang2pix(nside=self.nside,
            theta=src_gal.l.deg,
            phi=src_gal.b.deg,
            lonlat=True,
            nest=False)
        self.map[pixel] = 1.e6
        return
    # --------------------------------------------------------- #
    def _kwargs(self, kwargs):
        """ Evaluate the keyword arguments and fill attributes
        """
        self.nside = kwargs.get('nside', 256)
        self.freq = kwargs.get('freq', 50)
        return
# ============================================================= #
# ============================================================= #
