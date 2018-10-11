#! /usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
Class to handle pygsm sky models
        by A. Loh
"""

import os
import sys
import numpy as np

import healpy as hp
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage

try:
    from pygsm import GlobalSkyModel
except:
    print("\n\t=== WARNING: PyGSM is not installed ===")
    # raise ImportError("\n\t=== Impossible to import pygsm ===")    

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['SkyModel']

class SkyModel():
    def __init__(self):
        self.nra  = 1024
        self.ndec = 512
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def pointSource(self, ra, dec, intensity=1.e6, offset=0.1, sigma=1):
        """ Generate a fake sky model in equatorial coordinates
            Empty sky with a single bright pixel

            Parameters
            ----------
            ra : float
                Right-Ascension of the point source in degrees 
            dec : float
                Declination of the point source in degrees
            intensity : float, optional
                Value of the point-source (default=1.e6)
            offset : float, optional
                Value of sky background (default=0.1)
            sigma : float, optional
                sigma value in pixels for Gaussian smoothing of the point source (default=0)

            Returns
            -------
            self.skymodel : np.ndarray
                Sky model
        """
        self.skymodel = np.zeros( (self.ndec, self.nra) ) + offset
        decs = np.linspace(-90., 90.,   self.ndec) 
        ras  = np.linspace(180., -180., self.nra)
        if ra > 180.:
            ra -= 360. 
        decind  = min(range(decs.size), key=lambda i: np.abs(decs[i] - dec))
        raind   = min(range(ras.size),  key=lambda i: np.abs(ras[i]  - ra))
        self.skymodel[decind, raind] = intensity
        if sigma != 0.:
            self.skymodel = ndimage.gaussian_filter(self.skymodel, sigma=(sigma, sigma), order=0, mode='wrap')
        return

    def gsm2008(self, freq):
        """ Generate a PyGSM 2008 skymodel in equatorial coordinates

            Parameters
            ----------
            freq : float
                Frequency in MHz

            Returns
            -------
            self.skymodel : np.ndarray
                Sky model
        """
        gsm     = GlobalSkyModel(freq_unit='MHz')
        gsmmap  = gsm.generate(freq)
        gsmcart = hp.cartview(gsmmap, coord=['G', 'C'], xsize=self.nra,
            ysize=self.ndec, return_projected_map=True)
        plt.close('all')
        self.skymodel = gsmcart
        return








