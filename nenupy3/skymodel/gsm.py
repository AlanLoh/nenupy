#! /usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
Class to handle pygsm sky models
        by A. Loh
"""

import os
import sys
import numpy as np

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
__all__ = ['GSM']

class GSM():
    def __init__(self):
        self.nra = 1024
        self.ndec = 512
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def pointSource(self, ra, dec, intensity=1.e6, offset=0.1, sigma=0):
        """ Generate a fake sky model in equatorial coordinates
            Empyt sky with a single bright pixel

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
        decgrid = np.linspace(-90., 90.,   self.ndec) 
        ragrid  = np.linspace(180., -180., self.nra) 
        decind  = min(range(len(decgrid)), key=lambda i: np.abs(decgrid[i]-dec))
        raind   = min(range(len(ragrid)),  key=lambda i: np.abs(ragrid[i]-ra))
        self.skymodel[decind, raind] = intensity
        if sigma != 0.:
            self.skymodel = ndimage.gaussian_filter(self.skymodel, sigma=(sigma, sigma), order=0)
        return







