#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to compute a SST NenuFAR beam
        by A. Loh
"""

import os
import sys
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.time import Time

from . import PhasedArrayBeam
from .antenna import AntennaModel

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['SSTbeam']

class SSTbeam():
    """ Class to handle SST beams
        Parameters:
        freq (float)
        polar (string)
        az (float)
        el (float)
    """
    def __init__(self, obs=None, freq=50, polar='NW', az=180., el=90.):
        self.freq  = freq
        self.polar = polar
        self.az    = az
        self.el    = el

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self):
        """ Get the SST beam
        """
        model = AntennaModel(design='nenufar', freq=self.freq, polar=self.polar)
        beam  = PhasedArrayBeam(position=self._antPos(), model=model, azimuth=self.az, elevation=self.el)
        self.sstbeam = beam.getBeam()
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _antPos(self, rot=None):
        """ Return the antenna position within a mini-array
        """
        antpos = np.array([
            -5.50000000e+00, -9.52627850e+00,  0.00000000e+00,
             0.00000000e+00, -9.52627850e+00,  0.00000000e+00,
             5.50000000e+00, -9.52627850e+00,  0.00000000e+00,
            -8.25000000e+00, -4.76313877e+00,  0.00000000e+00,
            -2.75000000e+00, -4.76313877e+00,  0.00000000e+00,
             2.75000000e+00, -4.76313877e+00,  0.00000000e+00,
             8.25000000e+00, -4.76313877e+00,  0.00000000e+00,
            -1.10000000e+01,  9.53674316e-07,  0.00000000e+00,
            -5.50000000e+00,  9.53674316e-07,  0.00000000e+00,
             0.00000000e+00,  9.53674316e-07,  0.00000000e+00,
             5.50000000e+00,  9.53674316e-07,  0.00000000e+00,
             1.10000000e+01,  9.53674316e-07,  0.00000000e+00,
            -8.25000000e+00,  4.76314068e+00,  0.00000000e+00,
            -2.75000000e+00,  4.76314068e+00,  0.00000000e+00,
             2.75000000e+00,  4.76314068e+00,  0.00000000e+00,
             8.25000000e+00,  4.76314068e+00,  0.00000000e+00,
            -5.50000000e+00,  9.52628040e+00,  0.00000000e+00,
             0.00000000e+00,  9.52628040e+00,  0.00000000e+00,
             5.50000000e+00,  9.52628040e+00,  0.00000000e+00
            ]).reshape(19, 3)

        if rot is not None:
            rot = np.radians( rot )
            rotation = np.array([[np.cos(rot), -np.sin(rot), 0],
                                 [np.sin(rot),  np.cos(rot), 0],
                                 [0,            0,           1]])
            antpos = np.dot( antpos, rotation )
        return antpos



