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
import matplotlib.ticker as mtick

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
        - f (float): frequency in MHz
        - p (string): 'NE' or 'NW'
        - a (float): pointed azimuth in degrees
        - e (float): pointed elevation in degrees
        - r (float): mini-array rotation in degrees
    """
    def __init__(self, f=50, p='NW', a=180., e=90., r=None):
        self.freq  = f
        self.polar = p
        self.az    = a
        self.el    = e
        self.rot   = r

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self, **kwargs):
        """ Get the SST beam
        """
        model = AntennaModel(design='nenufar', freq=self.freq, polar=self.polar)
        beam  = PhasedArrayBeam(p=self._antPos(), m=model, a=self.az, e=self.el)
        self.sstbeam = beam.getBeam()
        return

    def plotBeam(self, **kwargs):
        """ Plot the SST Beam
        """
        self.getBeam(**kwargs)

        theta = np.linspace(0., 90., self.sstbeam.shape[1])
        phi   = np.radians( np.linspace(0., 360., self.sstbeam.shape[0]) )
        # ------ Plot ------ #
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='polar')
        normcb = mpl.colors.LogNorm(vmin=self.sstbeam.max() * 1.e-4, vmax=self.sstbeam.max())
        p = ax.pcolormesh(phi, theta, self.sstbeam.T, norm=normcb, **kwargs)
        ax.grid(linestyle='-', linewidth=0.5, color='white', alpha=0.4)
        plt.setp(ax.get_yticklabels(), rotation='horizontal', color='white')
        g = lambda x,y: r'%d'%(90-x)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter( g ))
        plt.show()
        plt.close('all')
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _antPos(self):
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

        if self.rot is not None:
            rot = np.radians( self.rot )
            rotation = np.array([[ np.cos(rot),  np.sin(rot), 0],
                                 [-np.sin(rot),  np.cos(rot), 0],
                                 [ 0,            0,           1]])
            antpos = np.dot( antpos, rotation )
        return antpos



