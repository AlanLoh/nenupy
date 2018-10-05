#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to compute a BST NenuFAR beam
        by A. Loh
"""

import os
import sys
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from . import SSTbeam
from . import PhasedArrayBeam
from .antenna import AntennaModel
from .antenna import miniarrays
from ..read import BST

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['BSTBeam']

class BSTbeam():
    """ Class to handle BST beams
    """
    def __init__(self, obsfile=None, **kwargs):
        if obsfile is None:
            self._evalkwargs(kwargs)
        else:
            self.obsfile = obsfile

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obsfile(self):
        """ BST observation
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        if o is None:
            return
        self._obsfile   = BST(o)
        self.miniarrays = self._obsfile.miniarrays
        self.rotations  = self._obsfile.marotation
        self.positions  = self._obsfile.maposition
        self.delays     = self._obsfile._delays
        self.freq       = self._obsfile.freq
        self.polar      = self._obsfile.polar
        self.azana      = self._obsfile.azana 
        self.elana      = self._obsfile.elana
        self.azdig      = self._obsfile.azdig
        self.eldig      = self._obsfile.eldig
        return

    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, (int, float)):
            return TypeError("\n\t=== Attribute 'freq' should be a number ===")
        elif (f<=0.) or (f>100.):
            return ValueError("\n\t=== Attribute 'freq' set to a value outside of NenuFAR's frequency range ===")
        else:
            self._freq = f
            return

    @property
    def polar(self):
        """ Polarization selection
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, (str)):
            return TypeError("\n\t=== Attribute 'polar' should be a string ===")
        elif p.upper() not in ['NW', 'NE']:
            return ValueError("\n\t=== Attribute 'polar' does not correspond to either 'NW' or 'NE' ===")
        else:
            self._polar = p.upper()
            return

    @property
    def miniarrays(self):
        """ Miniarrays selection
        """
        return self._miniarrays
    @miniarrays.setter
    def miniarrays(self, m):
        marecorded = miniarrays.ma[:, 0].astype(int)
        if m is None:
            print("\n\t==== WARNING: miniarrays are set by default ===")
            m = marecorded
        if isinstance(m, list):
            m = np.array(m)
        if not isinstance(m, np.ndarray):
            raise TypeError("\n\t=== Attribute 'miniarrays' should be an array ===")
        elif not all([ mi in marecorded for mi in m]):
            raise ValueError("\n\t=== Attribute 'miniarrays' contains miniarray indices not matching existing MAs (up to {}) ===".format(marecorded.max()))
        else:
            self._miniarrays = m
            return

    @property
    def rotations(self):
        """ MA rotation selection
        """
        return self._rotations
    @rotations.setter
    def rotations(self, r):
        if r is None:
            print("\n\t==== WARNING: MA rotations are set by default ===")
            r = miniarrays.ma[:, 1][self.miniarrays]
        if isinstance(r, list):
            r = np.array(r)
        if not isinstance(r, np.ndarray):
            raise TypeError("\n\t=== Attribute 'rotations' should be an array ===")
        elif r.size != self.miniarrays.size:
            raise ValueError("\n\t=== Attribute 'rotations' should be the same size as 'miniarrays' ===")
        else:
            self._rotations = r
            return

    @property
    def positions(self):
        """ MA position selection
        """
        return self._positions
    @positions.setter
    def positions(self, p):
        if p is None:
            print("\n\t==== WARNING: MA positions are set by default ===")
            p = miniarrays.ma[:, 2:5][self.miniarrays]
        if isinstance(p, list):
            p = np.array(p)
        if not isinstance(p, np.ndarray):
            raise TypeError("\n\t=== Attribute 'positions' should be an array ===")
        elif p.shape[0] != self.miniarrays.size:
            raise ValueError("\n\t=== Attribute 'positions' should be the same size as 'miniarrays' ===")
        else:
            self._positions = p
            return

    @property
    def delays(self):
        """ MA delay selection (in ns)
        """
        return self._delays
    @delays.setter
    def delays(self, d):
        if d is None:
            print("\n\t==== WARNING: MA delays are set by default ===")
            d = miniarrays.ma[:, 6][self.miniarrays]
        if isinstance(d, list):
            d = np.array(d)
        if not isinstance(d, np.ndarray):
            raise TypeError("\n\t=== Attribute 'delays' should be an array ===")
        elif d.size != self.miniarrays.size:
            raise ValueError("\n\t=== Attribute 'delays' should be the same size as 'miniarrays' ===")
        else:
            self._delays = d
            return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self, **kwargs):
        """ Get the BST beam
        """
        ma_beams = []
        for i in range(self.miniarrays.size):
            mabeam = SSTbeam(miniarray=self.miniarrays[i], freq=self.freq, polar=self.polar,
                azana=self.azana, elana=self.elana, rotation=self.rotations[i])
            mabeam.getBeam()
            ma_model = AntennaModel( design=mabeam.sstbeam, freq=self.freq)
            ma_beams.append( ma_model )
        
        beam = PhasedArrayBeam(p=self.positions, m=ma_beams, a=self.azdig, e=self.eldig)
        self.bstbeam = beam.getBeam()

    def plotBeam(self, **kwargs):
        """ Plot the BST Beam
        """
        self.getBeam(**kwargs)

        theta = np.linspace(0., 90., self.bstbeam.shape[1])
        phi   = np.radians( np.linspace(0., 360., self.bstbeam.shape[0]) )
        # ------ Plot ------ #
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='polar')
        normcb = mpl.colors.LogNorm(vmin=self.bstbeam.max() * 1.e-4, vmax=self.bstbeam.max())
        p = ax.pcolormesh(phi, theta, self.bstbeam.T, norm=normcb, **kwargs)
        ax.grid(linestyle='-', linewidth=0.5, color='white', alpha=0.4)
        plt.setp(ax.get_yticklabels(), rotation='horizontal', color='white')
        g = lambda x,y: r'%d'%(90-x)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter( g ))
        plt.show()
        plt.close('all')
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _evalkwargs(self, kwargs):
        """
        Parameters
        ----------
        kwargs : dictionnary
            Dictionnary of keys and values to look from in order to fill the class attributes
        """
        allowed = ['miniarrays', 'rotations', 'positions', 'delays', 'freq', 'polar', 'azana', 'elana', 'azdig', 'eldig']
        if not all([ki in allowed for ki in kwargs.keys()]):
            print("\n\t=== WARNING: unkwnown keywords, authorized: {} ===".format(allowed))
        self.miniarrays = kwargs.get('miniarrays', None)
        self.rotations  = kwargs.get('rotations', None)
        self.positions  = kwargs.get('positions', None)
        self.delays     = kwargs.get('delays', None)
        self.freq       = kwargs.get('freq', 50)
        self.polar      = kwargs.get('polar', 'NW')
        self.azana      = kwargs.get('azana', 180.)
        self.elana      = kwargs.get('elana', 90.)
        self.azdig      = kwargs.get('azdig', 180.)
        self.eldig      = kwargs.get('eldig', 90.)
        return




