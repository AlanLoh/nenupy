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
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
from astropy.io import fits

from . import PhasedArrayBeam
from .antenna import AntennaModel
from .antenna import miniarrays
from ..read import SST

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
    def __init__(self, obsfile=None, **kwargs):
        if obsfile is None:
            self._evalkwargs(kwargs)
        else:
            self.obsfile = obsfile

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obsfile(self):
        """ SST observation
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        if o is None:
            return
        self._obsfile  = SST(o)
        self.ma        = self._obsfile.ma
        self.rotation  = self._obsfile.marotation
        self.freq      = self._obsfile.freq
        self.polar     = self._obsfile.polar
        self.azana     = self._obsfile.azana 
        self.elana     = self._obsfile.elana
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
    def ma(self):
        """ Miniarray selection
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        marecorded = miniarrays.ma[:, 0].astype(int)
        if m is None:
            print("\n\t==== WARNING: ma is set by default ===")
            m = marecorded[0]
        if not isinstance(m, (int, np.int32, np.int64) ):
            raise TypeError("\n\t=== Attribute 'ma' should be an integer ===")
        elif not m in marecorded:
            raise ValueError("\n\t=== Attribute 'ma' contains miniarray index not matching existing MAs (up to {}) ===".format(marecorded.max()))
        else:
            self._ma = m
            return

    @property
    def rotation(self):
        """ MA rotation selection
        """
        return self._rotation
    @rotation.setter
    def rotation(self, r):
        if r is None:
            print("\n\t==== WARNING: MA rotation is set by default ===")
            r = miniarrays.ma[:, 1][self.ma]
        if isinstance(r, list):
            r = np.array(r)
        if not isinstance(r, (float, int, np.float32, np.float64, np.int32, np.int64)):
            raise TypeError("\n\t=== Attribute 'rotation' should be a number ===")
        else:
            self._rotation = r
            return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self):
        """ Get the SST beam
        """
        model = AntennaModel(design='nenufar', freq=self.freq, polar=self.polar)
        elevation = self._squintMA(self.elana)
        az, el = self._realPointing(self.azana, elevation)
        beam  = PhasedArrayBeam(p=self._antPos(), m=model, a=az, e=el)
        self.sstbeam = beam.getBeam()
        return

    def plotBeam(self, **kwargs):
        """ Plot the SST Beam
        """
        self.getBeam()

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
        antpos = miniarrays.antpos

        if self.rotation is not None:
            rot = np.radians( self.rotation )
            rotation = np.array([[ np.cos(rot),  np.sin(rot), 0],
                                 [-np.sin(rot),  np.cos(rot), 0],
                                 [ 0,            0,           1]])
            antpos = np.dot( antpos, rotation )
        return antpos

    def _realPointing(self, a, e):
        """ Find the real pointing direction from self.ra, self.dec
            Nenufar can only point towards directions on a 128 x 128 cells grid
            Therefore PZ computed for each -desired- direction at a resolution of 0.05deg
            the corresponding real observed direction.
        """ 
        modulepath = os.path.dirname( os.path.realpath(__file__) )
        thph = fits.getdata( os.path.join(modulepath, 'NenuFAR_thph.fits') )
        theta, phi = ( int((90.-e)/0.05 - 0.5), int(a/0.05 - 0.5)  )
        t, p = thph[:, theta, phi]
        return p, 90.-t

    def _squintMA(self, e):
        """ Compute the elevation to be pointed that take into account the squint if we want to observe ele
        """
        modulepath = os.path.dirname( os.path.realpath(__file__) )
        squint  = readsav( os.path.join(modulepath, 'squint_table.sav') )
        optfreq = 30
        indfreq = np.where(squint['freq']==optfreq)[0][0]
        newele  = interp1d(squint['elev_desiree'][indfreq,:], squint['elev_a_pointer'])(e)
        if newele < 20.: # squint is limited at 20 deg elevation
            newele = 20.
        return newele

    def _evalkwargs(self, kwargs):
        """
        Parameters
        ----------
        kwargs : dictionnary
            Dictionnary of keys and values to look from in order to fill the class attributes
        """
        allowed = ['ma', 'rotation', 'freq', 'polar', 'azana', 'elana']
        if not all([ki in allowed for ki in kwargs.keys()]):
            print("\n\t=== WARNING: unkwnown keywords, authorized: {} ===".format(allowed))
        self.ma        = kwargs.get('ma', None)
        self.rotation  = kwargs.get('rotation', None)
        self.freq      = kwargs.get('freq', 50)
        self.polar     = kwargs.get('polar', 'NW')
        self.azana     = kwargs.get('azana', 180.)
        self.elana     = kwargs.get('elana', 90.)
        return



