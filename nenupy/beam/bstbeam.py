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
from astropy.io import fits

from . import SSTbeam
from . import PhasedArrayBeam
from .antenna import AntennaModel
from .antenna import miniarrays
from ..read import BST
from ..utils import ProgressBar

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['BSTBeam']

class BSTbeam(object):
    """ Class to handle BST beams
    """
    def __init__(self, bst=None, **kwargs):
        self.bst = bst
        self._evalkwargs(kwargs)

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def bst(self):
        """ BST observation
        """
        return self._bst
    @bst.setter
    def bst(self, b):
        if isinstance(b, BST):
            self._bst       = b
            self.ma         = self._bst.ma
            self.marotation = self._bst.marotation
            self.maposition = self._bst.maposition
            self.delays     = self._bst.delays
            self.freq       = self._bst.freq
            self.polar      = self._bst.polar
            self.azana      = self._bst.azana 
            self.elana      = self._bst.elana
            self.azdig      = self._bst.azdig 
            self.eldig      = self._bst.eldig
        else:
            self._bst = b
        return

    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, (int, float, np.float32, np.float64)):
            raise TypeError("\n\t=== Attribute 'freq' should be a number ===")
        elif (f<=0.) or (f>100.):
            raise ValueError("\n\t=== Attribute 'freq' set to a value outside of NenuFAR's frequency range ===")
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
            raise TypeError("\n\t=== Attribute 'polar' should be a string ===")
        elif p.upper() not in ['NW', 'NE']:
            raise ValueError("\n\t=== Attribute 'polar' does not correspond to either 'NW' or 'NE' ===")
        else:
            self._polar = p.upper()
            return

    @property
    def ma(self):
        """ Miniarrays selection
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        if self.bst is None:
            marecorded = miniarrays.ma[:, 0].astype(int)
        else:
            marecorded = self.bst.ma
        if m is None:
            print("\n\t==== WARNING: miniarrays are set by default ===")
            m = marecorded
        if isinstance(m, list):
            m = np.array(m)
        if not isinstance(m, np.ndarray):
            raise TypeError("\n\t=== Attribute 'ma' should be an array ===")
        elif not all([ mi in marecorded for mi in m]):
            raise ValueError("\n\t=== Attribute 'ma' contains miniarray indices not matching existing MAs (up to {}) ===".format(marecorded.max()))
        else:
            self._ma = m
            return

    @property
    def marotation(self):
        """ MA rotation selection
        """
        if self.bst is None:
            return self._marotation[self.ma]
        else:
            return self._marotation
    @marotation.setter
    def marotation(self, r):
        if r is None:
            print("\n\t==== WARNING: MA rotations are set by default ===")
            self._marotation = miniarrays.ma[:, 1]
        if isinstance(r, list) or (isinstance(r, np.ndarray)):
            self._marotation = np.array(r)
            if self._marotation.size != self.ma.size:
                raise ValueError("\n\t=== Attribute 'marotation' should be the same size as 'ma' ===")
        return

    @property
    def maposition(self):
        """ MA position selection
        """
        if self.bst is None:
            return self._maposition[self.ma]
        else:
            return self._maposition
    @maposition.setter
    def maposition(self, p):
        if p is None:
            print("\n\t==== WARNING: MA positions are set by default ===")
            self._maposition = miniarrays.ma[:, 2:5]
        if isinstance(p, list) or (isinstance(p, np.ndarray)):
            self._maposition = np.array(p)
            if self._maposition.shape[0] != self.ma.size:
                raise ValueError("\n\t=== Attribute 'maposition' should be the same size as 'ma' ===")
        return

    @property
    def delays(self):
        """ MA delay selection (in ns)
        """
        if self.bst is None:
            return self._delays[self.ma]
        else:
            return self._delays
    @delays.setter
    def delays(self, d):
        if d is None:
            print("\n\t==== WARNING: MA delays are set by default ===")
            self._delays = miniarrays.ma[:, 5]
        if isinstance(d, list) or isinstance(d, np.ndarray):
            self._delays = np.array(d)
            if self._delays.size != self.ma.size:
                raise ValueError("\n\t=== Attribute 'delays' should be the same size as 'ma' ===")
        return

    @property
    def azana(self):
        """ Azimuth during the analogic pointing self.abeam
        """
        return self._azana
    @azana.setter
    def azana(self, a):
        if isinstance(a, (list, np.ndarray)):
            a = a[0]
        self._azana = a
        return

    @property
    def elana(self):
        """ Elevation during the analogic pointing self.abeam
        """
        return self._elana
    @elana.setter
    def elana(self, e):
        if isinstance(e, (list, np.ndarray)):
            e = e[0]
        self._elana = e
        return

    @property
    def azdig(self):
        """ Azimuth during the numeric pointing self.dbeam
        """
        return self._azdig
    @azdig.setter
    def azdig(self, a):
        if isinstance(a, (list, np.ndarray)):
            a = a[0]
        self._azdig = a
        return

    @property
    def eldig(self):
        """ Elevation during the numeric pointing self.dbeam
        """
        return self._eldig
    @eldig.setter
    def eldig(self, e):
        if isinstance(e, (list, np.ndarray)):
            e = e[0]
        self._eldig = e
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self, **kwargs):
        """ Get the BST beam
        """
        ma_beams = []
        bar = ProgressBar(valmax=self.ma.size, title='Computing BST beam')
        for i in range(self.ma.size):
            mabeam = SSTbeam(ma=self.ma[i], freq=self.freq, polar=self.polar,
                azana=self.azana, elana=self.elana, marotation=self.marotation[i])
            mabeam.getBeam()
            ma_model = AntennaModel( design=mabeam.beam, freq=self.freq)
            ma_beams.append( ma_model )
            bar.update()
        beam = PhasedArrayBeam(p=self.maposition, m=ma_beams, a=self.azdig, e=self.eldig)
        self.beam = beam.getBeam()

    def plotBeam(self, **kwargs):
        """ Plot the BST Beam
        """
        if not hasattr(self, 'beam'):
            self.getBeam(**kwargs)

        theta = np.linspace(0., 90., self.beam.shape[1])
        phi   = np.radians( np.linspace(0., 360., self.beam.shape[0]) )
        # ------ Plot ------ #
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='polar')
        normcb = mpl.colors.LogNorm(vmin=self.beam.max() * 1.e-4, vmax=self.beam.max())
        p = ax.pcolormesh(phi, theta, self.beam.T, norm=normcb, **kwargs)
        ax.grid(linestyle='-', linewidth=0.5, color='white', alpha=0.4)
        plt.setp(ax.get_yticklabels(), rotation='horizontal', color='white')
        g = lambda x,y: r'%d'%(90-x)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter( g ))
        plt.show()
        plt.close('all')
        return

    def saveBeam(self, savefile=None):
        """ Save the beam
        """
        if savefile is None:
            savefile = 'beam.fits'
        else:
            if not savefile.endswith('.fits'):
                raise ValueError("\n\t=== It should be a FITS ===")
        if not hasattr(self, 'beam'):
            self.getBeam()

        prihdr = fits.Header()
        prihdr.set('FREQ', str(self.freq))
        prihdr.set('POLAR', self.polar)
        # prihdr.set('MINI-ARR', str(self.ma))
        # prihdr.set('MA-ROT', str(self.rotation))
        datahdu = fits.PrimaryHDU( np.fliplr(self.beam).T, header=prihdr)
        hdulist = fits.HDUList([datahdu])
        hdulist.writeto(savefile, overwrite=True)
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
        allowed = ['ma', 'marotation', 'maposition', 'delays', 'freq', 'polar', 'azana', 'elana', 'azdig', 'eldig']
        if not all([ki in allowed for ki in kwargs.keys()]):
            print("\n\t=== WARNING: unkwnown keywords, authorized: {} ===".format(allowed))
        if not hasattr(self, 'ma'):
            self.ma         = kwargs.get('ma', None)
        if not hasattr(self, 'marotation'):
            self.marotation = kwargs.get('marotation', None)
        if not hasattr(self, 'maposition'):
            self.maposition = kwargs.get('maposition', None)
        if not hasattr(self, 'delays'):
            self.delays     = kwargs.get('delays', None)
        if not hasattr(self, 'freq'):
            self.freq       = kwargs.get('freq', 50)
        if not hasattr(self, 'polar'):
            self.polar      = kwargs.get('polar', 'NW')
        if not hasattr(self, 'azana'):
            self.azana      = kwargs.get('azana', 180.)
        if not hasattr(self, 'elana'):
            self.elana      = kwargs.get('elana', 90.)
        if not hasattr(self, 'azdig'):
            self.azdig      = kwargs.get('azdig', 180.)
        if not hasattr(self, 'eldig'):
            self.eldig      = kwargs.get('eldig', 90.)
        return




