#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to compute a phased-array beam
        by A. Loh
"""

import os
import sys
import numpy as np

from astropy import constants as const

from .antenna import AntennaModel

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['PhasedArrayBeam']

class PhasedArrayBeam():
    """ Class PhasedArrayBeam
        Parameters:
        - p (ndarray): antenna positions ([[x1, y1, z1], [x2, y2, z2], ..])
        - m (ndarray): antenna model
        - a (float): azimuth of the target location
        - e (float): elevation of the target location
    """
    def __init__(self, p=None, m=None, a=180, e=45):
        self.pos   = p
        self.model = m
        self.azim  = a
        self.elev  = e
        self.resol = 0.2 

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def azim(self):
        """ Azimuth in degrees
        """
        return self._azim
    @azim.setter
    def azim(self, a):
        if not isinstance(a, (float, int, np.float32, np.float64)):
            raise TypeError("\n\t=== Azimuth must be a number ===")
        elif (a < 0.) or (a > 360.):
            a %= 360.
        else:
            pass
        self._azim = a
        return

    @property
    def elev(self):
        """ Elevation in degrees
        """
        return self._elev
    @elev.setter
    def elev(self, e):
        if not isinstance(e, (float, int, np.float32, np.float64)):
            raise TypeError("\n\t=== Elevation must be a number ===")
        elif (e < 0.) or (e > 90.):
            raise ValueError("\n\t=== Elevation should be between 0 and 90 ===")
        else:
            pass
        self._elev = e
        return

    @property
    def pos(self):
        """ Antenna position array
        """
        return self._pos
    @pos.setter
    def pos(self, p):
        if p is None:
            return
        if not isinstance(p, (np.ndarray)):
            raise TypeError("\n\t=== Antenna position should be an array ===")
        elif len(p.shape) != 2:
            raise ValueError("\n\t=== Antenna position should be a 2D array ===")
        elif p.shape[1] != 3:
            raise ValueError("\n\t=== Antenna position array should be a (x, 3) shaped array ===")
        else:
            self._pos = p
        return

    @property
    def model(self):
        """ Antenna model
        """
        return self._model
    @model.setter
    def model(self, m):
        if m is None:
            return
        if not isinstance(m, (AntennaModel, list)):
             raise ValueError("\n\t=== model should be a AntennaModel object / list ===")
        else:
            self._model = m
            return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self):
        """ Compute the beam of the phased array
            Parameter:
            power (bool): return the power (False: return eiphi)
        """
        if isinstance(self.model, list):
            wavevec = 2 * np.pi * self.model[0].antenna_freq * 1.e6 / const.c.value
        else:
            wavevec = 2 * np.pi * self.model.antenna_freq * 1.e6 / const.c.value

        # ------ Sky grid ------ #
        thetagrid, phigrid = np.radians(np.meshgrid(np.arange(0, 90, self.resol),
            np.arange(0, 360, self.resol) ))
        thetagrid = np.fliplr(thetagrid)
        _xp = np.cos(thetagrid) * np.cos(phigrid)
        _yp = np.cos(thetagrid) * np.sin(phigrid)
        _zp = np.sin(thetagrid)

        # ------ Phase delay between antennae ------ #
        theta = np.repeat( np.radians( self.elev ), self.pos.shape[0])
        # theta = np.repeat( np.radians( 90. - self.elev ), self.pos.shape[0])
        phi   = np.repeat( np.radians( self.azim ), self.pos.shape[0])
        ux = np.cos(phi) * np.cos(theta)
        uy = np.sin(phi) * np.cos(theta)
        uz = np.sin(theta)
        dphix = wavevec * self.pos[:, 0] * ux
        dphiy = wavevec * self.pos[:, 1] * uy
        dphi  = dphix + dphiy

        # ------ Phase reference ------ #
        phi0  = wavevec * ( self.pos[:, 0] * _xp[:, :, np.newaxis] 
            + self.pos[:, 1] * _yp[:, :, np.newaxis] 
            + self.pos[:, 2] * _zp[:, :, np.newaxis] ).astype(np.float32)
        phase = phi0[:, :, :] - dphi[np.newaxis, np.newaxis, :]

        # ------ e^(i Phi) ------ #
        eiphi = np.sum( np.exp(1j * phase), axis=2 )
        eiphi = np.flipud(eiphi.T)
        if isinstance(self.model, list):
            antgain = np.zeros( (eiphi.shape) )
            for i in range(len(self.model)):
                antgain += np.flipud( self.model[i].antenna_gain(np.linspace(0, 360, eiphi.shape[1]),
                    np.linspace(0, 90, eiphi.shape[0])) )
            #antgain /= antgain.max()
        else:
            antgain = self.model.antenna_gain(np.linspace(0, 360, eiphi.shape[1]),
                np.linspace(0, 90, eiphi.shape[0]))

        beam = eiphi * eiphi.conjugate() * antgain
        beam = np.real(beam) / np.real(beam).max()
        return np.flipud(beam)



