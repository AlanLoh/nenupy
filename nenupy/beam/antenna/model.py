#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to handle antenna models
        by A. Loh
"""

import os
import sys
import numpy as np

from scipy.interpolate import interp2d

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['AntennaModel']

class AntennaModel():
    """ This class returns antenna models
        The outputs are response functions: antenna_gain(azim, elev)
        Parameters:
        - design (str): design of the antenna
        - kwargs: keywords used for specific antenna models
    """
    def __init__(self, design='nenufar', **kwargs):
        self.kwargs = kwargs
        self.design = design
        

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def design(self):
        """ Antenna design
        """
        return self._design
    @design.setter
    def design(self, d):
        if isinstance(d, np.ndarray):
            self._makeCustom(model=d)
        elif d.lower() == 'nenufar':
            self._design = 'nenufar'
            self._getNenuFAR()
        else:
            raise Exception("\n\t=== Nothing handled yet ===")
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def plotAntGain(self):
        """
        """
        from matplotlib import pyplot as plt
        gain = self.antenna_gain(np.linspace(0,360,360), np.linspace(0,90,90))
        plt.imshow(gain, origin='lower')
        plt.show()
        return

    def _makeCustom(self, model):
        gaininterp = interp2d( np.linspace(0., 360., model.shape[1]), 
             np.linspace(0., 90., model.shape[0]), model, kind='linear' )
        self.antenna_gain = gaininterp
        self.antenna_freq = self.kwargs['freq']
        return
    
    def _getNenuFAR(self):
        """
        """
        if 'freq' not in self.kwargs.keys():
            raise Exception("\n\t=== keyword 'freq' needs to be defined  ===")
        if 'polar' not in self.kwargs.keys():
            raise Exception("\n\t=== keyword 'polar' needs to be defined  ===")
        
        from scipy.io.idl import readsav
        from scipy.interpolate import interp2d
        
        antpol = {'NE': 'NE_SW', 'NW': 'NW_SE'}
        pol    = antpol[self.kwargs['polar']]
        f1 = int( np.floor( self.kwargs['freq']/10. ) ) * 10
        f2 = int( np.ceil(  self.kwargs['freq']/10. ) ) * 10

        modulepath = os.path.dirname( os.path.realpath(__file__) )

        gain_inf = 10**( readsav( os.path.join(modulepath, 'LSS_GRID_AG_'+pol+'_'+str(f1)+'_astro.sav') )['ga']/10. )
        gain_sup = 10**( readsav( os.path.join(modulepath, 'LSS_GRID_AG_'+pol+'_'+str(f2)+'_astro.sav') )['ga']/10. )
        if f1 != f2:
            gain = gain_inf * (f2-self.kwargs['freq']) / 10. + gain_sup * (self.kwargs['freq']-f1) / 10.
        else:
            gain = gain_inf

        gaininterp = interp2d( np.linspace(0., 360., gain.shape[1]), 
             np.linspace(0., 90., gain.shape[0]), gain, kind='linear' )
        
        self.antenna_gain = gaininterp
        self.antenna_freq = self.kwargs['freq']

        return


