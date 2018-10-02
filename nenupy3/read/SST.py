#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read NenuFAR SST data
"""

import os
import sys
import glob
import numpy as np

from astropy.io import fits
from astropy.time import Time


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['SST']


class SST():
    def __init__(self, obsfile=None):
        self.obsfile = obsfile
        self.ma      = 0
        self.freq    = 50 

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obsfile(self):
        """ SST observation file,
            obsfile set to 'fake' will produce a custom fake SST observation
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        if o == 'fake':
            self._createFakeObs()
            self._obsfile = o
            return
        
        if (o is None) or (o == '') or (os.path.isdir(o)):
            # Look at the current/specified directory
            if os.path.isdir(o):
                _opath = os.path.abspath(o)
                sstfiles = glob.glob( os.path.join(_opath, '*SST.fits') )
            else:
                sstfiles = glob.glob('*SST.fits')
            if len(sstfiles) == 0:
                raise IOError("\t=== No SST fits file in current directory, specify the file to read. ===")
            else:
                o = [os.path.abspath(f) for f in sstfiles]  
        elif isinstance(o, list):
            # List of files in input
            if all(os.path.isfile(f) for f in o):
                if all(f.endswith('SST.fits') for f in o):
                    o = [os.path.abspath(f) for f in o]
                else:
                    raise IOError("\t=== Multiple observation files (other than SST) are not understood - yet. ===")
            else:
                raise IOError("\t=== At least one file not found. ===")
        else:
            if not os.path.isfile(o):
                raise IOError("\t=== File {} not found. ===".format(os.path.abspath(o)))
            else:
                # The file has been correctly found
                o = [ os.path.abspath(o) ]
        self._obsfile = sorted(o)

        if not self._isSST():
            raise ValueError("\t=== Files might not all be SST observaitons ===")
        else:
            self._readSST()

        return

    @property
    def ma(self):
        """ Selected Mini-Array
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        if m is None:
            m = 0
        if not isinstance(m, int):
            try:
                m = int(m)
            except:
                print("\n\t=== WARNING: Mini-Array index {} not recognized ===".format(m))
                m = 0
        if m in self.miniarrays:
                self._ma = m
        else:
            print("\n\t=== WARNING: available Mini-Arrays are {} ===".format(self.miniarrays))
            self._ma = 0
        return

    @property
    def freq(self):
        """ Frequency selection in MHz
            Example:
            self.freq = 50
            self.freq = [20, 30]
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if isinstance(f, list):
            self._freq    = [0, 0]
            self._freq[0] = min(self.freqs, key=lambda x: np.abs(x-f[0]))
            self._freq[1] = min(self.freqs, key=lambda x: np.abs(x-f[1]))
        else:
            self._freq    = min(self.freqs, key=lambda x: np.abs(x-f))  
        return        

    # ------ Specific getters ------ #
    @property
    def marotation(self):
        """ Mini-array rotation in degrees
        """
        self._marotation = self._marot[ self.ma ]
        return self._marotation

    @property
    def maposition(self):
        self._maposition = self._mapos[ self.ma ]
        return self._maposition
    

    # ================================================================= #
    # =========================== Methods ============================= #
    def getData(self, **kwargs):
        """ 
        """
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _isSST(self):
        """ Check that self.obsfile contain only SST observations
        """
        isSST = True
        for sstfile in self.obsfile:
            with fits.open(sstfile, mode='readonly', ignore_missing_end=True, memmap=True) as f:
                if f[0].header['OBJECT'] != 'subband Statistics':
                    isSST = False
                else:
                    pass
        return isSST

    def _createFakeObs(self):
        """
        """
        return

    def _readSST(self):
        """ Read SST fits files and fill the class attributes
        """
        self.obsname = os.path.basename( self.obsfile[0] )[:-9]
        
        headi = fits.getheader(self.obsfile[0], ext=0, ignore_missing_end=True, memmap=True)
        headf = fits.getheader(self.obsfile[-1], ext=0, ignore_missing_end=True, memmap=True)
        self.obstart  = Time( headi['DATE-OBS'] + 'T' + headi['TIME-OBS'] )
        self.obstop   = Time( headf['DATE-END'] + 'T' + headf['TIME-END'] )
        self.exposure = self.obstop - self.obstart

        setup_ins = fits.getdata(self.obsfile[0], ext=1, ignore_missing_end=True, memmap=True)
        self.freqs      = np.squeeze( setup_ins['frq'] )
        self.miniarrays = np.squeeze( setup_ins['noMROn'] )
        self._marot     = np.squeeze( setup_ins['rotation'] )
        self._mapos     = np.squeeze( setup_ins['noPosition'] )
        self._mapos     = self._mapos.reshape( int(self._mapos.size/3), 3 )
        return



