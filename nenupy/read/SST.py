#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read SST NenuFAR data
"""

import os
import sys
import glob
import numpy as np

from astropy.io import fits


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
        self._obsfile = o

        if not self._isSST():
            raise ValueError("\t=== Files might not all be SST observaitons ===")
        else:
            self._readSST()

        return

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
        return



