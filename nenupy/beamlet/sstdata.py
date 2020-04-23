#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    SST_Data
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'SST_Data'
]


import numpy as np
from os.path import abspath, isfile
from astropy.time import Time
import astropy.units as u

from nenupy.beamlet import Beamlet
from nenupy.beamlet import SData

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- BST_Data -------------------------- #
# ============================================================= #
class SST_Data(Beamlet):
    """ Class to read *NenuFAR* SST data stored as FITS files.

        :param sstfile: Path to SST file.
        :type sstfile: str
    """

    def __init__(self, sstfile):
        super().__init__(
        )
        self.sstfile = sstfile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def sstfile(self):
        """ Path toward SST FITS file.
            
            :setter: SST file
            
            :getter: SST file
            
            :type: `str`

            :Example:
            
            >>> from nenupy.beamlet import SST_Data
            >>> sst = SST_Data(
                    sstfile='/path/to/SST.fits'
                )
            >>> sst.sstfile
            '/path/to/SST.fits'
        """
        return self._sstfile
    @sstfile.setter
    def sstfile(self, s):
        if not isinstance(s, str):
            raise TypeError(
                'String expected.'
                )
        s = abspath(s)
        if not isfile(s):
            raise FileNotFoundError(
                'Unable to find {}'.format(s)
                )
        self._sstfile = s
        self._load(self._sstfile)
        return


    @property
    def freqs(self):
        """ Recorded frequencies.

            :getter: Available frequencies
            
            :type: :class:`astropy.units.Quantity`
        """
        return self.meta['ins']['frq'].squeeze()*u.MHz

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #

