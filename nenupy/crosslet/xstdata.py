#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    XST_Data
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'XST_Data'
]


from astropy.io import fits
from astropy.time import Time, TimeDelta
import astropy.units as u
from os.path import abspath, isfile
import numpy as np

from nenupy.crosslet import Crosslet

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- XST_Data -------------------------- #
# ============================================================= #
class XST_Data(Crosslet):
    """
    """

    def __init__(self, xstfile):
        super().__init__(
        )
        self.xstfile = xstfile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def xstfile(self):
        """ NenuFAR XST
        """
        return self._xstfile
    @xstfile.setter
    def xstfile(self, x):
        if not isinstance(x, str):
            raise TypeError(
                'String expected.'
                )
        x = abspath(x)
        if not isfile(x):
            raise FileNotFoundError(
                'Unable to find {}'.format(x)
                )
        self._xstfile = x
        self._load()
        return

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self):
        """ Read XST file
        """
        log.info('Loading {}'.format(self._xstfile))
        self.mas = fits.getdata(
            self._xstfile,
            ext=1
        )['noMROn'][0]
        data_tmp = fits.getdata(
            self._xstfile,
            ext=7,
            memmap=True
        )
        self.vis = data_tmp['data']
        self.times = Time(data_tmp['jd'], format='jd')
        bw = 0.1953125
        self.sb_idx = np.unique(data_tmp['xstsubband'])
        self.freqs = self.sb_idx * bw * u.MHz
        return
# ============================================================= #

