#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *******
    Beamlet
    *******
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Beamlet'
]


import numpy as np
from astropy.io import fits
from astropy.time import Time

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# -------------------------- Beamlet -------------------------- #
# ============================================================= #
class Beamlet(object):
    """
    """

    def __init__(self):
        self.meta = {}
        self.times = None
        self.data = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def t_min(self):
        """ Observation time min
        """
        return np.min(self.times)


    @property
    def t_max(self):
        """ Observation time max
        """
        return np.max(self.times)


    @property
    def f_min(self):
        """ Observation freq min
        """
        try:
            freqs = np.unique(self.meta['bea']['freqList'])
        except KeyError:
            # Observation is probably a SST
            freqs = self.meta['ins']['frq'].squeeze()
        freqs = freqs[freqs != 0.]
        return np.min(freqs)


    @property
    def f_max(self):
        """ Observation freq max
        """
        try:
            freqs = np.unique(self.meta['bea']['freqList'])
        except KeyError:
            # Observation is probably a SST
            freqs = self.meta['ins']['frq'].squeeze()
        freqs = freqs[freqs != 0.]
        return np.max(freqs)
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self, filename):
        """
        """
        with fits.open(filename,
            mode='readonly',
            ignore_missing_end=True,
            memmap=True
        ) as f:
            # Metadata loading
            self.meta['hea'] = f[0].header
            self.meta['ins'] = f[1].data
            self.meta['obs'] = f[2].data
            self.meta['ana'] = f[3].data
            self.meta['bea'] = f[4].data
            self.meta['pan'] = f[5].data
            self.meta['pbe'] = f[6].data
            # Data loading 
            self.times = Time(f[7].data['JD'], format='jd')
            self.data = f[7].data['data']
        return

# ============================================================= #

