#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *******
    TV_Data
    *******
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'TV_Data'
]


import numpy as np
from os.path import abspath, isfile
from itertools import islice
from astropy.time import Time, TimeDelta
import astropy.units as u

from nenupy.crosslet import Crosslet

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# -------------------------- TV_Data -------------------------- #
# ============================================================= #
class TV_Data(Crosslet):
    """
    """

    def __init__(self, tvfile):
        super().__init__(
        )
        self.tvfile = tvfile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def tvfile(self):
        """ NenuFAR TV XST snapshot binary file
        """
        return self._tvfile
    @tvfile.setter
    def tvfile(self, x):
        if not isinstance(x, str):
            raise TypeError(
                'String expected.'
                )
        x = abspath(x)
        if not isfile(x):
            raise FileNotFoundError(
                'Unable to find {}'.format(x)
                )
        self._tvfile = x
        self._load()
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self):
        """ Read the binary file
            Visibilities are shaped as:
            (
                n_times,
                n_freqs,
                (n_ant*(n_ant-1)/2+n_ant)*4-n_ant
        """
        log.info('Loading {}'.format(self._tvfile))
        # Extract the ASCII header (5 first lines)
        with open(self._tvfile, 'rb') as f:
            header = list(islice(f, 0, 5))
        assert header[0] == b'HeaderStart\n',\
            'Wrong header start'
        assert header[-1] == b'HeaderStop\n',\
            'Wrong header stop'
        header = [s.decode('utf-8') for s in header]
        hd_size = sum([len(s) for s in header])

        # Parse informations into Crosslet attributes
        keys = ['freqs', 'mas', 'dt']
        search = ['Freq.List', 'Mr.List', 'accumulation']
        types = ['float64', 'int', 'int']
        for key, word, typ in zip(keys, search, types):
            unit = u.MHz if key == 'freqs' else 1
            for h in header:
                if word in h:
                    setattr(
                        self,
                        key,
                        np.array(
                            h.split('=')[1].split(','),
                            dtype=typ
                        )*unit
                    )

        # Deduce the dtype for decoding
        n_ma = self.mas.size
        n_sb = self.freqs.size
        dtype = np.dtype(
            [('jd', 'float64'),
            ('data', 'complex64', (n_sb, n_ma*n_ma*2 + n_ma))]
            )

        # Decoding the binary file
        tmp = np.memmap(
            filename=self._tvfile,
            dtype='int8',
            mode='r',
            offset=hd_size
            )
        decoded = tmp.view(dtype)

        self.dt = TimeDelta(self.dt, format='sec')
        self.vis = decoded['data'] / self.dt.sec
        self.times = Time(decoded['jd'], format='jd', precision=0)
# ============================================================= #

