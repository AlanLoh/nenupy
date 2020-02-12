#! /usr/bin/python3
# -*- coding: utf-8 -*-


""" 
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'XST'
    ]


import warnings
from astropy.io.fits import getdata
import numpy as np
from os.path import isfile, abspath
from astropy.time import Time

from .util import SpecData


# ============================================================= #
# ---------------------------- XST ---------------------------- #
# ============================================================= #
class XST(object):
    """
    """

    def __init__(self, xstfile):
        self.xstfile = xstfile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def xstfile(self):
        """
        """
        return self._xstfile
    @xstfile.setter
    def xstfile(self, x):
        x = abspath(x)
        if not isfile(x):
            raise FileNotFoundError(
                'Unable to find {}'.format(x)
            )
        self.ma = getdata(
            x,
            ext=1,
            memmap=True
        )['noMROn'][0]
        self._nma = self.ma.size
        self._ma_idx = np.arange(self._nma)
        self._xi, self._yi = np.tril_indices(self._nma * 2, 0)
        data_tmp = getdata(
            x,
            ext=7,
            memmap=True
        )
        self._time = data_tmp['jd']
        self._freq = data_tmp['xstsubband']
        self._frequencies = np.unique(self._freq)
        if self._frequencies.size != 16:
            warnings.warn(
                '\nWarning: number of subbands is not 16. First 16 selected.'
            )
        self._frequencies = self._frequencies[:16] * 195.3125 * 1.e-3
        self._data = data_tmp['data']
        #self._data = self._reshape(data_tmp['data'])
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self, ma1, ma2, pol='XX', tidx=None):
        """ Select data per baseline and polarization.
        """
        if tidx is not None:
            if not isinstance(tidx, np.ndarray):
                tidx = np.array([tidx])
        else:
            tidx = np.arange(self._data.shape[0])          
        fidx = np.arange(self._data.shape[1])

        pdict = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}

        cross_matrix = np.zeros(
            (self._nma * 2, self._nma * 2),
            dtype='bool'
        )
        if ma1 < ma2:
            # Asking for conjugate data
            auto = False
            tmp = ma1
            ma1 = ma2
            ma2 = tmp
        elif ma1 > ma2:
            auto = False
        else:
            auto = True
        
        mask = (self._xi//2 == ma1) & (self._yi//2 == ma2)
        
        cross_matrix[self._xi[mask], self._yi[mask]] = True
        bl_sel = cross_matrix[self._xi, self._yi].ravel()
        bl_idx = np.arange(bl_sel.size)[bl_sel]

        if auto:
            if pol=='XX':
                data = self._data[
                    np.ix_(
                        tidx,
                        fidx,
                        [bl_idx[0]]
                    )
                ]
            elif pol=='XY':
                data = self._data[
                    np.ix_(
                        tidx,
                        fidx,
                        [bl_idx[1]]
                    )
                ].conj()
            elif pol=='YX':
                data = self._data[
                    np.ix_(
                        tidx,
                        fidx,
                        [bl_idx[1]]
                    )
                ]
            elif pol=='YY':
                data = self._data[
                    np.ix_(
                        tidx,
                        fidx,
                        [bl_idx[2]]
                    )
                ]
            else:
                pass
        else:
            data = self._data[
                np.ix_(
                    tidx,
                    fidx,
                    [bl_idx[ pdict[pol] ]]
                )
            ]

        return data


    def beamform(self, ma1, ma2, part='re', tidx=None):
        """
        """
        if part.lower() == 're':
            xx = self.get(ma1=ma1, ma2=ma2, pol='XX', tidx=tidx)
            xy = self.get(ma1=ma1, ma2=ma2, pol='XY', tidx=tidx)
            yy = self.get(ma1=ma1, ma2=ma2, pol='YY', tidx=tidx)
            data = np.real(xx) + 2*np.real(xy) + np.real(yy)
        
        elif part.lower() == 'im':
            xx = self.get(ma1=ma1, ma2=ma2, pol='XX', tidx=tidx)
            xy = self.get(ma1=ma1, ma2=ma2, pol='XY', tidx=tidx)
            yy = self.get(ma1=ma1, ma2=ma2, pol='YY', tidx=tidx)
            data = np.imag(xx) + 2*np.imag(xy) + np.imag(yy)
        
        elif part.lower() == 'x':
            xx = self.get(ma1=ma1, ma2=ma2, pol='XX', tidx=tidx)
            auto_x1 = self.get(ma1=ma1, ma2=ma1, pol='XX', tidx=tidx)
            auto_x2 = self.get(ma1=ma2, ma2=ma2, pol='XX', tidx=tidx)
            data = np.real(auto_x1) + 2*np.real(xx) + np.real(auto_x2)

        elif part.lower() == 'y':
            yy = self.get(ma1=ma1, ma2=ma2, pol='YY', tidx=tidx)
            auto_y1 = self.get(ma1=ma1, ma2=ma1, pol='YY', tidx=tidx)
            auto_y2 = self.get(ma1=ma2, ma2=ma2, pol='YY', tidx=tidx)
            data = np.real(auto_y1) + 2*np.real(yy) + np.real(auto_y2)
        else:
            pass

        return SpecData(
            data=np.squeeze(data),
            time=Time(np.squeeze(self._time[tidx]), format='jd'),
            freq=self._frequencies,
            stokes=part
            )


    def reshape(self, tidx=None):
        """
        """
        if tidx is not None:
            if not isinstance(tidx, np.ndarray):
                tidx = np.array([tidx])
        reshaped_matrix = np.zeros(
            (
                self._time.shape[0] if tidx is None else tidx.size,
                self._freq.shape[1],
                self._nma,
                self._nma,
                4
            ),
            dtype='complex64'
        )
        for ma_i in range(self._nma):
            for ma_j in range(ma_i, self._nma):
                for pol_i, pol in enumerate(['XX', 'XY', 'YX', 'YY']):
                    reshaped_matrix[..., ma_i, ma_j, pol_i] = self.get(
                            ma1=ma_i,
                            ma2=ma_j,
                            pol=pol,
                            tidx=tidx
                        )
        return reshaped_matrix


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _reshape(self, data):
        """ From (time, freq, visib)
            to (time, freq, nant, nant, corr)
        """
        tmp = np.zeros(
            (
                self._time.size,
                self._freq.shape[1],
                self._nma*2,
                self._nma*2
            ),
            dtype='complex64'
        )
        tmp[..., self._xi, self._yi] = data
        return tmp



# ============================================================= #








