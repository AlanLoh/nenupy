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


from os.path import isfile
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy.base import MiniArrays
from nenupy.beamlet import SData

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- SST_Data -------------------------- #
# ============================================================= #
class SST_Data(object):
    """ Class to read *NenuFAR* SST data stored as FITS files.

        :param sstfile: Path to SST file.
        :type sstfile: str
    """

    def __init__(self, filename, **kwargs):
        self._autoUpdate = kwargs.get('autoUpdate', True)
        self.obsProperties = {}
        self.filename = filename
        self._freqRange = None
        self._timeRange = None
        self._ma = None
        self._polar = None


    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Boom')
        new = SST_Data(
            filename=self.filename + other.filename,
            autoUpdate=False
        )
        new.obsProperties = {**self.obsProperties, **other.obsProperties}
        return new


    def __radd__(self, other):
        if other==0:
            return self
        else:
            return self.__add__(other)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def filename(self):
        """
        """
        return sorted(self._filename)
    @filename.setter
    def filename(self, f):
        if not isinstance(f, list):
            f = [f]
        for fi in f:
            if not isfile(fi):
                raise FileNotFoundError(
                    'Unable to find {}'.format(fi)
                )
            self._fillObsDict(fi)
        self._filename = f


    @property
    def timeRange(self):
        """
        """
        if self._timeRange is None:
            self._timeRange = Time([self.tMin, self.tMax])
        return self._timeRange
    @timeRange.setter
    def timeRange(self, tr):
        if tr is None:
            tr = [self.tMin, self.tMax]
        self._timeRange = self._intervalBoundaries(
            tr,
            Time,
            precision=0
        )


    @property
    def freqRange(self):
        """
        """
        if self._freqRange is None:
            self._freqRange = [self.fMin, self.fMax]
        return self._freqRange
    @freqRange.setter
    def freqRange(self, fr):
        if fr is None:
            fr = [self.fMin, self.fMax]
        self._freqRange = self._intervalBoundaries(
            fr,
            u.Quantity,
            unit='MHz'
        )


    @property
    def polar(self):
        """
        """
        if self._polar is None:
            self._polar = self._getPropList('polars')[0][0]
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, str):
            raise TypeError(
                '`polar` should be a string.'
            )
        p = p.upper()
        if not all([p in ps for ps in self._getPropList('polars')]):
            raise IndexError(
                'Polarization `{}` unknown.'.format(p)
            )
        self._polar = p


    @property
    def ma(self):
        """
        """
        if self._ma is None:
            self._ma = self._getPropList('MAs')[0].names[0]
        return self._ma
    @ma.setter
    def ma(self, m):
        if not isinstance(m, int):
            raise TypeError(
                '`ma` should be an integer.'
            )
        if not all([m in mas for mas in self._getPropList('MAs')]):
            raise IndexError(
                'Mini-Array `{}` not used in current observation.'.format(m)
            )
        self._ma = m


    @property
    def tMin(self):
        """
        """
        return min(self._getPropList('tMin'))


    @property
    def tMax(self):
        """
        """
        return max(self._getPropList('tMax'))


    @property
    def fMin(self):
        """
        """
        return min(self._getPropList('fMin'))


    @property
    def fMax(self):
        """
        """
        return max(self._getPropList('fMax'))


    @property
    def mas(self):
        """
        """
        return self._getPropList('MAs')[0]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def select(self, **kwargs):
        """
        """
        allowedSelection = [
            'freqRange',
            'timeRange',
            'ma',
            'polar'
        ]
        for key, val in kwargs.items():
            if key in allowedSelection:
                setattr(self, key, val)
        for key in allowedSelection:
            log.info(
                'Selection on {}={}'.format(
                    key,
                    getattr(self, key)
                )
            )
        for sstfile in self.filename:
            if (self.timeRange[0] > self.obsProperties[sstfile]['tMax'])\
                or (self.timeRange[1] < self.obsProperties[sstfile]['tMin'])\
                or (self.freqRange[0] > self.obsProperties[sstfile]['fMax'])\
                or (self.freqRange[1] > self.obsProperties[sstfile]['fMax']):
                # File does not satisfy selection
                continue
            
            # Frequency selection
            freqs = self.obsProperties[sstfile]['freqs']
            if self.freqRange[0] == self.freqRange[1]:
                # Selection on closest value
                fMask = np.zeros(freqs.size, dtype=bool)
                fMask[np.argmin(np.abs(freqs - self.freqRange[0]))] = True
            else:
                # Selection based on the boundary values
                fMask = (freqs >= self.freqRange[0])\
                    & (freqs <= self.freqRange[1])

            # Time selection
            times = fits.getdata(sstfile, ext=7, memmap=True)['jd']
            if self.timeRange[0] == self.timeRange[1]:
                # Selection on closest value
                tMask = np.zeros(times.size, dtype=bool)
                tMask[np.argmin(np.abs(times - self.timeRange[0].jd))] = True
            else:
                # Selection based on the boundary values
                tMask = (times >= self.timeRange[0].jd)\
                    & (times <= self.timeRange[1].jd)
            
            # Polarization selection
            pMask = self.obsProperties[sstfile]['polars'] == self.polar
            
            data = fits.getdata(sstfile, ext=7, memmap=True)['data']
            sdataTemp = SData(
                data=np.swapaxes(data, 1, 3)[
                    np.ix_(tMask, fMask, pMask)
                ][..., self.ma], # MA selection
                time=Time(times[tMask], format='jd', precision=0),
                freq=freqs[fMask],
                polar=self.obsProperties[sstfile]['polars'][pMask]
            )

            if 'sdata' in locals():
                # Concatenate the data
                sdata = sdata | sdataTemp
            else:
                # Initialize the SData instance
                sdata = sdataTemp

        if not 'sdata' in locals():
            log.warning(
                'Empty selection.'
            )
            return
        return sdata


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _fillObsDict(self, filename):
        """
        """
        if not self._autoUpdate:
            return
        header = fits.getheader(filename)
        ins = fits.getdata(filename, ext=0, header=None)
        self.obsProperties[filename] = {
            'tMin': Time(header['DATE-OBS'] + 'T' + header['TIME-OBS'], precision=0),
            'tMax': Time(header['DATE-END'] + 'T' + header['TIME-END'], precision=0),
            'fMin': ins['frq'].min()*u.MHz,
            'fMax': ins['frq'].max()*u.MHz,
            'freqs': ins['frq'][0]*u.MHz,
            'polars': ins['spol'][0],
            'MAs': MiniArrays(ins['noMROn'][0])
        }


    def _getPropList(self, prop):
        """
        """
        obsFiles = self.obsProperties.keys()
        return [self.obsProperties[f][prop] for f in obsFiles]


    @staticmethod
    def _intervalBoundaries(bd, typ, **kwargs):
        """ For frequency or time
        """
        if typ is u.Quantity:
            bd = typ(bd, ndmin=1, **kwargs)
            bd = np.sort(bd)
        elif typ is Time:
            bd = typ(bd, **kwargs)
            if bd.ndim == 0:
                bd = typ([bd])
            bd = bd.sort()
        else:
            raise TypeError(
                '`_intervalBoundaries` only works with Time and Quantity objects.'
            )
        
        if bd.ndim > 1:
            raise IndexError(
                'Only dimension-1 arrays are understood.'
            )
        if bd.size > 2:
            raise IndexError(
                'Only length-2 arrays are allowed.'
            )
        elif bd.size == 1:
            bd = bd.insert(0, bd[0])
        
        return bd
# ============================================================= #

