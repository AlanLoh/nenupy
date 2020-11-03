#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **************
    Useful Classes 
    **************

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'MiniArrays',
    'ObsTime'
]


import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from nenupy.miscellaneous import accepts
from nenupy.instru import ma_info


# ============================================================= #
# ------------------------- MiniArrays ------------------------ #
# ============================================================= #
class MiniArrays(object):
    """
    """

    def __init__(self, names=None):
        self.names = names


    def __repr__(self):
        return self.names.__repr__()


    def __eq__(self, other):
        return self.names == other


    def __ne__(self, other):
        return self.names != other


    def __lt__(self, other):
        return self.names < other


    def __le__(self, other):
        return self.names <= other


    def __ge__(self, other):
        return self.names >= other


    def __gt__(self, other):
        return self.names > other


    def __len__(self):
        return self.names.__len__()


    def __getitem__(self, val):
        return MiniArrays(
            self.names[val]
        )


    @property
    def position(self):
        """
        """
        maPos = np.array(
            [a.tolist() for a in ma_info['pos']]
        )
        return maPos[self._selectedMA]


    @property
    def enu(self):
        """
        """
        from nenupy.astro import l93_to_etrs, etrs_to_enu

        return etrs_to_enu(
            l93_to_etrs(self.position)
        )    


    @property
    def rotation(self):
        """
        """
        return ma_info['rot'][self._selectedMA] * u.deg


    @property
    def delay(self):
        """
        """
        return ma_info['delay'][self._selectedMA]


    @property
    def attenuation(self):
        """
        """
        return ma_info['att'][self._selectedMA]


    @property
    def size(self):
        """
        """
        return self.names.size
    

    @property
    def names(self):
        """
        """
        return self._names
    @names.setter
    def names(self, m):
        if m is None:
            m = ma_info['ma']
        if np.isscalar(m):
            m = np.array([m])
        elif not isinstance(m, np.ndarray):
            m = np.array(m)

        if any(~np.isin(m, ma_info['ma'])):
            raise IndexError(
                'Unknown Mini-Arrays `{}`.'.format(
                    m[~np.isin(m, ma_info['ma'])]
                )
            )
        self._selectedMA = np.isin(
            ma_info['ma'],
            m
        )
        self._names = m
# ============================================================= #


# ============================================================= #
# -------------------------- ObsTime -------------------------- #
# ============================================================= #
class ObsTime(Time):
    """
    """

    @classmethod
    @accepts(type, Time, Time, int, strict=False)
    def linspace(cls, start, stop, num=50):
        """
        """
        if not all([t.isscalar for t in [start, stop]]):
            raise ValueError(
                '`start` and `stop` must be scalar quantities.'
            )
        dt = (stop - start)/(np.ceil(num) - 1)
        times = start + np.arange(num)*dt
        return cls(
            times,
            precision=max(start.precision, stop.precision)
        )


    @classmethod
    @accepts(type, Time, Time, TimeDelta, strict=False)
    def arange(cls, start, stop, dt):
        """
        """
        if not all([t.isscalar for t in [start, stop, dt]]):
            raise ValueError(
                '`start`, `stop` and `dt` must be scalar quantities.'
            )
        num = (stop - start)/dt
        times = start + np.arange(num)*dt
        return cls(
            times,
            precision=max(start.precision, stop.precision)
        )


    @accepts(Time, Time, Time, strict=False)
    def select(self, tMin=None, tMax=None):
        """
        """
        tMask = (self >= tMin) * (self <= tMax)
        return self[tMask]


    @accepts(Time, Time, strict=False)
    def closestTo(self, time):
        """
        """
        idx = np.argmin(
            np.abs(
                self - time
            )
        )
        return self[idx]
# ============================================================= #

