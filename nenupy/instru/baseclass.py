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
    'MiniArrays'
]


import numpy as np
import astropy.units as u

from nenupy.instru import ma_info
from nenupy.astro import l93_to_etrs, etrs_to_enu


# ============================================================= #
# ------------------------- MiniArrays ------------------------ #
# ============================================================= #
class MiniArrays(object):
    """
    """

    def __init__(self, miniArrays):
        self.miniArrays = miniArrays


    def __repr__(self):
        return self.miniArrays.__repr__()


    def __eq__(self, other):
        return self.miniArrays == other


    def __ne__(self, other):
        return self.miniArrays != other


    def __lt__(self, other):
        return self.miniArrays < other


    def __le__(self, other):
        return self.miniArrays <= other


    def __ge__(self, other):
        return self.miniArrays >= other


    def __gt__(self, other):
        return self.miniArrays > other


    def __getitem__(self, val):
        return MiniArrays(
            self.miniArrays[val]
        )


    @property
    def position(self):
        """
        """
        maPos = np.array(
            [a.tolist() for a in ma_info['pos']]
        )
        return maPos[
            np.isin(
                ma_info['ma'],
                self.miniArrays
            )
        ]

    @property
    def enu(self):
        """
        """
        return etrs_to_enu(
            l93_to_etrs(self.position)
        )    


    @property
    def rotation(self):
        """
        """
        return ma_info['rot'][
            np.isin(
                ma_info['ma'],
                self.miniArrays
            )
        ] * u.deg


    @property
    def delay(self):
        """
        """
        return ma_info['delay'][
            np.isin(
                ma_info['ma'],
                self.miniArrays
            )
        ]


    @property
    def attenuation(self):
        """
        """
        return ma_info['att'][
            np.isin(
                ma_info['ma'],
                self.miniArrays
            )
        ]


    @property
    def miniArrays(self):
        """
        """
        return self._miniArrays
    @miniArrays.setter
    def miniArrays(self, m):
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
        self._miniArrays = m
# ============================================================= #

