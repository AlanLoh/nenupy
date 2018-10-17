#! /usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
A-team coordinates in RA DEC
        by A. Loh
"""

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['vira', 'cyga', 'casa', 'hera', 'hyda', 'Source']


from astropy import coordinates as coord

vira = (187.70593075, +12.39112331)
cyga = (299.86815263, +40.73391583)
casa = (350.850000, +58.815000)
hera = (252.783433, +04.993031)
hyda = (139.523546, -12.095553)


class Source():
    def __init__(self, source):
        self.source = source

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def source(self):
        return self._source
    @source.setter
    def source(self, s):
        try:
            src = coord.SkyCoord.from_name(s)
            self._source = src
        except:
            self._source = None

    # ================================================================= #
    # ========================= Class Methods ========================= #
    @classmethod
    def VirA(cls):
        return cls(source='vira')

    @classmethod
    def CygA(cls):
        return cls(source='cyga')

    @classmethod
    def CasA(cls):
        return cls(source='casa')

    @classmethod
    def HerA(cls):
        return cls(source='hera')

    @classmethod
    def HydA(cls):
        return cls(source='hyda')


