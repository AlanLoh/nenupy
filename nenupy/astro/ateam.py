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
__all__ = ['Source']


from astropy import coordinates as coord

ATEAM = {'vir a': (187.70593075, +12.39112331),
         'cyg a': (299.86815263, +40.73391583),
         'cas a': (350.850000, +58.815000),
         'her a': (252.783433, +04.993031),
         'hyd a': (139.523546, -12.095553),
         'tau a': (83.63308, +22.01450)}


class Source(object):
    def __init__(self, source, time=None, location=None):
        self.time = time
        self.location = location
        self.source = source

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def source(self):
        return self._source
    @source.setter
    def source(self, s):
        # try:
        if not isinstance(s, coord.SkyCoord):
            if s.lower() in ATEAM.keys():
                src = coord.SkyCoord(ATEAM[s.lower()][0], ATEAM[s.lower()][1], frame="icrs", unit="deg")
            elif s.lower() in ['sun', 'moon', 'jupiter', 'saturn', 'mars', 'venus']:
                with coord.solar_system_ephemeris.set('builtin'):
                    src = coord.get_body(s, self.time, self.location)
            else:
                src = coord.SkyCoord.from_name(s)
        else:
            src = s
        self._source = src
        # except:
        #     self._source = None

    # ================================================================= #
    # ========================= Class Methods ========================= #
    # @classmethod
    # def VirA(cls):
    #     # return cls(source='vira')
    #     self.source = coord.SkyCoord(vira[0], vira[1], frame="icrs", unit="deg")
    #     return

    # # @classmethod
    # def CygA(cls):
    #     # return cls(source='cyga')
    #     self.source = coord.SkyCoord(cyga[0], cyga[1], frame="icrs", unit="deg")
    #     return

    # # @classmethod
    # def CasA(cls):
    #     # return cls(source='casa')
    #     self.source = coord.SkyCoord(casa[0], casa[1], frame="icrs", unit="deg")
    #     return

    # # @classmethod
    # def HerA(cls):
    #     # return cls(source='hera')
    #     self.source = coord.SkyCoord(hera[0], hera[1], frame="icrs", unit="deg")
    #     return

    # # @classmethod
    # def HydA(cls):
    #     # return cls(source='hyda')
    #     self.source = coord.SkyCoord(hyda[0], hyda[1], frame="icrs", unit="deg")
    #     return


