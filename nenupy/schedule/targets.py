#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    .. _schedule_targets:

    *******
    TARGETS
    *******
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_Target',
    'ESTarget',
    'SSTarget'
]


import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    Angle,
    SkyCoord,
    FK5,
    EarthLocation,
    solar_system_ephemeris,
    get_body
)
import numpy as np

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ============================================================= #
NENUFAR_LOC = EarthLocation(
    lat=47.376511 * u.deg,
    lon=2.192400 * u.deg,
    height=150 * u.m
)


SS_SOURCES = [
    'sun',
    'moon',
    'mercury',
    'venus',
    'mars',
    'jupiter',
    'saturn',
    'uranus',
    'neptune'
]
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------------- _Target -------------------------- #
# ============================================================= #
class _Target(object):
    """
    """

    def __init__(self, target):
        self._target = target
        self._lst = None
        self._fk5 = None
        self._hourAngle = None
        self._elevation = None
        self._azimuth = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def target(self):
        """
        """
        return self._target


    @property
    def hourAngle(self):
        """ Gets the Local Hour Angle.
        """
        if self._hourAngle is None:
            self._attrWarning('hourAngle')
        return self._hourAngle


    @property
    def elevation(self):
        """ Gets the elevation.
        """
        if self._elevation is None:
            self._attrWarning('elevation')
        return self._elevation


    @property
    def azimuth(self):
        """ Gets the azimuth.
        """
        if self._azimuth is None:
            self._attrWarning('azimuth')
        return self._azimuth


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def reset(self) -> None:
        """ Clear the Target instance from previous computations. """
        self._lst = None
        self._fk5 = None
        self._hourAngle = None
        self._elevation = None
        self._azimuth = None


    def computePosition(self, time):
        """
        """
        if not isinstance(time, Time):
            raise TypeError(
                f'<time> should be a {Time} object.'
            )
        if time.isscalar:
            time = Time([time.isot, time.isot])

        self._localSiderealTime(time)
        self._positionAtEquinox(time)
        self._computeHourAngle()
        self._computeHorizontalCoords()


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _localSiderealTime(self, time):
        """
        """
        # Number of days since 2000 January 1, 12h UT
        nDays = time.jd - 2451545.
        # Greenwich mean sidereal time
        gmst = 18.697374558 + 24.06570982441908 * nDays
        gmst %= 24.
        # Local Sidereal Time
        lst = gmst + NENUFAR_LOC.lon.hour
        if np.isscalar(lst):
            if lst < 0:
                lst += 24
        else:
            lst[lst < 0] += 24.   
        self._lst = Angle(lst, 'hour')


    def _positionAtEquinox(self, time):
        """
        """
        fk5 = self.target.transform_to(
            FK5(equinox=time)
        )
        self._fk5 = fk5


    def _computeHourAngle(self):
        """
        """
        twoPi = Angle(360.000000, unit='deg')
        ha = self._lst - self._fk5.ra
        if ha.isscalar:
            if ha.deg < 0:
                ha += twoPi
            elif ha.deg > 360:
                ha -= twoPi
        else:
            ha[ha.deg < 0] += twoPi
            ha[ha.deg > 360] -= twoPi
        self._hourAngle = ha


    def _computeHorizontalCoords(self):
        """
        """
        twoPi = Angle(360.000000, unit='deg')

        decRad = self._fk5.dec.rad
        sinDec = np.sin(decRad)
        haRad = self._hourAngle.rad
        latRad = NENUFAR_LOC.lat.rad
        sinLat = np.sin(latRad)
        cosLat = np.cos(latRad)
        sinEl = sinDec * sinLat +\
            np.cos(decRad) * cosLat * np.cos(haRad)
        self._elevation = Angle(
            np.arcsin(sinEl),
            unit='rad'
        ).to('deg')

        elRad = self._elevation.rad
        cosAz = (sinDec - np.sin(elRad) * sinLat)/\
            (np.cos(elRad) * cosLat)
        azRad = Angle(np.arccos(cosAz), unit='rad')

        if azRad.isscalar:
            if np.sin(self._hourAngle.rad) > 0:
                azRad *= -1
                azRad += twoPi
        else:
            posMask = np.sin(self._hourAngle.rad) > 0
            azRad[posMask] *= -1
            azRad[posMask] += twoPi

        self._azimuth = azRad.to('deg')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _attrWarning(self, attr):
        """
        """
        log.warning(
            'Target position must be computed using '
            '<.computePosition(time)> method, prior to asking'
            f' for the <{attr}> attribute.'
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- SSTarget -------------------------- #
# ============================================================= #
class SSTarget(_Target):
    """ Solar System target
    """

    def __init__(self, target):
        super().__init__(target=target)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def isCircumpolar(self):
        """
        """
        ninetyDeg = Angle(90, 'deg')
        decAndLat = self._fk5.dec + NENUFAR_LOC.lat
        return any(decAndLat > ninetyDeg) # not sure about that


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def fromName(cls, sourceName):
        """
        """
        if not isinstance(sourceName, str):
            raise TypeError(
                f"<sourceName> '{sourceName}' must be a {str}."
            )
        sourceName = sourceName.lower()
        if sourceName not in SS_SOURCES:
            raise ValueError(
                f"Solar System target '{sourceName}'' not in "
                f"{SS_SOURCES}."
            )

        log.debug(
            f"Solar System target '{sourceName}' loaded."
        )
        return cls(
            target=sourceName
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _positionAtEquinox(self, time):
        """ Over define this _Target method
        """
        with solar_system_ephemeris.set('builtin'):
            source = get_body(
                self.target,
                time,
                NENUFAR_LOC
            ) # GCRS
        ssSource = SkyCoord(source.ra, source.dec)

        fk5 = ssSource.transform_to(
            FK5(equinox=time)
        )
        self._fk5 = fk5
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- ESTarget -------------------------- #
# ============================================================= #
class ESTarget(_Target):
    """ ExtraSolar System target
    """

    def __init__(self, target):
        super().__init__(target=target)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def isCircumpolar(self):
        """
        """
        return self.target.dec + NENUFAR_LOC.lat > Angle(90, 'deg')


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def fromName(cls, sourceName):
        """
        """
        if not isinstance(sourceName, str):
            raise TypeError(
                f"<sourceName> '{sourceName}'' must be a {str}."
            )
        esSource = SkyCoord.from_name(sourceName)
        log.debug(
            f"ExtraSolar target '{sourceName}' loaded."
        )
        return cls(
            target=esSource
        )


    @classmethod
    def fromCoordinates(cls, coordinates):
        """
        """
        if isinstance(coordinates, str):
            esSource = SkyCoord(coordinates, unit=(u.hourangle, u.deg))
        else:
            esSource = SkyCoord(*coordinates, unit=u.deg)
        return cls(
            target=esSource
        )
# ============================================================= #
# ============================================================= #

