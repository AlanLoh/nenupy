#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "solar_system_source",
    "local_sidereal_time",
    "hour_angle",
    "radec_to_altaz",

]


from abc import ABC
from typing import Union, Tuple
from enum import Enum, auto

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    solar_system_ephemeris,
    get_body,
    Angle,
    Longitude,
    AltAz,
    FK5
)

from nenupy import nenufar_position


# ============================================================= #
# --------------------- SolarSystemSource --------------------- #
# ============================================================= #
class SolarSystemSource(Enum):
    """ Enumerator of available solar system sources. """

    SUN = auto()
    MOON = auto()
    MERCURY = auto()
    VENUS = auto()
    MARS = auto()
    JUPITER = auto()
    SATURN = auto()
    URANUS = auto()
    NEPTUNE = auto()
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- solar_system_source -------------------- #
# ============================================================= #
def solar_system_source(
        name: str,
        time: Time = Time.now(),
        observer: EarthLocation = nenufar_position
    ) -> SkyCoord:
    """ Returns a Solar System body in the ICRS reference system. """

    # Get the Solar System object in the GCRS reference system
    with solar_system_ephemeris.set('builtin'):
        source = get_body(
            body=name,
            time=time,
            location=observer
        )

    # Return the SkyCoord instance converted to ICRS
    # return source.transform_to('icrs')
    return SkyCoord(source.ra, source.dec)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- local_sidereal_time -------------------- #
# ============================================================= #
def local_sidereal_time(
        time: Time,
        observer: EarthLocation = nenufar_position,
        fast_compute: bool = True
    ) -> Longitude:
    """ """

    # Fast method to compute an approximation of the LST
    if fast_compute:
        # Number of days since 2000 January 1, 12h UT
        n_days = time.jd - Time("2000-01-01 12:00:00.000").jd

        sidereal_ref_time = 18.697374558 # at 2000-01-01 0 UT
        day_to_sidereal_hours = 24.06570982441908 # 1 sidereal day ~ 23h56m calendar hours.
        # Greenwich mean sidereal time
        gw_sidereal_hour = sidereal_ref_time + day_to_sidereal_hours * n_days
        gmst = Longitude(angle=gw_sidereal_hour, unit="hour")

        # Conversion at the given longitude
        return Longitude(gmst + observer.lon)

    # astropy computation accounting for precession and
    # for nutation and using the latest available
    # precession/nutation models
    else:    
        return time.sidereal_time(
            kind="apparent",
            longitude=observer.lon,
            model=None
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ hour_angle ------------------------- #
# ============================================================= #
def hour_angle(
        radec: SkyCoord,
        time: Time,
        observer: EarthLocation = nenufar_position,
        fast_compute: bool = True
    ) -> Longitude:
    """ """
    lst = local_sidereal_time(
        time=time,
        observer=observer,
        fast_compute=fast_compute)
    return Longitude(lst - radec.ra)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- radec_to_altaz ----------------------- #
# ============================================================= #
def radec_to_altaz(
        radec: SkyCoord,
        time: Time,
        observer: EarthLocation = nenufar_position,
        fast_compute: bool = True
    ) -> SkyCoord:
    """ """
    if fast_compute:
        radec = radec.transform_to(
            FK5(equinox=time)
        )
        two_pi = Angle(360.0, unit='deg')

        ha = hour_angle(
            radec=radec,
            time=time,
            observer=observer,
            fast_compute=fast_compute
        )

        sin_dec = np.sin(radec.dec.rad)
        cos_dec = np.cos(radec.dec.rad)
        sin_lat = np.sin(observer.lat.rad)
        cos_lat = np.cos(observer.lat.rad)

        # Compute elevation
        sin_elevation = sin_dec * sin_lat + cos_dec*cos_lat*np.cos(ha.rad)
        elevation = Angle(np.arcsin(sin_elevation), unit="rad")

        # Compute azimuth
        cos_azimuth = (sin_dec - np.sin(elevation.rad) * sin_lat)/\
            (np.cos(elevation.rad) * cos_lat)
        azimuth = Angle(np.arccos(cos_azimuth), unit="rad")

        if azimuth.isscalar:
            if np.sin(ha.rad) > 0:
                azimuth *= -1
                azimuth += two_pi
        else:
            posMask = np.sin(ha.rad) > 0
            azimuth[posMask] *= -1
            azimuth[posMask] += two_pi

        return SkyCoord(
            azimuth,
            elevation,
            frame=AltAz(
                obstime=time,
                location=observer
            )
        )
    else:
        return radec.transform_to(
            AltAz(
                obstime=time,
                location=observer
            )
        )

# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ AstroObject ------------------------ #
# ============================================================= #
class AstroObject(ABC):
    """ Abstract base class for all astronomy related classes. """

    coordinates: Union[
        SkyCoord,
        Tuple[float, float],
        Tuple[str, str]
    ]
    time: Time
    value: Union[
        float,
        int,
        np.ndarray
    ]
    observer: EarthLocation


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def is_circumpolar(self) -> bool:
        """ 
            Whether the celestial object is circumpolar at the 
            observer's latitude :math:`l` (defined in
            :attr:`~nenupy.astro2.astro_tools.AstroObject.observer`), i.e:

            .. math::
                l + \delta \geq 90\,{\rm deg}
            
            where :math:`\delta` is the object's declination (defined in
            :attr:`~nenupy.astro2.astro_tools.AstroObject.coordinates`)
        """
        return np.all((self.observer.lat + self.coordinates.dec) >= 90*u.deg)


    @property
    def culmination_azimuth(self) -> u.Quantity:
        """ """
        if not self.is_circumpolar:
            return 180*u.deg
        else:
            return 0*u.deg


    @property
    def horizontal_coordinates(self) -> SkyCoord:
        """ """
        return radec_to_altaz(
            radec=self.coordinates,
            time=self.time,
            observer=self.observer,
            fast_compute=True
        )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def local_sidereal_time(self, fast_compute: bool = True) -> Longitude:
        """ """
        return local_sidereal_time(
            time=self.time,
            observer=self.observer,
            fast_compute=fast_compute
        )


    def hour_angle(self, fast_compute: bool = True) -> Longitude:
        """ """
        return hour_angle(
            radec=self.coordinates,
            time=self.time,
            observer=self.observer,
            fast_compute=fast_compute
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #
# ============================================================= #
