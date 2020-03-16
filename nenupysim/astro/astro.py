#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **********************
    Astronomical Functions
    **********************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'nenufar_loc',
    'wavelength',
    'ho_coord',
    'eq_coord',
    'to_radec',
    'to_altaz',
    'ho_zenith',
    'eq_zenith',
    'radio_sources'
    ]


import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    Angle,
    SkyCoord,
    AltAz,
    Galactic,
    ICRS,
    solar_system_ephemeris,
    get_body
)
from astropy.constants import c as lspeed


# ============================================================= #
# ------------------------ nenufar_loc ------------------------ #
# ============================================================= #
def nenufar_loc():
    """ Returns the coordinate of NenuFAR array

        :returns: :class:`astropy.coordinates.EarthLocation`
            object
        :rtype: :class:`astropy.coordinates.EarthLocation`

        :Example:
        
        >>> from nenupysim.astro import nenufar_loc
        >>> location = nenufar_loc()
    """
    return EarthLocation(
        lat=47.375944 * u.deg,
        lon=2.193361 * u.deg,
        height=136.195 * u.m
    )
# ============================================================= #


# ============================================================= #
# ------------------------ wavelength ------------------------- #
# ============================================================= #
def wavelength(freq):
    """ Convert between MHz and wavelength in meters

        Returns
        -------
        wavelength : `np.ndarray`
            Wavelength in meters
    """
    if not hasattr(freq, '__len__'):
        freq = [freq]
    if not isinstance(freq, np.ndarray):
        freq = np.array(freq)
    freq *= u.MHz
    freq = freq.to(u.Hz)
    wavel = lspeed.value / freq.value
    return wavel
# ============================================================= #


# ============================================================= #
# ------------------------- ho_coord -------------------------- #
# ============================================================= #
def ho_coord(alt, az, time):
    """ Horizontal coordinates
    
        :param alt:
            Altitude in degrees
        :type alt: float
        :param az:
            Azimuth in degrees
        :type az: float
        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`astropy.time.Time`

        :returns: :class:`astropy.coordinates.AltAz` object
        :rtype: :class:`astropy.coordinates.AltAz`

        :Example:
        
        >>> from nenupysim.astro import ho_coord
        >>> altaz = ho_coord(
                alt=45,
                az=180,
                time='2020-01-01 12:00:00'
            )
    """
    return AltAz(
        az=az*u.deg,
        alt=alt*u.deg,
        location=nenufar_loc(),
        obstime=time
    )
# ============================================================= #


# ============================================================= #
# ------------------------- eq_coord -------------------------- #
# ============================================================= #
def eq_coord(ra, dec):
    """ Equatorial coordinates
        
        :param ra:
            Right ascension in degrees
        :type ra: float
        :param dec:
            Declination in degrees
        :type dec: float

        :returns: :class:`astropy.coordinates.ICRS` object
        :rtype: :class:`astropy.coordinates.ICRS`

        :Example:
        
        >>> from nenupysim.astro import eq_coord
        >>> radec = eq_coord(
                ra=51,
                dec=39,
            )
    """
    return ICRS(
        ra=ra*u.deg,
        dec=dec*u.deg
    )
# ============================================================= #


# ============================================================= #
# ------------------------- to_radec -------------------------- #
# ============================================================= #
def to_radec(altaz):
    """ Transform altaz coordinates to ICRS equatorial system
        
        :param altaz:
            Horizontal coordinates
        :type altaz: :class:`astropy.coordinates.AltAz`

        :returns: :class:`astropy.coordinates.ICRS` object
        :rtype: :class:`astropy.coordinates.ICRS`

        :Example:
        
        >>> from nenupysim.astro import eq_coord
        >>> radec = eq_coord(
                ra=51,
                dec=39,
            )
    """
    if not isinstance(altaz, AltAz):
        raise TypeError(
            'AltAz object expected.'
        )
    return altaz.transform_to(ICRS)
# ============================================================= #


# ============================================================= #
# ------------------------- to_altaz -------------------------- #
# ============================================================= #
def to_altaz(radec, time):
    """ Transform altaz coordinates to ICRS equatorial system
        
        :param radec:
            Equatorial coordinates
        :type altaz: :class:`astropy.coordinates.ICRS`
        :param time:
            Time at which the local coordinates should be 
            computed. It can either be provided as an 
            :class:`astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`astropy.time.Time`

        :returns: :class:`astropy.coordinates.AltAz` object
        :rtype: :class:`astropy.coordinates.AltAz`

        :Example:
        
        >>> from nenupysim.astro import eq_coord
        >>> radec = eq_coord(
                ra=51,
                dec=39,
            )
    """
    if not isinstance(radec, ICRS):
        raise TypeError(
            'ICRS object expected.'
        )
    altaz_frame = AltAz(
        obstime=time,
        location=nenufar_loc()
    )
    return radec.transform_to(altaz_frame)
# ============================================================= #


# ============================================================= #
# ------------------------- ho_zenith ------------------------- #
# ============================================================= #
def ho_zenith(time):
    """ Horizontal coordinates of local zenith above NenuFAR

        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`astropy.time.Time`

        :returns: :class:`astropy.coordinates.AltAz` object
        :rtype: :class:`astropy.coordinates.AltAz`

        :Example:
        
        >>> from nenupysim.astro import ho_zenith
        >>> zen_altaz = ho_zenith(time='2020-01-01 12:00:00')
    """
    return ho_coord(
        az=0.,
        alt=90.,
        time=time
    )
# ============================================================= #


# ============================================================= #
# ------------------------- eq_zenith ------------------------- #
# ============================================================= #
def eq_zenith(time):
    """ Equatorial coordinates of local zenith above NenuFAR
        
        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`astropy.time.Time`

        :Example:
        
        >>> from nenupysim.astro import ho_zenith
        >>> zen_radec = eq_zenith(time='2020-01-01 12:00:00')
    """
    altaz_zenith = ho_zenith(
        time=time
    )
    return to_radec(altaz_zenith)
# ============================================================= #


# ============================================================= #
# ----------------------- radio_sources ----------------------- #
# ============================================================= #
def radio_sources(time):
    """
    """
    if not isinstance(time, Time):
        time = Time(time)

    def solarsyst_eq(src, time):
        src = get_body(
            src,
            time,
            nenufar_loc()
        )
        return eq_coord(src.ra.deg, src.dec.deg)

    with solar_system_ephemeris.set('builtin'):
        src_radec = {
            'vira': eq_coord(187.70593075, +12.39112331),
            'cyga': eq_coord(299.86815263, +40.73391583),
            'casa': eq_coord(350.850000, +58.815000),
            'hera': eq_coord(252.783433, +04.993031),
            'hyda': eq_coord(139.523546, -12.095553),
            'taua': eq_coord(83.63308, +22.01450),
            'sun': solarsyst_eq('sun', time),
            'moon': solarsyst_eq('moon', time),
            'jupiter': solarsyst_eq('jupiter', time),
        }
    return {
        key: to_altaz(src_radec[key], time=time) for key in src_radec.keys()
    }
# ============================================================= #

