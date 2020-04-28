#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **********************
    Astronomical Functions
    **********************

    Below are defined a set of useful astronomical functions
    summarized as:
    
    * :func:`~nenupy.astro.astro.nenufar_loc`: NenuFAR Earth coordinates
    * :func:`~nenupy.astro.astro.lst`: Sidereal time
    * :func:`~nenupy.astro.astro.lha`: Local hour angle
    * :func:`~nenupy.astro.astro.wavelength`: Convert frequency to wavelength
    * :func:`~nenupy.astro.astro.ho_coord`: Define a :class:`~astropy.coordinates.AltAz` object
    * :func:`~nenupy.astro.astro.eq_coord`: Define a :class:`~astropy.coordinates.ICRS` object
    * :func:`~nenupy.astro.astro.to_radec`: Convert :class:`~astropy.coordinates.AltAz` to :class:`~astropy.coordinates.ICRS`
    * :func:`~nenupy.astro.astro.to_altaz`: Convert :class:`~astropy.coordinates.ICRS` to :class:`~astropy.coordinates.AltAz`
    * :func:`~nenupy.astro.astro.ho_zenith`: Get the local zenith in :class:`~astropy.coordinates.AltAz` coordinates
    * :func:`~nenupy.astro.astro.eq_zenith`: Get the local zenith in :class:`~astropy.coordinates.ICRS` coordinates
    * :func:`~nenupy.astro.astro.radio_sources`: Get main radio source poisitons
    * :func:`~nenupy.astro.astro.meridian_transit`: Next meridian transit time

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'nenufar_loc',
    'lst',
    'lha',
    'wavelength',
    'ho_coord',
    'eq_coord',
    'to_radec',
    'to_altaz',
    'ho_zenith',
    'eq_zenith',
    'radio_sources',
    'meridian_transit'
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

        :returns: :class:`~astropy.coordinates.EarthLocation`
            object
        :rtype: :class:`~astropy.coordinates.EarthLocation`

        :Example:
            >>> from nenupysim.astro import nenufar_loc
            >>> location = nenufar_loc()
    """
    # return EarthLocation( # old
    #     lat=47.375944 * u.deg,
    #     lon=2.193361 * u.deg,
    #     height=136.195 * u.m
    # )
    return EarthLocation(
        lat=47.376511 * u.deg,
        lon=2.192400 * u.deg,
        height=150 * u.m
    )

# ============================================================= #


# ============================================================= #
# ---------------------------- lst ---------------------------- #
# ============================================================= #
def lst(time):
    """ Local sidereal time

        :param time: Time
        :type time: :class:`~astropy.time.Time`

        :returns: LST time
        :rtype: :class:`~astropy.coordinates.Angle`
    """
    if not isinstance(time, Time):
        raise TypeError(
            'time is not an astropy Time.'
            )
    location = nenufar_loc()
    lon = location.to_geodetic().lon
    lst = time.sidereal_time('apparent', lon)
    return lst
# ============================================================= #


# ============================================================= #
# ---------------------------- lha ---------------------------- #
# ============================================================= #
def lha(time, ra):
    """ Local hour angle of an object in the observer's sky
        
        :param time: Time
        :type time: :class:`~astropy.time.Time`
        :param ra: Right Ascension
        :type ra: `float` or :class:`~astropy.coordinates.Angle` or :class:`~astropy.coordinates.Quantity`

        :returns: LHA time
        :rtype: :class:`~astropy.coordinates.Angle`
    """
    if not isinstance(ra, (u.Quantity, Angle)):
        ra = Angle(ra * u.deg)
    ha = lst(time) - ra
    twopi = Angle(360. * u.deg)
    if ha.isscalar:
        if ha.deg < 0:
            ha += twopi
        elif ha.deg > 360:
            ha -= twopi
    else:
        ha[ha.deg < 0] += twopi
        ha[ha.deg > 360] -= twopi
    return ha
# ============================================================= #


# ============================================================= #
# ------------------------ wavelength ------------------------- #
# ============================================================= #
def wavelength(freq):
    """ Convert radio frequency in wavelength.

        :param freq:
            Frequency (assumed in MHz unless a
            :class:`~astropy.units.Quantity` is provided)
        :type freq: `float`, :class:`~numpy.ndarray` or
            :class:`~astropy.units.Quantity`

        :returns: Wavelength in meters
        :rtype: :class:`~astropy.units.Quantity`
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    freq = freq.to(u.Hz)
    wavel = lspeed / freq
    return wavel.to(u.m)
# ============================================================= #


# ============================================================= #
# ------------------------- ho_coord -------------------------- #
# ============================================================= #
def ho_coord(alt, az, time):
    """ Horizontal coordinates
    
        :param alt:
            Altitude in degrees
        :type alt: `float` or :class:`~astropy.units.Quantity`
        :param az:
            Azimuth in degrees
        :type az: `float` or :class:`~astropy.units.Quantity`
        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`~astropy.time.Time`

        :returns: :class:`~astropy.coordinates.AltAz` object
        :rtype: :class:`~astropy.coordinates.AltAz`

        :Example:
            >>> from nenupysim.astro import ho_coord
            >>> altaz = ho_coord(
                    alt=45,
                    az=180,
                    time='2020-01-01 12:00:00'
                )
    """
    if not isinstance(az, u.Quantity):
        az *= u.deg
    if not isinstance(alt, u.Quantity):
        alt *= u.deg
    if not isinstance(time, Time):
        time = Time(time)
    return AltAz(
        az=az,
        alt=alt,
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
        :type ra: `float` or :class:`~astropy.units.Quantity`
        :param dec:
            Declination in degrees
        :type dec: `float` or :class:`~astropy.units.Quantity`

        :returns: :class:`~astropy.coordinates.ICRS` object
        :rtype: :class:`~astropy.coordinates.ICRS`

        :Example:
            >>> from nenupysim.astro import eq_coord
            >>> radec = eq_coord(
                    ra=51,
                    dec=39,
                )
    """
    if not isinstance(ra, u.Quantity):
        ra *= u.deg
    if not isinstance(dec, u.Quantity):
        dec *= u.deg
    return ICRS(
        ra=ra,
        dec=dec
    )
# ============================================================= #


# ============================================================= #
# ------------------------- to_radec -------------------------- #
# ============================================================= #
def to_radec(altaz):
    """ Transform altaz coordinates to ICRS equatorial system
        
        :param altaz:
            Horizontal coordinates
        :type altaz: :class:`~astropy.coordinates.AltAz`

        :returns: :class:`~astropy.coordinates.ICRS` object
        :rtype: :class:`~astropy.coordinates.ICRS`

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
        :type altaz: :class:`~astropy.coordinates.ICRS`
        :param time:
            Time at which the local coordinates should be 
            computed. It can either be provided as an 
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: `str`, :class:`~astropy.time.Time`

        :returns: :class:`~astropy.coordinates.AltAz` object
        :rtype: :class:`~astropy.coordinates.AltAz`

        :Example:
            >>> from nenupysim.astro import eq_coord
            >>> radec = eq_coord(
                    ra=51,
                    dec=39,
                )
    """
    if isinstance(radec, (ICRS, SkyCoord)):
        if isinstance(radec, SkyCoord):
            if not isinstance(radec.frame, ICRS):
                raise TypeError(
                    'frame should be ICRS'
                )
        pass
    else:
        raise TypeError(
            'ICRS or SkyCoord object expected.'
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
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: `str`, :class:`~astropy.time.Time`

        :returns: :class:`~astropy.coordinates.AltAz` object
        :rtype: :class:`~astropy.coordinates.AltAz`

        :Example:
            >>> from nenupysim.astro import ho_zenith
            >>> zen_altaz = ho_zenith(time='2020-01-01 12:00:00')
    """
    if not isinstance(time, Time):
        time = Time(time)
    if time.isscalar:
        return ho_coord(
            az=0.,
            alt=90.,
            time=time
        )
    else:
        return ho_coord(
            az=np.zeros(time.size),
            alt=np.ones(time.size) * 90.,
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
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: `str`, :class:`~astropy.time.Time`
        
        :returns: :class:`~astropy.coordinates.ICRS` object
        :rtype: :class:`~astropy.coordinates.ICRS`

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
    """ Main low-frequency radio source position in local
        coordinates frame at time ``time``.

        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: `str`, :class:`~astropy.time.Time`

        :returns:
            Dictionnary of radio source positions.
        :rtype: `dict`
    """
    if not isinstance(time, Time):
        time = Time(time)
    if not time.isscalar:
        raise ValueError(
            'Only scalar time allowed.'
        )

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


# ============================================================= #
# ----------------------- radio_sources ----------------------- #
# ============================================================= #
def meridian_transit(source, from_time, npoints=400):
    """ Find the next ``source``meridian transit time after the
        time ``from_time`` at NenuFAR location. This is a wrapper
        around the `astroplan` package and the dedicated function
        `target_meridian_transit_time()`.

        :param source:
            The fixed source instance to look for the transit.
            See also :func:`~nenupy.astro.astro.eq_coord` or 
            :meth:`~astropy.coordinates.SkyCoord.from_name`.
        :type source: :class:`~astropy.coordinates.SkyCoord`
        :param from_time:
            Time from which the next transit should be found.
        :type from_time: :class:`~astropy.time.Time`
        :param npoints:
            Number of points to look for the transit, the higher
            the more precise (but longer). ``400`` is a nice
            compromise.
        :type npoints: `int`
        
        :returns:
            Next meridian transit time of ``source``.
        :rtype: :class:`~astropy.time.Time`
    """
    from astroplan import Observer
    if not isinstance(source, SkyCoord):
        raise TypeError(
            'source must be a SkyCoord object'
        )
    if not isinstance(from_time, Time):
        raise TypeError(
            'from_time must be a Time object'
        )
    nenufar = Observer(
        name='NenuFAR',
        location=nenufar_loc()
    )
    transit = nenufar.target_meridian_transit_time(
        time=from_time,
        target=source,
        which='next',
        n_grid_points=npoints
    )
    return transit
# ============================================================= #

