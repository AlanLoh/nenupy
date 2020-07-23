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
    * :func:`~nenupy.astro.astro.radio_sources`: Get main radio source positions
    * :func:`~nenupy.astro.astro.getSource`: Get a particular source coordinates
    * :func:`~nenupy.astro.astro.altazProfile`: Retrieve horizontal coordinates versus time
    * :func:`~nenupy.astro.astro.meridian_transit`: Next meridian transit time
    * :func:`~nenupy.astro.astro.dispersion_delay`: Dispersion delay induced by wave propagation through an electron plasma

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
    'getSource',
    'altazProfile',
    'meridian_transit',
    'dispersion_delay',
    '_pos2ecef',
    '_ecef2enu'
    ]


import numpy as np
from astropy.time import Time, TimeDelta
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
from os.path import dirname, join
import json

from nenupy.instru import nenufar_loc
import nenupy.miscellaneous as misc

import logging
log = logging.getLogger(__name__)


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
@misc.time_args('time')
def lst(time):
    """ Local sidereal time

        :param time: Time
        :type time: :class:`~astropy.time.Time`

        :returns: LST time
        :rtype: :class:`~astropy.coordinates.Angle`
    """
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
        # if isinstance(radec, SkyCoord):
        #     if not isinstance(radec.frame, ICRS):
        #         raise TypeError(
        #             'frame should be ICRS'
        #         )
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
# ------------------------- getSource ------------------------- #
# ============================================================= #
def getSource(name, time=None):
    """ Retrieve a source equatorial coordinates.

        :param name:
            Source name.
        :type name: `str`
        :param time:
            Time at which the coordinates need to be calculated.
            This is particularly relevant for Solar System objects.
            Moreover, if ``time`` is not a scalar, the returned
            source coordinates will have dimensions matching the
            ``time`` argument.
        :type time: :class:`~astropy.time.Time`

        :returns:
            Source equatorial coordinates.
        :rtype: :class:`~astropy.coordinates.SkyCoord`

        :Example:
            >>> from nenupy.astro import getSource
            >>> from astropy.time import Time
            >>> getSource('Cyg X-3')
                <SkyCoord (ICRS): (ra, dec) in deg
                [(308.10741667, 40.95775)]>
            >>> getSource('Sun', Time(['2020-04-01 12:00:00', '2020-04-01 14:00:00']))
                <SkyCoord (GCRS:
                    obstime=['2020-04-01 12:00:00.000' '2020-04-01 14:00:00.000'],
                    obsgeoloc=[(4237729.57730929,  917401.20083485, 4662142.24721379),
                    (3208307.85440129, 2913433.12950414, 4664141.44068417)] m,
                    obsgeovel=[( -66.89938865, 308.36222659, 0.13071683),
                    (-212.45197538, 233.29547849, 0.41179784)] m / s): (ra, dec, distance) in (deg, deg, AU)
                    [(10.983783  , 4.71984404, 0.99938543),
                     (11.05889498, 4.75192569, 0.99941305)]>

        .. versionadded:: 1.1.0
    """
    sourceJson = join(
        dirname(__file__),
        'radio_sources.json',
    )
    with open(sourceJson) as jsonFile:
        sources = json.load(jsonFile)

    solarSystem = [
        'sun',
        'moon',
        'jupiter',
        'saturn',
        'mars',
        'venus',
        'uranus',
        'neptune'
    ]

    nTime = 1

    if name.upper() in sources.keys():
        if time is not None:
            if not time.isscalar:
                nTime = time.size
            else:
                nTime = 1
        src = sources[name.upper()]
        src = SkyCoord(
            ra=[src['ra']] * nTime * u.deg,
            dec=[src['dec']] * nTime * u.deg
        )
        log.info(
            f'Source {name} found in {sourceJson}.'
        )
    elif name.lower() in solarSystem:
        if not isinstance(time, Time):
            raise TypeError(
                'time should be a Time object'
            )
        with solar_system_ephemeris.set('builtin'):
            src = get_body(name, time, nenufar_loc())
        log.info(
            f'Source {name} found in Solar System Ephemeris.'
        )
    else:
        if time is not None:
            if not time.isscalar:
                nTime = time.size
            else:
                nTime = 1
        src = SkyCoord.from_name(name)
        if time is not None:
            if not time.isscalar:
                nTime = time.size
        else:
            nTime = 1
        src = SkyCoord(
            ra=[src.ra.deg] * nTime * u.deg,
            dec=[src.dec.deg] * nTime * u.deg
        )
        log.info(
            f'Source {name} found in Simbad.'
        )

    return src
# ============================================================= #


# ============================================================= #
# ----------------------- altazProfile ------------------------ #
# ============================================================= #
def altazProfile(sourceName, tMin=None, tMax=None, dt=None):
    """
    """
    if tMin is None:
        tMin = Time.now()
    elif not isinstance(tMin, Time):
        raise TypeError(
            'tMin should be a Time object'
        )

    if tMax is None:
        tMax = tMin + TimeDelta(24*3600, format='sec')
    elif not isinstance(tMax, Time):
        raise TypeError(
            'tMax should be a Time object'
        )

    if dt is None:
        dt = TimeDelta(5*60, format='sec')
    elif not isinstance(dt, TimeDelta):
        raise TypeError(
            'dt should be a TimeDelta object'
        )

    tCurrent = tMin
    times = []
    elevations = []
    azimuths = []
    while tCurrent <= tMax:
        if 'srcRaDec' not in locals():
            srcRaDec = getSource(
                name=sourceName,
                time=tCurrent
            )
        elif srcRaDec.frame.name == 'gcrs':
            # Solar system source, need to be updated/time
            srcRaDec = getSource(
                name=sourceName,
                time=tCurrent
            )
        else:
            # Do not ask again for same source
            pass

        srcAltAz = to_altaz(
            radec=srcRaDec,
            time=tCurrent
        )

        times.append(tCurrent.isot)
        azimuths.append(srcAltAz.az.deg)
        elevations.append(srcAltAz.alt.deg)
        tCurrent += dt

    return Time(times), np.array(azimuths), np.array(elevations)
# ============================================================= #


# ============================================================= #
# --------------------- meridian_transit ---------------------- #
# ============================================================= #
@misc.time_args('from_time')
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


# ============================================================= #
# --------------------- dispersion_delay ---------------------- #
# ============================================================= #
def dispersion_delay(freq, dm):
    r""" Dispersion delay induced to a radio wave of frequency
        ``freq`` (:math:`\nu`) propagating through an electron
        plasma of uniform density :math:`n_e`.
        
        The pulse travel time :math:`\Delta t_p` emitted at a
        distance :math:`d` is:

        .. math::
            \Delta t_p = \frac{d}{c} + \frac{e^2}{2\pi m_e c} \frac{\int_0^d n_e\, dl}{\nu^2}
    
        where :math:`\mathcal{D}\mathcal{M} = \int_0^d n_e\, dl`
        is the *Dispersion Measure*.

        Therefore, the time delay :math:`\Delta t_d` due to the
        dispersion is:

        .. math::
            \Delta t_d = \frac{e^2}{2 \pi m_e c} \frac{\mathcal{D}\mathcal{M}}{\nu^2} 

        and computed as:

        .. math::
            \Delta t_d = 4140 \left( \frac{\mathcal{D}\mathcal{M}}{\rm{pc}\,\rm{cm}^{-3}} \right) \left( \frac{\nu}{1\, \rm{MHz}} \right)^{-2}\, \rm{sec}

        :param freq:
            Observation frequency (assumed to be in MHz if no
            unit is provided).
        :type freq: `float` or :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
        :param dm:
            Dispersion Measure (assumed to be in pc/cm^-3 if no
            unit is provided.
        :type dm: `float` or :class:`~astropy.units.Quantity`

        :returns: Dispersion delay in seconds.
        :rtype: :class:`~astropy.units.Quantity`

        :Example:
            >>> from nenupy.astro import dispersion_delay
            >>> import astropy.units as u
            >>> dispersion_delay(
                    freq=50*u.MHz,
                    dm=12.4*u.pc/(u.cm**3)
                )
            20.5344 s

        .. versionadded:: 1.1.0
    """
    if not isinstance(dm, u.Quantity):
        dm *= u.pc / (u.cm**3)
    else:
        dm = dm.to(u.pc / (u.cm**3))
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    else:
        freq = freq.to(u.MHz)
    dm_ref = 1. * u.pc / (u.cm**3)
    freq_ref = 1. * u.MHz
    delay = 4140. * (dm/dm_ref) / ((freq/freq_ref)**2) * u.s
    return delay
# ============================================================= #


# ============================================================= #
# ------------------------- _pos2ecef ------------------------- #
# ============================================================= #
def _pos2ecef(earthLocation=None):
    """
    """
    return
#     if earthLocation is None:
#         earthLocation = nenufar_loc
#     if not isinstance(earthLocation, EarthLocation):
#         raise TypeError(
#             'earthLocation should be a EarthLocation instance.'
#         )
#     lon = earthLocation.lon.rad
#     lat = earthLocation.lat.rad, 
#     alt = earthLocation.height.value
#     gps_b = 6356752.31424518
#     gps_a = 6378137
#     e_squared = 6.69437999014e-3
#     gps_n = gps_a / np.sqrt(1 - e_squared * np.sin(lat)**2)
#     xyz = np.zeros((3))
#     xyz[0] = (gps_n + alt) * np.cos(lat) * np.cos(lon)
#     xyz[1] = (gps_n + alt) * np.cos(lat) * np.sin(lon)
#     xyz[2] = (gps_b**2 / gps_a**2 * gps_n + alt) * np.sin(lat)
#     return xyz
# ============================================================= #


# ============================================================= #
# ------------------------- _ecef2enu ------------------------- #
# ============================================================= #
def _ecef2enu(ecef, earthLocation=None):
    """
    """
    return
#     xyz = np.array(xyz)
#     if xyz.ndim > 1 and xyz.shape[1] != 3:
#         raise ValueError("The expected shape of ECEF xyz array is (Npts, 3).")
#     xyz_in = xyz

#     if xyz_in.ndim == 1:
#         xyz_in = xyz_in[np.newaxis, :]
# ============================================================= #


# gps_b = 6356752.31424518
# gps_a = 6378137
# e_squared = 6.69437999014e-3

# def XYZ_from_LatLonAlt(latitude, longitude, altitude):
#     """
#     from pyuvdata
#     Calculate ECEF x,y,z from lat/lon/alt values.

#     Parameters
#     ----------
#     latitude :  ndarray or float
#         latitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
#     longitude :  ndarray or float
#         longitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
#     altitude :  ndarray or float
#         altitude, numpy array (if Npts > 1) or value (if Npts = 1) in meters

#     Returns
#     -------
#     xyz : ndarray of float
#         numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.

#     """
#     latitude = np.array(latitude)
#     longitude = np.array(longitude)
#     altitude = np.array(altitude)
#     n_pts = latitude.size
#     if longitude.size != n_pts:
#         raise ValueError(
#             "latitude, longitude and altitude must all have the same length"
#         )
#     if altitude.size != n_pts:
#         raise ValueError(
#             "latitude, longitude and altitude must all have the same length"
#         )

#     # see wikipedia geodetic_datum and Datum transformations of
#     # GPS positions PDF in docs/references folder
#     gps_n = gps_a / np.sqrt(1 - e_squared * np.sin(latitude) ** 2)
#     xyz = np.zeros((n_pts, 3))
#     xyz[:, 0] = (gps_n + altitude) * np.cos(latitude) * np.cos(longitude)
#     xyz[:, 1] = (gps_n + altitude) * np.cos(latitude) * np.sin(longitude)
#     xyz[:, 2] = (gps_b ** 2 / gps_a ** 2 * gps_n + altitude) * np.sin(latitude)

#     xyz = np.squeeze(xyz)
#     return xyz


# def ENU_from_ECEF(xyz, latitude, longitude, altitude):
#     """
#     from pyuvdata
#     Calculate local ENU (east, north, up) coordinates from ECEF coordinates.

#     Parameters
#     ----------
#     xyz : ndarray of float
#         numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
#     latitude : float
#         Latitude of center of ENU coordinates in radians.
#     longitude : float
#         Longitude of center of ENU coordinates in radians.
#     altitude : float
#         Altitude of center of ENU coordinates in radians.

#     Returns
#     -------
#     ndarray of float
#         numpy array, shape (Npts, 3), with local ENU coordinates

#     """
#     xyz = np.array(xyz)
#     if xyz.ndim > 1 and xyz.shape[1] != 3:
#         raise ValueError("The expected shape of ECEF xyz array is (Npts, 3).")

#     xyz_in = xyz

#     if xyz_in.ndim == 1:
#         xyz_in = xyz_in[np.newaxis, :]

#     # check that these are sensible ECEF values -- their magnitudes need to be
#     # on the order of Earth's radius
#     ecef_magnitudes = np.linalg.norm(xyz_in, axis=1)
#     sensible_radius_range = (6.35e6, 6.39e6)
#     if np.any(ecef_magnitudes <= sensible_radius_range[0]) or np.any(
#         ecef_magnitudes >= sensible_radius_range[1]
#     ):
#         raise ValueError(
#             "ECEF vector magnitudes must be on the order of the radius of the earth"
#         )

#     xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)

#     xyz_use = np.zeros_like(xyz_in)
#     xyz_use[:, 0] = xyz_in[:, 0] - xyz_center[0]
#     xyz_use[:, 1] = xyz_in[:, 1] - xyz_center[1]
#     xyz_use[:, 2] = xyz_in[:, 2] - xyz_center[2]

#     enu = np.zeros_like(xyz_use)
#     enu[:, 0] = -np.sin(longitude) * xyz_use[:, 0] + np.cos(longitude) * xyz_use[:, 1]
#     enu[:, 1] = (
#         -np.sin(latitude) * np.cos(longitude) * xyz_use[:, 0]
#         - np.sin(latitude) * np.sin(longitude) * xyz_use[:, 1]
#         + np.cos(latitude) * xyz_use[:, 2]
#     )
#     enu[:, 2] = (
#         np.cos(latitude) * np.cos(longitude) * xyz_use[:, 0]
#         + np.cos(latitude) * np.sin(longitude) * xyz_use[:, 1]
#         + np.sin(latitude) * xyz_use[:, 2]
#     )
#     if len(xyz.shape) == 1:
#         enu = np.squeeze(enu)

#     return enu
