#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **********************
    Astronomical Functions
    **********************

    Below are defined a set of useful astronomical functions
    summarized as:
    
    * :func:`~nenupy.astro.astro.lst`: Sidereal time
    * :func:`~nenupy.astro.astro.lha`: Local hour angle
    * :func:`~nenupy.astro.astro.wavelength`: Convert frequency to wavelength
    * :func:`~nenupy.astro.astro.ho_coord`: Define a :class:`~astropy.coordinates.AltAz` object
    * :func:`~nenupy.astro.astro.eq_coord`: Define a :class:`~astropy.coordinates.ICRS` object
    * :func:`~nenupy.astro.astro.to_radec`: Convert :class:`~astropy.coordinates.AltAz` to :class:`~astropy.coordinates.ICRS`
    * :func:`~nenupy.astro.astro.toAltaz`: Convert equatorial coordinates to horizontal coordinates
    * :func:`~nenupy.astro.astro.ho_zenith`: Get the local zenith in :class:`~astropy.coordinates.AltAz` coordinates
    * :func:`~nenupy.astro.astro.eq_zenith`: Get the local zenith in :class:`~astropy.coordinates.ICRS` coordinates
    * :func:`~nenupy.astro.astro.radio_sources`: Get main radio source positions
    * :func:`~nenupy.astro.astro.getSource`: Get a particular source coordinates
    * :func:`~nenupy.astro.astro.altazProfile`: Retrieve horizontal coordinates versus time
    * :func:`~nenupy.astro.astro.meridianTransit`: Next meridian transit time
    * :func:`~nenupy.astro.astro.dispersion_delay`: Dispersion delay induced by wave propagation through an electron plasma

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'lst',
    'lha',
    'toFK5',
    'wavelength',
    'ho_coord',
    'eq_coord',
    'to_radec',
    'toAltaz',
    'ho_zenith',
    'eq_zenith',
    'radio_sources',
    'getSource',
    'altazProfile',
    'meridianTransit',
    'dispersion_delay',
    '_normalizeEarthRadius',
    'l93_to_etrs',
    'geo_to_etrs',
    'etrs_to_geo',
    'etrs_to_enu',
    'enu_to_etrs'
    ]


import numpy as np
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    Angle,
    Longitude,
    SkyCoord,
    AltAz,
    Galactic,
    ICRS,
    FK5,
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
# ---------------------------- lst ---------------------------- #
# ============================================================= #
@misc.accepts(Time, str, strict=(False, True))
def lst(time, kind='fast'):
    """ Computes the Local Sidereal Time at the longitude of the
        NenuFAR radio telescope.
        Viewed from NenuFAR, a fixed celestial object seen at one
        position in the sky will be seen at the same position on
        another night at the same sidereal time. LST angle
        indicates the Right Ascension on the sky that is
        currently crossing the Local Meridian.

        :param time:
            UT Time to be converted
        :type time: :class:`~astropy.time.Time`
        :param kind:
            ``'fast'`` computes an approximation of local sidereal
            time, ``'mean'`` accounts for precession and ``'apparent'``
            accounts for precession and nutation.
        :type kind: str

        :returns: Local Sidereal Time angle
        :rtype: :class:`~astropy.coordinates.Longitude`

        :Example:
            >>> from nenupy.astro import lst
            >>> from astropy.time import Time
            >>> lst(time=Time('2020-01-01 12:00:00'), kind='fast')
            <Longitude 18.85380195 hourangle>
            >>> lst(time=Time('2020-01-01 12:00:00'), kind='mean')
            <Longitude 18.85375283 hourangle>
            >>> lst(time=Time('2020-01-01 12:00:00'), kind='apparent')
            <Longitude 18.85347225 hourangle>

    """
    if kind.lower() == 'fast':
        # http://www.roma1.infn.it/~frasca/snag/GeneralRules.pdf
        # Number of days since 2000 January 1, 12h UT
        nDays = time.jd - 2451545.
        # Greenwich mean sidereal time
        gmst = 18.697374558 + 24.06570982441908 * nDays
        gmst %= 24.
        # Local Sidereal Time
        lst = gmst + nenufar_loc.lon.hour
        if np.isscalar(lst):
            if lst < 0:
                lst += 24
        else:
            lst[lst < 0] += 24.   
        return Longitude(lst, 'hour')
    else:
        location = nenufar_loc
        lon = location.to_geodetic().lon
        lst = time.sidereal_time(kind, lon)
        return lst
# ============================================================= #


# ============================================================= #
# ---------------------------- lha ---------------------------- #
# ============================================================= #
@misc.accepts(Longitude, (ICRS, SkyCoord, FK5), strict=True)
def lha(lst, skycoord):
    r""" Local Hour Angle of an object in the observer's sky. It
        is defined as the angular distance on the celestial
        sphere measured westward along the celestial equator from
        the meridian to the hour circle passing through a point.

        The local hour angle :math:`h` is computed with respect
        to the local sidereal time ``lst`` :math:`t_{\rm LST}`
        and the ``skycoord`` astronomical source's right 
        ascension :math:`\alpha`:

        .. math::
            h = t_{\rm LST} - \alpha

        with the rule that if :math:`h < 0`, a :math:`2\pi` angle
        is added or if :math:`h > 2\pi`, a :math:`2\pi` angle
        is subtracted.
        
        :param lst:
            Local Sidereal Time, such as returned by
            :func:`~nenupy.astro.astro.lst` for instance.
        :type lst: :class:`~astropy.coordinates.Longitude`
        :param skycoord:
            Sky coordinates to convert to Local Hour Angles. This
            must be converted to FK5 coordinates with the
            corresponding equinox in order to give accurate
            results (see :func:`~nenupy.astro.astro.toFK5`).
        :type skycoord: :class:`~astropy.coordinates.SkyCoord`

        :returns: LHA time
        :rtype: :class:`~astropy.coordinates.Angle`

        :Example:
            >>> from nenupy.astro import lst, lha, toFK5
            >>> from astropy.time import Time
            >>> from astropy.coordinates import SkyCoord
            >>> utcTime = Time(['2020-01-01 12:00:00', '2020-01-01 13:00:00'])
            >>> lstTime = lst(
                    time=utcTime,
                    kind='apparent'
                )
            >>> casA = SkyCoord.from_name('Cas A')
            >>> casA_2020 = toFK5(
                    skycoord=casA,
                    time=utcTime
                )
            >>> lHA = lha(
                    lst=lstTime,
                    skycoord=casA_2020
                )
            >>> lHA
            [19h26m53.9458s 20h27m03.8018s]

    """
    if skycoord.equinox is None:
        log.warning(
            (
                'Given skycoord for LHA computation does not '
                'have an equinox attribute, make sure the '
                'precession is taken into account.'
            )
        )
    ha = lst - skycoord.ra
    twopi = Angle(360.000000 * u.deg)
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
# --------------------------- toFK5 --------------------------- #
# ============================================================= #
@misc.accepts((ICRS, SkyCoord, FK5), Time, strict=(True, False))
def toFK5(skycoord, time):
    """ Converts sky coordinates ``skycoord`` to FK5 system with
        equinox given by ``time``.

        :param skycoord:
            Sky Coordinates to be converted to FK5 system.
        :type skycoord: :class:`~astropy.coordinates.SkyCoord`
        :param time:
            Time that defines the equinox to be accounted for.
        :type time: :class:`~astropy.time.Time`

        :returns: FK5 sky coordinates
        :rtype: :class:`~astropy.coordinates.SkyCoord`
    """
    return skycoord.transform_to(
        FK5(equinox=time)
    )
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
        location=nenufar_loc,
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
# -------------------------- toAltaz -------------------------- #
# ============================================================= #
@misc.accepts((SkyCoord, ICRS), Time, str, strict=(True, False, True))
def toAltaz(skycoord, time, kind='normal'):
    r""" Convert a celestial object equatorial coordinates
        ``skycoord`` to horizontal coordinates as seen from
        NenuFAR's location at a given ``time``.

        If ``kind='fast'`` is selected the computation is
        accelerated using Local Sidereal Time approximation
        (see :func:`~nenupy.astro.astro.lst`). The altitude
        :math:`\theta` and azimuth :math:`\varphi` are computed
        as follows:

        .. math::
            \cases{
                \sin(\theta) = \sin(\delta) \sin(l) + \cos(\delta) \cos(l) \cos(h)\\
                \cos(\varphi) = \frac{\sin(\delta) - \sin(l) \sin(\theta)}{\cos(l)\cos(\varphi)}
            }

        with :math:`\delta` the object's declination, :math:`l`
        the NenuFAR's latitude and :math:`h` the Local Hour Angle
        (see :func:`~nenupy.astro.astro.lha`).
        If :math:`\sin(h) \geq 0`, then :math:`\varphi = 2 \pi - \varphi`.
        Otherwise, :meth:`~astropy.coordinates.SkyCoord.transform_to`
        is used.

        :param skycoord:
            Celestial object equatorial coordinates
        :type skycoord: :class:`~astropy.coordinates.SkyCoord`
        :param time:
            Coordinated universal time
        :type time: :class:`~astropy.time.Time`
        :param kind:
            ``'fast'`` enables faster computation time for the
            conversion, mainly relying on an approximation of the
            local sidereal time. All other values would lead to
            accurate coordinates computation. Differences in
            coordinates values are of the order of :math:`10^{-2}`
            degrees or less.
        :type kind: `str`

        :returns:
            Celestial object's horizontal coordinates
        :rtype: :class:`~astropy.coordinates.SkyCoord`

        :Example:
            >>> from nenupy.astro import toAltaz
            >>> from astropy.time import Time
            >>> from astropy.coordinates import SkyCoord

            >>> utcTime = Time('2020-10-07 11:16:49')
            >>> casA_radec = SkyCoord.from_name('Cas A')
            >>> casA_altaz = toAltaz(
                    skycoord=casA_radec,
                    time=utcTime,
                    kind='fast'
                )
            >>> print(casA_altaz.az.deg, casA_altaz.alt.deg)
            9.024164094317975 17.2063660579154

            >>> casA_altaz = toAltaz(
                    skycoord=casA_radec,
                    time=utcTime
                )
            >>> print(casA_altaz.az.deg, casA_altaz.alt.deg)
            9.018801267468616 17.206414428075465

            >>> utcTime = Time(['2020-01-01', '2020-01-02'])
            >>> casA_altaz = toAltaz(casA_radec, utcTime, 'fast')
            >>> print(casA_altaz.az.deg, casA_altaz.alt.deg)
            [326.15940811 326.56313916] [30.23939922 29.86967785]

    """
    altazFrame = AltAz(
        obstime=time,
        location=nenufar_loc
    )
    if kind.lower() == 'fast':
        lstTime = lst(
            time=time,
            kind='fast'
        )
        skycoord = toFK5(
            skycoord=skycoord,
            time=time
        )
        lHA = lha(
            lst=lstTime,
            skycoord=skycoord
        )

        decRad = skycoord.dec.rad
        haRad = lHA.rad
        latRad = nenufar_loc.lat.rad

        sinDec = np.sin(decRad)
        cosDec = np.cos(decRad)
        sinLat = np.sin(latRad)
        cosLat = np.cos(latRad)
        sinHa = np.sin(haRad)
        cosHa = np.cos(haRad)

        # Elevation
        sinAlt = sinDec * sinLat + cosDec * cosLat * cosHa
        altRad = np.arcsin(sinAlt)

        # Azimuth
        cosAz = (sinDec - sinLat * np.sin(altRad))/\
                (cosLat * np.cos(altRad))
        azRad = np.arccos(cosAz)
        if np.isscalar(altRad):
            if sinHa >= 0.:
                azRad = 2*np.pi - azRad
        else:
            posMask = sinHa >= 0.
            azRad[posMask] = 2*np.pi - azRad[posMask]

        return SkyCoord(
            azRad * u.rad,
            altRad * u.rad,
            frame=altazFrame
        )
    else:
        return skycoord.transform_to(altazFrame)
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
            nenufar_loc
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
        key: toAltaz(src_radec[key], time=time) for key in src_radec.keys()
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
                    obsgeoloc=[
                        (4237729.57730929,  917401.20083485, 4662142.24721379),
                        (3208307.85440129, 2913433.12950414, 4664141.44068417)
                    ] m,
                    obsgeovel=[
                        ( -66.89938865, 308.36222659, 0.13071683),
                        (-212.45197538, 233.29547849, 0.41179784)
                    ] m / s): (ra, dec, distance) in (deg, deg, AU)
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
            dec=[src['dec']] * nTime * u.deg,
            frame='icrs'
        )
        log.debug(
            f'Source {name} found in {sourceJson}.'
        )
    elif name.lower() in solarSystem:
        if not isinstance(time, Time):
            raise TypeError(
                'time should be a Time object'
            )
        with solar_system_ephemeris.set('builtin'):
            src = get_body(name, time, nenufar_loc)
            if src.ra.isscalar:
                src = SkyCoord(
                    ra=[src.ra.deg] * u.deg,
                    dec=[src.dec.deg] * u.deg
                )
            else:
                src = SkyCoord(
                    ra=src.ra,
                    dec=src.dec
                )
        log.debug(
            f'Source {name} found in Solar System Ephemeris.'
        )
    else:
        if time is not None:
            if not time.isscalar:
                nTime = time.size
            else:
                nTime = 1
        src = SkyCoord.from_name(name) # ICRS
        if time is not None:
            if not time.isscalar:
                nTime = time.size
        else:
            nTime = 1
        src = SkyCoord(
            ra=[src.ra.deg] * nTime * u.deg,
            dec=[src.dec.deg] * nTime * u.deg,
            frame='icrs'
        )
        log.debug(
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

        srcAltAz = toAltaz(
            skycoord=srcRaDec,
            time=tCurrent
        )

        times.append(tCurrent.isot)
        azimuths.append(srcAltAz.az.deg)
        elevations.append(srcAltAz.alt.deg)
        tCurrent += dt

    return Time(times), np.array(azimuths), np.array(elevations)
# ============================================================= #


# ============================================================= #
# ---------------------- meridianTransit ---------------------- #
# ============================================================= #
@misc.accepts(SkyCoord, Time, TimeDelta, str, strict=(True, False, False, True))
def meridianTransit(source, fromTime, duration=TimeDelta(1), kind='fast'):
    """ Find the ``source`` meridian transit time(s) since the
        time ``fromTime`` at NenuFAR location within a time period
        ``duration``.

        :param source:
            The source instance to look for the transit.
            See also :func:`~nenupy.astro.astro.eq_coord` or 
            :meth:`~astropy.coordinates.SkyCoord.from_name`.
        :type source: :class:`~astropy.coordinates.SkyCoord`
        :param fromTime:
            Time from which the next transit should be found.
        :type fromTime: :class:`~astropy.time.Time`
        :param duration:
            Duration to check transits since ``fromTime``
        :type duration: :class:`~astropy.time.TimeDelta`
        :param kind:
            Manner to compute the Local Sidereal Time, allowed
            values are ``'fast'``, ``'mean'`` and ``'apparent'``,
            see :func:`~nenupy.astro.astro.lst`.
        :type kind: str
        
        :returns:
            Next meridian transit time of ``source``.
        :rtype: :class:`~astropy.time.Time`
    """
    def _timeRange(tMin, tMax, dt):
        """
        """
        if not (isinstance(tMin, Time) and isinstance(tMax, Time)):
            raise TypeError(
                'tMin and tMax should be astropy.Time instances.'
            )
        if not isinstance(dt, TimeDelta):
            raise TypeError(
                'dt should be an astropy.TimeDelta instance.'
            )
        nTimes = (tMax - tMin) / dt
        times = tMin + np.arange(nTimes + 1) * dt
        return times

    def _jumpIndices(angleArray):
        """ ``angleArray`` is expected to contain increasing
            angular values.
            This function enables to detect the jumps between 360
            and 0 deg for instance (or 180 and -180 deg).
        """
        indices = np.where(
            (np.roll(angleArray, -1) - angleArray)[:-1] < 0
        )[0]
        return indices

    def _getLHA(times, source, kind):
        """
        """
        sourcefk5 = toFK5(
            skycoord=source,
            time=times
        )
        lstTime = lst(
            time=times,
            kind=kind
        )
        hourAngles = lha(
            lst=lstTime,
            skycoord=sourcefk5
        )
        return hourAngles

    transitTimes = []

    extDuration = TimeDelta(3600, format='sec')
    largeDt = TimeDelta(1800, format='sec')
    mediumDt = TimeDelta(60, format='sec')
    smallDt = TimeDelta(1, format='sec')

    # Broad search
    times = _timeRange(
        tMin=fromTime - extDuration,
        tMax=fromTime + duration + extDuration,
        dt=largeDt
    )
    hourAngles = _getLHA(times, source, kind)
    indices = _jumpIndices(hourAngles)
    
    # Iterate over the transit(s)
    for index in indices:
        # Medium search
        medTimes = _timeRange(
            tMin=times[index],
            tMax=times[index + 1],
            dt=mediumDt
        )
        hourAngles = _getLHA(medTimes, source, kind)
        medIndices = _jumpIndices(hourAngles)
        medIndex = medIndices[0]

        # Finest search
        smallTimes = _timeRange(
            tMin=medTimes[medIndex],
            tMax=medTimes[medIndex + 1],
            dt=smallDt
        )
        hourAngles = _getLHA(smallTimes, source, kind)
        smallIndices = _jumpIndices(hourAngles)

        transitTimes.append(
            (smallTimes[smallIndices[0]] + smallDt/2.).isot
        )

    return Time(transitTimes)
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
# ------------------------ l93_to_etrs ------------------------ #
# ============================================================= #
def l93_to_etrs(positions):
    """
    """
    from pyproj import Transformer
    t = Transformer.from_crs(
        crs_from='EPSG:2154', # RGF93
        crs_to='EPSG:4896'# ITRF2005 / ETRS used in MS
    )
    positions[:, 0], positions[:, 1], positions[:, 2] = t.transform(
        xx=positions[:, 0],
        yy=positions[:, 1],
        zz=positions[:, 2]
    )
    return positions
# ============================================================= #


# ============================================================= #
# ------------------- _normalizeEarthRadius ------------------- #
# ============================================================= #
def _normalizeEarthRadius(lat):
    """ Normalized radius of the WGS84 ellipsoid at a given latitude
        From https://github.com/brentjens/lofar-antenna-positions/blob/master/lofarantpos/geo.py
        lat in radians
    """
    wgs84_f = 1./298.257223563
    cosLat = np.cos(lat)
    sinlat = np.sin(lat)
    return 1./np.sqrt(cosLat**2 + ((1. - wgs84_f)**2) * (sinlat**2))
# ============================================================= #


# ============================================================= #
# ------------------------ geo_to_xyz ------------------------- #
# ============================================================= #
@misc.accepts(EarthLocation, strict=True)
def geo_to_etrs(earthlocation=nenufar_loc):
    """
    """
    gps_b = 6356752.31424518
    gps_a = 6378137
    e_squared = 6.69437999014e-3
    latRad = earthlocation.lat.rad
    lonRad = earthlocation.lon.rad
    alt = earthlocation.height.value
    if earthlocation.isscalar:
        xyz = np.zeros((1, 3))
    else:
        xyz = np.zeros((earthlocation.size, 3))
    gps_n = gps_a / np.sqrt(1 - e_squared * np.sin(latRad) ** 2)
    xyz[:, 0] = (gps_n + alt) * np.cos(latRad) * np.cos(lonRad)
    xyz[:, 1] = (gps_n + alt) * np.cos(latRad) * np.sin(lonRad)
    xyz[:, 2] = (gps_b**2/gps_a**2*gps_n + alt) * np.sin(latRad)
    return xyz
# ============================================================= #


# ============================================================= #
# ------------------------ etrs_to_geo ------------------------ #
# ============================================================= #
@misc.accepts(np.ndarray, strict=True)
def etrs_to_geo(positions):
    """
    """
    assert (len(positions.shape)==2) and positions.shape[1]==3,\
        'positions should be an array of shape (n, 3)'

    wgs84_a = 6378137.0
    wgs84_f = 1./298.257223563
    wgs84_e2 = wgs84_f*(2.0 - wgs84_f)
    
    x, y, z = np.transpose(positions)
    lonRad = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    # Iterate to latitude solution
    phi_previous = 1e4
    phi = np.arctan2(z, r)
    while (np.abs(phi - phi_previous) > 1.6e-12).any():
        phi_previous = phi
        earthRadius = _normalizeEarthRadius(phi)
        sinPhi = np.sin(phi)
        phi = np.arctan2(
            z + wgs84_e2*wgs84_a*earthRadius*sinPhi,
            r
        )
    latRad = phi
    cosLat = np.cos(latRad)
    sinLat = np.sin(latRad)
    heightM = r*cosLat + z*sinLat - wgs84_a*np.sqrt(1. - wgs84_e2*sinLat**2)

    return EarthLocation(
        lon=lonRad*u.rad,
        lat=latRad*u.rad,
        height=heightM*u.m
    )
# ============================================================= #


# ============================================================= #
# ------------------------ etrs_to_enu ------------------------ #
# ============================================================= #
@misc.accepts(np.ndarray, EarthLocation, strict=True)
def etrs_to_enu(positions, earthlocation=nenufar_loc):
    r""" Local east, north, up (ENU) coordinates centered on the 
        position ``earthlocation`` (default is at the location of
        NenuFAR).

        The conversion from cartesian coordinates :math:`(x, y, z)`
        to ENU :math:`(e, n, u)` is done as follows:

        .. math::
                \pmatrix{
                    e \\
                    n \\
                    u
                } =
                \pmatrix{
                    -\sin(b) & \cos(l) & 0\\
                    -\sin(l) \cos(b) & -\sin(l) \sin(b) & \cos(l)\\
                    \cos(l)\cos(b) & \cos(l) \sin(b) & \sin(l)
                }
                \pmatrix{
                    \delta x\\
                    \delta y\\
                    \delta z
                }

        where :math:`b` is the longitude, :math:`l` is the
        latitude and :math:`(\delta x, \delta y, \delta z)` are
        the cartesian coordinates with respect to the center
        ``earthlocation``.
    """
    assert (len(positions.shape)==2) and positions.shape[1]==3,\
        'positions should be an array of shape (n, 3)'
    xyz = positions.copy()
    xyzCenter = geo_to_etrs(earthlocation)
    xyz -= xyzCenter

    cosLat = np.cos(earthlocation.lat.rad)
    sinLat = np.sin(earthlocation.lat.rad)
    cosLon = np.cos(earthlocation.lon.rad)
    sinLon = np.sin(earthlocation.lon.rad)
    transformation = np.array([
        [       -sinLon,          cosLon,      0],
        [-sinLat*cosLon, - sinLat*sinLon, cosLat],
        [ cosLat*cosLon,   cosLat*sinLon, sinLat]
    ])

    return np.matmul(xyz, transformation.T)
# ============================================================= #


# ============================================================= #
# ------------------------ enu_to_etrs ------------------------ #
# ============================================================= #
@misc.accepts(np.ndarray, EarthLocation, strict=True)
def enu_to_etrs(positions, earthlocation=nenufar_loc):
    """
    """
    assert (len(positions.shape)==2) and positions.shape[1]==3,\
        'positions should be an array of shape (n, 3)'
    enu = positions.copy()

    cosLat = np.cos(earthlocation.lat.rad)
    sinLat = np.sin(earthlocation.lat.rad)
    cosLon = np.cos(earthlocation.lon.rad)
    sinLon = np.sin(earthlocation.lon.rad)
    transformation = np.array([
        [       -sinLon,          cosLon,      0],
        [-sinLat*cosLon, - sinLat*sinLon, cosLat],
        [ cosLat*cosLon,   cosLat*sinLon, sinLat]
    ])

    xyz = np.matmul(enu, np.linalg.inv(transformation).T)
    xyzCenter = geo_to_etrs(earthlocation)
    xyz += xyzCenter

    return xyz
# ============================================================= #

