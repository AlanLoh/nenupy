#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ******************
    Astronomical tools
    ******************
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
    "altaz_to_radec",
    "sky_temperature",
    "dispersion_delay",
    "wavelength",
    "l93_to_etrs",
    "geo_to_etrs",
    "etrs_to_enu",
    "AstroObject"
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
    Latitude,
    AltAz,
    FK5,
    ICRS
)
from pyproj import Transformer

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
        return gmst + observer.lon

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
    r""" Local Hour Angle of an object in the ``observer``'s sky. It
        is defined as the angular distance on the celestial
        sphere measured westward along the celestial equator from
        the meridian to the hour circle passing through a point.

        The local hour angle :math:`h` is computed with respect
        to the local sidereal time :math:`t_{\rm LST}`
        and the astronomical source (defined as ``radec``) right 
        ascension :math:`\alpha`:

        .. math::
            h = t_{\rm LST} - \alpha

        with the rule that if :math:`h < 0`, a :math:`2\pi` angle
        is added or if :math:`h > 2\pi`, a :math:`2\pi` angle
        is subtracted.

        :param radec:
            Sky coordinates to convert to Local Hour Angles.
        :type radec:
            :class:`~astropy.coordinates.SkyCoord`
        :param time:
            UTC time a which the hour angle is computed.
        :type time:
            :class:`~astropy.time.Time`
        :param observer:
            Earth location where the observer is at.
            Default is NenuFAR's position.
        :type observer:
            :class:`~astropy.coordinates.EarthLocation`
        :param fast_compute:
            If set to ``True``, an approximation is made while 
            computing the local sidereal time.
            Default is ``True``.
        :type fast_compute:
            `bool`

        :returns: LHA time
        :rtype: :class:`~astropy.coordinates.Longitude`

        :Example:
            >>> blabla

    """
    lst = local_sidereal_time(
        time=time,
        observer=observer,
        fast_compute=fast_compute
    )
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
    if not time.isscalar:
        time = time.reshape((time.size, 1))
        radec = radec.reshape((1, radec.size))

    if fast_compute:
        radec = radec.transform_to(
            FK5(equinox=time)
        )

        ha = hour_angle(
            radec=radec,
            time=time,
            observer=observer,
            fast_compute=fast_compute
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
        sin_elevation = sin_dec*sin_lat + cos_dec*cos_lat*np.cos(ha.rad)
        elevation = Latitude(np.arcsin(sin_elevation), unit="rad")

        # Compute azimuth
        cos_azimuth = (sin_dec - np.sin(elevation.rad)*sin_lat)/\
            (np.cos(elevation.rad)*cos_lat)
        azimuth = Longitude(np.arccos(cos_azimuth), unit="rad")

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
# ---------------------- altaz_to_radec ----------------------- #
# ============================================================= #
def altaz_to_radec(
        altaz: SkyCoord,
        fast_compute: bool = False
    ) -> SkyCoord:
    """ """
    if fast_compute: # does not work perfectly, don't know why...
        sin_lat = np.sin(altaz.location.lat.rad)
        cos_lat = np.cos(altaz.location.lat.rad)
        
        cos_az = np.cos(altaz.az.rad)
        sin_az = np.sin(altaz.az.rad)
        
        sin_alt = np.sin(altaz.alt.rad)
        cos_alt = np.cos(altaz.alt.rad)
        
        sin_dec = sin_lat*sin_alt + cos_lat*cos_alt*cos_az
        dec = np.arcsin(sin_dec)
        cos_dec = np.cos(dec)
        
        hour_angle = np.arccos( (sin_alt - sin_lat*sin_dec)/(cos_lat*cos_dec) )

        lst = local_sidereal_time(
            time=altaz.obstime,
            observer=altaz.location,
            fast_compute=True
        )
        mask = np.sin(altaz.az.rad) > 0.
        hour_angle[mask] -= np.radians(360.)
        hour_angle[mask] *= -1
        ra = lst.deg - np.degrees(hour_angle)
        return SkyCoord(ra*u.deg, Latitude(dec, unit="rad"))
    else:
        return altaz.transform_to(
            ICRS
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- sky_temperature ---------------------- #
# ============================================================= #
def sky_temperature(frequency: u.Quantity = 50*u.MHz) -> u.Quantity:
    r""" Sky temperature at a given frequency ``freq`` (strongly
        dominated by Galactic emission).

        .. math::
            T_{\rm sky} = T_0 \lambda^{2.55}

        with :math:`T_0 = 60 \pm 20\,\rm{K}` for Galactic
        latitudes between 10 and 90 degrees.

        :param frequency:
            Frequency at which computing the sky temperature.
            Default is ``50 MHz``.
        :type frequency:
            :class:`~astropy.units.Quantity`

        :returns:
            Sky temperature in Kelvins
        :rtype:
            :class:`~astropy.units.Quantity`

        .. seealso::
            `LOFAR website <http://old.astron.nl/radio-observatory/astronomers/lofar-imaging-capabilities-sensitivity/sensitivity-lofar-array/sensiti>`_, 
            Haslam et al. (1982) and Mozdzen et al. (2017, 2019)
    """
    wavelength = frequency.to(
        u.m,
        equivalencies=u.spectral()
    ).value
    t0 = 60. * u.K
    t_sky = t0 * wavelength**2.55
    return t_sky
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- dispersion_delay ---------------------- #
# ============================================================= #
def dispersion_delay(frequency, dispersion_measure):
    r""" Dispersion delay induced to a radio wave of ``frequency``
        (:math:`\nu`) propagating through an electron
        plasma of uniform density :math:`n_e`.
        
        The pulse travel time :math:`\Delta t_p` emitted at a
        distance :math:`d` is:

        .. math::
            \Delta t_p = \frac{d}{c} + \frac{e^2}{2\pi m_e c} \frac{\int_0^d n_e\, dl}{\nu^2}
    
        where :math:`\mathcal{D}\mathcal{M} = \int_0^d n_e\, dl`
        is the *Dispersion Measure* (``dispersion_measure``).
        Therefore, the time delay :math:`\Delta t_d` due to the
        dispersion is:

        .. math::
            \Delta t_d = \frac{e^2}{2 \pi m_e c} \frac{\mathcal{D}\mathcal{M}}{\nu^2} 

        and computed as:

        .. math::
            \Delta t_d = 4140 \left( \frac{\mathcal{D}\mathcal{M}}{\rm{pc}\,\rm{cm}^{-3}} \right) \left( \frac{\nu}{1\, \rm{MHz}} \right)^{-2}\, \rm{sec}

        :param frequency:
            Observation frequency.
        :type frequency:
            :class:`~astropy.units.Quantity`
        :param dispersion_measure:
            Dispersion Measure (in units equivalent to :math:`{\rm pc}/{\rm cm}^3`
        :type dispersion_measure:
            :class:`~astropy.units.Quantity`

        :returns: Dispersion delay in seconds.
        :rtype: :class:`~astropy.units.Quantity`

        :Example:
            >>> from nenupy.astro import dispersion_delay
            >>> import astropy.units as u
            >>> dispersion_delay(
                    frequency=50*u.MHz,
                    dispersion_measure=12.4*u.pc/(u.cm**3)
                )
            20.5344 s

    """
    dispersion_measure = dispersion_measure.to(u.pc / (u.cm**3))
    frequency = frequency.to(u.MHz)
    dm_ref = 1. * u.pc / (u.cm**3)
    freq_ref = 1. * u.MHz
    delay = 4140. * (dispersion_measure/dm_ref) / ((frequency/freq_ref)**2) * u.s
    return delay
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ wavelength ------------------------- #
# ============================================================= #
def wavelength(frequency: u.Quantity = 50*u.MHz):
    r""" Converts an electromagnetic frequency to a wavelength.

        .. math::
            \lambda = \frac{c}{\nu}
        
        where :math:`\lambda` is the wavelength, :math:`c` is the
        speed of light and :math:`\nu` is the frequency.

        :param frequency:
            Frequency to convert in wavelength.
        :type frequency:
            :class:`~astropy.units.Quantity`
        
        :returns:
            Wavelength in meters.
        :rtype:
            :class:`~astropy.units.Quantity`
        
        :Example:
            >>> from nenupy.astro import wavelength
            >>> import astropy.units as u
            >>> wavelength(frequency=10*u.MHz)
            29.979246 m

    """
    return frequency.to(u.m, equivalencies=u.spectral())
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ l93_to_etrs ------------------------ #
# ============================================================= #
def l93_to_etrs(positions= np.ndarray):
    """
    """
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


# ============================================================= #
# ------------------------ geo_to_etrs ------------------------ #
# ============================================================= #
def geo_to_etrs(location: EarthLocation = nenufar_position) -> np.ndarray:
    """
    """
    gps_b = 6356752.31424518
    gps_a = 6378137
    e_squared = 6.69437999014e-3
    lat_rad = location.lat.rad
    lon_rad = location.lon.rad
    alt = location.height.value
    if location.isscalar:
        xyz = np.zeros((1, 3))
    else:
        xyz = np.zeros((location.size, 3))
    gps_n = gps_a / np.sqrt(1 - e_squared * np.sin(lat_rad) ** 2)
    xyz[:, 0] = (gps_n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    xyz[:, 1] = (gps_n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    xyz[:, 2] = (gps_b**2/gps_a**2*gps_n + alt) * np.sin(lat_rad)
    return xyz
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ etrs_to_enu ------------------------ #
# ============================================================= #
def etrs_to_enu(positions: np.ndarray, location: EarthLocation = nenufar_position) -> np.ndarray:
    r""" Local east, north, up (ENU) coordinates centered on the 
        position ``location`` (default is at the location of
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
        ``location``.
    """
    assert (len(positions.shape)==2) and positions.shape[1]==3,\
        'positions should be an array of shape (n, 3)'
    xyz = positions.copy()
    xyz_center = geo_to_etrs(location)
    xyz -= xyz_center

    cos_lat = np.cos(location.lat.rad)
    sin_lat = np.sin(location.lat.rad)
    cos_lon = np.cos(location.lon.rad)
    sin_lon = np.sin(location.lon.rad)
    transformation = np.array([
        [        -sin_lon,           cos_lon,       0],
        [-sin_lat*cos_lon, - sin_lat*sin_lon, cos_lat],
        [ cos_lat*cos_lon,   cos_lat*sin_lon, sin_lat]
    ])

    return np.matmul(xyz, transformation.T)
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
    frequency: u.Quantity
    polarization: np.ndarray
    value: Union[
        float,
        int,
        np.ndarray
    ]
    observer: EarthLocation


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def frequency(self):
        """ test de doc """
        return self._frequency
    @frequency.setter
    def frequency(self, f):
        self._frequency = f


    @property
    def horizontal_coordinates(self) -> SkyCoord:
        """ """
        # if not hasattr(self, "_computed_ho_coord"):
        #     # Only do that once
        #     self._computed_ho_coord = getattr(
        #         self,
        #         "custom_ho_coordinates",
        #         radec_to_altaz(
        #             radec=self.coordinates,
        #             time=self.time,
        #             observer=self.observer,
        #             fast_compute=True
        #         )
        #     )
        # return self._computed_ho_coord
        return getattr(
            self,
            "custom_ho_coordinates",
            radec_to_altaz(
                radec=self.coordinates,
                time=self.time,
                observer=self.observer,
                fast_compute=True
            )
        )


    @property
    def custom_ho_coordinates(self) -> SkyCoord:
        """ Allows to modify horizontal coordinates without messing up with the actual coordinates object. """
        return self._custom_ho_coordinates
    @custom_ho_coordinates.setter
    def custom_ho_coordinates(self, coord: SkyCoord):
        self._custom_ho_coordinates = coord


    @property
    def ground_projection(self):
        """ """
        altaz = self.horizontal_coordinates
        if altaz.ndim == 1:
            altaz = altaz.reshape((altaz.size, 1))

        az_rad = altaz.az.rad
        alt_rad = altaz.alt.rad

        cos_az = np.cos(az_rad)
        sin_az = np.sin(az_rad)
        cos_alt = np.cos(alt_rad)
        sin_alt = np.sin(alt_rad)

        ground_proj = np.array(
            [
                cos_az*cos_alt,
                sin_az*cos_alt,
                sin_alt
            ]
        )

        # Reshape to put the time axis in front
        ground_proj = np.moveaxis(ground_proj, -2, 0)
        # Add frequency and polarization dimensions
        ground_proj = np.expand_dims(
            ground_proj,
            axis=(1, 2)
        )

        # return ground_proj
        visible_mask = altaz.alt.deg < 0.
        visible_mask = np.repeat(visible_mask[:, None, :], 3, axis=1)
        visible_mask = np.expand_dims(
            visible_mask,
            axis=(1, 2)
        )
        return np.ma.masked_array(ground_proj, mask=visible_mask)


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
