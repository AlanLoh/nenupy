#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ****************
    Celestial Target
    ****************

    .. inheritance-diagram:: nenupy.astro.target.FixedTarget nenupy.astro.target.SolarSystemTarget
        :parts: 3

    .. autosummary::

        ~FixedTarget
        ~SolarSystemTarget

"""


from __future__ import annotations


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "Target",
    "FixedTarget",
    "SolarSystemTarget"
]


from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import logging
log = logging.getLogger(__name__)

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, FK5, AltAz

from nenupy import nenufar_position
from nenupy.astro import common_sources
from nenupy.astro.astro_tools import (
    AstroObject,
    hour_angle,
    solar_system_source
)


# ============================================================= #
# -------------------------- Target --------------------------- #
# ============================================================= #
class Target(AstroObject, ABC):
    """ Abstract class to handle target objects.

        .. versionadded:: 2.0.0

        .. rubric:: Attributes Summary

        .. autosummary::

            ~Target.coordinates
            ~Target.time
            ~Target.observer
            ~Target.is_circumpolar
            ~Target.culmination_azimuth

        .. rubric:: Methods Summary

        .. autosummary::

            ~Target.meridian_transit
            ~Target.next_meridian_transit
            ~Target.previous_meridian_transit
            ~Target.azimuth_transit
            ~Target.rise_time
            ~Target.next_rise_time
            ~Target.previous_rise_time
            ~Target.set_time
            ~Target.next_set_time
            ~Target.previous_set_time

        .. rubric:: Attributes and Methods Documentation
    
    """

    def __init__(self,
            coordinates: SkyCoord,
            observer: EarthLocation = nenufar_position,
            time: Time = Time.now()
        ):
        self.coordinates = coordinates
        self.observer = observer
        self.time = time


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time(self) -> Time:
        """ """
        return self._time
    @time.setter
    def time(self, t):
        if t.isscalar:
            t = t.reshape(1,)
        self._time = t


    @property
    def is_circumpolar(self) -> bool:
        r""" Whether the celestial object is circumpolar at the 
            observer's latitude.

            .. math::
                l + \delta \geq 90\,{\rm deg}
            
            where :math:`l` is the latitude (defined in
            :attr:`~nenupy.astro.astro_tools.AstroObject.observer`),
            :math:`\delta` is the object's declination (defined in
            :attr:`~nenupy.astro.astro_tools.AstroObject.coordinates`).
        """
        return np.all((self.observer.lat + self.coordinates.dec) >= 90*u.deg)


    @property
    def culmination_azimuth(self) -> u.Quantity:
        """ """
        if not self.is_circumpolar:
            return 180*u.deg
        else:
            return 0*u.deg


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def meridian_transit(self,
            t_min: Time = Time.now(),
            duration: TimeDelta = TimeDelta(86400, format='sec'),
            precision: TimeDelta = TimeDelta(5, format='sec'),
            fast_compute: bool = True
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` meridian transit time(s).
            This method returns all the transit times found in the time
            window ranging from ``t_min`` to ``t_min + duration``.

            :param t_min:
                Starting time of the temporal window within which
                meridian transits are looked for.
                Default is current time.
            :type t_min:
                :class:`~astropy.time.Time`
            :param duration:
                Width of the temporal window within which
                meridian transits are looked for.
                Default is ``1 day``.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param precision:
                Temporal precision of the returned meridian transit values.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`
            :param fast_compute:
                If set to ``True``, a fast approximation is used during
                the computation of Local Sidereal Time.
                Default is ``True``.
            :type fast_compute:
                `bool`

            :returns:
                Meridian transit times.
                If no transit times are found (because the requested
                time window doesn't contain any) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.meridian_transit(
                        t_min=Time("2021-01-01"),
                        duration=TimeDelta(86400*2, format="sec")
                    )
                <Time object: scale='utc' format='iso' value=['2021-01-01 13:05:47.868' '2021-01-02 13:01:51.882']>

            .. seealso::
                :ref:`ephemerides_sec`

        """
        def find_ha_transit(times: Time):
            """ """
            fk5 = self._get_source_coordinates(
                time=times
            ).transform_to(FK5(equinox=times))
            ha = hour_angle(
                radec=fk5,
                time=times,
                observer=self.observer,
                fast_compute=fast_compute
            )
            return np.where(
                (np.roll(ha, shift=-1, axis=1) - ha)[:, :-1] < 0
            )
        return self._find_crossing_times(
            finding_function=find_ha_transit,
            t_min=t_min,
            duration=duration,
            precision=precision
        )


    def azimuth_transit(self,
            azimuth: u.Quantity = 180*u.deg,
            t_min: Time = Time.now(),
            duration: TimeDelta = TimeDelta(86400, format='sec'),
            precision: TimeDelta = TimeDelta(5, format='sec'),
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` transit time(s) at a given ``azimuth`` value.
            This method returns all the transit times found in the time
            window ranging from ``t_min`` to ``t_min + duration``.

            :param azimuth:
                Azimuth at which the transit is computed.
                Default is ``180 deg`` (i.e. South).
            :type azimuth:
                :class:`~astropy.units.Quantity`
            :param t_min:
                Starting time of the temporal window within which
                azimuth transits are looked for.
                Default is current time.
            :type t_min:
                :class:`~astropy.time.Time`
            :param duration:
                Width of the temporal window within which
                azimuth transits are looked for.
                Default is ``1 day``.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param precision:
                Temporal precision of the returned azimuth transit values.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Azimuth transit times.
                If no transit times are found (either because the requested
                time window doesn't contain any or because the source apparent
                sky position does not cross the desired ``azimuth``) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.azimuth_transit(
                        azimuth=100*u.deg,
                        t_min=Time("2021-01-01"),
                        duration=TimeDelta(86400*2, format="sec")
                    )
                <Time object: scale='utc' format='iso' value=['2021-01-01 11:22:12.463' '2021-01-02 11:18:16.477']>

            .. seealso::
                :ref:`ephemerides_sec`

        """
        def find_az_transit(times: Time):
            """ """
            # altaz_coordinates = radec_to_altaz(
            #     radec=self._get_source_coordinates(time=times),
            #     time=times,
            #     observer=self.observer,
            #     fast_compute=fast_compute
            # ).reshape(times.shape)
            altaz_coordinates = self._get_source_coordinates(time=times).transform_to(
                AltAz(
                    obstime=times,
                    location=self.observer
                )
            )
            azimuths = altaz_coordinates.az.rad
            az = azimuth.to(u.rad).value
            if self.is_circumpolar:
                complexAzStarts = np.angle(
                    np.cos(azimuths[:, :-1]) + 1j*np.sin(azimuths[:, :-1])
                )
                complexAzStops = np.angle(
                    np.cos(azimuths[:, 1:]) + 1j*np.sin(azimuths[:, 1:])
                )
                mask = (complexAzStarts <= az) &\
                    (complexAzStops >= az)
                mask |= (complexAzStarts >= az) &\
                    (complexAzStops <= az)
            else:
                mask = (azimuths[:, :-1] <= az) &\
                    (azimuths[:, 1:] >= az)
            return np.where(mask)
        return self._find_crossing_times(
            finding_function=find_az_transit,
            t_min=t_min,
            duration=duration,
            precision=precision
        )


    def next_meridian_transit(self,
            time: Time = Time.now(),
            precision: TimeDelta = TimeDelta(5, format='sec'),
            fast_compute: bool = True
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` next meridian transit time.
            This method returns the next transit time found after ``time``.

            :param time:
                Relative time used to searching for the next meridian transit.
            :type time:
                :class:`~astropy.time.Time`
            :param precision:
                Temporal precision of the returned meridian transit value.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`
            :param fast_compute:
                If set to ``True``, a fast approximation is used during
                the computation of Local Sidereal Time.
                Default is ``True``.
            :type fast_compute:
                `bool`

            :returns:
                Next meridian transit time.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.next_meridian_transit(
                        time=Time("2021-01-01 12:00:00")
                    )
                <Time object: scale='utc' format='iso' value=2021-01-01 13:05:47.868>

            .. seealso::
                :ref:`ephemerides_sec`, :meth:`~nenupy.astro.target.Target.meridian_transit`

        """
        return self.meridian_transit(
            t_min=time,
            duration=TimeDelta(48*3600, format='sec'),
            precision=precision,
            fast_compute=fast_compute
        )[0]


    def previous_meridian_transit(self,
            time: Time = Time.now(),
            precision: TimeDelta = TimeDelta(5, format='sec'),
            fast_compute: bool = True
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` previous meridian transit time.
            This method returns the previous transit time found before ``time``.

            :param time:
                Relative time used to searching for the previous meridian transit.
                Default is current time.
            :type time:
                :class:`~astropy.time.Time`
            :param precision:
                Temporal precision of the returned meridian transit value.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`
            :param fast_compute:
                If set to ``True``, a fast approximation is used during
                the computation of Local Sidereal Time.
                Default is ``True``.
            :type fast_compute:
                `bool`

            :returns:
                Previous meridian transit time.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.previous_meridian_transit(
                        time=Time("2021-01-01 12:00:00")
                    )
                Time object: scale='utc' format='iso' value=2020-12-31 13:09:43.855>

            .. seealso::
                :ref:`ephemerides_sec`, :meth:`~nenupy.astro.target.Target.meridian_transit`

        """
        return self.meridian_transit(
            t_min=time - TimeDelta(48*3600, format='sec'),
            duration=TimeDelta(48*3600, format='sec'),
            precision=precision,
            fast_compute=fast_compute
        )[-1]


    def rise_time(self,
        t_min: Time = Time.now(),
        elevation: u.Quantity = 0*u.deg,
        duration: TimeDelta = TimeDelta(86400, format='sec'),
        precision: TimeDelta = TimeDelta(5, format='sec'),
        ):
        """ Computes the :class:`~nenupy.astro.target.Target` rise time(s) above ``elevation``.
            This method returns all the rise times found in the time
            window ranging from ``t_min`` to ``t_min + duration``.

            :param t_min:
                Starting time of the temporal window within which
                rise times are looked for.
                Default is current time.
            :type t_min:
                :class:`~astropy.time.Time`
            :param elevation:
                Elevation above which the rise time is computed.
                Default is ``0 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param duration:
                Width of the temporal window within which
                rise times are looked for.
                Default is ``1 day``.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param precision:
                Temporal precision of the returned meridian transit values.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Rise times above a given elevation.
                If no rise times are found (because the requested
                time window doesn't contain any) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.rise_time(
                        t_min=Time("2021-01-01"),
                        elevation=0*u.deg,
                        duration=TimeDelta(86400*2, format="sec")
                    )
                <Time object: scale='utc' format='iso' value=['2021-01-01 02:28:51.926' '2021-01-02 02:24:56.599']>

            .. seealso::
                :ref:`ephemerides_sec`

        """

        def _find_elevation_rise_time(times):
            """ """
            altaz_coordinates = self._get_source_coordinates(time=times).transform_to(
                AltAz(
                    obstime=times,
                    location=self.observer
                )
            )
            elevations = altaz_coordinates.alt
            return np.where(
                (elevations[:, :-1] <= elevation) & (elevations[:, 1:] >= elevation)
            )
        return self._find_crossing_times(
            finding_function=_find_elevation_rise_time,
            t_min=t_min,
            duration=duration,
            precision=precision
        )


    def next_rise_time(self,
            time: Time = Time.now(),
            elevation: u.Quantity = 0*u.deg,
            precision: TimeDelta = TimeDelta(5, format='sec')
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` next rise time above ``elevation``.
            This method returns the next rise time found after ``time``.

            :param time:
                Relative time used to searching for the next rise time.
                Default is current time.
            :type time:
                :class:`~astropy.time.Time`
            :param elevation:
                Elevation above which the rise time is computed.
                Default is ``0 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param precision:
                Temporal precision of the returned rise time value.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Next rise time.
                If no rise time is found (because the source does not
                cross the elevation) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.next_rise_time(
                        time=Time("2021-01-01"),
                        elevation=40*u.deg,
                    )
                <Time object: scale='utc' format='iso' value=2021-01-01 08:20:16.447>

            .. seealso::
                :ref:`ephemerides_sec`, :meth:`~nenupy.astro.target.Target.rise_time`

        """
        try:
            return self.rise_time(
                t_min=time,
                elevation=elevation,
                duration=TimeDelta(48*3600, format='sec'),
                precision=precision
            )[0]
        except IndexError:
            return Time([], format="jd")


    def previous_rise_time(self,
            time: Time = Time.now(),
            elevation: u.Quantity = 0*u.deg,
            precision: TimeDelta = TimeDelta(5, format='sec')
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` previous rise time above ``elevation``.
            This method returns the previous rise time found after ``time``.

            :param time:
                Relative time used to searching for the previous rise time.
                Default is current time.
            :type time:
                :class:`~astropy.time.Time`
            :param elevation:
                Elevation above which the rise time is computed.
                Default is ``0 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param precision:
                Temporal precision of the returned rise time value.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Previous rise time.
                If no rise time is found (because the source does not
                cross the elevation) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.previous_rise_time(
                        time=Time("2021-01-01"),
                        elevation=40*u.deg,
                    )
                <Time object: scale='utc' format='iso' value=2020-12-31 08:24:12.434>

            .. seealso::
                :ref:`ephemerides_sec`, :meth:`~nenupy.astro.target.Target.rise_time`

        """
        try:
            return self.rise_time(
                t_min=time - TimeDelta(48*3600, format='sec'),
                elevation=elevation,
                duration=TimeDelta(48*3600, format='sec'),
                precision=precision
            )[-1]
        except IndexError:
            return Time([], format="jd")


    def set_time(self,
        t_min: Time = Time.now(),
        elevation: u.Quantity = 0*u.deg,
        duration: TimeDelta = TimeDelta(86400, format='sec'),
        precision: TimeDelta = TimeDelta(5, format='sec'),
        ):
        """ Computes the :class:`~nenupy.astro.target.Target` set time(s) below ``elevation``.
            This method returns all the set times found in the time
            window ranging from ``t_min`` to ``t_min + duration``.

            :param t_min:
                Starting time of the temporal window within which
                set times are looked for.
                Default is current time.
            :type t_min:
                :class:`~astropy.time.Time`
            :param elevation:
                Elevation below which the set time is computed.
                Default is ``0 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param duration:
                Width of the temporal window within which
                set times are looked for.
                Default is ``1 day``.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param precision:
                Temporal precision of the returned meridian transit values.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Set times below a given elevation.
                If no set times are found (because the requested
                time window doesn't contain any) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.set_time(
                        t_min=Time("2021-01-01"),
                        elevation=0*u.deg,
                        duration=TimeDelta(86400*2, format="sec")
                    )
                <Time object: scale='utc' format='iso' value=['2021-01-01 23:42:41.174' '2021-01-02 23:38:45.188']>

            .. seealso::
                :ref:`ephemerides_sec`

        """

        def _find_elevation_set_time(times):
            """ """
            altaz_coordinates = self._get_source_coordinates(time=times).transform_to(
                AltAz(
                    obstime=times,
                    location=self.observer
                )
            )
            elevations = altaz_coordinates.alt
            return np.where(
                (elevations[:, :-1] >= elevation) & (elevations[:, 1:] <= elevation)
            )
        return self._find_crossing_times(
            finding_function=_find_elevation_set_time,
            t_min=t_min,
            duration=duration,
            precision=precision
        )


    def next_set_time(self,
            time: Time = Time.now(),
            elevation: u.Quantity = 0*u.deg,
            precision: TimeDelta = TimeDelta(5, format='sec')
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` next set time below ``elevation``.
            This method returns the next set time found after ``time``.

            :param time:
                Relative time used to searching for the next set time.
                Default is current time.
            :type time:
                :class:`~astropy.time.Time`
            :param elevation:
                Elevation below which the set time is computed.
                Default is ``0 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param precision:
                Temporal precision of the returned set time value.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Next set time.
                If no set time is found (because the source does not
                cross the elevation) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.next_set_time(
                        time=Time("2021-01-01"),
                        elevation=40*u.deg,
                    )
                <Time object: scale='utc' format='iso' value=2021-01-01 17:51:17.312>

            .. seealso::
                :ref:`ephemerides_sec`, :meth:`~nenupy.astro.target.Target.set_time`

        """
        try:
            return self.set_time(
                t_min=time,
                elevation=elevation,
                duration=TimeDelta(48*3600, format='sec'),
                precision=precision
            )[0]
        except IndexError:
            return Time([], format="jd")


    def previous_set_time(self,
            time: Time = Time.now(),
            elevation: u.Quantity = 0*u.deg,
            precision: TimeDelta = TimeDelta(5, format='sec')
        ) -> Time:
        """ Computes the :class:`~nenupy.astro.target.Target` previous set time below ``elevation``.
            This method returns the next set time found before ``time``.

            :param time:
                Relative time used to searching for the previous set time.
                Default is current time.
            :type time:
                :class:`~astropy.time.Time`
            :param elevation:
                Elevation below which the set time is computed.
                Default is ``0 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param precision:
                Temporal precision of the returned set time value.
                Default is ``5 sec``.
            :type precision:
                :class:`~astropy.time.TimeDelta`

            :returns:
                Previous set time.
                If no set time is found (because the source does not
                cross the elevation) an empty
                :class:`~astropy.time.Time` object is returned.
            :rtype:
                :class:`~astropy.time.Time`

            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import astropy.units as u
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> cyg_a.previous_set_time(
                        time=Time("2021-01-01"),
                        elevation=40*u.deg,
                    )
                <Time object: scale='utc' format='iso' value=2020-12-31 17:55:12.639>

            .. seealso::
                :ref:`ephemerides_sec`, :meth:`~nenupy.astro.target.Target.set_time`

        """
        try:
            return self.set_time(
                t_min=time - TimeDelta(48*3600, format='sec'),
                elevation=elevation,
                duration=TimeDelta(48*3600, format='sec'),
                precision=precision
            )[-1]
        except IndexError:
            return Time([], format="jd")



    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @abstractmethod
    def _get_source_coordinates(self, time: Time) -> SkyCoord:
        """ Abstract method that must be replaced by subclasses. """
        pass


    @staticmethod
    def _find_crossing_times(
            finding_function: Callable,
            t_min: Time,
            duration: TimeDelta,
            precision: TimeDelta
        ) -> np.ndarray:
        """ """
        # Set t_min to a higher dimension in preparation for multiple matches
        t_min = t_min.reshape((1,))

        # At each iteration, dt will be reduced by down_factor
        down_factor = 5
        dt = duration/down_factor

        # If duration is too big, intial dt is quite big as well
        # therefore, we set the max dt to 6h
        max_dt = TimeDelta(6*3600, format="sec")
        if dt > max_dt:
            down_factor = int(np.ceil(duration/max_dt))
            dt = duration/down_factor

        # Loop until the precision is reached
        while dt*down_factor > precision:

            # Prepare the time array upon which the coordinates are computed
            n_steps = np.ceil(duration/dt)
            times = t_min[:, None] + np.arange(n_steps + 1) * dt

            # Find the indices depending on the function to apply
            transit_indices = finding_function(times)

            # Update t_min at the spots where the transit(s) have been found
            t_min = times[transit_indices]

            if t_min.size == 0:
                # Nothing has been found
                return Time([], format='jd')
            elif t_min.isscalar:
                t_min = t_min.reshape((1,))

            # Next loop will occur on the last time step only
            duration = dt
            dt /= down_factor

        return times[transit_indices] + dt/2.
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ FixedTarget ------------------------ #
# ============================================================= #
class FixedTarget(Target):
    """ Class to handle astronomical targets outside the Solar System.

        .. versionadded:: 2.0.0

        :param coordinates:
        :type coordinates:
            :class:`~astropy.coordinates.SkyCoord`
        :param observer:
        :type observer:
            :class:`~astropy.coordinates.EarthLocation`
        :param time:
        :type time:
            :class:`~astropy.time.Time`

        .. rubric:: Attributes Summary

        .. autosummary::

            ~Target.coordinates
            ~Target.time
            ~Target.observer
            ~Target.is_circumpolar
            ~Target.culmination_azimuth

        .. rubric:: Methods Summary

        .. autosummary::

            ~FixedTarget.from_name
            ~Target.meridian_transit
            ~Target.next_meridian_transit
            ~Target.previous_meridian_transit
            ~Target.azimuth_transit
            ~Target.rise_time
            ~Target.next_rise_time
            ~Target.previous_rise_time
            ~Target.set_time
            ~Target.next_set_time
            ~Target.previous_set_time

        .. rubric:: Attributes and Methods Documentation
        
    """

    def __init__(self,
            coordinates: SkyCoord,
            time: Time = Time.now(),
            observer: EarthLocation = nenufar_position
        ):
        super().__init__(
            coordinates=coordinates,
            observer=observer,
            time=time
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def horizontal_coordinates(self):
        """ """
        return super().horizontal_coordinates[:, 0]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_name(cls,
            name: str,
            time: Time = Time.now(),
            observer: EarthLocation = nenufar_position
        ) -> FixedTarget:
        """ Instantiates a :class:`~nenupy.astro.target.FixedTarget`
            object from a name that could be resolved by `Simbad <http://simbad.u-strasbg.fr/simbad/>`_.

            :param name:
                Source name.
            :type name:
                `str`
            :param time:
                Time at which the source is looked at.
                Default is current time.
            :type time:
                :class:`~astropy.time.Time`
            :param observer:
                Earth location from where the source is observed.
                Default is NenuFAR's location.
            :type observer:
                :class:`~astropy.coordinates.EarthLocation`

            :returns:
                :class:`~nenupy.astro.target.FixedTarget` instance.
            :rtype:
                :class:`~nenupy.astro.target.FixedTarget`
            
            :Example:
                >>> from nenupy.astro.target import FixedTarget
                >>> cyg_a = FixedTarget.from_name("Cyg A")

        """

        if name.lower() in common_sources.keys():
            src = common_sources[name.lower()]
            source = SkyCoord(src["ra"], src["dec"], unit="deg")
        else:
            # Retrieve the Simbad coordinates
            source = SkyCoord.from_name(name)

        return cls(
            coordinates=source,
            observer=observer,
            time=time
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_source_coordinates(self, time: Time):
        """ """
        return SkyCoord(
            ra=np.repeat(self.coordinates.ra.deg, time.size).reshape(time.shape),
            dec=np.repeat(self.coordinates.dec.deg, time.size).reshape(time.shape),
            unit="deg",
            frame=self.coordinates.frame
        )
        # return self.coordinates
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- SolarSystemTarget --------------------- #
# ============================================================= #
class SolarSystemTarget(Target):
    """ Class to handle Solar System targets.

        .. versionadded:: 2.0.0

        .. rubric:: Attributes Summary

        .. autosummary::

            ~Target.coordinates
            ~Target.time
            ~Target.observer
            ~Target.is_circumpolar
            ~Target.culmination_azimuth

        .. rubric:: Methods Summary

        .. autosummary::

            ~SolarSystemTarget.from_name
            ~Target.meridian_transit
            ~Target.next_meridian_transit
            ~Target.previous_meridian_transit
            ~Target.azimuth_transit
            ~Target.rise_time
            ~Target.next_rise_time
            ~Target.previous_rise_time
            ~Target.set_time
            ~Target.next_set_time
            ~Target.previous_set_time

        .. rubric:: Attributes and Methods Documentation
        
    """

    def __init__(self,
            name: str,
            coordinates: SkyCoord,
            time: Time = Time.now(),
            observer: EarthLocation = nenufar_position
        ):
        super().__init__(
            coordinates=coordinates,
            observer=observer,
            time=time
        )
        self.name = name


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def horizontal_coordinates(self):
        """ """
        return super().horizontal_coordinates[np.identity(self.time.size, dtype=bool)]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_name(cls,
            name: str,
            time: Time = Time.now(),
            observer: EarthLocation = nenufar_position
        ) -> SolarSystemTarget:
        """ """

        # Get the ICRS instance of the Solar System object
        source = solar_system_source(
            name=name,
            time=time,
            observer=observer
        )

        return cls(
            name=name,
            coordinates=source,
            observer=observer,
            time=time
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_source_coordinates(self, time: Time):
        """ """
        return solar_system_source(
            name=self.name,
            time=time,
            observer=self.observer
        )
# ============================================================= #
# ============================================================= #
