#! /usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import annotations


"""
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "ExtraSolarTarget,"
    "SolarSystemTarget"
]


from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import logging
log = logging.getLogger(__name__)

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, FK5

from nenupy import nenufar_position
from nenupy.astro2.astro_tools import (
    AstroObject,
    hour_angle,
    solar_system_source,
    radec_to_altaz
)


# ============================================================= #
# -------------------------- Target --------------------------- #
# ============================================================= #
class Target(AstroObject, ABC):
    """ """

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


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def meridian_transit(self,
            t_min: Time = Time.now(),
            duration: TimeDelta = TimeDelta(86400, format='sec'),
            precision: TimeDelta = TimeDelta(5, format='sec'),
            fast_compute: bool = True
        ) -> Time:
        """ """
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
        return self._find_transit_times(
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
            fast_compute: bool = True
        ) -> Time:
        """
            from nenupy.astro2.target import SolarSystemTarget, ExtraSolarTarget

            from astropy.time import Time, TimeDelta
            import astropy.units as u
            times = Time("2021-01-01 12:00:00") + np.arange(12*48) * TimeDelta(300, format='sec')

            #sol = SolarSystemTarget.from_name('sun', time=Time.now())
            casa = ExtraSolarTarget.from_name('cas a', time=times)
            #sol.local_sidereal_time(Time.now()).hour

            casa.azimuth_transit(azimuth=0*u.deg)
        
        """
        def find_az_transit(times: Time):
            """ """
            altaz_coordinates = radec_to_altaz(
                radec=self._get_source_coordinates(time=times),
                time=times,
                observer=self.observer,
                fast_compute=fast_compute
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

                mask = (az >= complexAzStarts) &\
                    (az <= complexAzStops)
                mask |= (az <= complexAzStarts) &\
                    (az >= complexAzStops)
            else:
                mask = (az >= azimuths[:, :-1]) &\
                    (az <= azimuths[:, 1:])
            return np.where(mask)
        return self._find_transit_times(
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
        """ """
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
        """ """
        return self.meridian_transit(
            t_min=time - TimeDelta(48*3600, format='sec'),
            duration=TimeDelta(48*3600, format='sec'),
            precision=precision,
            fast_compute=fast_compute
        )[-1]


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @abstractmethod
    def _get_source_coordinates(self, time: Time) -> SkyCoord:
        """ Abstract method that must be replaced by subclasses. """
        pass


    @staticmethod
    def _find_transit_times(
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
# --------------------- ExtraSolarTarget ---------------------- #
# ============================================================= #
class ExtraSolarTarget(Target):
    """ """

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

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_name(cls,
            name: str,
            time: Time = Time.now(),
            observer: EarthLocation = nenufar_position
        ) -> SolarSystemTarget:
        """ """

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
        # return SkyCoord(
        #     ra=np.repeat(self.coordinates.ra.deg, time.size).reshape(time.shape),
        #     dec=np.repeat(self.coordinates.dec.deg, time.size).reshape(time.shape),
        #     unit="deg",
        #     frame=self.coordinates.frame
        # )
        return self.coordinates
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- SolarSystemTarget --------------------- #
# ============================================================= #
class SolarSystemTarget(Target):
    """ """

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