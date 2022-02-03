#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Pointing
    ********

    .. inheritance-diagram:: nenupy.astro.pointing.Pointing
        :parts: 3

    .. autosummary::

        ~Pointing

"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "Pointing"
]

from astropy.coordinates.builtin_frames.icrs import ICRS
import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)

from astropy.coordinates import (
    SkyCoord,
    AltAz,
    EarthLocation
)
from astropy.time import Time, TimeDelta
import astropy.units as u

from nenupy import nenufar_position
from nenupy.astro import AstroObject, altaz_to_radec, radec_to_altaz
from nenupy.astro.target import Target
# from nenupy.io.bst import BST


# ============================================================= #
# ------------------------- Pointing -------------------------- #
# ============================================================= #
class Pointing(AstroObject):
    """ Class to handle instrument pointing.

        :param coordinates:
            Pointing coordinates (in the equatorial frame).
        :type coordinates:
            :class:`~astropy.coordinates.SkyCoord`
        :param time:
            Pointing times.
        :type time:
            :class:`~astropy.time.Time`
        :param duration:
            Pointing duration.
        :type duration:
            :class:`~astropy.time.TimeDelta`
        :param observer:
            Earth location from where the pointing is made.
        :type observer:
            :class:`~astropy.coordinates.EarthLocation`

        .. seealso::
            :ref:`pointing_doc` for more details on how to instantiate
            and use this class.

        .. versionadded:: 2.0.0

        .. rubric:: Attributes Summary

        .. autosummary::

            ~Pointing.time
            ~Pointing.duration
            ~Pointing.horizontal_coordinates

        .. rubric:: Methods Summary

        .. autosummary::

            ~Pointing.plot
            ~Pointing.from_file
            ~Pointing.target_tracking
            ~Pointing.target_transit
            ~Pointing.zenith_tracking

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self,
            coordinates: SkyCoord,
            time: Time,
            duration: TimeDelta = TimeDelta(1, format="sec"),
            observer: EarthLocation = nenufar_position
        ):
        self.coordinates = coordinates
        self.time = time
        self.duration = duration
        self.observer = observer


    def __str__(self):
        return str(self.horizontal_coordinates)


    def __getitem__(self, time: Time):
        """ """
        starts = (self.time).jd
        stops = (self.time + self.duration).jd
        
        if self.coordinates.isscalar:
            coordinates = self.coordinates.reshape(1,)
        else:
            coordinates = self.coordinates
        ras = coordinates.ra.deg
        decs = coordinates.dec.deg

        ra = []
        dec = []
        custom_az = []
        custom_el = []
        for t in time.jd:
            # Find the corresponding RA/Dec
            mask = (t >= starts) & (t < stops)
            if np.all(~mask):
                # No match
                t = Time(t, format="jd")
                log.warning(
                    f"Default zenith pointing at {t.isot}."
                )
                zenith = SkyCoord(
                    0*u.deg,
                    90*u.deg,
                    frame=AltAz(
                        obstime=t,
                        location=self.observer
                    )
                ).transform_to(ICRS)
                ra.append(zenith.ra)
                dec.append(zenith.dec)

                if hasattr(self, "custom_ho_coordinates"):
                    custom_az.append(0*u.deg)
                    custom_el.append(90*u.deg)
            else:
                # there is a match
                ra.append(ras[mask][0])
                dec.append(decs[mask][0])
                if hasattr(self, "custom_ho_coordinates"):
                    custom_az.append(self.custom_ho_coordinates[mask].az[0])
                    custom_el.append(self.custom_ho_coordinates[mask].alt[0])

        pointing = Pointing(
            coordinates=SkyCoord(
                ra,
                dec,
                unit='deg'
            ),
            time=time,
        )

        if hasattr(self, "custom_ho_coordinates"):
            #pointing.custom_ho_coordinates = self.custom_ho_coordinates
            pointing.custom_ho_coordinates = SkyCoord(
                custom_az,
                custom_el,
                frame=AltAz(
                    obstime=time,#.reshape(time.size, 1),
                    location=nenufar_position
                )
            )

        return pointing

    # def __getitem__(self, time: Time):
    #     """ """
    #     starts = self.time
    #     stops = starts + self.duration
    #     # Ease the comparison by starting a bit earlier
    #     starts -= TimeDelta(0.01, format='sec')

    #     az = []
    #     el = []
    #     horizontal_coordinates = self.horizontal_coordinates
    #     for t in time:
    #         idx = np.argwhere(
    #             (t >= starts) & (t < stops)
    #         )
    #         if idx.size == 1 or idx.size == 2:
    #             idx = idx[0][0]
    #             ho_coords = horizontal_coordinates
    #             az.append(ho_coords[idx][0].az.deg)
    #             el.append(ho_coords[idx][0].alt.deg)
    #         elif idx.size == 0:
    #             log.warning(
    #                 f"Default zenith pointing at {t.isot}."
    #             )
    #             az.append(180)
    #             el.append(90)
    #         else:
    #             raise Exception(f'Weird... {idx}')

    #     return Pointing(
    #         coordinates=SkyCoord(
    #             az,
    #             el,
    #             unit='deg',
    #             frame=AltAz(
    #                 obstime=time,
    #                 location=nenufar_position
    #             )
    #         ),
    #         time=time,
    #     )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time(self):
        """ Pointing start times.

            :setter: Pointing times.
            
            :getter: Pointing times.
            
            :type: :class:`~astropy.time.Time`
        """
        return self._time
    @time.setter
    def time(self, t):
        if t.isscalar:
            t = t.reshape((1,))
        self._time = t


    @property
    def duration(self):
        """ Pointing durations.
            If this attribute is scalar, it means that the same duration
            should be applied to all the start times defined in
            :attr:`~nenupy.astro.pointing.Pointing.time`.

            :setter: Pointing durations.
            
            :getter: Pointing durations.
            
            :type: :class:`~astropy.time.TimeDelta`
        """
        return self._duration
    @duration.setter
    def duration(self, d):
        if (d.size != 1) and (d.size != self.time.size):
            raise IndexError(
                f"'duration' of size {d.size} does not match 'time' of size {self.time.size}."
            )
        self._duration = d


    @property
    def horizontal_coordinates(self):
        """ Horizontal coordinates as seen from :attr:`~nenupy.astro.astro_tools.AstroObject.observer`
            at :attr:`~nenupy.astro.pointing.Pointing.time`.

            :getter: Horizontal coordinates.
            
            :type: :class:`~astropy.coordinates.SkyCoord`
        """
        # Coord/time dimensions of a pointing instance are expected
        # to be identical. Therefore, only the diagonal terms are needed.
        # Otherwise, radec_to_altaz will recompute altaz for each
        # radec and each time by default.
        altaz = super().horizontal_coordinates
        if altaz.size == 1:
            return altaz
        elif altaz.shape[0] == altaz.size:
            return altaz
        else:
            return altaz[np.identity(altaz.shape[0], dtype=bool)]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, **kwargs):
        """ Plots the elevation and azimuth versus time for the current pointing.

            :param figsize:
                Size of the figure. Default is ``(10, 5)``.
            :type figsize:
                `tuple`
            :param figname:
                File name of the figure to save. Default is ``''``,
                i.e. show the figure without saving it.
            :type figname:
                `str`
            :param title:
                Set the title of the figure.
            :type title:
                `str`
            :param display_duration:
                Switch the display of :attr:`~nenupy.astro.pointing.Pointing.duration`.
                If set to ``True`` a grey time window is added to all pointing point,
                representing the duration of each individual pointing.
                Default is ``False``.
            :type display_duration:
                `bool`
        """
        altaz = self.horizontal_coordinates

        fig, axs = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=kwargs.get('figsize', (10, 5))
        )
        fig.subplots_adjust(hspace=0)
        
        axs[0].set_title(kwargs.get("title", ""))
        
        # Elevation plot
        axs[0].set_ylabel("Elevation (deg)")
        if kwargs.get("display_duration", False):
            starts = self.time
            stops = self.time + self.duration
            for start, stop in zip(starts, stops):
                axs[0].axvspan(start.datetime, stop.datetime, alpha=0.5, facecolor="grey", edgecolor=None)

        axs[0].plot(self.time.datetime, altaz.alt.deg, marker="o", markersize=3)

        # Azimuth plot
        axs[1].set_ylabel("Azimuth (deg)")
        if kwargs.get("display_duration", False):
            for start, stop in zip(starts, stops):
                axs[1].axvspan(start.datetime, stop.datetime, alpha=0.5, facecolor="grey", edgecolor=None)
        axs[1].plot(self.time.datetime, altaz.az.deg, marker="o", markersize=3)
        axs[1].set_xlabel(f"Time (since {self.time[0].isot})")
        
        if kwargs.get("figname", "") != "":
            plt.savefig(
                kwargs.get("figname"),
                dpi=300,
                transparent=True,
                bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close('all')


    # def to_stmoc(self):
    #     """ """


    @classmethod
    def from_bst(cls,
            bst,
            beam: int = 0,
            analog: bool = True,
            max_points: int = 100
        ):
        """ """
        bst.beam = beam

        if analog:
            time, az, el = bst.analog_pointing
        else:
            time, az, el = bst.digital_pointing
        
        if time.size == 1:
            # for a transit with multiple ABeams
            az = np.append(az, [0]*u.deg)
            el = np.append(el, [90]*u.deg)
            time = time.insert(1, bst.time[-1])

        if time.size > max_points:
            julian_days = time.jd
            jd_rebin = np.linspace(julian_days[0], julian_days[-1], max_points)
            az = np.interp(jd_rebin, julian_days, az)
            el = np.interp(jd_rebin, julian_days, el)
            time = Time(jd_rebin, format='jd')

        altaz_coords = SkyCoord(
            az[:-1],
            el[:-1],
            frame=AltAz(
                obstime=time[:-1],
                location=nenufar_position
            )
        )
        pointing = cls(
            coordinates=altaz_to_radec(altaz=altaz_coords),
            time=time[:-1],
            duration=time[1:] - time[:-1],
            observer=nenufar_position
        )
        pointing.custom_ho_coordinates = altaz_coords

        return pointing


    @classmethod
    def from_file(cls,
            file_name,
            beam_index: int = 0,
            include_corrections: bool = True
        ):
        """ Instantiates a :class:`~nenupy.astro.pointing.Pointing` object from
            a NenuFAR pointing file.
            Several beam pointings (analog and/or numerical) could be described 
            in ``file_name``. The argument ``beam_index`` allows for the selection
            of one of them.
        
            :param file_name:
                NenuFAR pointing file, either analog (ending with ``.altazA``) or 
                numerical (ending with ``.altazB``).
            :type file_name:
                `str`
            :param beam_index:
                Beam number to take into account.
            :type beam_index:
                `int`
            
            :return:
                Pointing derived from a NenuFAR pointing file.
            :rtype:
                :class:`~nenupy.astro.pointing.Pointing`

            :Example:
                >>> from nenupy.astro.pointing import Pointing
                >>> pointing = Pointing.from_file(
                        file_name=".../20211104_170000_20211104_200000_JUPITER_TRACKING.altazA",
                        beam_index=1
                    )

        """
        if file_name.endswith('.altazA'):
            try:
                pointing = np.loadtxt(
                    file_name,
                    skiprows=3,
                    comments=";",
                    dtype={
                        'names': ('time', 'anabeam', 'az', 'el', 'az_cor', 'el_cor', 'freq', 'el_eff'),
                        'formats': ('U20', 'i4', 'f4', 'f4', 'f4', 'f4', 'U5', 'f4')
                    }
                )
                pointing = pointing[pointing["anabeam"] == beam_index]
                azimuths = pointing["az_cor"] if include_corrections else pointing["az"]
                elevations = pointing["el_eff"] if include_corrections else pointing["el"]
            except ValueError:
                # No correction
                pointing = np.loadtxt(
                    file_name,
                    skiprows=3,
                    comments=";",
                    dtype={
                        'names': ('time', 'anabeam', 'az', 'el', 'freq', 'el_eff'),
                        'formats': ('U20', 'i4', 'f4', 'f4', 'U5', 'f4')
                    }
                )
                pointing = pointing[pointing["anabeam"] == beam_index]
                azimuths = pointing["az"]
                elevations = pointing["el_eff"] if include_corrections else pointing["el"]
            except IndexError:
                # No beamsquint
                pointing = np.loadtxt(
                    file_name,
                    skiprows=3,
                    comments=";",
                    dtype={
                        'names': ('time', 'anabeam', 'az', 'el', 'az_cor', 'el_cor'),
                        'formats': ('U20', 'i4', 'f4', 'f4', 'f4', 'f4')
                    }
                )
                pointing = pointing[pointing["anabeam"] == beam_index]
                azimuths = pointing["az_cor"] if include_corrections else pointing["az"]
                elevations = pointing["el_cor"] if include_corrections else pointing["el"]

            times = Time(pointing["time"])
            azimuths *= u.deg
            elevations *= u.deg
            if times.size == 1:
                # for a transit with multiple ABeams
                azimuths = np.append(azimuths, [0]*u.deg)
                elevations = np.append(elevations, [90]*u.deg)
                times = times.insert(1, times[-1] + TimeDelta(1, format="sec"))

            duration = times[1:] - times[:-1]
            times = times[:-1]
            altaz_coords = SkyCoord(
                azimuths[:-1],
                elevations[:-1],
                frame=AltAz(
                    obstime=times,
                    location=nenufar_position
                )
            )
        elif file_name.endswith('.altazB'):
            pointing = np.loadtxt(
                file_name,
                skiprows=2,
                comments=";",
                dtype={
                    'names': ('time', 'anabeam', 'digibeam', 'az', 'el', 'l', 'm', 'n'),
                    'formats': ('U20', 'i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4')
                }
            )
            pointing = pointing[pointing["digibeam"] == beam_index]
            times = Time(pointing["time"])
            duration = times[1:] - times[:-1]
            # Add the last duration at the end (supposed to be 10 seconds)
            duration = duration.insert(-1, TimeDelta(10, format="sec", scale=duration.scale))
            altaz_coords = SkyCoord(
                pointing['az'],
                pointing["el"],
                unit="deg",
                frame=AltAz(
                    obstime=times,
                    location=nenufar_position
                )
            )
        
        # return cls(
        #     coordinates=altaz_to_radec(altaz=altaz_coords),
        #     time=times,
        #     duration=duration,
        #     observer=nenufar_position
        # )
        pointing = cls(
            coordinates=altaz_to_radec(altaz=altaz_coords),
            time=times,
            duration=duration,
            observer=nenufar_position
        )
        pointing.custom_ho_coordinates = altaz_coords
        return pointing


    @classmethod
    def target_tracking(cls,
            target: Target,
            time: Time,
            duration: TimeDelta = TimeDelta(3600, format="sec"),
            observer: EarthLocation = nenufar_position,
        ):
        """ Instantiates a :class:`~nenupy.astro.pointing.Pointing`
            object that tracks a given celestial source target.
        
            :param target:
                Celestial source target to track.
            :type target:
                :class:`~nenupy.astro.target.Target`
            :param time:
                Start times of the pointing aiming at ``target``.
            :type time:
                :class:`~astropy.time.Time`
            :param duration:
                Duration of each individual pointing. If this argument is
                a scalar, then it will be applied to every start time (defined
                in ``time``).
                Default is one hour.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param observer:
                Earth location from where the target is observed.
                Default is NenuFAR's location.
            :type observer:
                :class:`~astropy.coordinates.EarthLocation`

            :return:
                Pointing derived while tracking a specific target.
            :rtype:
                :class:`~nenupy.astro.pointing.Pointing`

            :Example:
                >>> from nenupy.astro.pointing import Pointing
                >>> from nenupy.astro.target import FixedTarget
                >>> from astropy.time import Time, TimeDelta
                >>> import numpy as np
                >>> cyg_a = FixedTarget.from_name("Cyg A")
                >>> pointing = Pointing.target_tracking(
                        target=cyg_a,
                        time=Time("2021-01-01 00:00:00") + np.arange(10)*TimeDelta(1800, format="sec"),
                        duration=TimeDelta(np.ones(10)*1200, format="sec")
                    )

        """
        return cls(
            coordinates=target._get_source_coordinates(time),
            time=time,
            duration=duration,
            observer=observer
        )


    @classmethod
    def target_transit(cls,
            target: Target,
            t_min: Time,
            duration: TimeDelta = TimeDelta(3600, format="sec"),
            dt: TimeDelta = TimeDelta(10, format="sec"),
            azimuth: u.Quantity = 180*u.deg,
            observer: EarthLocation = nenufar_position,
        ):
        """ Instantiates a :class:`~nenupy.astro.pointing.Pointing`
            object around a source transit.
            The next transit at a given ``azimuth`` is search from ``t_min``.
            The pointing is then centered at the transit, for a ``duration`` period.
            The pointing is made in steps numbered as ``duration/dt``.
        
            :param target:
                Celestial source target transiting.
            :type target:
                :class:`~nenupy.astro.target.Target`
            :param t_min:
                Time from which the next transit is searched for.
            :type t_min:
                :class:`~astropy.time.Time`
            :param duration:
                Total duration of the pointing, centered on the transit time.
                Default is one hour.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param dt:
                Time steps of individual pointings.
                Default is 10 sec.
            :type dt:
                :class:`~astropy.time.TimeDelta`
            :param azimuth:
                Azimuth at which the transit is computed.
                A ``ValueError`` exception is raised if the selected
                ``target`` does not cross the required ``azimuth`` value.
                Default is 180 deg (i.e., South).
            :type azimuth:
                :class:`~astropy.units.Quantity`
            :param observer:
                Earth location from where the target is observed.
                Default is NenuFAR's location.
            :type observer:
                :class:`~astropy.coordinates.EarthLocation`

            :return:
                Pointing centered around a given target transit.
            :rtype:
                :class:`~nenupy.astro.pointing.Pointing`

            :Example:
                >>> from nenupy.astro.pointing import Pointing
                >>> from astropy.time import Time, TimeDelta
                >>> pointing = Pointing.target_transit(
                        target=cyg_a,
                        time=Time("2021-01-01 00:00:00"),
                        duration=TimeDelta(7200, format="sec"),
                        azimuth=180*u.deg
                    )

        """
        transit_time = target.azimuth_transit(azimuth=azimuth, t_min=t_min)
        if transit_time.size == 0:
            raise ValueError(
                f"The selected target does not cross azimuth={azimuth}."
            )
        elif transit_time.size != 1:
            # Possibly 2 crossings if the source is circumpolar
            # get the one at maximal elevation
            target_altaz = radec_to_altaz(radec=target.coordinates, time=transit_time)[:, 0]
            transit_time = transit_time[target_altaz.alt.argmax()]
        
        transit_altaz = radec_to_altaz(radec=target.coordinates, time=transit_time)
        time_steps = np.floor(duration/dt)
        time_steps = time_steps + 1 if time_steps%2 == 0 else time_steps
        dt_shifts = np.arange(time_steps) - (time_steps - 1)/2

        pointing_times = transit_time + dt_shifts*dt

        return cls(
            coordinates=SkyCoord(
                np.repeat(transit_altaz.az.deg, pointing_times.size),
                np.repeat(transit_altaz.alt.deg, pointing_times.size),
                unit="deg",
                frame=AltAz(
                    obstime=pointing_times,
                    location=nenufar_position
                )
            ).transform_to(ICRS),
            time=pointing_times,
            duration=dt,
            observer=observer
        )


    # @classmethod
    # def snapshot(cls,
    #         target: Target,
    #         time: Time,
    #         duration: TimeDelta = TimeDelta(1, format="sec"),
    #         observer: EarthLocation = nenufar_position
    #     ):
    #     """ Instantiates a :class:`~nenupy.astro.pointing.Pointing`
    #         object as a simple snapshot (i.e., single pointing).
        
    #         :param target:
    #             Celestial source target to track.
    #         :type target:
    #             :class:`~nenupy.astro.target.Target`
    #         :param time:
    #             Start time of the pointing aiming at ``target``.
    #         :type time:
    #             :class:`~astropy.time.Time`
    #         :param duration:
    #             Duration of each individual pointing. If this argument is
    #             a scalar, then it will be applied to every start time (defined
    #             in ``time``).
    #             Default is one hour.
    #         :type duration:
    #             :class:`~astropy.time.TimeDelta`
    #         :param observer:
    #             Earth location from where the target is observed.
    #             Default is NenuFAR's location.
    #         :type observer:
    #             :class:`~astropy.coordinates.EarthLocation`
            
    #         :return:
    #             Pointing derived from a NenuFAR pointing file.
    #         :rtype:
    #             :class:`~nenupy.astro.pointing.Pointing`

    #         :Example:
    #             >>> from nenupy.astro.pointing import Pointing
    #             >>> pointing = Pointing.from_file(
    #                     file_name=".../20211104_170000_20211104_200000_JUPITER_TRACKING.altazA",
    #                     beam_index=1
    #                 )

    #     """
    #     return cls(
    #         coordinates=target._get_source_coordinates(time=time),
    #         time=time,
    #         duration=duration,
    #         observer=observer
    #     )


    @classmethod
    def zenith_tracking(cls,
            time: Time,
            duration: TimeDelta = TimeDelta(1, format="sec"),
            observer: EarthLocation = nenufar_position
        ):
        """ Instantiates a :class:`~nenupy.astro.pointing.Pointing`
            object at the local zenith.
        
            :param time:
                Start times of the zenith pointing.
            :type t_min:
                :class:`~astropy.time.Time`
            :param duration:
                Duration of each individual pointing. If this argument is
                a scalar, then it will be applied to every start time (defined
                in ``time``).
                Default is one hour.
            :type duration:
                :class:`~astropy.time.TimeDelta`
            :param observer:
                Earth location from where the target is observed.
                Default is NenuFAR's location.
            :type observer:
                :class:`~astropy.coordinates.EarthLocation`

            :return:
                Pointing fixed at the local zenith.
            :rtype:
                :class:`~nenupy.astro.pointing.Pointing`

            :Example:
                >>> from nenupy.astro.pointing import Pointing
                >>> from astropy.time import Time, TimeDelta
                >>> pointing = Pointing.target_transit(
                        target=cyg_a,
                        time=Time("2021-01-01 00:00:00"),
                        duration=TimeDelta(7200, format="sec"),
                        azimuth=180*u.deg
                    )

        """
        az = 0
        el = 90
        if not time.isscalar:
            az = np.repeat(az, time.size)
            el = np.repeat(el, time.size)
        return cls(
            coordinates=altaz_to_radec(
                altaz=SkyCoord(
                    az,
                    el,
                    unit="deg",
                    frame=AltAz(
                        obstime=time,
                        location=observer
                    )
                ),
                fast_compute=False
            ),
            time=time,
            duration=duration,
            observer=observer
        )
# ============================================================= #
# ============================================================= #
