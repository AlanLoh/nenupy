#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***************************
    Bright Source Contamination
    ***************************
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "BeamLobes",
    "SourceInLobes"
]


import logging

log = logging.getLogger(__name__)

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from dask.diagnostics import ProgressBar
from typing import List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mocpy import MOC

from nenupy import DummyCtMgr
from nenupy.astro.pointing import Pointing
from nenupy.astro.sky import HpxSky
from nenupy.astro.astro_tools import SolarSystemSource, solar_system_source, local_sidereal_time
from nenupy.instru import MiniArray, NenuFAR_Configuration


# Mini-Array indices that present a different orientation.
MA_INDICES = np.array([0, 1, 3, 7, 11, 13])
MA_ROTATIONS = np.array([0, 30, 20, 50, 10, 40])


# ============================================================= #
# ----------------------- SourceInLobes ----------------------- #
# ============================================================= #
class SourceInLobes:
    """ SourceInLobes object """

    def __init__(self, time: Time, frequency: u.Quantity, value: np.ndarray):
        self.time = time
        self.frequency = frequency
        self.value = value


    def __add__(self, other):
        """ Adds two SourceInLobes objects. """
        same_frequencies = np.all(self.frequency.to(u.MHz) == other.frequency.to(u.MHz))
        same_times = np.all(self.time.jd == other.time.jd)
        if (not same_frequencies) or (not same_times):
            raise Exception(
                f"Addition of {SourceInLobes} objects with different time and/or frequency ranges."
            )
        return SourceInLobes(
            time=self.time.copy(),
            frequency=self.frequency.copy(),
            value=self.value + other.value 
        )


    @property
    def lst_time(self):
        """ Returns the times associated with the SourceInLobe object in Local Sidereal Time."""
        return local_sidereal_time(self.time)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, time_unit: str = "utc", **kwargs):
        """
        :param time_unit:
            To select in which units the time axe is to be displayed : 'utc' or 'lst'. Default is 'utc'. 
            For pcolormesh to plot correctly the data, the time list in the wanted unit should be monotonous.
        :type time_unit:
            str
            
        """
        fig = plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        roll_index = 0
        if time_unit == "utc":
            try:
                roll_index = np.argwhere(np.diff(self.time.jd) < 0)[0, 0] + 1
            except IndexError:
                pass
            time_to_plot = self.time.datetime
            time_label = f"Time (UTC since {self.time[0].isot.split('.')[0]})"

        elif time_unit == "lst":
            try:
                roll_index = np.argwhere(np.diff(self.lst_time.rad) < 0)[0, 0] + 1
            except IndexError:
                pass
            time_to_plot = self.lst_time.rad
            time_label = "Local Sidereal Time (rad)"

        plt.pcolormesh(
            np.roll(time_to_plot, -roll_index),
            self.frequency.to(u.MHz).value,
            np.roll(self.value, -roll_index, axis=1),
            shading="auto",
            cmap=kwargs.get("cmap", "Blues")
        )
        plt.grid(alpha=0.5)
        plt.ylabel("Frequency (MHz)")
        plt.xlabel(time_label)
        plt.title(kwargs.get("title", ""))
        cb = plt.colorbar(pad=0.03)
        cb.set_label("Sources in beam lobes")

        # Format of the time axis tick labels
        ax = plt.gca()
        if time_unit == "utc":
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M:%S")
            )
            fig.autofmt_xdate()

        # Add minor ticks
        ax.minorticks_on()

        # Save or show the figure
        figname = kwargs.get("figname", "")
        if figname != "":
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches="tight",
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close("all")


    def export(self, filename: str , in_bytes: bool = False) -> None:
        """ Exports the time-frequency contamination array in FITS. The mask being either in floats or in bytes.
            :param filename:
                FITS file name.
            :type filename:
                `str`
            :param in_bytes:
                Set to `True` to convert the mask array to bytes. Default is `False`.
            :type in_bytes:
                `bool`
        """
        t_size, f_size = self.time.size, self.frequency.size

        if in_bytes :
            self.value = np.array(np.floor(self.value*255)).astype(int)
            dtype = [("contamination", "B", (t_size,))]
        else :
            dtype = [("contamination", "i8" if self.value.dtype==np.int32 else "f8", (t_size,))]
        
        primary_hdu = fits.PrimaryHDU()

        time_jd = self.time.jd
        time_hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name="time_jd", format="f8", array=time_jd)]
        )
        time_lst_deg = local_sidereal_time(self.time).deg
        lst_hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name="time_lst_deg", format="f8", array=time_lst_deg)]
        )
        freq_mhz = self.frequency.to_value(u.MHz)
        freq_hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name="frequency_mhz", format="f8", array=freq_mhz)]
        )

        data = np.zeros(f_size, dtype=dtype)
        data["contamination"] = self.value
        contamination_hdu = fits.BinTableHDU(data)

        hdu_list = fits.HDUList([primary_hdu, time_hdu, lst_hdu, freq_hdu, contamination_hdu])
        hdu_list.writeto(filename, overwrite=True)
        log.info(
            f"{SourceInLobes} object saved in {filename}."
        )


    @classmethod
    def from_fits(cls, filename: str):
        """ """
        with fits.open(filename) as hdus:
            time_jd = hdus[1].data["time_jd"]
            frequency_mhz = hdus[3].data["frequency_mhz"]
            values = hdus[4].data["contamination"]

        if values.dtype == "B":
            values = values.astype("float64")
            values /= 255

        return cls(
            time=Time(time_jd, format="jd"),
            frequency=frequency_mhz*u.MHz,
            value=values
        )

    # def export(self, filename: str):
    #     """ Exports the time-frequency contamination array in FITS.
            
    #         :param filename:
    #             FITS file name.
    #         :type filename:
    #             `str`
    #     """
    #     t_size, f_size = self.time.size, self.frequency.size
    #     dtype = [
    #         ('time_jd', 'f8', (t_size,)),
    #         ('time_lst_deg', 'f8', (t_size,)),
    #         ('frequency_mhz', 'f8'),
    #         ('contamination', 'i8' if self.value.dtype==np.int32 else 'f8', (t_size,))
    #     ]
    #     data = np.zeros(f_size, dtype=dtype)
    #     data["time_jd"] = self.time.jd
    #     data["time_lst_deg"] = local_sidereal_time(self.time).deg
    #     data["frequency_mhz"] = self.frequency.to(u.MHz).value
    #     data["contamination"] = self.value
    #     hdu = fits.BinTableHDU(data)
    #     hdu.writeto(filename, overwrite=True)
    #     log.info(
    #         f"{SourceInLobes} object saved in {filename}."
    #     )


    # @classmethod
    # def from_fits(cls, filename: str):
    #     """ """
    #     with fits.open(filename) as hdus:
    #         time_jd = hdus[1].data["time_jd"][0, :]
    #         frequency_mhz = hdus[1].data["frequency_mhz"]
    #         values = hdus[1].data["contamination"]

    #     return cls(
    #         time=Time(time_jd, format="jd"),
    #         frequency=frequency_mhz*u.MHz,
    #         value=values
    #     )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- BeamLobes ------------------------- #
# ============================================================= #
class BeamLobes:
    """ """

    def __init__(self,
            time: Time,
            frequency: u.Quantity,
            pointing: Pointing,
            nenufar_config: NenuFAR_Configuration = NenuFAR_Configuration(beamsquint_correction=False),
            miniarray_rotations: List[int] = None,
            use_antenna_gain: bool = True
        ):
        self.time = time
        self.frequency = frequency
        self.pointing = pointing
        self.configuration=nenufar_config
        self.miniarray_rotations = miniarray_rotations
        
        self.moc = None
        self.tf_coverage = None
        self._src_display = {}

        # Instantiation of HpxSky with the required time and frequency
        # axes. The resolution is set to 2 deg which should be more than
        # enough while dealing with Mini-Array beams. PLus we don't need
        # full resolution for estimating the grating lobes positions.
        self.sky = HpxSky(
            resolution=2*u.deg,
            frequency=frequency,
            time=time
        )
        log.debug(
            "'HpxSky' object instantiated "
            f"(time steps: {self.sky.time.size}, "
            f"frequency steps: {self.sky.frequency.size}, "
            f"coordinates: {self.sky.coordinates.size})."
        )

        # Fill in the value attribute with the array factor
        # computed over the 6 different MA orientations.
        self.sky.value = self._compute_array_factor(use_antenna_gain=use_antenna_gain)
        # self._array_factor = self._compute_array_factor()
        # self.sky.value = np.sum(self._array_factor, axis=0)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def miniarray_rotations(self):
        """ """
        return self._miniarray_rotations
    @miniarray_rotations.setter
    def miniarray_rotations(self, mr: List[int]):
        if mr is None:
            # Consider all possible rotations
            mr = np.arange(0, 60, 10)
        else:
            # Check that the rotation format is correct
            if not all([rot%10 == 0 for rot in mr]):
                raise ValueError(
                    f"Syntax error: miniarray_rotations={mr}. It should be a list of integers, multiples of 10."
                )
            # Reduce to the modulo 60 deg, and only keep unique values
            mr = np.unique(np.array(mr)%60)
        self._miniarray_rotations = mr



    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute_weight_moc(self, sources: List[str], thresholds: np.ndarray = np.logspace(-2, 0, 10, endpoint=False)) -> np.ndarray:
        """ Compute moc with arrays of values between 0 and 1."""

        # Get an array of sky positions (including all sources, all times)
        sky_positions = self._get_skycoord_array(sources)

        threshold_moc = np.zeros((thresholds.size, self.frequency.size, self.time.size), dtype=bool)

        for i, thr in enumerate(thresholds):

            moc = self.compute_moc(relative_threshold=thr, return_array=True)
            
            source_in_grating_lobes = np.zeros(moc.shape, dtype=int)
            for f_idx in range(self.frequency.size):
                for t_idx in range(self.time.size):
                    source_in_grating_lobes[f_idx, t_idx] = np.any(
                        moc[f_idx, t_idx].contains_lonlat(
                            sky_positions[:, t_idx].ra,
                            sky_positions[:, t_idx].dec
                        )
                    )

            threshold_moc[i, :, :] = source_in_grating_lobes

        self.moc = np.nanmean(threshold_moc, axis=0) > 0 # TODO : a tester....

        return SourceInLobes(
            time=self.time,
            frequency=self.frequency,
            value=np.nanmean(threshold_moc, axis=0)
        )

    def compute_moc(self, relative_threshold: float = 0.5, return_array: bool = False):
        r"""
            Computes the MOC as sky patches where the array factor
            is above :math:`{\rm max}(\mathcal{F}_{\rm MA}) \times r`.
            :math:`\mathcal{F}_{\rm MA}` is the Mini-Array(s) normalized
            array factor and :math:`r` is the ``relative_threshold``.

            :param relative_threshold:
                All the AF values above this threshold are included in the MOC.
            :type maximum_ratio:
                float
        """
        # Number of Mini-Array that have been summed
        # n_ma = self.miniarray_rotations.size
        mocs = []
        for f_idx in range(self.frequency.size):
            # Compute the threshold above which the HEALPix cells are
            # to be included in the 'main sensitivity' of the analog beam
            # (i.e., primary beam, and grating lobes).
            threshold = self.sky.value[:, f_idx, 0].max()*relative_threshold

            # Find the cells and select only those that are above the horizon.
            above_threshold = self.sky.value[:, f_idx, 0] >= threshold
            above_horizon = self.sky.horizontal_coordinates[:].alt.deg >= 0
            cells_in_lobes = np.where(above_threshold & above_horizon)
            
            # Update the array in time
            mocs.append([
                MOC.from_skycoords(
                    self.sky.coordinates[cells_in_lobes[1][cells_in_lobes[0]==t_idx]],
                    max_norder=5
                ) for t_idx in range(self.time.size)
            ])

        if return_array:
            return np.array(mocs)

        self.moc = np.array(mocs)
        log.debug("MOC computed (stored in '.moc' attribute).")


    def gsm_in_lobes(self, temperature_threshold: float = 15000) -> SourceInLobes:
        """ """
        from nenupy.astro.skymodel import HpxGSM
        gsm = HpxGSM(frequency=self.frequency)

        gsm_moc = []
        for f_idx in range(self.frequency.size):
            above_threshold = gsm.value[0, f_idx, 0].compute() >= temperature_threshold
            cells_in_lobes = np.where(above_threshold)
            gsm_moc.append(
                MOC.from_skycoords(
                    gsm.coordinates[cells_in_lobes[0]],
                    max_norder=5
                )
            )

        # For each time and frequency value, compute the number of sources
        # that fall within the MOC
        gsm_in_grating_lobes = np.zeros(self.moc.shape, dtype=int)
        for f_idx in range(self.frequency.size):
            for t_idx in range(self.time.size):
                intersecting_moc = self.moc[f_idx, t_idx].intersection(gsm_moc[f_idx])
                gsm_in_grating_lobes[f_idx, t_idx] = int(not intersecting_moc.empty())

                # TODO : improvement fraction of intersection vs. beam size 
                # gsm_in_grating_lobes[f_idx, t_idx] = intersecting_moc.sky_fraction / grating_lobes.moc[f_idx, t_idx].sky_fraction # TODO : turn dtype to float !
            
        self._src_display = {
            "moc": gsm_moc,
        }

        return SourceInLobes(
            time=self.time,
            frequency=self.frequency,
            value=gsm_in_grating_lobes
        )


    def sources_in_lobes(self, sources: List[str]) -> SourceInLobes:
        """ """
        # Get an array of sky positions (including all sources, all times)
        sky_positions = self._get_skycoord_array(sources)

        # Prepare a dictionnary for plotting purposes
        self._src_display = {
            "names": [
                (sky_positions[:, t_idx], sources, "tab:red") for t_idx in range(self.time.size)
            ],
            "positions": [
                (sky_positions[:, t_idx], 10, "tab:red") for t_idx in range(self.time.size)
            ]
        }

        # For each time and frequency value, compute the number of sources
        # that fall within the MOC
        source_in_grating_lobes = np.zeros(self.moc.shape, dtype=int)
        for f_idx in range(self.frequency.size):
            for t_idx in range(self.time.size):
                source_in_grating_lobes[f_idx, t_idx] = np.sum(
                    self.moc[f_idx, t_idx].contains_lonlat(  # contains_lonlat
                        sky_positions[:, t_idx].ra,
                        sky_positions[:, t_idx].dec
                    )
                )

        return SourceInLobes(
            time=self.time,
            frequency=self.frequency,
            value=source_in_grating_lobes
        )

    def plot(self,
            time: Time,
            frequency: u.Quantity,
            **kwargs
        ):
        """ Plots the figure.

            :meth:`~nenupy.astro.sky.SkySliceBase.plot`

        """
        try:
            t_idx = np.where(self.time.jd >= time.jd)[0][0]
        except IndexError:
            t_idx = -1
        try:
            f_idx = np.where(self.frequency >= frequency)[0][0]
        except IndexError:
            f_idx = -1
        
        if "moc" in self._src_display:
            # Plot the array factor moc and the GSM moc
            return self.sky[t_idx, f_idx, 0].plot(
                moc=[self.moc[f_idx, t_idx], self._src_display["moc"][f_idx]],
                **kwargs
            )
        else:
            # Plot the array factor MOC and highlight the source positions
            return self.sky[t_idx, f_idx, 0].plot(
                moc=self.moc[f_idx, t_idx],
                text=self._src_display["names"][t_idx],
                scatter=self._src_display["positions"][t_idx],
                **kwargs
            )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _compute_array_factor(self, use_antenna_gain: bool = True) -> np.ndarray:
        """ Computes the normalized array factor, summed over the 6 possible different NenuFAR Mini-Array rotations.
            If less rotations are selected, the sum is done on less MA array factors.
            The returned AF is normalized.
        """
        # Mini-Array selection
        ma_mask = np.isin(MA_ROTATIONS, self.miniarray_rotations)
        af = 0
        below_horizon_mask = self.pointing.horizontal_coordinates.alt.deg <= 0

        # Compute the array factor for each Mini-Array
        for m in MA_INDICES[ma_mask]:
            ma = MiniArray(index=m)
            af += ma.array_factor(
                sky=self.sky,
                pointing=ma.analog_pointing(
                    self.pointing,
                    configuration=self.configuration
                )#self.pointing
            )

        if use_antenna_gain:
            af *= ma._antenna_gain(sky=self.sky, pointing=self.pointing)

        log.info("Computing the array factor...")
        with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
            array_factor = af.compute()

        # Set the array factor to NaN if the pointing is below the horizon
        if np.any(below_horizon_mask):
            array_factor[below_horizon_mask, ...] = np.nan
            log.warning("Some time samples match a sub-horizon pointing, the corresponding array-factor is set to NaN.")

        return array_factor / np.max(array_factor, axis=3)[:, :, :, None]


    def _get_skycoord_array(self, sources: List[str]) -> SkyCoord:
        """ """
        ra = np.zeros((len(sources), self.time.size))
        dec = np.zeros(ra.shape)
        for i, source in enumerate(sources):
            if source.upper() in SolarSystemSource._member_names_:
                src = solar_system_source(source, time=self.time)
                ra[i, :] = src.ra.deg
                dec[i, :] = src.dec.deg
            else:
                src = SkyCoord.from_name(source)
                ra[i, :] = np.repeat(src.ra.deg, self.time.size)
                dec[i, :] = np.repeat(src.dec.deg, self.time.size)
        return SkyCoord(ra, dec, unit="deg")
# ============================================================= #
# ============================================================= #

