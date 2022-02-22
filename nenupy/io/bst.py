#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST file
    ********

    .. inheritance-diagram:: nenupy.io.bst.BST nenupy.io.bst.BST_Slice
        :parts: 3

    .. autosummary::

        ~BST
        ~BST_Slice

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "BST",
    "BST_Slice"
]


import profile
from astropy.time import Time
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from nenupy.io.io_tools import StatisticsData

import logging
log = logging.getLogger(__name__)



# ============================================================= #
# ------------------ AnalogBeamConfiguration ------------------ #
# ============================================================= #
class AnalogBeamConfiguration:
    """ """

    def __init__(self, pointing=None):
        self.pointing = pointing

    @classmethod
    def from_statistics_metadata(cls, metadata: dict):
        """ """
        metadata["pan"]
        return cls()

# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- BST_Slice ------------------------- #
# ============================================================= #
class BST_Slice:
    """ Class to handle data sub-set from BST.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~BST_Slice.time
            ~BST_Slice.frequency
            ~BST_Slice.value
            ~BST_Slice.beam
            ~BST_Slice.analog_pointing_times
            ~BST_Slice.digital_pointing_times

        .. rubric:: Methods Summary

        .. autosummary::

            ~BST_Slice.plot
            ~BST_Slice.rebin
            ~BST_Slice.fit_transit

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self,
            time: Time,
            frequency: u.Quantity,
            value: np.ndarray,
            analog_pointing_times: Time = None,
            digital_pointing_times: Time = None
        ):
        self._time = time
        self._frequency = frequency
        self._value = value
        self._analog_pointing_times = analog_pointing_times
        self._digital_pointing_times = digital_pointing_times


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time(self) -> Time:
        """ BST data times.
            
            :getter: Time array.
            
            :type: :class:`~astropy.time.Time`
        """
        return self._time


    @property
    def frequency(self) -> u.Quantity:
        """ BST data frequencies.
            
            :getter: Frequency array.
            
            :type: :class:`~astropy.units.Quantity`
        """
        return self._frequency


    @property
    def value(self) -> np.ndarray:
        """ Data values.
            
            :getter: Values array.
            
            :type: :class:`~numpy.ndarray`
        """
        return self._value


    @property
    def analog_pointing_times(self) -> Time:
        """ Analog pointing start times corresponding to this data set.
            
            :getter: Starting times array.
            
            :type: :class:`~astropy.time.Time`
        """
        return self._analog_pointing_times


    @property
    def digital_pointing_times(self) -> Time:
        """ Digital pointing start times corresponding to this data set.
            
            :getter: Starting times array.
            
            :type: :class:`~astropy.time.Time`
        """
        return self._digital_pointing_times


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, **kwargs):
        r""" Plots the data, while automatically taking into account
            its shape (lightcurve, spectrum or dynamic spectrum). 

            Several parameters, listed below, can be tuned to adapt the plot
            to the user requirements:

            .. rubric:: Data display keywords

            :param decibel:
                If set to ``True``, the data will be displayed in
                a decibel scale (i.e., :math:`{\rm dB} = 10 \log_{10}({\rm data})`).
                Default is ``True``.
            :type decibel:
                `bool`
            :param xmin:
                Minimal x-axis value (could be either time or
                frequency units depending on the data shape).
                Default is automatic scaling.
            :type xmin:
                :class:`~astropy.units.Quantity` or :class:`~astropy.time.TimeDatetime`
            :param xmax:
                Maximal x-axis value (could be either time or
                frequency units depending on the data shape).
                Default is automatic scaling.
            :type xmax:
                :class:`~astropy.units.Quantity` or :class:`~astropy.time.TimeDatetime`
            :param ymin:
                Minimal y-axis value (could be either data amplitude or
                frequency units depending on the data shape).
                Default is automatic scaling.
            :type ymin:
                `float` or :class:`~astropy.units.Quantity`
            :param ymax:
                Maximal y-axis value (could be either data amplitude or
                frequency units depending on the data shape).
                Default is automatic scaling.
            :type ymax:
                `float` or :class:`~astropy.units.Quantity`
            :param vmin:
                *Dynamic spectrum plot only*.
                Minimal data value to display.
            :type vmin:
                `float`
            :param vmax:
                *Dynamic spectrum plot only*.
                Maximal data value to display.
            :type vmax:
                `float`

            .. rubric:: Overplot keywords

            :param vlines:
                *Temporal plot only*.
                Adds vertical lines at specific times.
                The expected format is an array of :class:`~astropy.time.TimeDatetime`.
                Default is ``[]``.
            :type vlines:
                [:class:`~astropy.time.TimeDatetime`]
            :param analog_pointing:
                *Temporal plot only*.
                Overplots vertical dot-dashed black lines at analog pointing start times.
                Default is ``False``.
            :type analog_pointing:
                `bool`
            :param digital_pointing:
                *Temporal plot only*.
                Overplots vertical dotted black lines at analog pointing start times.
                Default is ``False``.
            :type digital_pointing:
                `bool`
            :param hatched_overlay:
                *Dynamic spectrum plot only*.
                Produces a hatched overlay on top of the dynamic spectrum.
                The expected format is ``(time, frequency, hatched_array)``
                where ``hatched_array`` is a boolean :class:`~numpy.ndarray`,
                shaped as (frequency, time),
                set to ``True`` where a hatched cell should be drawn.
                Default is ``None``.
            :type hatched_overlay:
                (:class:`~astropy.time.Time`, :class:`~astropy.units.Quantity`, :class:`~numpy.ndarray`)

            .. rubric:: Plotting layout keywords

            :param figname:
                Name of the file (absolute or relative path) to save the figure.
                Default is ``''`` (i.e., only show the figure).
            :type figname:
                `str`
            :param figsize:
                Set the figure size.
                Default is ``(15, 7)``.
            :type figsize:
                `tuple`
            :param title:
                Set the figure title.
                Default is ``''``.
            :type title:
                `str`
            :param colorbar_label:
                *Dynamic spectrum plot only*.
                Label of the color bar.
                Default is ``'Amp'`` if ``decibel=False`` and ``'dB'`` otherwise.
            :type colorbar_label:
                `str`
            :param cmap:
                *Dynamic spectrum plot only*.
                Color map used to represent the data.
                Default is ``'YlGnBu_r'``.
            :type cmap:
                `str`

            :Exampe:
                .. code-block:: python

                    from nenupy.io.bst import BST
                    from astropy.time import Time, TimeDelta
                    import astropy.units as u
                    import numpy as np

                    # Select BST data
                    bst = BST("/path/to/BST.fits")
                    data = bst.get()
                    
                    # Prepare a boolean array to overlay a hatched pattern
                    hatch_array = np.zeros((30, 300), dtype=bool)
                    hatch_array[5:20, 100:200] = True
                    # Specify time and frequency arrays
                    time_dts = np.arange(300)*TimeDelta(1, format='sec')
                    times = Time("2022-01-24T11:01:00") + time_dts
                    frequencies = np.linspace(47, 52, 30)*u.MHz

                    # Plot
                    data.plot(
                        hatched_overlay=(
                            times,
                            frequencies,
                            hatch_array
                        )
                    )

        """
        # Initialize the figure
        fig = plt.figure(figsize=kwargs.get("figsize", (15, 7)))
        ax = fig.add_subplot(111)
        
        data = self.value.T
        if kwargs.get("decibel", True):
            data = 10*np.log10(data)

        if len(data.shape) == 2:
            self._plot_dynamic_spectrum(data, ax, fig, **kwargs)
        elif (len(data.shape) == 1) and (data.size == self.frequency.size):
            self._plot_spectrum(data, ax, fig, **kwargs)
        elif (len(data.shape) == 1) and (data.size == self.time.size):
            self._plot_lightcurve(data, ax, fig, **kwargs)
        else:
            raise ValueError("Problem...")

        # Axes limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(kwargs.get("xmin", xmin), kwargs.get("xmax", xmax))
        ax.set_ylim(kwargs.get("ymin", ymin), kwargs.get("ymax", ymax))

        # Title
        ax.set_title(kwargs.get("title", ""))

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


    def rebin(self, dt: u.Quantity = None, df: u.Quantity = None):
        """ Rebins the data in time and frequency.

            :param dt:
                Time bin widths.
                Default is ``None`` (i.e., no rebin in time).
            :type dt:
                :class:`~astropy.units.Quantity`
            :param df:
                Frequency bin widths.
                Default is ``None`` (i.e., no rebin in frequency).
            :type df:
                :class:`~astropy.units.Quantity`

            :returns:
                Rebinned data.
            :rtype:
                :class:`~nenupy.io.bst.BST_Slice`
            
            :Example:
                .. code-block:: python

                    from nenupy.io.bst import BST
                    import astropy.units as u

                    bst = BST("/path/to/BST.fits")
                    data = bst.get()
                    rebin_data = data.rebin(
                        dt=3*u.s,
                        df=2*u.MHz
                    )

        """
        time = self.time.copy()
        frequency = self.frequency.copy()
        value = self.value.copy()

        # Dynamic spectrum
        if len(value.shape) == 2:
            rebin_t_indices = self._rebin_time_indices(dt=dt)
            if rebin_t_indices is not None:
                value = np.nanmean(
                    value[rebin_t_indices, :],
                    axis=1
                )
                time = Time(np.nanmean(time.jd[rebin_t_indices], axis=1), format='jd')
            rebin_f_indices = self._rebin_frequency_indices(df=df)
            if rebin_f_indices is not None:
                value = np.nanmean(
                    value[:, rebin_f_indices],
                    axis=2
                )
                frequency = np.nanmean(frequency[rebin_f_indices], axis=1)
            value = value.squeeze()

        # Spectrum
        elif (len(value.shape) == 1) and (value.size == frequency.size):
            rebin_indices = self._rebin_frequency_indices(df=df)
            if rebin_indices is not None:
                value = np.nanmean(value[rebin_indices], axis=1)
                frequency = np.nanmean(frequency[rebin_indices], axis=1)

        # Light curve
        elif (len(value.shape) == 1) and (value.size == time.size):
            rebin_indices = self._rebin_time_indices(dt=dt)
            if rebin_indices is not None:
                value = np.nanmean(value[rebin_indices], axis=1)
                time = Time(np.nanmean(time.jd[rebin_indices], axis=1), format='jd')
        else:
            raise ValueError("Problem...")
        
        return BST_Slice(
            time=time,
            frequency=frequency,
            value=value,
            analog_pointing_times=self.analog_pointing_times,
            digital_pointing_times=self.digital_pointing_times
        )


    def fit_transit(self):
        """ Do a fit.
        """
        from scipy.optimize import curve_fit

        def analog_switch_load(t, coeff_a=1., coeff_b=1.):
            """
                f(t) = a log_10(t) + b
            """
            return coeff_a*np.log10(t) + coeff_b

        def gaussian(t, amp=1., mu=1., sig=1.):
            return amp*np.exp(-np.power(t - mu, 2.) / (2*np.power(sig, 2.)))

        def poly(t, c1=1.):#, c2=1.):#, c3):
            return c1*t#, c1*np.power(t, 2.)# + c3

        def combined(t, coeff_a, coeff_b, amp, mu, sig, c1):
            return analog_switch_load(t, coeff_a, coeff_b) + gaussian(t, amp, mu, sig) + poly(c1)

        data = self.value.copy()
        max_data = data.max()
        data /= max_data

        x_data = np.arange(data.size) + 1

        popt, pcov = curve_fit(
            combined,
            x_data,
            data,
            p0=[1e-2, data.min(), 1, np.mean(x_data), 10, 1e-6],
            method="trf",
            bounds=(
                [0,   data.min(), 0,            0,          0.1, -1e3],
                [1e1, data.max(), 1, x_data.max(), x_data.max(),  1e3])
        )

        fitted_values = max_data*combined(x_data, *popt)

        # Chi square
        stdev = np.std(self.value)
        chi_square = np.sum((self.value - fitted_values/stdev)**2)
        degree_of_freedom = self.value.size - 6

        # Compute the fitted transit time
        interpolated_time_jd = np.interp(popt[3], x_data, self.time.jd)
        transit_time = Time(interpolated_time_jd, format="jd")

        return BST_Slice(
            time=self.time,
            frequency=self.frequency,
            value=fitted_values,
            analog_pointing_times=self.analog_pointing_times,
            digital_pointing_times=self.digital_pointing_times
        ), transit_time, chi_square


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _plot_spectrum(self, data, ax, fig, **kwargs):
        """ Plots a spectrum.
        """
        ax.plot(self.frequency.to(u.MHz).value, data)

        # X label
        ax.set_xlabel("Frequency (MHz)")

        # Y label
        ax.set_ylabel("dB" if kwargs.get("decibel", True) else "Amp")


    def _plot_lightcurve(self, data, ax, fig, **kwargs):
        """ Plots a ligthcurve.
        """
        ax.plot(self.time.datetime, data)

        # Display pointings
        if kwargs.get("analog_pointing", False):
            for time_i in self.analog_pointing_times:
                ax.axvline(time_i.datetime, color="black", linestyle="-.")
        if kwargs.get("digital_pointing", False):
            for time_i in self.digital_pointing_times:
                ax.axvline(time_i.datetime, color="black", linestyle=":")
        
        for vline in kwargs.get("vlines", []):
            ax.axvline(vline, linestyle="--")

        # X label
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )
        fig.autofmt_xdate()
        ax.set_xlabel(f"Time (UTC since {self.time[0].isot})")

        # Y label
        ax.set_ylabel("dB" if kwargs.get("decibel", True) else "Amp")


    def _plot_dynamic_spectrum(self, data, ax, fig, **kwargs):
        """ Plots a dynamic spectrum.
        """
        im = ax.pcolormesh(
            self.time.datetime,
            self.frequency.to(u.MHz).value,
            data,
            shading="auto",
            cmap=kwargs.get("cmap", "YlGnBu_r"),
            vmin=kwargs.get("vmin", np.nanpercentile(data, 5)),
            vmax=kwargs.get("vmax", np.nanpercentile(data, 95))
        )

        # Display pointings
        if kwargs.get("analog_pointing", False):
            for time_i in self.analog_pointing_times:
                ax.axvline(time_i.datetime, color="black", linestyle="-.")
        if kwargs.get("digital_pointing", False):
            for time_i in self.digital_pointing_times:
                ax.axvline(time_i.datetime, color="black", linestyle=":")
            
        for vline in kwargs.get("vlines", []):
            ax.axvline(vline, linestyle="--")

        if kwargs.get("hatched_overlay", None) is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            overlay_time, overlay_frequency, overlay_values = kwargs["hatched_overlay"]
            ax.contourf(
                overlay_time.datetime,
                overlay_frequency.to(u.MHz).value,
                overlay_values,
                levels=[0, 1],
                hatches=[None, '/'],
                colors='none',
                extend='both',
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        cbar = plt.colorbar(im, pad=0.03)
        cbar.set_label(kwargs.get("colorbar_label", "dB" if kwargs.get("decibel", True) else "Amp"))

        # X label
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )
        fig.autofmt_xdate()
        ax.set_xlabel(f"Time (UTC since {self.time[0].isot})")

        # Y label
        ax.set_ylabel(f"Frequency (MHz)")


    def _rebin_frequency_indices(self, df: u.Quantity) -> np.ndarray:
        """ Get the indices of self.value to rebin in frequencies.
        """
        # Check that there is an even distribution of frequencies
        if df is None:
            return None
        elif not np.unique(np.diff(self.frequency)).size == 1:
            log.error(
                "Impossible to rebin, the frequency range is not uniformly distributed."
            )
            raise Exception()
        else:
            df_mhz = df.to(u.MHz).value
            data_df = self.frequency[1] - self.frequency[0]
            data_df_mhz = data_df.to(u.MHz).value
            n_bins = int(np.floor(df_mhz/data_df_mhz))
            if n_bins > self.frequency.size:
                n_bins = self.frequency.size
            elif n_bins == 0:
                log.warning(
                    f"No frequency rebin applied, {df.to(u.MHz)} <= {data_df.to(u.MHz)}."
                )
                return None
        return np.arange(n_bins)[None, :] + n_bins*np.arange(self.frequency.size//n_bins)[:, None]


    def _rebin_time_indices(self, dt: u.Quantity = None) -> np.ndarray:
        """ Get the indices of self.value to rebin in times.
        """
        # TO DO IN DASK but not optimized yet...
        # import dask.array as da
        # arr = da.from_array(np.arange(100))
        # nbins = 9
        # bins = da.from_array(np.arange(nbins))[None, :] + da.from_array(nbins*np.arange(arr.size//nbins))[:, None]
        # arr.vindex[bins].compute()
        if dt is None:
            return None
        else:
            obs_dt = (self.time[1] - self.time[0]).sec * u.s
            n_bins = int(
                np.floor(
                    (dt/obs_dt).to(u.dimensionless_unscaled).value
                )
            )
            if n_bins > self.time.size:
                n_bins = self.time.size
            elif n_bins == 0:
                log.warning(
                    f"No time rebin applied, {dt.to(u.s)} <= {obs_dt.to(u.s)}."
                )
                return None
        return np.arange(n_bins)[None, :] + n_bins*np.arange(self.time.size//n_bins)[:, None]
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------------- BST ---------------------------- #
# ============================================================= #
class BST(StatisticsData):
    """ Beamlet STatistics reading class.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~BST.analog_beams
            ~BST.digital_beams
            ~BST.analog_beam
            ~BST.beam
            ~BST.analog_pointing
            ~BST.digital_pointing
            ~BST.frequencies
            ~BST.mini_arrays

        .. rubric:: Methods Summary

        .. autosummary::

            ~BST.get

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self, file_name, beam=0):
        super().__init__(file_name=file_name)
        self.beam = beam


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def analog_beams(self):
        """ Lists the analog beam indices used for the recording of BST data.

            :getter: List of analog beam indices.

            :type: :class:`~numpy.ndarray`
        """
        return np.arange(self._meta_data["ana"].size)


    @property
    def digital_beams(self):
        """ Lists the digital beam indices used for the recording of BST data.

            :getter: List of digital beam indices.

            :type: :class:`~numpy.ndarray`
        """
        return np.arange(self._meta_data["bea"].size)


    @property
    def analog_beam(self):
        """ Prints the analog beam index currently corresponding to the digital
            :attr:`~nenupy.io.bst.BST.beam` index.

            :getter: Analog beam index.

            :type: `int`
        """
        log.info(
            f"Retrieving analog beam index associated with beam #{self.beam}."
        )
        return self._meta_data["bea"]["NoAnaBeam"][self.beam]


    @property
    def analog_pointing(self):
        """ Retrieves the analog pointing associated to the current :attr:`~nenupy.io.bst.BST.analog_beam` selected.

            :getter: Analog pointing (time, azimuth, elevation).

            :type:
                `tuple`(:class:`~astropy.time.Time`, :class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`)
        """
        analog_beam = self.analog_beam
        log.info(
            f"Retrieving analog pointing associated with analog beam #{analog_beam}."
        )
        analog_mask = self._meta_data["pan"]["noAnaBeam"] == analog_beam
        pointing = self._meta_data["pan"][analog_mask]
        return Time(pointing["timestamp"]), pointing["az"]*u.deg, pointing["el"]*u.deg


    @property
    def digital_pointing(self):
        """ Retrieves the digital pointing associated to the current :attr:`~nenupy.io.bst.BST.beam` selected.

            :getter: Digital pointing (time, azimuth, elevation).

            :type:
                `tuple`(:class:`~astropy.time.Time`, :class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`)
        """
        log.info(
            f"Retrieving digital pointing associated with beam #{self.beam}."
        )
        digital_mask = self._meta_data["pbe"]["noBeam"] == self.beam
        pointing = self._meta_data["pbe"][digital_mask]
        return Time(pointing["timestamp"]), pointing["az"]*u.deg, pointing["el"]*u.deg


    @property
    def frequencies(self) -> u.Quantity:
        """ Retrieves the sub-band middle frequency of all the sub-bands recorded for the selected :attr:`~nenupy.io.bst.BST.beam`.

            :getter: Sub-band mid frequencies.

            :type: :class:`~astropy.units.Quantity`
        """
        log.info(
            f"Retrieving frequencies associated with beam #{self.beam}."
        )
        beamlets = self._meta_data["bea"]["nbBeamlet"][self.beam]
        subband_half_width = 195.3125*u.kHz
        freqs = self._meta_data["bea"]["freqList"][self.beam][:beamlets]*u.MHz
        return freqs - subband_half_width/2


    @property
    def mini_arrays(self) -> np.ndarray:
        """ Retrieves the list of Mini-Arrays used to record BST data for the selected :attr:`~nenupy.io.bst.BST.analog_beam`.

            :getter: Mini-Arrays list.

            :type: :class:`~numpy.ndarray`
        """
        analog_beam = self.analog_beam
        log.info(
            f"Retrieving Mini-Arrays associated with analog beam #{analog_beam}."
        )
        analog_config = self._meta_data["ana"][analog_beam]
        nb_mini_arrays = analog_config["nbMRUsed"]
        return analog_config["MRList"][:nb_mini_arrays]


    @property
    def beam(self) -> int:
        """ Digital beam index.

            :setter: Beam index.
            
            :getter: Beam index.
            
            :type: `int`
        """
        return self._beam
    @beam.setter
    def beam(self, b: int):
        if b not in self.digital_beams:
            log.error(
                f"Selected beam #{b} should be one of {self.digital_beams}."
            )
            raise IndexError()
        self._beam = b


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self,
            frequency_selection: str = None,
            time_selection: str = None,
            polarization: str = "NW",
            beam: int = 0
        ) -> BST_Slice:
        """ Sub-selects BST data.
            ``frequency_selection`` and ``time_selection``
            arguments accept `str` values formatted as, e.g.,
            ``'>={value}'`` or ``'>={value_1} & <{value_2}'`` or ``'=={value}'``.

            :param frequency_selection:
                Frequency selection. The expected ``'{value}'`` format is frequency units, e.g. ``'>=50MHz'`` or ``'< 1 GHz'``.
                Default is ``None`` (i.e., no selection upon frequency).
            :type frequency_selection:
                `str`
            :param time_selection:
                Time selection. The expected ``'{value}'`` format is ISOT, e.g. ``'>=2022-01-01T12:00:00'``.
                Default is ``None`` (i.e., no selection upon time).
            :type time_selection:
                `str`
            :param polarization:
                Polarization selection, must be either ``'NW'`` or ``'NE'``.
                Default is ``'NW'``.
            :type polarization:
                `str`
            :param beam:
                Digital beam index selection.
                Default is ``0``.
            :type beam:
                `int`
            
            :returns:
                BST data subset.
            :rtype:
                :class:`~nenupy.io.bst.BST_Slice`
            
            :Example:
                .. code-block:: python

                    from nenupy.io.bst import BST

                    bst = BST("/path/to/BST.fits")
                    data = bst.get(
                        frequency_selection="<=52MHz",
                        time_selection='>=2022-01-24T11:08:10 & <2022-01-24T11:14:08',
                        polarization="NW",
                        beam=8
                    )

        """
        self.beam = beam

        # Frequency selection
        frequencies = self.frequencies
        if frequency_selection is None:
            frequency_selection = f">={frequencies.min()} & <= {frequencies.max()}"
        frequency_mask = self._parse_frequency_condition(frequency_selection)(frequencies)
        n_beamlets = self._meta_data["bea"]["nbBeamlet"][self.beam]
        beamlets = self._meta_data["bea"]['BeamletList'][self.beam][:n_beamlets]
        freq_idx = beamlets[frequency_mask]

        # Time selection
        if time_selection is None:
            time_selection = f">={self.time[0].isot} & <= {self.time[-1].isot}"
        time_mask = self._parse_time_condition(time_selection)(self.time)

        # Polarization selection
        polars = self._meta_data['ins']['spol'][0]
        if polarization not in polars:
            log.warning(
                f"`polarization` - unknown '{polarization}', setting default value ('NW')."
            )
            polarization = "NW"
        polar_idx = np.where(polars == polarization)[0]

        log.info(
            "BST selection applied\n"
            f"\t- time ({np.sum(time_mask)}): '{time_selection}'\n"
            f"\t- frequency ({np.sum(frequency_mask)}): '{frequency_selection}'\n"
            f"\t- polarization (1): '{polarization}'\n"
            f"\t- beam (1): {self.beam}"
        )

        return BST_Slice(
            time=self.time[time_mask],
            frequency=frequencies[frequency_mask],
            value=np.squeeze(self.data[
                np.ix_(
                    time_mask,
                    polar_idx,
                    freq_idx
                )
            ]),
            analog_pointing_times=self.analog_pointing[0],
            digital_pointing_times=self.digital_pointing[0]
        )
# ============================================================= #
# ============================================================= #

