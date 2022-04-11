#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST file
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "StatisticsData",
    "ST_Slice"
]


from abc import ABC
import operator
import re
from typing import Callable
from astropy.io import fits
from astropy.time import Time, TimeDelta
from scipy.signal import find_peaks
from astropy.modeling import fitting
from astropy.modeling.models import custom_model
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from nenupy.instru.instrument_tools import sb2freq

import logging
log = logging.getLogger(__name__)


ops = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
}


# ============================================================= #
# ---------------------- StatisticsData ----------------------- #
# ============================================================= #
class StatisticsData(ABC):
    """ """

    def __init__(self, file_name: str):
        self.file_name = file_name
        #self.instrument = None
        #self.pointing = None

        self._meta_data = {}
        self._lazy_load()
    

    def _lazy_load(self):
        """ """
        with fits.open(self.file_name,
            mode='readonly',
            ignore_missing_end=True,
            memmap=True
        ) as f:
            # Metadata loading
            # self.meta['hea'] = f[0].header
            self._meta_data['ins'] = f[1].data
            self._meta_data['obs'] = f[2].data
            self._meta_data['ana'] = f[3].data
            self._meta_data['bea'] = f[4].data
            self._meta_data['pan'] = f[5].data
            self._meta_data['pbe'] = f[6].data

            # # Data loading 
            self.time = Time(f[7].data['JD'], format='jd')
            self.data = f[7].data['data']
            try:
                # For XST data, the frequencies are in the data extension
                self.frequencies = sb2freq(
                    np.unique(f[7].data['xstsubband']).astype("int")
                ) + 195.3125*u.kHz/2 # mid frequency
            except KeyError:
                pass

        return


    @staticmethod
    def _parse_condition(conditions, converter):
        """ """
        condition_list = conditions.replace(" ", "").split("&")

        cond = []
        for condition in condition_list:
            try:
                op = re.search('((>=)|(<=)|(==)|(<)|(>))', condition).group(0)
                val = re.search(f'(?<={op})(.*)', condition).group(0)
            except AttributeError:
                log.error(
                    f"Selection syntax '{condition}' not understood."
                )
                raise
            val = converter(val)
            op = ops[op]
            cond.append( lambda x, op=op, val=val: op(converter(x), val) )

        if len(cond) == 2:
            return lambda x, cond1=cond[0], cond2=cond[1]: operator.and_(cond1(x), cond2(x))
        elif len(cond) == 1:
            return cond[0]
        else:
            raise Exception


    def _parse_time_condition(self, conditions):
        """ """
        return self._parse_condition(conditions, lambda t: Time(t).jd)


    def _parse_frequency_condition(self, conditions):
        """ """
        return self._parse_condition(conditions, lambda f: u.Quantity(f))
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- ST_Slice -------------------------- #
# ============================================================= #
class InconsistentShapeError(Exception):
    """ Error raised when an operation between two ST_Slice
        objects is performed although they have different time
        and freequency axes.
    """

    def __init__(self):
        self.message = (
            "Operation between two ST_Slice instances with "
            "un-identical time and frequency axes is prohibited."
        )
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class ST_Slice:
    """ Class to handle data sub-set from Statistical data.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~ST_Slice.time
            ~ST_Slice.frequency
            ~ST_Slice.value
            ~ST_Slice.analog_pointing_times
            ~ST_Slice.digital_pointing_times

        .. rubric:: Methods Summary

        .. autosummary::

            ~ST_Slice.plot
            ~ST_Slice.rebin
            ~ST_Slice.fit_transit
            ~ST_Slice.flatten_frequency
            ~ST_Slice.flatten_time
            ~ST_Slice.clear_pointing_switch

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self,
            time: Time,
            frequency: u.Quantity,
            value: np.ndarray,
            analog_pointing_times: Time = Time([], format='jd'),
            digital_pointing_times: Time = Time([], format='jd')
        ):
        self._time = time
        self._frequency = frequency
        self._value = value
        self._analog_pointing_times = analog_pointing_times
        self._digital_pointing_times = digital_pointing_times


    def __and__(self, other):
        """ Concatenates two ST_Slice in frequency.
        """
        if not all(self.time == other.time):
            raise ValueError(
                f"The {ST_Slice} objects to concatenate in "
                "frequency do not have equal time axes."
            )

        log.info(
            f"Concatenating in frequency ({self.frequency.size}, {other.frequency.size})."
        )

        if self.frequency.max() < other.frequency.min():
            new_data = np.hstack((self.value, other.value))
            new_freq = np.concatenate((self.frequency, other.frequency))
        else:
            new_data = np.hstack((other.value, self.value))
            new_freq = np.concatenate((other.frequency, self.frequency))
        
        unique_freqs_nb = np.unique(new_freq).size
        if unique_freqs_nb != new_freq.size:
            log.warning(
                f"There are {new_freq.size - unique_freqs_nb} overlaps in the frequency axis."
            )

        return ST_Slice(
            time=self.time,
            frequency=new_freq,
            value=new_data,
            analog_pointing_times=self.analog_pointing_times,
            digital_pointing_times=self.digital_pointing_times
        )


    def __or__(self, other):
        """ Concatenates two ST_Slice in time.
        """
        if not all(self.frequency == other.frequency):
            raise ValueError(
                f"The {ST_Slice} objects to concatenate in "
                "time do not have equal frequency axes."
            )

        log.info(
            f"Concatenating in time ({self.time.size}, {other.time.size})."
        )

        if self.time.max() < other.time.min():
            new_data = np.vstack((self.value, other.value))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_ana_times = Time(np.concatenate((self.analog_pointing_times, other.analog_pointing_times)))
            new_digi_times = Time(np.concatenate((self.digital_pointing_times, other.digital_pointing_times)))
        else:
            new_data = np.vstack((other.value, self.value))
            new_time = Time(np.concatenate((other.time, self.time)))
            new_ana_times = Time(np.concatenate((other.analog_pointing_times, self.analog_pointing_times)))
            new_digi_times = Time(np.concatenate((other.digital_pointing_times, self.digital_pointing_times)))

        unique_times_nb = np.unique(new_time).size
        if unique_times_nb != new_time.size:
            log.warning(
                f"There are {new_time.size - unique_times_nb} overlaps in the time axis."
            )

        return ST_Slice(
            time=new_time,
            frequency=self.frequency,
            value=new_data,
            analog_pointing_times=new_ana_times,
            digital_pointing_times=new_digi_times
        )


    def __getitem__(self, slice_tuple):
        """ (time, frequency) """
        # Expects an explicit tuple of length 2
        if not (isinstance(slice_tuple, tuple) and\
                (len(slice_tuple) == 2) and\
                all([isinstance(s, slice) for s in slice_tuple])
            ):
            raise IndexError("Only tuple of two slices allowed.")
        return ST_Slice(
            time=self.time[slice_tuple[0]],
            frequency=self.frequency[slice_tuple[1]],
            value=self.value[slice_tuple],
            analog_pointing_times=self.analog_pointing_times,
            digital_pointing_times=self.digital_pointing_times
        )


    def __add__(self, other):
        """ """
        return self._operation_with_other(other, np.add)


    def __sub__(self, other):
        """ """
        return self._operation_with_other(other, np.subtract)


    def __mul__(self, other):
        """ """
        return self._operation_with_other(other, np.multiply)
    

    def __truediv__(self, other):
        """ """
        return self._operation_with_other(other, np.divide)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time(self) -> Time:
        """ Data record times.
            
            :getter: Time array.
            
            :type: :class:`~astropy.time.Time`
        """
        return self._time


    @property
    def frequency(self) -> u.Quantity:
        """ Data record frequencies.
            
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


    def rebin(self, dt: u.Quantity = None, df: u.Quantity = None, method: str = "mean"):
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
            :param method:
                Type of method for rebin purpose (either ``'mean'`` or ``'median'``).
                Default is ``'mean'``.
            :type method:
                `str`

            :returns:
                Rebinned data.
            :rtype:
                :class:`~nenupy.io.bst.ST_Slice`
            
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

        # Define the type of rebin
        if method.lower() == "mean":
            rebin_method = np.nanmean
        elif method.lower() == "median":
            rebin_method = np.nanmedian
        else:
            raise ValueError("`method` should either be 'mean' or 'median'.")

        # Dynamic spectrum
        if len(value.shape) == 2:
            rebin_t_indices = self._rebin_time_indices(dt=dt)
            if rebin_t_indices is not None:
                value = rebin_method(
                    value[rebin_t_indices, :],
                    axis=1
                )
                time = Time(np.nanmean(time.jd[rebin_t_indices], axis=1), format='jd')
            rebin_f_indices = self._rebin_frequency_indices(df=df)
            if rebin_f_indices is not None:
                value = rebin_method(
                    value[:, rebin_f_indices],
                    axis=2
                )
                frequency = np.nanmean(frequency[rebin_f_indices], axis=1)
            value = value.squeeze()

        # Spectrum
        elif (len(value.shape) == 1) and (value.size == frequency.size):
            rebin_indices = self._rebin_frequency_indices(df=df)
            if rebin_indices is not None:
                value = rebin_method(value[rebin_indices], axis=1)
                frequency = np.nanmean(frequency[rebin_indices], axis=1)

        # Light curve
        elif (len(value.shape) == 1) and (value.size == time.size):
            rebin_indices = self._rebin_time_indices(dt=dt)
            if rebin_indices is not None:
                value = rebin_method(value[rebin_indices], axis=1)
                time = Time(np.nanmean(time.jd[rebin_indices], axis=1), format='jd')
        else:
            raise ValueError("Problem...")
        
        return ST_Slice(
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

        return ST_Slice(
            time=self.time,
            frequency=self.frequency,
            value=fitted_values,
            analog_pointing_times=self.analog_pointing_times,
            digital_pointing_times=self.digital_pointing_times
        ), transit_time, chi_square


    def flatten_frequency(self):
        """ """
        return self/np.nanmedian(self.value, axis=0)[None, :]


    def flatten_time(self):
        """ """
        return self/np.nanmedian(self.value, axis=1)[:, None]


    def clear_pointing_switch(self,
            flatten_frequency: bool = True,
            pointing_dt: TimeDelta = TimeDelta(6*60, format="sec"),
            peak_sample_error: int = 2,
            return_correction: bool = False
        ):
        r"""

        .. math::
            p(t) = a \log(t) + bt^2 + ct + d


        """

        # Prepare the data by computing the median time profile
        if flatten_frequency:
            # The median time profile is computed after the data have been flattened in frequency.
            # This decreases fit domination by the most sensitive part of the spectrum.
            median_frequency_profile = np.nanmedian(self.value, axis=0)
            median_time_profile = np.nanmedian(self.value/median_frequency_profile[None, :], axis=1)
        else:
            # Keep the frequency responce while performing the median in time.
            median_time_profile = np.nanmedian(self.value, axis=1)
        time_profile_max = median_time_profile.max()
        median_time_profile_normalized = median_time_profile/time_profile_max

        # ------ Find fixed analog pointing time slots ------
        # NenuFar analog pointing is applied once every `pointing_dt` (usually 6 min).
        # Find the minimal distance, in sample unit, between two peaks.
        data_dt_sec = (self.time[1] - self.time[0]).sec
        pointing_dt_sec = pointing_dt.sec
        pointing_dstance = int(np.round(pointing_dt_sec/data_dt_sec))
        # The minimal distance is taken `peak_sample_error` samples short to give an error margin
        minimal_sample_distance_between_peaks = pointing_dstance - peak_sample_error
        # Find peaks over the gradient of the time profile
        time_profile_gradient = np.gradient(median_time_profile_normalized)
        peak_indices, _ = find_peaks(
            -time_profile_gradient,
            height=np.std(time_profile_gradient)*2,
            distance=minimal_sample_distance_between_peaks
        )
        # The gradient shift the peak indices by -1
        peak_indices += 1
        # Add first and last indices if they have not been picked up
        peak_indices = np.insert(peak_indices, 0, 0)
        peak_indices = np.append(peak_indices, self.value.shape[0])
        peak_indices = np.unique(peak_indices)

        # ------ Fit each pointing interval ------
        # Define the fitting function
        @custom_model
        def nenufar_switch_load(time, a=1., b=1., c=1., d=1.):
            """ """
            return a*np.log10(time) + b + c*time**2 + d*time
        # Loop over each pointing slot to fit the function
        switch_correction = np.ones(self.time.size)
        for start_idx, stop_idx in zip(peak_indices[:-1], peak_indices[1:]):
            # Select the time profile portion between two peaks
            interval_profile = median_time_profile_normalized[start_idx:stop_idx]
            # Perform the fitting
            switch_model = nenufar_switch_load(1., interval_profile.min())
            fitter = fitting.LevMarLSQFitter()
            times = 1 + np.arange(interval_profile.size)
            switch_model_fit = fitter(switch_model, times, interval_profile)
            # Update the fit correction
            switch_correction[start_idx:stop_idx] *= switch_model_fit(times)

        # ------ Return the corrected data ------
        new_st_slice = self/switch_correction[:, None]
        if return_correction:
            return new_st_slice, switch_correction
        else:
            return new_st_slice


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _overplot_pointings(self, ax, **kwargs):
        """ Overplot vertical lines
        """
        if kwargs.get("analog_pointing", False):
            for time_i in self.analog_pointing_times:
                ax.axvline(time_i.datetime, color="black", linestyle="-.", alpha=0.5)
        if kwargs.get("digital_pointing", False):
            for time_i in self.digital_pointing_times:
                ax.axvline(time_i.datetime, color="black", linestyle=":", alpha=0.5)


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
        self._overplot_pointings(ax, **kwargs)
        
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
        self._overplot_pointings(ax, **kwargs)
            
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


    def _operation_with_other(self, other, operation: Callable):
        """ """
        if type(other) is type(self):
            # If `other` is a ST_Slice

            # Check if they have the same frequency and time axes
            same_frequency_size = self.frequency.size == other.frequency.size
            same_time_size = self.time.size == other.time.size
            if (not same_frequency_size) or (not same_time_size):
                raise InconsistentShapeError

            # Check if they have the same frequency and time axes
            same_frequencies = np.all(self.frequency == other.frequency)
            same_times = np.all(self.time == other.time)
            if (not same_frequencies) or (not same_times):
                raise InconsistentShapeError
            
            # Find out if any of the instances have their pointings filled up
            analog_pointings = [self.analog_pointing_times, other.analog_pointing_times]
            digital_pointings = [self.digital_pointing_times, other.digital_pointing_times]
            analog_id_max = np.argmax(list(map(len, analog_pointings)))
            digital_id_max = np.argmax(list(map(len, digital_pointings)))
    
            # Return a new object, while performing a numpy operation
            return ST_Slice(
                time=self.time,
                frequency=self.frequency,
                value=operation(self.value, other.value),
                analog_pointing_times=analog_pointings[analog_id_max],
                digital_pointing_times=digital_pointings[digital_id_max]
            )
        else:
            # Perform a normal numpy operation
            return ST_Slice(
                time=self.time,
                frequency=self.frequency,
                value=operation(self.value, other),
                analog_pointing_times=self.analog_pointing_times,
                digital_pointing_times=self.digital_pointing_times
            )
# ============================================================= #
# ============================================================= #

