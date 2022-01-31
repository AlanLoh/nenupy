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
    "BST"
]


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
    """ """

    def __init__(self, time, frequency, value):
        self.time = time
        self.frequency = frequency
        self.value = value

    


    def plot(self, **kwargs):
        """ """
        fig = plt.figure(figsize=kwargs.get("figsize", (15, 7)))
        ax = fig.add_subplot(111)
        
        data = self.value.T
        if kwargs.get("decibel", True):
            data = 10*np.log10(data)
        
        if len(data.shape) == 2:
            self._plot_dynamic_spectrum(data, ax, fig)
        elif (len(data.shape) == 1) and (data.size == self.frequency.size):
            self._plot_spectrum(data, ax, fig)
        elif (len(data.shape) == 1) and (data.size == self.time.size):
            self._plot_lightcurve(data, ax, fig)
        else:
            raise ValueError("Problem...")

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
    

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _plot_spectrum(self, data, ax, fig, **kwargs):
        """ """
        ax.plot(self.frequency.to(u.MHz).value, data)

        # X label
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )
        fig.autofmt_xdate()
        ax.set_xlabel("Frequency (MHz)")

        # Y label
        ax.set_ylabel("dB" if kwargs.get("decibel", True) else "Amp")


    def _plot_lightcurve(self, data, ax, fig, **kwargs):
        """ """
        ax.plot(self.time.datetime, data)

        # X label
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )
        fig.autofmt_xdate()
        ax.set_xlabel(f"Time (UTC since {self.time[0].isot})")

        # Y label
        ax.set_ylabel("dB" if kwargs.get("decibel", True) else "Amp")


    def _plot_dynamic_spectrum(self, data, ax, fig, **kwargs):
        """ """
        im = ax.pcolormesh(
            self.time.datetime,
            self.frequency.to(u.MHz).value,
            data,
            shading="auto",
            cmap=kwargs.get("cmap", "YlGnBu_r"),
            vmin=kwargs.get("vmin", np.nanpercentile(data, 5)),
            vmax=kwargs.get("vmax", np.nanpercentile(data, 95))
        )

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
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------------- BST ---------------------------- #
# ============================================================= #
class BST(StatisticsData):
    """ """

    def __init__(self, file_name, beam=0):
        super().__init__(file_name=file_name)
        self.beam = beam
    

    @property
    def analog_beams(self):
        """ """
        return np.arange(self._meta_data["ana"].size)


    @property
    def digital_beams(self):
        """ """
        return np.arange(self._meta_data["bea"].size)


    @property
    def analog_beam(self):
        """ """
        return self._meta_data["bea"]["NoAnaBeam"][self.beam]


    @property
    def analog_pointing(self):
        """ """
        analog_mask = self._meta_data["pan"]["noAnaBeam"] == self.analog_beam
        pointing = self._meta_data["pan"][analog_mask]
        return Time(pointing["timestamp"]), pointing["az"]*u.deg, pointing["el"]*u.deg


    @property
    def digital_pointing(self):
        """ """
        digital_mask = self._meta_data["pbe"]["noBeam"] == self.beam
        pointing = self._meta_data["pbe"][digital_mask]
        return Time(pointing["timestamp"]), pointing["az"]*u.deg, pointing["el"]*u.deg


    @property
    def frequencies(self):
        """ """
        beamlets = self._meta_data["bea"]["nbBeamlet"][self.beam]
        subband_half_width = 195.3125*u.kHz
        freqs = self._meta_data["bea"]["freqList"][self.beam][:beamlets]*u.MHz
        return freqs - subband_half_width/2
    

    @property
    def mini_arrays(self):
        """ """
        analog_config = self._meta_data["ana"][self.analog_beam]
        nb_mini_arrays = analog_config["nbMRUsed"]
        return analog_config["MRList"][:nb_mini_arrays]


    @property
    def beam(self):
        """ """
        return self._beam
    @beam.setter
    def beam(self, b):
        if b not in self.digital_beams:
            log.error(f"Selected beam {b} should be one of {self.digital_beams}.")
            raise IndexError()
        self._beam = b


    def get(self,
            frequency_selection: str = None,
            time_selection: str = None,
            polarization: str = "NW",
            beam: int = 0
        ):
        """ """
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
        polar_idx = np.where(polars == polarization)[0]
    
        return BST_Slice(
            time=self.time[time_mask],
            frequency=frequencies[frequency_mask],
            value=np.squeeze(self.data[
                np.ix_(
                    time_mask,
                    polar_idx,
                    freq_idx
                )
            ])
        )
# ============================================================= #
# ============================================================= #

