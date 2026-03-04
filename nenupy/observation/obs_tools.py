#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *****************
    Observation Tools
    *****************
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2025, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "in_analog_beam_max_frequency",
    "plot_current_pointing"
]


from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import patheffects

from nenupy.astro.astro_tools import SolarSystemSource, altaz_to_radec, solar_system_source, radec_to_altaz
from nenupy.astro.target import FixedTarget, SolarSystemTarget
from nenupy.astro.skymodel import Skymodel
from nenupy.astro.sky import Sky
from nenupy.astro.pointing import Pointing
from nenupy import nenufar_position
from nenupy.instru import MiniArray, Polarization


# ============================================================= #
# --------------- in_analog_beam_max_frequency ---------------- #
# ============================================================= #
def in_analog_beam_max_frequency(
        source1: Union[str, SkyCoord],
        source2: Union[str, SkyCoord],
        time: Time = Time.now(),
    ) -> u.Quantity:
    """ Given two sources at any time(s), computes the maximal
        frequency(ies) in order to observe them simultaneously within
        the same NenuFAR analog beam.

        :Example:
            .. code-block:: python

                in_analog_beam_max_frequency(
                    source1="Sun",
                    source2="Moon",
                    time=Time("2022-02-21T12:00:00")
                )

                in_analog_beam_max_frequency(
                    source1="Sun",
                    source2="PSR J2330-2005",
                    time=Time(["2022-02-21T12:00:00", "2022-02-21T15:00:00"])
                )

                in_analog_beam_max_frequency(
                    source1="Saturn",
                    source2="Sun",
                    time=Time("2022-02-04T06:00:00") + np.arange(10)*TimeDelta(2, format="jd")
                )

    """

    def _select_target_type(source_name):
        if isinstance(source_name, SkyCoord):
            return FixedTarget(source_name, time=time)
        # Check whether the source name matches a solar system object
        if source_name.upper() in SolarSystemSource._member_names_:
            return SolarSystemTarget.from_name(source_name, time=time)
        else:
            return FixedTarget.from_name(source_name, time=time)

    # Initialize the two sources
    src1 = _select_target_type(source1)
    src2 = _select_target_type(source2)
    # Compute their angular separations
    src_separations = src1.separation(src2)
    if src_separations.isscalar:
        src_separations = src_separations.reshape((1,))

    # Evaluate the analog beam Half Width at Half Maximum over the frequency range
    frequencies = np.linspace(15, 85, 200)*u.MHz
    ma = MiniArray()
    analog_beam_fwhm = ma.angular_resolution(frequency=frequencies)
    analog_beam_hwhm = analog_beam_fwhm/2

    # Compute the maximal frequencies at which the two sources
    # are still within the HWHM of the analog beam
    max_frequencies = u.Quantity(np.zeros(time.size), unit="MHz")
    within_analog_beam = analog_beam_hwhm[None, :] >= src_separations[:, None]
    for i in range(time.size):
        try:
            max_frequencies[i] = frequencies[within_analog_beam[i, :]].max()
        except ValueError:
            # The separation is greater than the analog beam at any freq
            pass
    return max_frequencies
# ============================================================= #
# ============================================================= #

# ============================================================= #
# ------------------- plot_current_pointing ------------------- #
# ============================================================= #
def plot_current_pointing(figname: str, time: Time, frequency: u.Quantity = 50 * u.MHz, analog_pointing_file: str = None, source_positions: bool = False, **kwargs) -> None:
    """
    Display the GSM and the analog beam of NenuFAR.
    The map is projected in horizontal coordinates, in log scale.
    The analog beam of NenuFAR is simulated as the sum of the analog beams of each of the six different MA rotations.
    It is displayed as a contour plot, in log scale as well. 

    Parameters
    ----------
    figname : `str`
        Name of the figure to plot.
        If a figure name is provided (e.g., with an extension .png) the figure is saved.
        If `"return"` the figure and ax are returned.
        If `None` or `""` the figure is just shown.
    time : :class:`~astropy.time.Time`
        Time at which the sky and the pointing are represented.
        If the value is not within the ``analog_pointing_file``, a zenithal pointing is assumed.
    frequency : :class:`~astropy.units.Quantity`, optional
        Frequency at which the sky model and the beam simulation should be computed, by default 50 MHz 
    analog_pointing_file : `str`, optional
        NenuFAR analog pointing file (can be retrieved for each observation), by default `None`
    source_positions : `bool`, optional
        Display the positions of the main radio sources, by default `False`
    figsize : `tuple`, optional
        Size of the figure, by default `(7, 7)`
    n_azimuths : `int`, optional
        Number of azimuths explored, by default `500`
    n_elevations : `int`, optional
        Number of elevations explored, by default `300`
    cmap : `str`, optional
        Colormap, by default `"magma"`
    contour_values : `bool`, optional
        Show the values of the contours, by default `True`
    dpi : `int`, optional
        Figure DPI if saved, by default `150`
    """
    
    # Check time input
    if not time.isscalar:
        raise ValueError("time should be a scalar.")

    # Get the current pointing
    if analog_pointing_file is None:
        phase_center_altaz = SkyCoord(
            0, 90,
            unit="deg",
            frame=AltAz(
                obstime=time,
                location=nenufar_position
            )
        )
    else:
        pointing = Pointing.from_file(
            analog_pointing_file, include_corrections=False
        )[time.reshape((1,))]
        phase_center_altaz = pointing.custom_ho_coordinates[0]
    
    # Check frequency input
    if not isinstance(frequency, u.Quantity):
        frequency *= u.MHz
        log.warning("No unit provided for the frequency, considered to be MHz.")
    elif frequency.unit.physical_type != "frequency":
        raise ValueError(f"frequency {frequency} is expressed in other physcial units.")
    if not frequency.isscalar:
        raise ValueError("Frequency should be a scalar.")

    # Generate the GSM map
    sm = Skymodel(frequency=frequency)
    azimuths, elevations, radec, gsm_map = sm.altaz_map_at(
        time=time,
        n_azimuths=kwargs.get("n_azimuths", 500),
        n_elevations=kwargs.get("n_elevations", 300),
        return_coords=True
    )

    # Freeze the min and max values of the GSM displayed
    # no matter the selected time
    data_min = np.percentile(sm.data, 1)
    data_max = np.percentile(sm.data, 99.9)

    # Initialize a polar plot
    fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
    ax = fig.add_subplot(polar=True)

    # Plot the GSM
    im = ax.pcolormesh(
        np.radians(azimuths), elevations, gsm_map,
        norm="log", cmap=kwargs.get("cmap", "magma"),
        vmin=data_min, vmax=data_max,
        edgecolors="face", shading="gouraud"
    )

    # Analog beam simulation
    analog_beam = np.zeros(gsm_map.shape)
    # Sum the analog beams of the 6 different MA rotations
    # the following MA indices each represent a unique 10deg rotation
    for ma_index in [0, 11, 3, 1, 13, 7]:
        ma = MiniArray(index=ma_index)
        af_sky = ma.beam(
            sky=Sky(
                coordinates=radec,
                time=time,
                frequency=frequency,
                polarization=Polarization.NW
            ),
            pointing=Pointing(
                coordinates=altaz_to_radec(phase_center_altaz),
                time=time
            )
        )
        # Sum and normalize the beams
        analog_beam += af_sky[0, 0, 0].value.compute().reshape(gsm_map.shape)
    analog_beam_normalized = analog_beam / analog_beam.max()

    # Plot the contours of the analog beam on top of the GSM
    cont = ax.contour(
        np.radians(azimuths), elevations,
        analog_beam_normalized,
        # levels=np.logspace(-1.5, 0, 6),
        levels=np.logspace(np.log(np.percentile(analog_beam_normalized, 98)), 0, 7),
        colors="#97bf61",
        linewidths=0.7
    )
    if kwargs.get("contour_values", True):
        ax.clabel(cont, inline=1, fontsize=6)

    # Polar axes modifications
    ax.set_rlim(90, 0) # horizon at the plot border
    ax.set_theta_zero_location("N") # North on top
    ax.grid(alpha=0.5)

    # Insert degrees symbols for elevation ticks
    rad2fmt = lambda x, pos : f"{x:.0f}°"
    ax.yaxis.set_major_formatter(FuncFormatter(rad2fmt))
    ax.tick_params(axis="y", colors="gray", labelsize=9)

    # Add NSEW points
    nesw_labels = np.array(["N", "E", "S", "W"])
    nesw_az = np.radians(np.array([0, 90, 180, 270]))
    nesw_alt = np.ones(4) * 4
    for label, az, alt in zip(nesw_labels, nesw_az, nesw_alt):
        ax.text(
            x=az,
            y=alt,
            s=label,
            color="tab:orange",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="black")],
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True
        )

    # Overplot source positions
    if source_positions:
        sources = ["Sun", "Jupiter", "Cas A", "Cyg A", "Tau A", "Vir A", "3C 196", "3C 218", "3C 219", "3C 254", "3C 273", "3C 295", "3C 338", "3C 348", "3C 380", "3C 84"]
        for source in sources:
            try:
                src = solar_system_source(name=source, time=time)
            except:
                src = SkyCoord.from_name(source)
            src_altaz = radec_to_altaz(src, time=time)
            if src_altaz.alt.deg <= 0:
                continue
            ax.scatter(
                src_altaz.az.rad, src_altaz.alt.deg,
                s=50, facecolors="none", edgecolors="gray"
            )
            ax.text(
                src_altaz.az.rad, src_altaz.alt.deg,
                f"  {source}", fontsize=7, color="white"
            ).set_clip_on(True)

    # GSM colorbar
    cb = fig.colorbar(im, ax=ax, pad=0.08, shrink=0.7, anchor=(0, 0.7))
    cb.set_label("Brightness temperature (K)")

    # Credits
    ax.annotate("nenupy analog beam simulation\nGSM (de Oliveira-Costa et al., 2008)",
        xy=(0, 0),
        xytext=(0.91, 0.075),
        textcoords="figure fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9
    )
    time.precision = 0
    ax.set_title(
        f"NenuFAR pointing (az={phase_center_altaz.az.deg:.1f}, "
        f"el={phase_center_altaz.alt.deg:.1f}), {time.iso}, {frequency:.1f}",
        fontsize=11
    )

    fig.tight_layout()

    # Save or not the figure
    if (figname is None) or (figname == ""):
        plt.show()
    elif figname.lower() == "return":
        return fig, ax
    else:
        fig.savefig(
            figname,
            dpi=kwargs.get("dpi", 150),
            bbox_inches="tight",
            transparent=True
        )
    plt.close("all")
# ============================================================= #
# ============================================================= #
