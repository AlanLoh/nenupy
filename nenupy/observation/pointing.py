#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **********************
    Observation Monitoring
    **********************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'readPointing',
    'plotPointing'
]


import numpy as np
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from os.path import basename, isfile

from nenupy.astro import altazProfile, ho_coord, getSource, toAltaz


# ============================================================= #
# ------------------------ readPointing ----------------------- #
# ============================================================= #
def readPointing(pointing_file=None):
    """
    """
    if pointing_file is None:
        return None
    if not isfile(pointing_file):
        raise FileNotFoundError(
            f'Unable to find {pointing_file}.'
        )
    if pointing_file.endswith('.altazA'):
        pointing = np.loadtxt(
            pointing_file,
            skiprows=3,
            dtype={
                'names': (
                    'time',
                    'anabeam',
                    'az',
                    'el',
                    'az_cor',
                    'el_cor',
                    'freq',
                    'el_eff'
                ),
                'formats': (
                    'S20',
                    'i4',
                    'f4',
                    'f4',
                    'f4',
                    'f4',
                    'S5',
                    'f4'
                )
            }
        )
    elif pointing_file.endswith('.altazB'):
        pointing = np.loadtxt(
            pointing_file,
            skiprows=3,
            dtype={
                'names': (
                    'time',
                    'anabeam',
                    'digibeam',
                    'az',
                    'el',
                    'l',
                    'm',
                    'n',
                ),
                'formats': (
                    'S20',
                    'i4',
                    'i4',
                    'f4',
                    'f4',
                    'f4',
                    'f4',
                    'f4'
                )
            }
        )
    else:
        raise ValueError(
            'Wrong file format.'
        )
    return pointing
# ============================================================= #


# ============================================================= #
# ------------------------ plotPointing ----------------------- #
# ============================================================= #
def plotPointing(altaza=None, altazb=None, sourceName=None):
    """
    """
    fig, axs = plt.subplots(
        2 if sourceName is None else 3,
        1,
        sharex=True,
        figsize=(15, 10)
    )
    fig.subplots_adjust(hspace=0)

    if altaza is not None:
        if not altaza.endswith('.altazA'):
            raise TypeError(
                'Wrong file format for altaza.'
            )
        title = basename(altaza).replace('.altazA', '')
        aza = readPointing(altaza)
        azaTime = Time(aza['time'])
        tmin = azaTime[0]
        tmax = azaTime[-1]
        for abeam in np.unique(aza['anabeam']):
            mask = aza['anabeam'] == abeam
            axs[0].plot(
                azaTime[mask][:-1].datetime, # Do not show last pointing / back to zenith
                aza['az'][mask][:-1],
                linewidth='5.5',
                label=f'Analog requested (#{abeam})'
            )
            axs[0].plot(
                azaTime[mask][:-1].datetime,
                aza['az_cor'][mask][:-1],
                linewidth='3',
                label=f'Analog corrected (#{abeam})'
            )
            axs[1].plot(
                azaTime[mask][:-1].datetime,
                aza['el'][mask][:-1],
                linewidth='5.5',
                label=f'Analog requested (#{abeam})'
            )
            axs[1].plot(
                azaTime[mask][:-1].datetime,
                aza['el_cor'][mask][:-1],
                linewidth='3',
                label=f'Analog corrected (#{abeam})'
            )
            axs[1].plot(
                azaTime[mask][:-1].datetime,
                aza['el_eff'][mask][:-1],
                linewidth='2',
                label=f'Analog beamsquint corrected (#{abeam})'
            )

    if altazb is not None:
        if not altazb.endswith('.altazB'):
            raise TypeError(
                'Wrong file format for altazb.'
            )
        title = basename(altazb).replace('.altazB', '')
        azb = readPointing(altazb)
        azbTime = Time(azb['time'])
        tmin = azbTime[0]
        tmax = azbTime[-1]
        for dbeam in np.unique(azb['digibeam']):
            mask = azb['digibeam'] == dbeam
            axs[0].plot(
                azbTime[mask].datetime,
                azb['az'][mask],
                linewidth='1',
                label=f'Numeric (#{dbeam})'
            )
            axs[1].plot(
                azbTime[mask].datetime,
                azb['el'][mask],
                linewidth='1',
                label=f'Numeric (#{dbeam})'
            )

    if (altaza is None) and (altazb is None):
        raise ValueError(
            'At least one pointing file should be given.'
        )

    if sourceName is not None:
        srcTime, srcAz, srcEl = altazProfile(
            sourceName=sourceName,
            tMin=tmin,
            tMax=tmax,
            dt=(tmax - tmin)/20. - TimeDelta(0.5, format='sec')
        )
        axs[0].plot(
            srcTime.datetime,
            srcAz,
            color='black',
            linestyle='--',
            alpha=0.8,
            label=f'{sourceName}'
        )
        axs[1].plot(
            srcTime.datetime,
            srcEl,
            color='black',
            linestyle='--',
            alpha=0.8,
            label=f'{sourceName}'
        )

        if altaza is not None:
            for abeam in np.unique(aza['anabeam']):
                mask = aza['anabeam'] == abeam
                analog = ho_coord(
                    alt=aza['el'][mask][:-1],
                    az=aza['az'][mask][:-1],
                    time=azaTime[mask][:-1]
                )
                analog_cor = ho_coord(
                    alt=aza['el_cor'][mask][:-1],
                    az=aza['az_cor'][mask][:-1],
                    time=azaTime[mask][:-1]
                )
                analog_bscor = ho_coord(
                    alt=aza['el_eff'][mask][:-1],
                    az=aza['az_cor'][mask][:-1],
                    time=azaTime[mask][:-1]
                )
                source = toAltaz(
                    skycoord=getSource(
                        name=sourceName,
                        time=azaTime[mask][:-1]
                    ),
                    time=azaTime[mask][:-1]
                )
            axs[2].plot(
                azaTime[mask][:-1].datetime,
                source.separation(analog).deg,
                label=f'Analog (#{abeam})'
            )
            axs[2].plot(
                azaTime[mask][:-1].datetime,
                source.separation(analog_cor).deg,
                label=f'Analog corrected (#{abeam})'
            )
            axs[2].plot(
                azaTime[mask][:-1].datetime,
                source.separation(analog_bscor).deg,
                label=f'Analog beamsquint corrected (#{abeam})'
            )

        if altazb is not None:
            for dbeam in np.unique(azb['digibeam']):
                mask = azb['digibeam'] == dbeam
                numeric = ho_coord(
                    alt=azb['el'][mask],
                    az=azb['az'][mask],
                    time=azbTime[mask]
                )
                source = toAltaz(
                    skycoord=getSource(
                        name=sourceName,
                        time=azbTime[mask]
                    ),
                    time=azbTime[mask]
                )
            axs[2].plot(
                azbTime[mask].datetime,
                source.separation(numeric).deg,
                label=f'Numeric (#{dbeam})'
            )

        axs[2].set_ylabel(f'Separation from {sourceName} (deg)')
        axs[2].legend()
        axs[2].set_xlabel(f'UTC Time (since {tmin.iso})')
    else:
        axs[1].set_xlabel(f'UTC Time (since {tmin.iso})')

    axs[0].set_title('Pointing - ' + title)
    axs[0].set_ylabel('Azimuth (deg)')
    axs[0].legend()
    axs[1].set_ylabel('Elevation (deg)')
    axs[1].legend()
    plt.show()
# ============================================================= #

