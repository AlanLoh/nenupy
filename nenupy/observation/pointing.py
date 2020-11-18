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
    'plotPointing',
    'AnalogPointing'
]


import os
import numpy as np
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from os.path import basename, isfile

from nenupy.astro import altazProfile, ho_coord, getSource, toAltaz
from nenupy.miscellaneous import accepts


# ============================================================= #
# ------------------------ readPointing ----------------------- #
# ============================================================= #
def readPointing(pointing_file=None):
    """
        .. versionadded:: 1.1.0

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
                    'U20',
                    'i4',
                    'f4',
                    'f4',
                    'f4',
                    'f4',
                    'U5',
                    'f4'
                )
            }
        )
        pointing['freq'] = list(
            map(
                lambda x: x.replace('MHz', ''),
                pointing['freq']
            )
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
                    'U20',
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
        .. versionadded:: 1.1.0

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


# ============================================================= #
# ---------------------- AnalogPointing ----------------------- #
# ============================================================= #
class AnalogPointing(object):
    """

        .. versionadded:: 1.1.0

    """

    def __init__(self, filename, **kwargs):
        self._autoUpdate = kwargs.get('autoUpdate', True)
        self.pointingOrders = {}
        self.filename = filename


    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('{} exepected.'.format(self.__class__))
        new = AnalogPointing(
            filename=self.filename + other.filename,
            autoUpdate=False
        )
        new.pointingOrders = {**self.pointingOrders, **other.pointingOrders}
        return new


    def __radd__(self, other):
        if other==0:
            return self
        else:
            return self.__add__(other)


    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('{} exepected.'.format(self.__class__))
        newFileList = self.filename.copy()
        newPointingOrders = self.pointingOrders.copy()
        for fileToRemove in list(map(basename, other.filename)):
            selfBaseNames = list(map(basename, newFileList))
            try:
                fileIndex = selfBaseNames.index(fileToRemove)
                removedFile = newFileList.pop(fileIndex)
            except ValueError:
                # SST file in other not in self
                pass
            try:
                fileKeys = list(newPointingOrders.keys())
                baseKeys = list(map(basename, fileKeys))
                fileIndex = baseKeys.index(fileToRemove)
                del newPointingOrders[fileKeys[fileIndex]]
            except (ValueError, KeyError):
                # SST file in other not in self
                pass
        new = AnalogPointing(
            filename=newFileList,
            autoUpdate=False
        )
        new.pointingOrders = newPointingOrders
        return new

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def filename(self):
        """
        """
        return sorted(self._filename)
    @filename.setter
    def filename(self, f):
        if not isinstance(f, list):
            f = [f]
        for fi in f:
            self._fill(fi)
        self._filename = f


    @property
    def tMax(self):
        """
        """
        return max([max(self.pointingOrders[f]['time']) for f in self.pointingOrders.keys()])


    @property
    def pointingTimes(self):
        """
        """
        times = []
        for key in sorted(self.pointingOrders.keys()):
            times += self.pointingOrders[key]['time'].isot.tolist()
        return Time(times, precision=0)
    
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @accepts(object, Time)
    def atTime(self, time):
        """
        """
        if time >= self.tMax:
            lastFile = sorted(self.pointingOrders.keys())[-1]
            pointing = self.pointingOrders[lastFile]
            return {
                    'time': pointing['time'][-1],
                    'abeam': pointing['abeam'][-1],
                    'pDesired': pointing['pDesired'][-1],
                    'pCorrected': pointing['pCorrected'][-1],
                    'pEff': pointing['pEff'][-1],
                    'beamSquintFreq': pointing['beamSquintFreq'][-1]
                }

        for key in self.pointingOrders.keys():
            pointing = self.pointingOrders[key]
            times = pointing['time']
            mask = [(times[i] <= time) and (times[i+1] > time) for i in range(times.size - 1)]
            mask += [False]
            if any(mask):
                return {
                    'time': pointing['time'][mask][0],
                    'abeam': pointing['abeam'][mask][0],
                    'pDesired': pointing['pDesired'][mask][0],
                    'pCorrected': pointing['pCorrected'][mask][0],
                    'pEff': pointing['pEff'][mask][0],
                    'beamSquintFreq': pointing['beamSquintFreq'][mask][0]
                }

        return None


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _fill(self, filename):
        """
        """
        if not self._autoUpdate:
            return

        if not filename.endswith('.altazA'):
            raise ValueError(
                '`*.altazA file(s) required.`'
            )
        pointing = readPointing(filename)

        times = Time(pointing['time'], precision=0)
        
        self.pointingOrders[os.path.basename(filename)] = {
            'time': times,
            'abeam': pointing['anabeam'],
            'pDesired': ho_coord(
                alt=pointing['el'],
                az=pointing['az'],
                time=times
            ),
            'pCorrected': ho_coord(
                alt=pointing['el_cor'],
                az=pointing['az_cor'],
                time=times
            ),
            'pEff': ho_coord(
                alt=pointing['el_eff'],
                az=pointing['az_cor'],
                time=times
            ),
            'beamSquintFreq': pointing['freq'].astype(float)

        }
# ============================================================= #

