#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **********
    Near-Field
    **********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'NearField'
]


import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import LinearLocator
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

import nenupy
from nenupy.astro import (
    l93_to_etrs,
    etrs_to_enu
)
from nenupy.instru import getMAL93

import logging
log = logging.getLogger(__name__)


# Nancay buildings in ENU
buildingsENU = np.array([
    [27.75451691, -51.40993459, 7.99973228],
    [20.5648047, -59.79299576, 7.99968629],
    [167.86485612, 177.89170175, 7.99531119]
])


# ============================================================= #
# ------------------------- NearField ------------------------- #
# ============================================================= #
class NearField(object):
    """
    """

    def __init__(
        self,
        nfImage,
        antNames,
        meanFreq,
        obsTime,
        simuSources,
        radius
        ):
        self.nfImage = nfImage
        self.antNames = antNames
        self.meanFreq = meanFreq
        self.obsTime = obsTime
        self.simuSources = simuSources
        self.radius = radius


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def nfImage(self):
        """
        """
        return self._nfImage
    @nfImage.setter
    def nfImage(self, nfi):
        if not isinstance(nfi, np.ndarray):
            raise TypeError(
                'nfImage should be a numpy array.'
            )
        if len(nfi.shape) != 2:
            raise ValueError(
                'nfImage should be a 2D image.'
            )
        if nfi.shape[0] != nfi.shape[1]:
            raise ValueError(
                'nfImage should be a square.'
            )
        self._nfImage = nfi 
        return

    @property
    def antNames(self):
        """
        """
        return self._antNames
    @antNames.setter
    def antNames(self, names):
        if not isinstance(names, np.ndarray):
            raise TypeError(
                'antNames should be a numpy array.'
            )
        self._antNames = names


    @property
    def meanFreq(self):
        """
        """
        return self._meanFreq
    @meanFreq.setter
    def meanFreq(self, freq):
        if not isinstance(freq, u.Quantity):
            raise TypeError(
                'freq should be an astropy quantity.'
            )
        if not freq.isscalar:
            raise ValueError(
                'freq should be scalar.'
            )
        self._meanFreq = freq


    @property
    def obsTime(self):
        """
        """
        return self._obsTime
    @obsTime.setter
    def obsTime(self, time):
        if not isinstance(time, Time):
            raise TypeError(
                'obsTime should be an astropy Time.'
            )
        if not time.isscalar:
            raise ValueError(
                'obsTime should be scalar.'
            )
        self._obsTime = time


    @property
    def simuSources(self):
        """
        """
        return self._simuSources
    @simuSources.setter
    def simuSources(self, simu):
        if not isinstance(simu, dict):
            raise TypeError(
                'simuSources should be a dictionnary.'
            )
        for key in simu.keys():
            if not isinstance(simu[key], np.ndarray):
                raise TypeError(
                    'simuSources[{}] is not a numpy array'.format(
                        key
                    )
                )
            if simu[key].shape != self.nfImage.shape:
                raise ValueError(
                    'simuSources[{}] has a shape {}, different than nfImage {}'.format(
                        key,
                        simu[key].shape,
                        self.nfImage.shape
                    )
                )
        self._simuSources = simu


    @property
    def radius(self):
        """
        """
        return self._radius
    @radius.setter
    def radius(self, r):
        if not isinstance(r, u.Quantity):
            raise TypeError(
                'radius should be an astropy quantity.'
            )
        if not r.isscalar:
            raise ValueError(
                'radius should be scalar.'
            )
        self._radius = r


    @property
    def nPix(self):
        """
        """
        return self.nfImage.shape[0]
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def fromFile(cls, filename):
        """
        """
        reservedNames = [
            'PRIMARY',
            'NEAR-FIELD',
            'MINI-ARRAYS'
        ]
        hdus = fits.open(filename)
        nf = cls(
            nfImage=hdus['NEAR-FIELD'].data,
            antNames=hdus['MINI-ARRAYS'].data,
            meanFreq=hdus['NEAR-FIELD'].header['FREQUENC']*u.MHz,
            obsTime=Time(hdus['NEAR-FIELD'].header['DATE-OBS']),
            simuSources={
                hdu.header['SOURCE']: hdu.data for hdu in hdus if hdu.name not in reservedNames
            },
            radius=hdus['NEAR-FIELD'].header['RADIUS']*u.m
        )
        return nf


    def saveFits(self, filename):
        """
        """
        # Header
        primaryHeader = fits.Header()
        primaryHeader['OBSERVER'] = 'Isaac Newton'
        primaryHeader['AUTHOR'] = 'nenupy {}'.format(nenupy.__version__)
        primaryHeader['DATE'] = Time.now().isot
        primaryHeader['INSTRUME'] = 'XST'
        primaryHeader['OBSERVER'] = 'NenuFAR-TV'
        primaryHeader['ORIGIN'] = 'Station de Radioastronomie de Nancay, LESIA, Observatoire de Paris'
        primaryHeader['REFERENC'] = 'Alan Loh and the NenuFAR team, nenupy, 2020 (DOI: 10.5281/zenodo.3775196.)'
        primaryHeader['TELESCOP'] = 'NenuFAR'

        primaryHDU = fits.PrimaryHDU(
            header=primaryHeader
        )

        # Near-Field
        nearfieldHeader = fits.Header()
        nearfieldHeader['NAXIS'] = 2
        nearfieldHeader['NAXIS1'] = self.nPix
        nearfieldHeader['NAXIS2'] = self.nPix
        nearfieldHeader['DATE-OBS'] = (
            self.obsTime.isot,
            'Mean observation UTC date'
        )
        nearfieldHeader['DATAMIN'] = self.nfImage.min()
        nearfieldHeader['DATAMAX'] = self.nfImage.max()
        nearfieldHeader['FREQUENC'] = (
            self.meanFreq.to(u.MHz).value,
            'Mean observing frequency in MHz.'
        )
        nearfieldHeader['DESCRIPT'] = 'Near-Field image.'
        nearfieldHeader['RADIUS'] = (
            self.radius.value,
            'Radius of the ground (in m).'
        )

        nearfieldHDU = fits.ImageHDU(
            data=self.nfImage,
            header=nearfieldHeader,
            name='Near-Field'
        )
        
        # Mini-Arrays
        antennaHeader = fits.Header()
        antennaHeader['DESCRIPT'] = 'Mini-Array names'
        antennaHDU = fits.ImageHDU(
            data=self.antNames,
            header=antennaHeader,
            name='Mini-Arrays'
        )

        # HDU list
        hduList = fits.HDUList(
            [
                primaryHDU,
                nearfieldHDU,
                antennaHDU
            ]
        )

        for src in self.simuSources:
            hduName = src.replace(' ', '_')
            srcHeader = fits.Header()
            srcHeader['NAXIS'] = 2
            srcHeader['NAXIS1'] = self.nPix
            srcHeader['NAXIS2'] = self.nPix
            srcHeader['DATE-OBS'] = (
                self.obsTime.isot,
                'Mean observation UTC date'
            )
            srcHeader['DATAMIN'] = self.simuSources[src].min()
            srcHeader['DATAMAX'] = self.simuSources[src].max()
            srcHeader['FREQUENC'] = (
                self.meanFreq.to(u.MHz).value,
                'Mean observing frequency in MHz.'
            )
            srcHeader['SOURCE'] = (
                src,
                'Name of the source imprint on the near-field'
            )
            srcHeader['DESCRIPT'] = 'Normalized sky source imprint on the near-field.'
            srcHeader['RADIUS'] = (
                self.radius.value,
                'Radius of the ground (in m).'
            )
            srcHDU = fits.ImageHDU(
                data=self.simuSources[src],
                name=hduName,
                header=srcHeader
            )
            hduList.append(srcHDU)

        hduList.writeto(filename, overwrite=True)


    def plot(self, figname=''):
        """
        """
        radius = self.radius.to(u.m).value

        # Mini-Array positions in ENU coordinates
        mapos_l93 = getMAL93(self.antNames)
        mapos_etrs = l93_to_etrs(mapos_l93)
        maposENU = etrs_to_enu(mapos_etrs)

        # Display
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot the image of the near-field dB scaled
        nfImage_db = 10*np.log10(self.nfImage)
        ax.imshow(
            np.flipud(nfImage_db), # This needs to be understood...
            cmap='YlGnBu_r',
            extent=[-radius, radius, -radius, radius],
            zorder=0
        )
        # Show the contour of the simulated source imprints
        groundGranularity = np.linspace(-radius, radius, self.nPix)
        posx, posy = np.meshgrid(groundGranularity, groundGranularity)
        dist = np.sqrt(posx**2 + posy**2)
        for src in self.simuSources.keys():
            srcImprint = self.simuSources[src]
            srcImprint /= srcImprint.max()
            ax.contour(
                srcImprint,
                np.arange(0.8, 1, 0.04),
                #colors='black',
                cmap='copper',
                alpha=0.5,
                extent=[-radius, radius, -radius, radius],
                zorder=5
            )
            maxY, maxX = np.unravel_index(
                srcImprint.argmax(),
                srcImprint.shape
            )
            borderMin = 0.1*self.nPix
            borderMax = self.nPix - 0.1*self.nPix
            if (maxX <= borderMin) or (maxY <= borderMin) or (maxX >= borderMax) or (maxY >= borderMax):
                dist[dist<=np.median(dist)] = 0
                maxY, maxX = np.unravel_index(
                    ((1 - dist/dist.max())*srcImprint).argmax(),
                    srcImprint.shape
                )
            ax.text(
                groundGranularity[maxX],
                groundGranularity[maxY],
                ' {}'.format(src),
                color='#b35900',
                fontweight='bold',
                va='center',
                ha='center',
                zorder=30
            )
        # Colorbar
        cax = inset_axes(ax,
           width='5%',
           height='100%',
           loc='lower left',
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax.transAxes,
           borderpad=0,
           )
        cb = ColorbarBase(
            cax,
            cmap=get_cmap(name='YlGnBu_r'),
            orientation='vertical',
            norm=Normalize(
                vmin=np.min(nfImage_db),
                vmax=np.max(nfImage_db)
            ),
            ticks=LinearLocator(),
            format='%.2f'
        )
        cb.solids.set_edgecolor('face')
        cb.set_label('dB')
        # NenuFAR array info
        ax.scatter(
            maposENU[:, 0],
            maposENU[:, 1],
            20,
            color='black',
            zorder=10
        )
        for i in range(maposENU.shape[0]):
            ax.text(
                maposENU[i, 0],
                maposENU[i, 1],
                ' {}'.format(self.antNames[i]),
                color='black',
                zorder=10
            )
        ax.scatter(
            buildingsENU[:, 0],
            buildingsENU[:, 1],
            20,
            color='tab:red',#'tab:orange',
            zorder=10
        )
        # Plot axis labels       
        ax.set_xlabel(r'$\Delta x$ (m)')
        ax.set_ylabel(r'$\Delta y$ (m)')
        ax.set_title('{} MHz -- {}'.format(
            self.meanFreq.to(u.MHz).value,
            self.obsTime.isot)
        )

        # Save or not the plot
        if figname == '':
            plt.show()
        else:
            fig.savefig(
                figname,
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
        plt.close('all')

        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #

# ============================================================= #

