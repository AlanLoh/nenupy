#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *************
    HEALPix LOFAR
    *************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HpxLOFAR'
]


from os.path import join, dirname, abspath
import numpy as np
from healpy.pixelfunc import ang2pix
from healpy.sphtfunc import smoothing
import astropy.units as u
from astropy.table import Table

from nenupy.astro import HpxSky
from nenupy.instru import _HiddenPrints


# ============================================================= #
# ------------------------- HpxLOFAR -------------------------- #
# ============================================================= #
class HpxLOFAR(HpxSky):
    """
    """

    def __init__(self, freq=50, resolution=1, smooth=True):
        super().__init__(
            resolution=resolution
        )
        self._load_lofar()
        self._smooth = smooth
        self.freq = freq


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, u.Quantity):
            f *= u.MHz
        self._freq = f
        self._populate()
        return


    # @property
    # def skymap(self):
    #     if self.visible_sky:
    #         mask = np.ones(self._skymap.size, dtype=bool)
    #         mask[self._is_visible] = False
    #         return np.ma.masked_array(
    #             smoothing(
    #                 self._skymap, 
    #                 fwhm=self.resolution.to(u.rad)/2
    #             ),
    #             mask=mask,
    #             fill_value=-1.6375e+30
    #         )
    #     else:
    #         return smoothing(
    #             self._skymap, 
    #             fwhm=self.resolution.to(u.rad)/2
    #         )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #



    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load_lofar(self):
        """
        """
        skymodel_file = join(
            dirname(abspath(__file__)),
            'lfsky.fits'
        )
        self.model_table = Table.read(skymodel_file)
        return


    @staticmethod
    def _extrapol_spec(freq, rflux, rfreq, index):
        """ Given the `rflux` in Jy at the `rfreq` in MHz,
            and the spectral index, extrapolate the `flux`
            at `freq` MHz
        """
        if freq is None:
            return rflux
        freq *= u.MHz
        rfreq *= u.Hz
        return (rflux * (freq/rfreq.to(u.MHz))**index).value


    def _populate(self):
        """ Resize the GSM to match tjhe desired resolution
        """
        indices = ang2pix(
            theta=self.model_table['ra'],
            phi=self.model_table['dec'],
            nside=self.nside,
            lonlat=True
        )
        # There are possibly overlaps, find a way to avoid that!
        self.skymap[indices] = self._extrapol_spec(
            freq=self._freq.to(u.MHz).value,
            rflux=self.model_table['flux'],
            rfreq=self.model_table['rfreq'],
            index=self.model_table['index']
        )
        if self._smooth:
            with _HiddenPrints():
                self.skymap = smoothing(
                    self.skymap, 
                    fwhm=self.resolution.to(u.rad).value*3
                )
        return
# ============================================================= #

