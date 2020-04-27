#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    XST_Data
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'XST_Data'
]


from astropy.io import fits
from astropy.time import Time, TimeDelta
import astropy.units as u
from os.path import abspath, isfile
import numpy as np

from nenupy.crosslet import Crosslet

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- XST_Data -------------------------- #
# ============================================================= #
class XST_Data(Crosslet):
    """
    """

    def __init__(self, xstfile):
        super().__init__(
        )
        self.xstfile = xstfile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def xstfile(self):
        """ NenuFAR XST
        """
        return self._xstfile
    @xstfile.setter
    def xstfile(self, x):
        if not isinstance(x, str):
            raise TypeError(
                'String expected.'
                )
        x = abspath(x)
        if not isfile(x):
            raise FileNotFoundError(
                'Unable to find {}'.format(x)
                )
        self._xstfile = x
        self._load()
        return

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def time_split_file(self, newname, start_idx=0, stop_idx=-1):
        """
        """
        hdus = fits.open(self.xstfile, memmap=True)
        newtimes = hdus[7].data['jd'][start_idx:stop_idx]
        newdata = hdus[7].data['DATA'][start_idx:stop_idx, :, :]
        log.info(
            'Time splitting {} from {} to {} time steps.'.format(
                self.xstfile,
                hdus[7].data['jd'].size,
                newtimes.size
            )
        )
        metadata_hdus = hdus[:-1]
        newrec = fits.FITS_rec.from_columns(
            [
                fits.Column(
                    name='jd',
                    array=newtimes,
                    format='1D'
                ),
                fits.Column(
                    name='xstSubband',
                    array=hdus[7].data['xstSubband'],
                    format='16I'
                ),
                fits.Column(
                    name='DATA',
                    array=newdata,
                    format='97680C',
                    dim='(6105, 16)'
                ),
            ],
            nrows=newtimes.size
        )
        newheader = hdus[7].header.copy()
        newheader['NAXIS2'] = newtimes.size
        new_hdu7 = fits.BinTableHDU(
            data=newrec,
            header=newheader,
            name='XST'
        )
        new_hdu = fits.HDUList(
            metadata_hdus + [new_hdu7]
        )
        new_hdu.writeto(newname, overwrite=True)
        log.info(
            'File {} created.'.format(newname)
        )
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self):
        """ Read XST file
        """
        log.info('Loading {}'.format(self._xstfile))
        self.mas = fits.getdata(
            self._xstfile,
            ext=1
        )['noMROn'][0]
        data_tmp = fits.getdata(
            self._xstfile,
            ext=7,
            memmap=True
        )
        self.vis = data_tmp['data']
        self.times = Time(data_tmp['jd'], format='jd')
        bw = 0.1953125
        self.sb_idx = np.unique(data_tmp['xstsubband'])
        self.freqs = self.sb_idx * bw * u.MHz
        return
# ============================================================= #

