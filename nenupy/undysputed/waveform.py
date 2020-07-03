#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Waveform
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_RawLane',
    'Waveform'
]


import numpy as np
from os.path import abspath, isfile
import astropy.units as u
from astropy.time import Time
import dask.array as da

from nenupy.instru import sb2freq

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- _RawLane -------------------------- #
# ============================================================= #
class _RawLane(object):
    """
    """

    def __init__(self, rawlanefile):
        self.lane_index = None
        # self.data = None
        # self.dt = None
        # self.df = None
        # self.fft0 = None
        # self.fft1 = None
        # self.beam_arr = None
        # self.chan = None
        # self.unix = None
        self.rawlanefile = rawlanefile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def rawlanefile(self):
        """
        """
        return self._rawlanefile
    @rawlanefile.setter
    def rawlanefile(self, r):
        r = abspath(r)
        if not isfile(r):
            raise FileNotFoundError(
                f'{r} not found.'
            )
        if not r.endswith('raw'):
            raise ValueError(
                'Wrong file type, expected *.raw'
            )
        self._rawlanefile = r
        self._load()
        return


    # @property
    # def data(self):
    #     return self._data
    # @data.setter
    # def data(self, d):
    #     if d is None:
    #         self._data = d
    #         return


    # @property
    # def subband_width(self):
    #     """
    #         Should be equal to 0.1953125 MHz
    #     """
    #     return self.df * self.fftlen


    # @property
    # def tmin(self):
    #     """
    #     """
    #     return Time(self._times[0], format='unix', precision=7)


    # @property
    # def tmax(self):
    #     """
    #     """
    #     return Time(self._times[-1], format='unix', precision=7)


    # @property
    # def fmin(self):
    #     """
    #     """
    #     half_sb = self.subband_width/2
    #     channels = self.chan[0, :] # Assumed all identical!
    #     return np.min(channels)*self.subband_width - half_sb


    # @property
    # def fmax(self):
    #     """
    #     """
    #     half_sb = self.subband_width/2
    #     channels = self.chan[0, :] # Assumed all identical!
    #     return np.max(channels)*self.subband_width + half_sb
    

    # # --------------------------------------------------------- #
    # # ------------------------ Methods ------------------------ #
    # def get_stokes(self, stokes='I'):
    #     """
    #     """
    #     stokes_data = {
    #         'i': np.sum(self.fft0, axis=4),
    #         'q': self.fft0[..., 0] - self.fft0[..., 1],
    #         'u': self.fft1[..., 0] * 2,
    #         'v': - self.fft1[..., 1] * 2,
    #         'l': np.sqrt((self.fft0[..., 0] - self.fft0[..., 1])**2 + (self.fft1[..., 0] * 2)**2),
    #         'xx': self.fft0[..., 0],
    #         'yy': self.fft0[..., 1]
    #     }
    #     try:
    #         selected_stokes = stokes_data[stokes.lower()]
    #     except KeyError:
    #         log.error(
    #             f'Available Stokes: {stokes_data.keys()}'
    #         )
    #         raise
    #     selected_stokes = self._to2d(selected_stokes)
    #     # selected_stokes = self._bp_correct(
    #     #     data=selected_stokes,
    #     #     method=bpcorr
    #     # )
    #     return selected_stokes


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self):
        """
        """
        
        # Lane index just before '.raw', warning: hardcoded!
        self.lane_index = self._rawlanefile[-5]
        
        # Header structure to decode it:
        header_struct = [
            ('nobpb', 'int32'), # NUMBER_OF_BEAMLET_PER_BANK = number of channels
            ('nb_samples', 'int32'), # NUMBER of SAMPLES (fftlen*nfft)
            ('bytespersample', 'int32') # BYTES per SAMPLE (4/8 for 8/16bits data)
        ]
        with open(self._rawlanefile, 'rb') as rf:
            header_dtype = np.dtype(header_struct)
            header = np.frombuffer(
                rf.read(header_dtype.itemsize),
                count=1,
                dtype=header_dtype,
            )[0]
        
        # Storing metadata
        for key in [h[0] for h in header_struct]:
            setattr(self, key.lower(), header[key])
        bytes_in = self.bytespersample * self.nb_samples * self.nobpb

        # Find bit mode
        if self.bytespersample == 4:
            bit_mode = 'int8'
        elif self.bytespersample == 8:
            bit_mode = 'int16'
        else:
            raise Exception(
                f'Unknown bytespersample: {self.bytespersample}'
            )

        # Prepare data structure
        n_pol = 2
        reim = 2
        data_shape = (self.nb_samples * self.nobpb * n_pol * reim, )
        dt_block = np.dtype(
            [
                ('eisb', 'uint64'),
                ('tsb', 'uint64'),
                ('bsnb', 'uint64'),
                ('data', bit_mode, data_shape),
            ]
        )
        dt_lane_beam_chan = np.dtype(
            [
                ('lane', 'int32'),
                ('beam', 'int32'),
                ('chan', 'int32'),
            ]
        )
        dt_header = np.dtype(
            [
                ('nobpb', 'int32'),
                ('nb_samples', 'int32'),
                ('bytespersample', 'int32'),
                ('lbc_alloc', dt_lane_beam_chan, (self.nobpb, )),
            ]
        )

        # Read definitive header
        with open(self._rawlanefile, 'rb') as rf:
            header = np.frombuffer(
                rf.read(dt_header.itemsize),
                count=1,
                dtype=dt_header,
            )[0]

        # Set the missing attributes
        setattr(self, 'lbc_alloc', header['lbc_alloc'])
        self.fftlen = 256
        self.nfft = self.nb_samples // self.fftlen

        # Memmory read the data
        with open(self._rawlanefile, 'rb') as rf:
            tmp = np.memmap(
                rf,
                dtype=dt_block,
                mode='r',
                offset=dt_header.itemsize
            )

        # Time
        times = tmp['tsb'].astype("double")
        max_bsn = 200e6 / 1024
        times += tmp['bsnb'].astype("double") / max_bsn
        self.times = Time(times, format='unix', precision=6)

        # Frequency
        centralFrequencies = sb2freq([i[-1] for i in self.lbc_alloc])
        return
# ============================================================= #


# ============================================================= #
# ------------------------- Waveform -------------------------- #
# ============================================================= #
class Waveform(object):
    """
    """

    def __init__(self, rawfiles=[]):
        self.rawfiles = rawfiles


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #

