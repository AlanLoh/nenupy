#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *******
    Dynspec
    *******
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_Lane',
    'Dynspec'
]


from os.path import isfile, abspath, join, dirname
import numpy as np
import astropy.units as u
from astropy.time import Time
from dask.array import from_array
from dask.diagnostics import ProgressBar

from nenupy.beamlet import SData

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# --------------------------- Lane ---------------------------- #
# ============================================================= #
class _Lane(object):
    """
    """

    def __init__(self, lanefile):
        self.lane_index = None
        self.data = None
        self.dt = None
        self.df = None
        self.fft0 = None
        self.fft1 = None
        self.beam = None
        self.chan = None
        self.unix = None
        self.lanefile = lanefile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def lanefile(self):
        """
        """
        return self._lanefile
    @lanefile.setter
    def lanefile(self, l):
        l = abspath(l)
        if not isfile(l):
            raise FileNotFoundError(
                f'{l} not found.'
            )
        self._lanefile = l
        self._load()
        return


    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, d):
        if d is None:
            self._data = d
            return


    @property
    def subband_width(self):
        """
            Should be equal to 0.1953125 MHz
        """
        return self.df * self.fftlen


    @property
    def tmin(self):
        """
        """
        return Time(self._times[0], format='unix', precision=7)


    @property
    def tmax(self):
        """
        """
        return Time(self._times[-1], format='unix', precision=7)


    @property
    def fmin(self):
        """
        """
        half_sb = self.subband_width/2
        channels = self.chan[0, :] # Assumed all identical!
        return np.min(channels)*self.subband_width - half_sb


    @property
    def fmax(self):
        """
        """
        half_sb = self.subband_width/2
        channels = self.chan[0, :] # Assumed all identical!
        return np.max(channels)*self.subband_width + half_sb
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get_stokes(self, stokes='I'):
        """
        """
        stokes_data = {
            'i': np.sum(self.fft0, axis=4),
            'q': self.fft0[..., 0] - self.fft0[..., 1],
            'u': self.fft1[..., 0] * 2,
            'v': - self.fft1[..., 1] * 2,
            'xx': self.fft0[..., 0],
            'yy': self.fft0[..., 1]
        }
        try:
            selected_stokes = stokes_data[stokes.lower()]
        except KeyError:
            log.error(
                f'Available Stokes: {stokes_data.keys()}'
            )
            raise
        selected_stokes = self._to2d(selected_stokes)
        selected_stokes = self._bp_correct(selected_stokes)
        return selected_stokes


    def apply(self, func, *args):
        """
        """
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self):
        """
        """
        # Lane index just before '.spectra', warning: hardcoded!
        self.lane_index = self._lanefile[-9]
        # Header structure to decode it:
        header_struct = [
            ('idx', 'uint64'),
            ('TIMESTAMP', 'uint64'),
            ('BLOCKSEQNUMBER', 'uint64'),
            ('fftlen', 'int32'),
            ('nfft2int', 'int32'),
            ('fftovlp', 'int32'),
            ('apodisation', 'int32'),
            ('nffte', 'int32'),
            ('nbchan', 'int32')
        ]
        with open(self._lanefile, 'rb') as rf:
            header_dtype = np.dtype(header_struct)
            header = np.frombuffer(
                rf.read(header_dtype.itemsize),
                count=1,
                dtype=header_dtype,
            )[0]
        # Storing metadata
        for key in [h[0] for h in header_struct]:
            setattr(self, key.lower(), header[key])
        self.dt = 5.12e-6 * self.fftlen * self.nfft2int * u.s
        self.block_dt = self.dt * self.nffte
        self.df = (1.0 / 5.12e-6 / self.fftlen) * u.Hz
        log.info(
            f'Header of {self._lanefile} correctly parsed.'
        )
        # Read data
        beamlet_dtype = np.dtype([
            ('lane', 'int32'),
            ('beam', 'int32'),
            ('channel', 'int32'),
            ('fft0', 'float32', (self.nffte, self.fftlen, 2)),
            ('fft1', 'float32', (self.nffte, self.fftlen, 2))
        ])
        global_struct = header_struct +\
            [('data', beamlet_dtype, (self.nbchan))]
        global_dtype = np.dtype(global_struct)
        itemsize = global_dtype.itemsize
        with open(self._lanefile, 'rb') as rf:
            tmp = np.memmap(rf, dtype='int8', mode='r')
        n_blocks = tmp.size * tmp.itemsize // (itemsize)
        data = from_array(
            tmp[: n_blocks * itemsize].view(global_dtype)
        )['data'] # dask powered
        # Store data in appropriated attributes
        self.fft0 = data['fft0']
        self.fft1 = data['fft1']
        self.beam = data['beam']
        self.chan = data['channel']
        ntb, nfb = data['lane'].shape # time blocks, freq blocks
        # self.unix = np.arange(ntb, dtype='float64')
        # self.unix *= self.block_dt.to(u.s).value
        # self.unix += self.timestamp # Maybe I dont need .unix...
        log.info(
            f'Data of {self._lanefile} correctly parsed.'
        )
        # Time array
        n_times = ntb * self.nffte
        self._times = from_array(
            np.arange(n_times, dtype='float64')
        )
        self._times *= self.dt.value
        self._times += self.timestamp
        # Frequency array
        n_freqs = nfb * self.fftlen
        self._freqs = from_array(
            np.tile(np.arange(self.fftlen) - self.fftlen/2, nfb)
        )
        self._freqs = self._freqs.reshape((nfb, self.fftlen))
        self._freqs *= self.df.to(u.Hz).value
        self._freqs += self.chan[0, :][:, np.newaxis] * self.subband_width.value # assumed constant over time
        self._freqs = self._freqs.ravel()
        return


    def _to2d(self, data):
        """ Inverts the halves of each beamlet and reshape the
            array in 2D (time, frequency).
        """
        ntb, nfb = data.shape[:2]
        data = np.swapaxes(data, 1, 2)
        n_times = ntb * self.nffte
        n_freqs = nfb * self.fftlen
        # Invert the halves of the beamlet
        if self.fftlen % 2. != 0.0:
            raise ValueError(
                f'Problem with fftlen value: {self.fftlen}!'
            )
        data = data.reshape(
            (
                n_times,
                int(n_freqs/self.fftlen),
                2,
                int(self.fftlen/2)
            )
        )
        data = data[:, :, ::-1, :].reshape((n_times, n_freqs))
        return data


    def _bandpass(self):
        """ Computes the bandpass correction for a beamlet.
        """
        kaiser_file = join(
            dirname(abspath(__file__)),
            'bandpass_coeffs.dat'
        )
        kaiser = np.loadtxt(kaiser_file)

        n_tap = 16
        over_sampling = self.fftlen // n_tap
        n_fft = over_sampling * kaiser.size

        g_high_res = np.fft.fft(kaiser, n_fft)
        mid = self.fftlen // 2
        middle = np.r_[g_high_res[-mid:], g_high_res[:mid]]
        right = g_high_res[mid:mid + self.fftlen]
        left = g_high_res[-mid - self.fftlen:-mid]

        midsq = np.abs(middle)**2
        leftsq = np.abs(left)**2
        rightsq = np.abs(right)**2
        g = 2**25/np.sqrt(midsq + leftsq + rightsq)
        return g**2.


    def _bp_correct(self, data):
        """ Applies the bandpass correction to each beamlet
        """
        bp = self._bandpass()
        ntimes, nfreqs = data.shape
        data = data.reshape(
            (
                ntimes,
                int(nfreqs/bp.size),
                bp.size
            )
        )
        data *= bp[np.newaxis, np.newaxis]
        return data.reshape((ntimes, nfreqs))
# ============================================================= #


# ============================================================= #
# -------------------------- Dynspec -------------------------- #
# ============================================================= #
class Dynspec(object):
    """
    """

    def __init__(self, lanefiles=[]):
        self.lanes = []
        self.lanefiles = lanefiles
        self.beam = 0
        self.dispersion_measure = None
        self.rebin_dt = None
        self.rebin_df = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def lanefiles(self):
        """
        """
        return self._lanefiles
    @lanefiles.setter
    def lanefiles(self, l):
        if not isinstance(l, list):
            l = [l]
        l = map(abspath, l)
        for li in map(abspath, l):
            if not isfile(li):
                raise FileNotFoundError(
                    f'{li} not found.'
                )
            else:
                log.info(f'Found {li}')
                self.lanes.append(
                    _Lane(li)
                )
        return


    @property
    def time_range(self):
        """
        """
        if not hasattr(self, '_time_range'):
            return [self.tmin.unix, self.tmax.unix]
        return self._time_range
    @time_range.setter
    def time_range(self, t):
        if not isinstance(t, list):
            raise TypeError(
                'time_range expects a list.'
            )
        if not len(t) == 2:
            raise ValueError(
                'time_range should be a length-2 list.'
            )
        if not all([isinstance(ti, Time) for ti in t]):
            t = [Time(ti, precision=7) for ti in t]
        if t[0] >= t[1]:
            raise ValueError(
                'time_range start >= stop.'
            )
        log.info(
            f'Time-range set: {t[0].isot} to {t[1].isot}.'
        )
        self._time_range = [t[0].unix, t[1].unix]
        return


    @property
    def freq_range(self):
        """
        """
        if not hasattr(self, '_freq_range'):
            return [self.fmin.to(u.Hz), self.fmax.to(u.Hz)]
        return self._freq_range
    @freq_range.setter
    def freq_range(self, f):
        if not isinstance(f, list):
            raise TypeError(
                'freq_range expects a list.'
            )
        if not len(f) == 2:
            raise ValueError(
                'freq_range should be a length-2 list.'
            )
        if not all([isinstance(fi, u.Quantity) for fi in f]):
            f = [fi*u.MHz for fi in f]
        if f[0] >= f[1]:
            raise ValueError(
                'freq_range min >= max.'
            )
        log.info(
            f'Freq-range set: {f[0].to(u.MHz)} to {f[1].to(u.MHz)}.'
        )
        self._freq_range = [f[0].to(u.Hz), f[1].to(u.Hz)]
        return


    @property
    def dispersion_measure(self):
        """
        """
        return self._dispersion_measure
    @dispersion_measure.setter
    def dispersion_measure(self, dm):
        if not (dm is None):
            if not isinstance(dm, u.Quantity):
                dm *= u.pc / (u.cm**3)
            dm = dm.to(u.pc / (u.cm**3))
            log.info(
                f'DM set to {dm}'
            )
        self._dispersion_measure = dm
        return


    @property
    def rebin_dt(self):
        """
        """
        return self._rebin_dt
    @rebin_dt.setter
    def rebin_dt(self, dt):
        if not (dt is None):
            if not isinstance(dt, u.Quantity):
                dt *= u.s
            dt = dt.to(u.s)
            log.info(
                f'Time averaging on {dt}'
            )
        self._rebin_dt = dt
        return


    @property
    def rebin_df(self):
        """
        """
        return self._rebin_df
    @rebin_df.setter
    def rebin_df(self, df):
        if not (df is None):
            if not isinstance(df, u.Quantity):
                df *= u.MHz
            df = df.to(u.Hz)
            log.info(
                f'Frequency averaging on {df}'
            )
        self._rebin_df = df
        return


    @property
    def tmin(self):
        """
        """
        return min(li.tmin for li in self.lanes)


    @property
    def tmax(self):
        """
        """
        return max(li.tmax for li in self.lanes)


    @property
    def fmin(self):
        """
        """
        fm = min(li.fmin for li in self.lanes)
        return fm.compute().to(u.MHz)


    @property
    def fmax(self):
        """
        """
        fm = max(li.fmax for li in self.lanes)
        return fm.compute().to(u.MHz)
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self, stokes='I'):
        """
        """
        for li in self.lanes:
            data = li.get_stokes(stokes)
            # Time selection
            # way more efficient to compute indices than masking
            tmin_idx = np.argmin(
                np.abs(li._times - self.time_range[0])
            ).compute()
            tmax_idx = np.argmin(
                np.abs(li._times - self.time_range[1])
            ).compute()
            # Freq selection
            # way more efficient to compute indices than masking
            fmin_idx = np.argmin(
                np.abs(li._freqs - self.freq_range[0].value)
            ).compute()
            fmax_idx = np.argmin(
                np.abs(li._freqs - self.freq_range[1].value)
            ).compute()
            log.info(
                f'Retrieving data selection from lane {li.lane_index}...'
            )
            # High-rate data selection
            with ProgressBar():
                data = data[
                    tmin_idx:tmax_idx,
                    fmin_idx:fmax_idx
                ].compute()
            selfreqs = li._freqs[fmin_idx:fmax_idx].compute()*u.Hz
            # Dedispersion if FRB or Pulsar
            data = self._dedisperse(
                data=data,
                freqs=selfreqs,
                dt=li.dt
            )
            # Rebin data to downweight the output
            data, seltimes, selfreqs = self._rebin(
                data=data,
                times=li._times[tmin_idx:tmax_idx],
                freqs=selfreqs,
                dt=li.dt,
                df=li.df,
            )
            # Build a SData instance
            sd = SData(
                data=data[..., np.newaxis],
                time=seltimes,
                freq=selfreqs,
                polar=stokes,
            )
            # Stack in frequency the SData instances at
            # each lane reading
            if not 'spec' in locals():
                spec = sd
            else:
                spec = spec & sd
        return spec

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _dedisperse(self, data, freqs, dt):
        """
        """
        if not (self.dispersion_measure is None):
            log.info(
                'Starting de-dispersion...'
            )
            if data.shape[1] != freqs.size:
                raise ValueError(
                    'Problem with frequency axis.'
                )
            dm = self.dispersion_measure.value # pc/cm^3
            fmhz = freqs.to(u.MHz).value # MHz
            fmaxmhz = self.freq_range[1].to(u.MHz).value
            delays = 4140 * dm / (fmhz**2) * u.s # sec
            delays -= 4140 * dm / (fmaxmhz**2) *u.s # start of dynspec
            cell_delays = np.round(delays/dt).astype(int)
            for i in range(freqs.size):
                data[:, i] = np.roll(data[:, i], -cell_delays[i])
                # mask right edge of dynspec
                data[-cell_delays[i]:, i] = np.nan
            log.info(
                'Data are de-dispersed.'
            )
        return data


    def _rebin(self, data, times, freqs, dt, df):
        """

        """
        ntimes_i, nfreqs_i = data.shape
        if not (self.rebin_dt is None):
            # Rebin in time
            log.info(
                'Time-averaging...'
            )
            ntimes = int(np.floor(ntimes_i/(self.rebin_dt/dt)))
            tleftover = ntimes_i % ntimes
            data = data[:-tleftover if tleftover != 0 else ntimes_i, :].reshape(
                (ntimes, int((ntimes_i - tleftover)/ntimes), nfreqs_i)
            )
            times = times[:-tleftover if tleftover != 0 else ntimes_i].reshape(
                (ntimes, int((ntimes_i - tleftover)/ntimes))
            )
            data = np.mean(data, axis=1)
            times = np.mean(times, axis=1)
            times = Time(
                times,
                format='unix',
                precision=7
            )
            ntimes_i, nfreqs_i = data.shape
            log.info(
                'Data are time-averaged.'
            )
        if not (self.rebin_df is None):
            # Rebin in frequency
            log.info(
                'Frequency-averaging...'
            )
            data = data
            nfreqs = int(np.floor(nfreqs_i/(self.rebin_df/df)))
            fleftover = nfreqs_i % nfreqs
            #fleftover = fleftover if fleftover != 0 else nfreqs_i
            data = data[:, :-fleftover if fleftover != 0 else nfreqs_i].reshape(
                (ntimes_i, nfreqs, int((nfreqs_i - fleftover)/nfreqs))
            )
            freqs = freqs[:-fleftover if fleftover != 0 else nfreqs_i].reshape(
                (nfreqs, int((nfreqs_i - fleftover)/nfreqs))
            )
            data = np.mean(data, axis=2)
            freqs = np.mean(freqs, axis=1)
            log.info(
                'Data are frequency-averaged.'
            )
        return data, times, freqs

# ============================================================= #

