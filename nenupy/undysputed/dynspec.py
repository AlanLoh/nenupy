#! /usr/bin/python3
# -*- coding: utf-8 -*-


r"""
    *******
    Dynspec
    *******

    **UnDySPuTeD**

    *UnDySPuTeD* (stands for Unified Dynamic Spectrum Pulsar and
    Time Domain receiver) is the receiver of the NenuFAR
    beamformer mode, fed by the (up-to-)96 core Mini-Arrays (2
    polarizations) from the *LANewBa* backend. 

    **DynSpec data**

    The raw data flow from *LANewBa* consists of 195312.5 pairs
    of complex X and Y values per second per beamlet. These data
    are downsampled in channels per subband (of 195.3125 kHz)
    numbered from 16 to 2048 channels, ``fftlen``, (to achieve a
    frequency resolution of :math:`\delta \nu = 12 - 0.1\, \rm{kHz}`
    respectively). After computation of cross and auto-correlations,
    the data are downsampled again in time, integrating from 4 to 1024
    spectra, ``nfft2int`` (implying a time resolution :math:`195312.5^{-1} \times \rm{fftlen} \times \rm{fft2int}`,
    :math:`\delta t = 0.3 - 83.9\, \rm{ms}`).

    .. seealso::
        `DynSpec data product <https://nenufar.obs-nancay.fr/en/astronomer/#data-products>`_

    **DynSpec data files**
    
    Each NenuFAR/*UnDySPuTeD*/DynSpec observation results in the
    production of several proprietary formatted files (``'*.spectra'``),
    each corresponding to an individual lane of the *UnDySPuTeD* receiver.
    Depending on the observation configuration, the bandwidth and/or
    the different observed beams (i.e., beamforming in different sky
    directions) can be spread accross these files.

    The class :class:`~nenupy.undysputed.dynspec.Dynspec` offers
    the possibility to read and analyze all these observation files
    at once, through an API aimed at minimizing user interaction
    over complicated settings.
    
    For the purpose of the following 'how-to' guide, a 1h-observation
    of the famous pulsar `PSR B1919+21 <https://en.wikipedia.org/wiki/PSR_B1919%2B21>`_
    is analyzed. :mod:`~nenupy.undysputed.dynspec.Dynspec` needs
    first to be imported, as well as :mod:`~astropy.units` in order
    to use physcial units (to avoid mistakes).
    Attribute :attr:`~nenupy.undysputed.dynspec.Dynspec.lanefiles`
    is filled with a `list` of all the available lane files in order to create
    ``ds``, an instance of :class:`~nenupy.undysputed.dynspec.Dynspec`:

    >>> from nenupy.undysputed import Dynspec
    >>> import astropy.units as u
    >>> ds = Dynspec(
            lanefiles=[
                'B1919+21_TRACKING_20200214_090835_0.spectra',
                'B1919+21_TRACKING_20200214_090835_1.spectra'
            ]
        )

    .. note::
        `nenupy` logger could be activated at wish to enhance the verbosity:

        >>> import logging
        >>> logging.getLogger('nenupy').setLevel(logging.INFO)
    
    **Observation properties**

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


from os.path import isfile, abspath, join, dirname, basename
import numpy as np
import astropy.units as u
from astropy.time import Time
from dask.array import from_array
from dask.diagnostics import ProgressBar

from nenupy.beamlet import SData
from nenupy.astro import dispersion_delay

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
        self.beam_arr = None
        self.chan = None
        # self.unix = None
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
        if not l.endswith('spectra'):
            raise ValueError(
                'Wrong file type, expected *.spectra'
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
        self.beam_arr = data['beam'][0].compute() # asummed same for all time step
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
        # Beam indices
        unique_b = np.unique(self.beam_arr)
        self.beam_idx = {}
        for b in unique_b:
            b_idx = np.where(self.beam_arr == b)[0]
            self.beam_idx[str(b)] = (
                b_idx[0] * self.fftlen, # freq start of beam
                (b_idx[-1] + 1) * self.fftlen # freq stop of beam
            )
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
        .. versionadded:: 1.1.0
    """

    def __init__(self, lanefiles=[]):
        self.lanes = []
        self.lanefiles = lanefiles
        self.beam = 0
        self.dispersion_measure = None
        self.rebin_dt = None
        self.rebin_df = None
        self.jump_correction = True


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def lanefiles(self):
        """ UnDySPuTeD time-frequency files input (i.e. one file
            per lane) that are stored in the attribute 
            :attr:`~nenupy.undysputed.dynspec.Dynspec.lanes` as
            :class:`~nenupy.undysputed.dynspec._Lane` instances.

            Subsequent selections in time, frequency, beam and
            data operation processes will be run on these files
            while querying only those required. It means that any
            file listed in input is not necessarily read, and
            therefore having more files in input than needed
            does not increase memory usage or computing time.

            :setter: `list` of ``'*.spectra'`` files.

            :getter: `list` of ``'*.spectra'`` files.
            
            :type: `list`

            :Example:
                >>> from nenupy.undysputed import Dynspec
                >>> ds = Dynspec(
                        lanefiles=['lane_0.spectra', 'lane_0.spectra']
                    )
                >>> ds.lanefiles
                ['/path/to/lane_0.spectra', '/path/to/lane_0.spectra']

            .. warning::
                Files are checked to belong to the same
                observation based on their prefix.
    
            .. versionadded:: 1.1.0
        """
        return list(self._lanefiles)
    @lanefiles.setter
    def lanefiles(self, l):
        if not isinstance(l, list):
            l = [l]
        l = list(map(abspath, l))
        # Check that all the input files belong to the same obs
        names = list(map(basename,l))
        obsname = [ni[::-1].split('_', 1)[1] for ni in names]
        if obsname != obsname[::-1]:
            raise ValueError(
                'Input files seem not to belong to the same observation.'
            )
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
        self._lanefiles = l
        return


    @property
    def time_range(self):
        """
            .. versionadded:: 1.1.0
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
            .. versionadded:: 1.1.0
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
    def beam(self):
        """
            .. versionadded:: 1.1.0
        """
        return self._beam
    @beam.setter
    def beam(self, b):
        if not isinstance(b, (int, np.integer)):
            raise TypeError(
                'beam should be an integer.'
            )
        if b not in self.beams:
            raise ValueError(
                f'Available beam indices are {self.beams}.'
            )
        self._beam = b
        log.info(
            f'Beam index set: {b}.'
        )
        return


    @property
    def dispersion_measure(self):
        """
            .. versionadded:: 1.1.0
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
            .. versionadded:: 1.1.0
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
            .. versionadded:: 1.1.0
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
            .. versionadded:: 1.1.0
        """
        return min(li.tmin for li in self.lanes)


    @property
    def tmax(self):
        """
            .. versionadded:: 1.1.0
        """
        return max(li.tmax for li in self.lanes)


    @property
    def fmin(self):
        """
            .. versionadded:: 1.1.0
        """
        fm = min(li.fmin for li in self.lanes)
        return fm.compute().to(u.MHz)


    @property
    def fmax(self):
        """
            .. versionadded:: 1.1.0
        """
        fm = max(li.fmax for li in self.lanes)
        return fm.compute().to(u.MHz)


    @property
    def beams(self):
        """
            .. versionadded:: 1.1.0
        """
        un_beams = []
        for li in self.lanes:
            lane_beams = [int(lbi) for lbi in li.beam_idx.keys()]
            un_beams += lane_beams
        return np.unique(un_beams)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self, stokes='I'):
        """
            .. versionadded:: 1.1.0
        """
        for li in self.lanes:
            try:
                beam_start, beam_stop = li.beam_idx[str(self.beam)]
            except KeyError:
                # No beam 'self.beam' found on this lane
                continue
            data = li.get_stokes(stokes)
            # Time selection
            # way more efficient to compute indices than masking
            tmin_idx = np.argmin(
                np.abs(li._times - self.time_range[0])
            ).compute()
            tmax_idx = np.argmin(
                np.abs(li._times - self.time_range[1])
            ).compute()
            # Freq/beam selection
            # way more efficient to compute indices than masking
            fmin_idx = np.argmin(
                np.abs(li._freqs[beam_start:beam_stop] - self.freq_range[0].value)
            ).compute()
            fmax_idx = np.argmin(
                np.abs(li._freqs[beam_start:beam_stop] - self.freq_range[1].value)
            ).compute()
            if (fmin_idx - fmax_idx) == 0:
                # No data selected on this lane
                continue
            log.info(
                f'Retrieving data selection from lane {li.lane_index}...'
            )
            # High-rate data selection
            with ProgressBar():
                data = data[:, beam_start:beam_stop][
                    tmin_idx:tmax_idx,
                    fmin_idx:fmax_idx
                ].compute()
            selfreqs = li._freqs[beam_start:beam_stop][fmin_idx:fmax_idx].compute()*u.Hz
            # Correct 6 min jumps
            data = self._correct_jumps(
                data=data,
                dt=li.dt 
            )
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
            .. versionadded:: 1.1.0
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
            delays = dispersion_delay(
                freq=freqs,
                dm=self.dispersion_measure)
            delays -= dispersion_delay( # relative delays
                freq=self.freq_range[1],
                dm=self.dispersion_measure
            )
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
            .. versionadded:: 1.1.0
        """
        ntimes_i, nfreqs_i = data.shape
        if not (self.rebin_dt is None):
            # Rebin in time
            tbins = int(np.floor(self.rebin_dt/dt))
            log.info(
                f'Time-averaging {tbins} spectra, dt={tbins*dt}...'
            )
            ntimes = int(np.floor(ntimes_i/tbins))
            tleftover = ntimes_i % ntimes
            log.info(
                f'Last {tleftover} spectra are left over for time-averaging.'
            )
            data = data[:-tleftover if tleftover != 0 else ntimes_i, :].reshape(
                (ntimes, int((ntimes_i - tleftover)/ntimes), nfreqs_i)
            )
            times = times[:-tleftover if tleftover != 0 else ntimes_i].reshape(
                (ntimes, int((ntimes_i - tleftover)/ntimes))
            )
            data = np.mean(data, axis=1)
            times = np.mean(times, axis=1)
            ntimes_i, nfreqs_i = data.shape
            log.info(
                'Data are time-averaged.'
            )
        if not (self.rebin_df is None):
            # Rebin in frequency
            fbins = int(np.floor(self.rebin_df/df))
            log.info(
                f'Frequency-averaging {fbins} channels: df={fbins*df}...'
            )
            nfreqs = int(np.floor(nfreqs_i/fbins))
            fleftover = nfreqs_i % nfreqs
            log.info(
                f'Last {fleftover} channels are left over for frequency-averaging.'
            )
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
        times = Time(
            times,
            format='unix',
            precision=7
        )
        return data, times, freqs


    def _correct_jumps(self, data, dt):
        """
        """
        if not self.jump_correction:
            return data
        # Work on cleanest profile
        #std_freq = np.std(data, axis=0)
        #profile = data[:, np.argmin(std_freq)].copy()
        profile = np.median(data, axis=1)
        # detect jumps
        dt = dt.to(u.s).value
        deriv = np.gradient(profile, dt)
        low_threshold = np.nanmedian(deriv) - 8 * np.nanstd(deriv)
        lower_thr_indices = np.where(deriv < low_threshold)[0] # detection
        # Sort jump indices
        jump_idx = [0]
        for idx in lower_thr_indices:
            if idx > jump_idx[-1] + 10:
                jump_idx.append(idx)
        jump_idx.append(data.shape[0] - 1)
        # Make sure there are no > 6min gaps
        sixmin = 6*60.000000 + dt
        for idx in range(1, len(jump_idx)):
            delta_t = dt * (jump_idx[idx] - jump_idx[idx-1])
            if delta_t > sixmin:
                missing_idx = int(np.round(sixmin/dt))
                jump_idx.insert(idx, missing_idx + jump_idx[idx-1])
        # flatten
        boundaries = list(map(list, zip(jump_idx, jump_idx[1:]))) 
        for i, boundary in enumerate(boundaries):
            idi, idf = boundary
            med_data_freq = profile[idi:idf]
            data[idi:idf, :] /= med_data_freq[:, np.newaxis]
            if i == 0:
                data[idi:idf, :] *= np.median(med_data_freq)
            else:
                data[idi:idf, :] *= previous_endpoint
            previous_endpoint = np.median(data[idf-1, :])
        return data
# ============================================================= #

