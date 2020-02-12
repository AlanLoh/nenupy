#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np 
import os
import warnings

from astropy.io import fits
import astropy.units as u

# ============================================================= #
# ------------------------ NenuFAR_Obs ------------------------ #
# ============================================================= #
class NenuFAR_Obs(object):
    """
    """
    def __init__(self, fitsfile):
        self.fitsfile = fitsfile
        # self.type = None


    @property
    def fitsfile(self):
        """ Path toward the NenuFAR observation FITS file.
        """
        return self._fitsfile
    @fitsfile.setter
    def fitsfile(self, f):
        if f is None:
            self._fitsfile = None
            return
        assert os.path.isfile(f),\
            'File {} not found'.format(f)
        assert f.endswith('.fits'),\
            'File {} is not a FITS.'.format(f)
        self._meta_read(fitsfile=f)
        # self._time_read(fitsfile=f)
        # self._obs_type()
        self._fitsfile = os.path.abspath(f)
        return

    @property
    def times(self):
        if not hasattr(self, '_times'):
            self._times = self._time_read(fitsfile=self.fitsfile)
        return self._times
    

    def _meta_read(self, fitsfile=None):
        """ Read the metadata of a NenuFAR osbervation.
            This stores everything in a `_meta` dict.

            Parameters
            ----------

            fitsfile: str
                BST FITS file to extract metadata from.
        """
        if fitsfile is None:
            fitsfile = self.fitsfile
        self._meta = {}
        with fits.open(fitsfile,
                mode='readonly',
                ignore_missing_end=True,
                memmap=True) as f:
            self._meta['hea'] = f[0].header
            self._meta['ins'] = f[1].data
            self._meta['obs'] = f[2].data
            self._meta['ana'] = f[3].data
            self._meta['bea'] = f[4].data
            self._meta['pan'] = f[5].data
            self._meta['pbe'] = f[6].data
        return

    def _time_read(self, fitsfile=None):
        """ Read the data extension of a NenuFAR observation.
            This stores data in the `_alldata` attribute and
            time in the `times` attribute.

            Parameters
            ----------

            fitsfile: str
                BST FITS file to read times from.
        """
        if fitsfile is None:
            fitsfile = self.fitsfile
        with fits.open(fitsfile,
                mode='readonly',
                ignore_missing_end=True,
                memmap=True) as f:
            # self.times = self._convert(f[7].data['JD'])
            times = self._convert(f[7].data['JD'])
            self._alldata = f[7].data['data']
            try:
                self._xstsb = f[7].data['xstSubband']
            except:
                pass
        # return
        return times

    # def _obs_type(self):
    #     """ Discriminate between a transit or a tracking observation
    #     """
    #     self.type = None
    #     return


    @staticmethod
    def _convert(jds):
        """ Convert the NenuFAR Julian Days in `np.datetime64`
            ms precision

            Parameters
            ----------

            jds : np.ndarray
                Array of Julian Days to be converted


            Returns
            -------
            
            iso_datetime : np.datetime64
                Array of numpy datetime.
        """
        def _jd_to_date(jd):
            """
            """
            jd += 0.5
            F, I = np.modf(jd)
            I = int(I)
            A = np.trunc((I - 1867216.25)/36524.25)
            if I > 2299160:
                B = I + 1 + A - np.trunc(A / 4.)
            else:
                B = I
            C = B + 1524
            D = np.trunc((C - 122.1) / 365.25)
            E = np.trunc(365.25 * D)
            G = np.trunc((C - E) / 30.6001)
            day = C - E + F - np.trunc(30.6001 * G)
            if G < 13.5:
                month = G - 1
            else:
                month = G - 13
            if month > 2.5:
                year = D - 4716
            else:
                year = D - 4715
            time, day = np.modf(day)
            minutes, hours = np.modf(time * 24)
            seconds, mins = np.modf(minutes * 60)
            secs = seconds * 60
            if '{:.3f}'.format(secs) == '60.000':
                mins += 1
                secs = 0
            if '{:02d}'.format(int(mins)) == '60':
                hours += 1
                mins = 0
            if '{:02d}'.format(int(hours)) == '24':
                hours = 0
                day += 1
            date = '{0:02d}-{1:02d}-{2:02d}'\
                .format(int(year), int(month), int(day))
            hour = '{0:02d}:{1:02d}:{2:06.3f}'\
                .format(int(hours), int(mins), secs)
            return 'T'.join([date, hour])

        return np.array(list(map(_jd_to_date, jds)),
            dtype='datetime64')
# ============================================================= #


# ============================================================= #
# ------------------------ NenuFAR_Obs ------------------------ #
# ============================================================= #
class NenuFAR(NenuFAR_Obs):
    """ Instrumental configuration of the NenuFAR array
    """
    def __init__(self, obs):
        super().__init__(obs)
        self._read_configuration()

    def _read_configuration(self):
        """
        """
        instru = self._meta['ins']
        for name in instru.names:
            setattr(self, name.lower(), instru[name])
        return
# ============================================================= #


# ============================================================= #
# ------------------------- SpecData -------------------------- #
# ============================================================= #
class SpecData(object):
    """ A class to handle dynamic spectrum data as parsed
        via the `BST` class.

        e.g.:
            from BST_v2 import BST
            b = BST('/Users/aloh/Desktop/NenuFAR_Data/20190228_071900_BST.fits')
            b.freq=[33, 60]
            b.select()
            b.data.fmean().tmean().db
    """
    def __init__(self, data, time, freq, polar):
        self.time = time
        self.freq = freq
        self.polar = polar
        self.data = data

    def __repr__(self):
        return self.data.__repr__()

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, d):
        ts, ps, fs = d.shape
        assert self.time.size == ts,\
            'time axis inconsistent'
        assert self.polar.size == ps,\
            'polar axis inconsistent'
        assert self.freq.size == fs,\
            'frequency axis inconsistent'
        self._data = d
        return

    @property
    def mjd(self):
        """ Return MJD dates
        """
        from astropy.time import Time
        isot = Time(self.time)
        return isot.mjd

    @property
    def jd(self):
        """ Return JD dates
        """
        from astropy.time import Time
        isot = Time(self.time)
        return isot.jd

    @property
    def amp(self):
        """ Linear amplitude of the data
        """
        return self.data.squeeze()
    
    @property
    def db(self):
        """ Convert the amplitude in decibels
        """
        return 10 * np.log10(self.data.squeeze())

    def fmean(self, freq1=None, freq2=None, method='mean', clean=False):
        """ Average over the frequency.
            
            Parameters
            ----------

            freq1 : float
                Lower frequency bound in MHz.

            freq2 : float
                Upper frequency bound in MHz.

            method : str
                Method used to average (either 'mean' or 'median')

            clean : bool
                Apply a cleaning before averaging

            Returns
            -------

            averaged_data : SpecData
                A new `SpecData` instance containging the averaged quantities.
        """
        if freq1 is None:
            freq1 = self.freq.min()
        else:
            freq1 *= u.MHz
        if freq2 is None:
            freq2 = self.freq.max()
        else:
            freq2 *= u.MHz
        fmask = (self.freq >= freq1) & (self.freq <= freq2)
        
        data = self.data[:, :, fmask]
        if clean:
            # Find out the noisiest profiles
            sigmas = np.std(data, axis=0)
            max_sig = np.percentile(sigmas, 90, axis=1)
            data = data[:, np.newaxis, sigmas < max_sig]

            # Smoothin' the time profiles
            from scipy.signal import savgol_filter
            tf = np.zeros(data.shape)
            for i in range(self.polar.size):
                # for j in range(self.freq.size):
                    # tf[:, i, j] = savgol_filter(data[:, i, j], 51, 1)
                for j in range(sigmas[sigmas < max_sig].size):
                    tf[:, i, j] = savgol_filter(data[:, i, j], 201, 1)

            # Rescale everything not to bias the mean
            data = tf - (np.median(tf, axis=0) - np.median(tf))

        average = np.mean(data, axis=2)\
            if method == 'mean'\
            else np.median(data, axis=2)

        return SpecData(
            data=np.expand_dims(average, axis=2),
            time=self.time.copy(),
            freq=np.expand_dims(np.mean(self.freq[fmask]), axis=0),
            polar=self.polar.copy())


    def tmean(self, t1=None, t2=None, method='mean',):
        """ Average over the time.
            
            Parameters
            ----------

            t1 : str
                Lower time bound in ISO/ISOT format.

            t2 : str
                Upper time bound in ISO/ISOT format.

            Returns
            -------

            averaged_data : SpecData
                A new `SpecData` instance containging the averaged quantities.
        """
        if t1 is None:
            t1 = self.time[0]
        else:
            t1 = np.datetime64(t1)
        if t2 is None:
            t2 = self.time[-1]
        else:
            t2 = np.datetime64(t2)
        tmask = (self.time >= t1) & (self.time <= t2)
        tmasked = self.time[tmask]
        dt = (tmasked[-1] - tmasked[0])
        #mean = np.mean(self.data[tmask, :, :], axis=0)
        average = np.mean(self.data[tmask, :, :], axis=0)\
            if method == 'mean'\
            else np.median(self.data[tmask, :, :], axis=0)
        return SpecData(
            #data=np.expand_dims(mean, axis=0),
            data=np.expand_dims(average, axis=0),
            time=np.array([tmasked[0] + dt/2.]),
            freq=self.freq.copy(),
            polar=self.polar.copy())

    def stream(self, start=None, stop=None):
        """ Set up a DAS2 stream. 
        """
        if start is None:
            start = self.time[0]
        else:
            start = np.datetime64(start)
        if stop is None:
            stop = self.time[-1]
        else:
            stop = np.datetime64(stop)

        wfile = open('/Users/aloh/Desktop/blabla.d2s', 'w')

        # ------ Stream Header ------ #
        title = 'NenuFAR BST'
        dt = (self.time[1] - self.time[0]).astype('timedelta64[s]').astype(float)
        header = f'<stream version="2.2">\n'\
            f'\t<properties '\
            f'title="{title}" '\
            f'Datum:xTagWidth="{dt} s" '\
            f'DatumRange:xRange="{self.time[0]} to {self.time[-1]} UTC" '\
            f'String:renderer="spectrogram"/>\n'\
            f'</stream>\n'
        # print(f'[00]{len(header):06d}{header}')
        wfile.write(f'[00]{len(header):06d}{header}')

        # ------ Packet Header ------ #
        yaxis = ','.join(['{}'.format(item.value) for item in self.freq]) # convert to strings
        p_header = f'<packet>\n'\
                   f'\t<x type="time23" units="us2000"/>\n'\
                   f'\t<yscan nitems="{self.freq.size}" '\
                   f'type="ascii12" '\
                   f'yUnits="{self.freq.unit}" '\
                   f'name="BST" '\
                   f'zUnits="dB"\n'\
                   f'\t\tyTags="{yaxis}">\n'\
                   f'\t\t<properties '\
                   f'String:yLabel="Frequency" '\
                   f'String:zLabel="dB" '\
                   f'double:zFill="0"/>\n'\
                   f'\t</yscan>\n'\
                   f'</packet>\n'
        # print(f'[01]{len(p_header):06d}{p_header}')
        wfile.write(f'[01]{len(p_header):06d}{p_header}')

        # ------ Packets ------ #
        for i in range(self.time.size):
            if self.time[i] >= start and self.time[i] <= stop:
                data = ''.join([f'{item:12.4e}' for item in self.db[i]])
                wfile.write(f':01:{self.time[i]}{data}\n')

        wfile.close()
        return

    def filter(self, kernel=7):
        """ Remove the spikes

            Parameters
            ----------

            kernel : array_like
                A scalar or an N-length list giving the size 
                of the median filter window in each dimension. 
                Elements of kernel_size should be odd. 
                If kernel_size is a scalar, then this scalar is 
                used as the size in each dimension. Default size 
                is 3 for each dimension.
        """
        from scipy.signal import medfilt
        if (self.data.shape[2] == 1) and (not isinstance(kernel, list)):
            kernel = [kernel, 1]
        filtered_data = np.zeros(self.data.shape)
        for i in range(self.polar.size):    
            tf = medfilt(self.data[:, i, :], kernel)
            filtered_data[:, i, :] = tf
        return SpecData(
            data=filtered_data,
            time=self.time.copy(),
            freq=self.freq.copy(),
            polar=self.polar.copy())

    def bg_remove(self, kernel=11, sigma=3):
        """ Remove the background

            Parameters
            ----------

            kernel : array_like
                A scalar or an N-length list giving the size 
                of the median filter window in each dimension. 
                Elements of kernel_size should be odd. 
                If kernel_size is a scalar, then this scalar is 
                used as the size in each dimension. Default size 
                is 3 for each dimension.
        """
        from scipy.signal import medfilt
        if (self.data.shape[2] == 1) and (not isinstance(kernel, list)):
            kernel = [kernel, 1]
        filtered_data = np.zeros(self.data.shape)
        for i in range(self.polar.size):
            tf = medfilt(self.data[:, i, :], kernel)
            sig = np.std(self.data[:, i, :])
            bad = (np.abs(self.data[:, i, :] - tf) / sig) > sigma
            tf[~bad] = self.data[:, i, :].copy()[~bad]
            filtered_data[:, i, :] = tf
        return SpecData(
            data=filtered_data,
            time=self.time.copy(),
            freq=self.freq.copy(),
            polar=self.polar.copy())

    def to_sound(self, output='/Users/aloh/Desktop/test_file.wav'):
        """ Convert a dynamic spectrum to a sound file
        """
        import wave
        import struct

        sample_rate = 44100
        min_freq = 22000
        max_freq = 44100
        obs_dt = (self.time[-1] - self.time[0]).astype('timedelta64[s]')
        duration = int(obs_dt.astype('int') / 3600 * 5) # 5 sec / hour
        duration = 10

        wavef = wave.open(output,'w')
        wavef.setnchannels(1) # mono
        wavef.setsampwidth(2) 
        wavef.setframerate(sample_rate)

        max_frame = int(duration * sample_rate)
        max_intensity = 32767 # Defined by WAV
        
        step_size = 400 # Hz, each pixel's portion of the spectrum
        stepping_spectrum = int((max_freq-min_freq)/step_size)
        
        data = np.log10(self.data[:, 0, :]) # only consider first polarisation
        tsize, fsize = data.shape
        data /= data.max() # normalised the data
        data *= max_intensity # scale to WAV standards
        
        for frame in range(max_frame):
            signal_value, count = 0, 0
            for step in range(stepping_spectrum):
                intensity = data[frame%tsize, step%fsize]
                current_f = (step * step_size) + min_freq
                next_f = ((step+1) * step_size) + min_freq
                if next_f - min_freq > max_freq: # If we're at the end of the spectrum
                    next_f = max_freq
                for freq in range(current_f, next_f, 1000): # substep of 1000 Hz is good
                    signal_value += intensity*np.cos(freq * 2 * np.pi * float(frame) / sample_rate)
                    count += 1
            if count == 0: count = 1
            signal_value /= count
            
            w_data = struct.pack('<h', int(signal_value))
            wavef.writeframesraw( w_data )
            
        wavef.writeframes(''.encode())
        wavef.close()
        
        return
# ============================================================= #


# ============================================================= #
# ---------------------------- BST ---------------------------- #
# ============================================================= #
class BST(NenuFAR_Obs):
    """
    """
    def __init__(self, bst):
        super().__init__(bst)
        self.bstfile = self.fitsfile
        self.track_beam = None
        self.pointing = 0
        self.polar = None
        self.freq = None
        self.time = None
        self._keywords = [
            'track_dbeam',
            'pointing',
            'freq',
            'time',
            'polar']

    # ------------------------- Setter -------------------------#
    @property
    def bstfile(self):
        """ Path toward the NenuFAR BST FITS file.
        """
        return self._bstfile
    @bstfile.setter
    def bstfile(self, b):
        assert 'beamlet' in self._meta['hea']['Object'],\
            '{} is not a BST file.'.format(b)
        self._read_bst(bst=b)
        self._bstfile = b
        return

    @property
    def pointing(self):
        """ Pointing beam index selection

            Parameters
            ----------
            pointing: int or mask_array
                If an `int` is given, it is understood as the index
                of pointing (e.g. one particular numerical beam in a tracking).
                If a `mask_array` is given, it is understood as a condition
                upon the numerical beams (e.g. `(self.dbeams==0)` to select
                all the pointings corresponding to the numerical beam index 0).
        """
        return self._digibeams[self._pointing]
        # return self._pointing
    @pointing.setter
    def pointing(self, b):
        if isinstance(b, int):
            assert b < self._n_points,\
                'Maximum beam index is {}'.format(self._n_points-1)
            tmp_b = np.zeros(self._digibeams.size, dtype=bool)
            #tmp_b[ self._digibeams == b ] = True ## 
            tmp_b[b] = True
            b = tmp_b
        else:
            b = np.array(b)
            if b.dtype == bool:
                assert b.size == self._n_points,\
                    'Provide a mask array of size {}'.format(self._n_points)
            else:
                assert np.all(np.array(b) < self._n_points),\
                    'Some indices are greater than {}'.format(self._n_points)
        self._pointing = b
        return

    @property
    def polar(self):
        """ Polarization selection.
            Could be 'NE' or 'NW' 
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        if p is None:
            p = 'NW'
        p = p.upper()
        assert p in self._meta['ins']['spol'],\
            'Polarization should be NE or NW'
        obs_pols = self._meta['ins']['spol'].ravel()
        self._pmask = (obs_pols == p)
        self._polar = obs_pols[self._pmask]
        return

    @property
    def time(self):
        """ Time selection, must be given in ISO/ISOT format.
        """
        self._tmask = np.ones(self.times.size, dtype=bool)
        self._tmask *= (
            (self.times >= self.beam_start[0]) &\
            (self.times <= self.beam_stop[-1])
            )

        if self._time.size == 2:
            if (self._time[0] > self.beam_stop[-1]) or\
                (self._time[1] < self.beam_start[0]):
                pass 
            else:
                self._tmask *= (
                    (self.times >= self._time[0]) &\
                    (self.times <= self._time[1])
                    )
        else:
            if (self._time[0] > self.beam_stop[-1]) or\
                (self._time[0] < self.beam_start[0]):
                pass 
            else:
                self._tmask *= (
                    (self.times >= self._time[0]) |\
                    (self.times <= self._time[0])
                    )
            idx = (np.abs(self.times - self._time[0])).argmin()
            self._tmask[idx] = True
        
        return self.times[self._tmask]
    @time.setter
    def time(self, t):
        if t is None:
            t = [self.obs_start, self.obs_stop]
        t = np.array([t], dtype='datetime64').ravel()
        assert t.size <= 2,\
            'time attribute must be of size 2 maximum.'
        for ti in t:
            assert (self.obs_start <= ti) &\
                (ti <= self.obs_stop),\
                '{} not within {} -- {}'.format(
                        ti,
                        self.obs_start,
                        self.obs_stop)
        if t.size == 2:
            assert t[0] < t[1],\
            '{} should be before {}.'.format(t[0], t[1])
        self._time = t
        return

    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        freqs = self._meta['bea']['freqlist'][self.db] * u.MHz
        if len(freqs.shape) == 2:
            freqs = freqs[0, :]
        self._fmask = np.zeros(freqs.size, dtype=bool)
        if self._freq.size == 2:
            idx1 = (np.abs(freqs - self._freq[0])).argmin()
            idx2 = (np.abs(freqs - self._freq[1])).argmin()
            self._fmask[idx1:idx2+1] = True
        else:
            idx = (np.abs(freqs - self._freq[0])).argmin()
            self._fmask[idx] = True
        selected_freq = freqs[self._fmask]
        # Shift the mask accordingly to the db index
        self._fmask = np.roll(
            self._fmask,
            self._meta['bea']['BeamletList'][self.db][0]
            )
        return selected_freq
    @freq.setter
    def freq(self, f):
        if f is None:
            f = self.freq_min.value
        # Convert to scalar if `Quantity` are entered
        if isinstance(f, list):
            f = [fi.value if isinstance(fi, u.Quantity) else fi for fi in f]
            f = sorted(f)
        else:
            f = [f.value] if isinstance(f, u.Quantity) else [f]
        f = np.array(f) * u.MHz
        assert f.size <= 2,\
            'freq attribute must be of size 2 maximum.'
        for i, fi in enumerate(f):
            if self.freq_min > fi:
                warnings.warn(
                    '\nWarning: {} < {}, setting to fmin.'.format(
                        fi,
                        self.freq_min
                        )
                    )
                f[i] = self.freq_min.copy()
            if self.freq_max < fi:
                warnings.warn(
                    '\nWarning: {} > {}, setting to fmax.'.format(
                        fi,
                        self.freq_max
                        )
                    )
                f[i] = self.freq_max.copy()
        self._freq = f
        return

    @property
    def track_dbeam(self):
        """ Choose one numerical beam index to follow
        """
        return self._track_dbeam
    @track_dbeam.setter
    def track_dbeam(self, t):
        if t is not None:
            self.pointing = self._digibeams == t
        self._track_dbeam = t
        return    


    # ------------------------- Getter -------------------------#
    @property
    def obs_start(self):
        """ First time record
        """
        return self.times[0]

    @property
    def obs_stop(self):
        """ Last time record
        """
        return self.times[-1]

    @property
    def beam_start(self):
        """ Start of the selected numerical pointing
        """
        start = self._meta['pbe']['timestamp'][self._pointing]
        start = np.array(
            [s.replace('Z', '') for s in start],
            dtype='datetime64[s]')
        return start

    @property
    def beam_stop(self):
        """ End of the selected numerical pointing
        """
        stop = self._meta['pbe']['timestamp'][np.roll(self._pointing, 1)]
        stop = np.array(
                [s.replace('Z', '') for s in stop],
                dtype='datetime64[s]')
        for i, s in enumerate(stop):
            if s <= self.beam_start[0]:
                stop[i] = self.obs_stop
        # if not self._pointing[-1]:
        #     # Last element is False: don't go to the last timestamp
        #     stop = self._meta['pbe']['timestamp'][np.roll(self._pointing, 1)]
        #     # if not isinstance(stop, str):
        #     #     stop = stop[-1]
        #     # stop = np.datetime64(stop.replace('Z', ''), 's')
        #     stop = np.array(
        #         [s.replace('Z', '') for s in stop],
        #         dtype='datetime64[s]')
        # else:
        #     stop = self.obs_stop
        # for i, s in enumerate(stop):
        #     if s <= self.beam_start:
        #         stop[i] = self.obs_stop
        return stop

    @property
    def duration(self):
        """ Duration of the selected beam
        """
        return self.beam_stop - self.beam_start

    @property
    def freq_min(self):
        """ Minimum observed frequency in MHz
        """
        freqs = self._meta['bea']['freqlist'][self.db]
        fmin = freqs[freqs > 0].min()
        return fmin * u.MHz

    @property
    def freq_max(self):
        """ Maximum observd frequency in MHz
        """
        f_max = self._meta['bea']['freqlist'][self.db].max()
        return f_max * u.MHz

    @property
    def db(self):
        """ Corresponding numerical beam index
        """
        return self._meta['pbe']['noBeam'][self._pointing][0]

    @property
    def ab(self):
        """ Corresponding analogic beam
        """
        return self._dig2ana[self.db]
    

    @property
    def azana(self):
        """ Pointed azimuth in degrees
            for the analog beam
        """
        a_mask = self._ana_start >= self.beam_start
        a_mask *= self._ana_stop <= self.beam_stop
        # a_mask = self.abeams == np.unique(self.ab)
        az = self.all_azana[a_mask]
        return az * u.deg

    @property
    def elana(self):
        """ Pointed elevation in degrees
            for the analog beam
        """
        a_mask = self._ana_start >= self.beam_start
        a_mask *= self._ana_stop <= self.beam_stop
        # a_mask = self.abeams == np.unique(self.ab)
        el = self.all_elana[a_mask]
        return el * u.deg

    @property
    def azdig(self):
        """ Pointed azimuth in degrees
            for the numerical beam
        """
        az = self.all_azdig[self._pointing]
        return az * u.deg

    @property
    def eldig(self):
        """ Pointed elevation in degrees
            for the numerical beam
        """
        el = self.all_eldig[self._pointing]
        return el * u.deg


    @property
    def ma(self):
        """ Mini-array used
        """
        n = self._meta['ana']['nbMRUsed'][self.ab]
        return self._meta['ana']['MRList'][self.ab, :n]


    @property
    def ma_position(self):
        """ Positions of mini-arrays
        """
        pos = self._meta['ins']['noPosition']
        pos = pos.reshape((pos.size//3, 3))
        return pos[self.ma]


    @property
    def ma_rotation(self):
        """ Rotations of mini-arrays
        """
        rot = self._meta['ins']['rotation']
        rot = rot[0]
        return rot[self.ma]
    

    # ------------------------- Method -------------------------#
    def select(self, **kwargs):
        """ Select and load BST data.
            
            Parameters
            ----------

            beam : int
                Beam index.

            freq : (float, array_like)
                Frequency selection in MHz.
                Single frequency:
                    `>>> freq=55`
                Frequency range selection:
                    `>>>freq=[20, 30]`

            time : (str or array_like)
                Time selection in ISO/ISOT format.
                Single time selection:
                    `>>> time='2019-08-07 12:00:00'`
                Time range selection:
                    `>>> time=['2019-08-07 12:00:00', '2019-08-07 13:00:00']`

            polar : str
                Polarization selection


            Returns
            -------

            data : SpecData
                Selected data are loaded and stored in the `data` attribute.
                This is a `SpecData` object.
        """
        for key in self._keywords:
            if key in kwargs.keys():
                self.__setattr__(key, kwargs[key])

        self.data = SpecData(
            time=self.time,
            freq=self.freq,
            polar=self.polar,
            data=self._alldata[np.ix_(self._tmask,
                self._pmask,
                self._fmask)]
            )
        return self.data


    def to_stmoc(self):
        """ Method to create a CDS STMOC from the current BST observation.
        """
        # self._meta['pbe']['timestamp']
        # self.all_azdig
        # self.all_eldig
        return


    def fit_transit(self, data=None):
        """ After a data selection (via `select()`),
            this performs a gaussian fitting.

            Example
            -------
            ```
            >>> from BST_v2 import BST
            >>> import matplotlib.pyplot as plt
            >>> b = BST('filename_BST.fits')
            >>> data = b.select(freq=50, pointing=0)
            >>> fit = b.fit_transit()
            >>> plt.plot(data.time, data.db, label='raw')
            >>> plt.plt(fit.time, fit.db, label='fitted')
            >>> plt.legend()
            >>> plt.show()
            ```
        """
        # import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        def gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        def linear(x, a, b):
            return a * x + b

        # Remove data spikes
        if data is None:
            if self.freq.size > 1:
                filtered_data = self.data.fmean(method='median', clean=True).amp
            else:
                filtered_data = self.data.filter(kernel=31).amp 
        else:
            assert isinstance(data, SpecData),\
                'data must be of type SpecData'
            if data.freq.size > 1:
                filtered_data = data.fmean(method='median', clean=True).amp
            else:
                filtered_data = data.filter(kernel=31).amp
        main_time = self.data.time

        # plt.figure()
        # plt.plot(self.data.time, filtered_data, label='raw')

        # Subtract a linear fit to the data prior to searching for max
        filtered_data_to_fit = filtered_data/filtered_data.max()
        popt_lin, pcov_lin = curve_fit(
            f=linear,
            xdata=self.data.jd,
            ydata=filtered_data_to_fit,
            p0=[0, filtered_data_to_fit.min()] # estimated parameters after normalization
            )
        # plt.plot(self.data.time, linear(self.data.jd, *popt_lin)*filtered_data.max(), label='fit lin')

        # Find the highest amplitude point
        # this should roughly be the transit source
        ind_max = np.argmax(filtered_data - linear(self.data.jd, *popt_lin))

        # Just consider a reduced time interval to fit a gaussian
        # +/- 15 min from the maximum
        dt = np.timedelta64(5, 'm')
        t_mask = main_time >= main_time[ind_max] - dt
        t_mask *= main_time <= main_time[ind_max] + dt
        
        # Data to fit
        time = self.data.jd[t_mask]
        data = filtered_data[t_mask]
        # Normalize them, which helps the fit
        t_fit = (time - np.median(time)) / (time - np.median(time)).max()
        d_fit = (data - data.min()) / (data - data.min()).max()
        
        # Perform the fit
        popt, pcov = curve_fit(
            f=gauss,
            xdata=t_fit,
            ydata=d_fit,
            sigma=np.ones(d_fit.size)*0.1,
            absolute_sigma=True,
            p0=[1, 0, 1] # estimated parameters after normalization
            )

        # Show the fitted data and rescale them
        d_rescaled = gauss(t_fit, *popt) * (data - data.min()).max() + data.min()

        # t_fit_extended = (self.data.jd - np.median(time)) / (time - np.median(time)).max()
        # d_rescaled_extended = gauss(t_fit_extended, *popt) * (data - data.min()).max() + data.min()


        # plt.plot(main_time[t_mask], d_rescaled, label='fit gauss')
        # plt.show()

        fit = SpecData(
            time=main_time[t_mask],
            freq=np.mean(self.freq),
            polar=self.polar,
            data=d_rescaled.reshape(d_rescaled.size, 1, 1)
            )

        # fit_extended = SpecData(
        #     time=main_time,
        #     freq=self.freq,
        #     polar=self.polar,
        #     data=d_rescaled_extended.reshape(d_rescaled_extended.size, 1, 1)
        #     )

        # plt.savefig('/Users/aloh/Desktop/Angular_Shift_2019_July/{}.png'.format(self.pointing[0]))
        # plt.close('all')

        peak_amp = popt[0] * (data - data.min()).max() + data.min()
        peak_t = popt[1] * (time - np.median(time)).max() + np.median(time)
        return fit, peak_amp, peak_t#, popt[2]

    def fit_transit_v0(self, data=None):
        """ After a data selection (via `select()`),
            this performs a gaussian fitting.

            Example
            -------
            ```
            >>> from BST_v2 import BST
            >>> import matplotlib.pyplot as plt
            >>> b = BST('filename_BST.fits')
            >>> data = b.select(freq=50, pointing=0)
            >>> fit = b.fit_transit()
            >>> plt.plot(data.time, data.db, label='raw')
            >>> plt.plt(fit.time, fit.db, label='fitted')
            >>> plt.legend()
            >>> plt.show()
            ```
        """
        from scipy.optimize import curve_fit
        def gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        # Remove data spikes
        if data is None:
            filtered_data = self.data.filter(kernel=31).amp
        else:
            assert isinstance(data, SpecData),\
                'data must be of type SpecData'
            filtered_data = data.filter(kernel=31).amp
        main_time = self.data.time

        # Find the highest amplitude point
        # this should roughly be the transit source
        ind_max = np.argmax(filtered_data)

        # Just consider a reduced time interval to fit a gaussian
        # +/- 15 min from the maximum
        dt = np.timedelta64(5, 'm')
        t_mask = main_time >= main_time[ind_max] - dt
        t_mask *= main_time <= main_time[ind_max] + dt
        
        # Data to fit
        time = self.data.jd[t_mask]
        data = filtered_data[t_mask]
        # Normalize them, which helps the fit
        t_fit = (time - np.median(time)) / (time - np.median(time)).max()
        d_fit = (data - data.min()) / (data - data.min()).max()
        
        # Perform the fit
        popt, pcov = curve_fit(
            f=gauss,
            xdata=t_fit,
            ydata=d_fit,
            sigma=np.ones(d_fit.size)*0.1,
            absolute_sigma=True,
            p0=[1, 0, 1] # estimated parameters after normalization
            )

        # Show the fitted data and rescale them
        d_rescaled = gauss(t_fit, *popt) * (data - data.min()).max() + data.min()

        fit = SpecData(
            time=main_time[t_mask],
            freq=self.freq,
            polar=self.polar,
            data=d_rescaled.reshape(d_rescaled.size, 1, 1)
            )

        return fit


    def _read_bst(self, bst):
        """ Low-level BST reader, mainly used to parse beam informations.
        """
        self.abeams = self._meta['ana']['NoAnaBeam']
        self._n_anapoints = self._meta['ana']['NbPointingAB']
        self.dbeams = self._meta['bea']['noBeam']
        self._n_digpoints = self._meta['bea']['NbPointingB']
        self._dig2ana = self._meta['bea']['NoAnaBeam']
        self._n_points = np.sum(self._n_digpoints)

        self._anabeams = self._meta['pan']['noAnaBeam']
        self.all_azana = self._meta['pan']['AZ']
        self.all_elana = self._meta['pan']['EL']
        self._digibeams = self._meta['pbe']['noBeam']
        self.all_azdig = self._meta['pbe']['AZ']
        self.all_eldig = self._meta['pbe']['EL']

        self._ana_start = np.array(
            self._meta['pan']['timestamp'].replace('Z', ''),
            dtype='datetime64[s]'
            )
        self._ana_stop = np.append(self._ana_start[1:],
            self.times[-1].astype('datetime64[s]'))

        return
# ============================================================= #



















# ============================================================= #
# ---------------------------- BST ---------------------------- #
# ============================================================= #
class BST_v0(NenuFAR_Obs):
    """
    """
    def __init__(self, bst):
        super().__init__(bst)
        self.bstfile = self.fitsfile
        self.db = 0
        self.ab = 0
        self.polar = 'NW'
        self.freq = 50
        self.time = None

        self._keywords = [
            'db',
            'ab',
            'freq',
            'time',
            'polar']

    # ------------------------- Setter -------------------------#
    @property
    def bstfile(self):
        """ Path toward the NenuFAR BST FITS file.
        """
        return self._bstfile
    @bstfile.setter
    def bstfile(self, b):
        assert 'beamlet' in self._meta['hea']['Object'],\
            '{} is not a BST file.'.format(b)
        self._read_bst(bst=b)
        self._bstfile = b
        return

    @property
    def db(self):
        """ Numerical beam index
        """
        return self._db
    @db.setter
    def db(self, d):
        # assert d in self._num_beams['index'],\
        #     '{} not in {}'.format(d, self._num_beams['index'])
        assert d in self._num_beams['name'],\
            '{} not in {}'.format(d, self._num_beams['name'])
        self._db = d
        return

    @property
    def ab(self):
        """ Analog beam index
        """
        return self._db
    @ab.setter
    def ab(self, a):
        assert a in self._ana_beams['name'],\
            '{} not in {}'.format(a, self._ana_beams['name'])
        # assert a in self._ana_beams['index'],\
        #     '{} not in {}'.format(a, self._ana_beams['index'])
        self._ab = a
        return

    @property
    def polar(self):
        """ Polarization selection.
            Could be 'NE' or 'NW' 
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        p = p.upper()
        assert p in self._meta['ins']['spol'],\
            'Polarization should be NE or NW'
        obs_pols = self._meta['ins']['spol'].ravel()
        self._pmask = (obs_pols == p)
        self._polar = obs_pols[self._pmask]
        return
    
    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        freqs = self._meta['bea']['freqlist'][self.db] * u.MHz
        self._fmask = np.zeros(freqs.size, dtype=bool)
        if self._freq.size == 2:
            idx1 = (np.abs(freqs - self._freq[0])).argmin()
            idx2 = (np.abs(freqs - self._freq[1])).argmin()
            self._fmask[idx1:idx2+1] = True
        else:
            idx = (np.abs(freqs - self._freq[0])).argmin()
            self._fmask[idx] = True
        selected_freq = freqs[self._fmask]
        # Shift the mask accordingly to the db index
        self._fmask = np.roll(
            self._fmask,
            self._meta['bea']['BeamletList'][self.db][0]
            )
        return selected_freq
    @freq.setter
    def freq(self, f):
        f = np.array(sorted([f])).ravel() * u.MHz
        assert f.size <= 2,\
            'freq attribute must be of size 2 maximum.'
        for i, fi in enumerate(f):
            if self.freq_min > fi:
                warnings.warn(
                    '\nWarning: {} < {}, setting to fmin.'.format(
                        fi,
                        self.freq_min
                        )
                    )
                f[i] = self.freq_min.copy()
            if self.freq_max < fi:
                warnings.warn(
                    '\nWarning: {} > {}, setting to fmax.'.format(
                        fi,
                        self.freq_max
                        )
                    )
                f[i] = self.freq_max.copy()
        self._freq = f
        return
    
    @property
    def time(self):
        """ Time selection, must be given in ISO/ISOT format.
        """
        self._tmask = np.zeros(self.times.size, dtype=bool)
        if self._time.size == 2:
            idx1 = (np.abs(self.times - self._time[0])).argmin()
            idx2 = (np.abs(self.times - self._time[1])).argmin()
            self._tmask[idx1:idx2+1] = True
        else:
            idx = (np.abs(self.times - self._time[0])).argmin()
            self._tmask[idx] = True
        self._tmask *= (
            (self.times >= self.a_start) &\
            (self.times <= self.a_stop)
            )
        # self._tmask *= (
        #     (self.times >= self.d_start) &\
        #     (self.times <= self.d_stop)
        #     )
        return self.times[self._tmask]
    @time.setter
    def time(self, t):
        if t is None:
            t = [self.obs_start, self.obs_stop]
        t = np.array([t], dtype='datetime64').ravel()
        assert t.size <= 2,\
            'time attribute must be of size 2 maximum.'
        for ti in t:
            assert (self.obs_start <= ti) &\
                (ti <= self.obs_stop),\
                '{} not within {} -- {}'.format(
                        ti,
                        self.obs_start,
                        self.obs_stop)
        if t.size == 2:
            assert t[0] < t[1],\
            '{} should be before {}.'.format(t[0], t[1])
        self._time = t
        return
    

    # ------------------------- Getter -------------------------#
    @property
    def obs_start(self):
        """ First time record
        """
        return self.times[0]

    @property
    def obs_stop(self):
        """ Last time record
        """
        return self.times[-1]

    @property
    def d_start(self):
        """ Start of the selected numerical pointing
        """
        return self._num_beams['start'][self.db]

    @property
    def d_stop(self):
        """ End of the selected numerical pointing
        """
        return self._num_beams['stop'][self.db]

    @property
    def a_start(self):
        """ Start of the selected numerical pointing
        """
        return self._ana_beams['start'][self.ab]

    @property
    def a_stop(self):
        """ End of the selected numerical pointing
        """
        return self._ana_beams['stop'][self.ab]

    @property
    def freq_min(self):
        """ Minimum observed frequency in MHz
        """
        freqs = self._meta['bea']['freqlist'][self.db]
        fmin = freqs[freqs > 0].min()
        return fmin * u.MHz

    @property
    def freq_max(self):
        """ Maximum observd frequency in MHz
        """
        f_max = self._meta['bea']['freqlist'][self.db].max()
        return f_max * u.MHz


    

    def select(self, **kwargs):
        """ Select and load BST data.
            
            Parameters
            ----------

            ab : int
                Analog beam index.

            db : int
                Numerical beam index.

            freq : (float, array_like)
                Frequency selection in MHz.
                Single frequency:
                    `>>> freq=55`
                Frequency range selection:
                    `>>>freq=[20, 30]`

            time : (str or array_like)
                Time selection in ISO/ISOT format.
                Single time selection:
                    `>>> time='2019-08-07 12:00:00'`
                Time range selection:
                    `>>> time=['2019-08-07 12:00:00', '2019-08-07 13:00:00']`

            polar : str
                Polarization selection


            Returns
            -------

            data : SpecData
                Selected data are loaded and stored in the `data` attribute.
                This is a `SpecData` object.
        """
        for key in self._keywords:
            if key in kwargs.keys():
                self.__setattr__(key, kwargs[key])

        self.data = SpecData(
            time=self.time,
            freq=self.freq,
            polar=self.polar,
            data=self._alldata[np.ix_(self._tmask,
                self._pmask,
                self._fmask)]
            )
        return

    def _read_bst(self, bst):
        """ Low-level BST reader, mainly used to parse beam informations.
        """

        # ------ Parse Analog Beams ------ #
        ana_start = np.array(
            self._meta['pan']['timestamp'].replace('Z', ''),
            dtype='datetime64[s]'
            )
        ana_stop = np.append(ana_start[1:],
            self.times[-1].astype('datetime64[s]'))
        self._ana_beams = {
                'name': np.arange(ana_start.size).astype('int16'),
                'index': self._meta['pan']['NoAnaBeam'],
                'start': ana_start,
                'stop': ana_stop,
                'duration': (ana_stop - ana_start).astype('timedelta64[s]'),
                'az': self._meta['pan']['AZ'],
                'el': self._meta['pan']['EL']
            }

        # ------ Parse Numeric Beams ------ #
        num_start = np.array(
            self._meta['pbe']['timestamp'].replace('Z', ''),
            dtype='datetime64[s]'
            )
        num_stop = np.append(num_start[1:],
            self.times[-1].astype('datetime64[s]'))
        self._num_beams = {
                'name': np.arange(num_start.size).astype('int16'),
                'index': self._meta['pbe']['NoBeam'],
                'toana': self._meta['bea']['NoAnaBeam']\
                    if self._meta['bea']['NoAnaBeam'].size==num_start.size\
                    else np.arange(num_start.size).astype('int16'),
                'start': num_start,
                'stop': num_stop,
                'duration': (num_stop - num_start),#.astype('timedelta64[s]'),
                'az': self._meta['pbe']['AZ'],
                'el': self._meta['pbe']['EL']
            }
# ============================================================= #


# ============================================================= #
# ---------------------------- XST ---------------------------- #
# ============================================================= #
class XST(NenuFAR_Obs):
    """
    """
    def __init__(self, xst):
        super().__init__(xst)
        self.xstfile = self.fitsfile
        self.db = 0
        self.ab = 0
        self.polar = 'XX'
        self.ma1 = None
        self.ma2 = None 
        self.freq = None
        self.time = None

        self._keywords = [
            'db',
            'ab',
            'ma1',
            'ma2',
            'freq',
            'time',
            'polar']

    @property
    def xstfile(self):
        """ Path toward the NenuFAR XST FITS file.
        """
        return self._xstfile
    @xstfile.setter
    def xstfile(self, x):
        assert 'crosscorrelation' in self._meta['hea']['Object'],\
            '{} is not a XST file.'.format(x)
        self._read_xst(xst=x)
        self._xstfile = x
        return

    @property
    def db(self):
        """ Numerical beam index
        """
        return self._db
    @db.setter
    def db(self, d):
        assert d in self._num_beams['index'],\
            '{} not in {}'.format(d, self._num_beams['index'])
        self._db = d
        return

    @property
    def ab(self):
        """ Analog beam index
        """
        return self._db
    @ab.setter
    def ab(self, a):
        assert a in self._ana_beams['name'],\
            '{} not in {}'.format(a, self._ana_beams['name'])
        # assert a in self._ana_beams['index'],\
        #     '{} not in {}'.format(a, self._ana_beams['index'])
        self._ab = a
        return

    @property
    def polar(self):
        """ Polarization selection.
            Could be 'XX' or 'XY', 'YX', 'YY' 
        """
        pols = np.array(['XX', 'XY', 'YX', 'YY'])
        return pols[self._pmask]
    @polar.setter
    def polar(self, p):
        p = p.upper()
        pols = np.array(['XX', 'XY', 'YX', 'YY'])
        assert p in pols,\
            'Polarization should be XX or XY or YX or YY'
        self._pmask = pols == p
        self._polar = p
        return
    
    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        freqs = self._xstsb[0] * u.MHz # assume all frequencies are identical
        if self._freq.size == 2:
            self._fmask = (freqs >= self._freq[0]) & (freqs <= self._freq[1])
        else:
            self._fmask = freqs == self._freq[0]
        return freqs[self._fmask]
    @freq.setter
    def freq(self, f):
        fmin = self._xstsb.min() * u.MHz
        fmax = self._xstsb.max() * u.MHz
        if f is not None:
            f = np.array(sorted([f])).ravel() * u.MHz
            assert f.size <= 2,\
                'freq attribute must be of size 2 maximum.'
        else:
            f = np.array([fmin.value, fmax.value]) * u.MHz
        for i, fi in enumerate(f):
            if fmin > fi:
                warnings.warn(
                    '\nWarning: {} < {}, setting to fmin.'.format(
                        fi,
                        fmin
                        )
                    )
                f[i] = fmin
            if fmax < fi:
                warnings.warn(
                    '\nWarning: {} > {}, setting to fmax.'.format(
                        fi,
                        fmax
                        )
                    )
                f[i] = fmax
        self._freq = f
        return

    @property
    def ma1(self):
        """ First correlated Mini-Array(s)
        """
        mas = self._meta['ins']['noMR'].ravel()
        self._ma1mask = np.isin(mas, self._ma1)
        return self._ma1
    @ma1.setter
    def ma1(self, m):
        mas = self._meta['ins']['noMR'].ravel()
        if isinstance(m, np.ndarray):
            pass
        elif m is None:
            m = mas
        elif isinstance(m, list):
            m = np.array(m)
        else:  
            m = np.array([m])
        assert all(np.isin(m, mas)),\
            'Some MA(s) not in selection.'
        self._ma1 = m
        return
    
    @property
    def ma2(self):
        """ Second correlated Mini-Array(s)
        """
        mas = self._meta['ins']['noMR'].ravel()
        self._ma2mask = np.isin(mas, self._ma2)
        return self._ma1
    @ma2.setter
    def ma2(self, m):
        mas = self._meta['ins']['noMR'].ravel()
        if isinstance(m, np.ndarray):
            pass
        elif m is None:
            m = mas
        elif isinstance(m, list):
            m = np.array(m)
        else:  
            m = np.array([m])
        assert all(np.isin(m, mas)),\
            'Some MA(s) not in selection.'
        self._ma2 = m
        return
    
    @property
    def time(self):
        """ Time selection, must be given in ISO/ISOT format.
        """
        self._tmask = np.zeros(self.times.size, dtype=bool)
        if self._time.size == 2:
            idx1 = (np.abs(self.times - self._time[0])).argmin()
            idx2 = (np.abs(self.times - self._time[1])).argmin()
            self._tmask[idx1:idx2+1] = True
        else:
            idx = (np.abs(self.times - self._time[0])).argmin()
            self._tmask[idx] = True
        self._tmask *= (
            (self.times >= self.d_start) &\
            (self.times <= self.d_stop)
            )
        return self.times[self._tmask]
    @time.setter
    def time(self, t):
        if t is None:
            t = [self.obs_start, self.obs_stop]
        t = np.array([t], dtype='datetime64').ravel()
        assert t.size <= 2,\
            'time attribute must be of size 2 maximum.'
        for ti in t:
            assert (self.obs_start <= ti) &\
                (ti <= self.obs_stop),\
                '{} not within {} -- {}'.format(
                        ti,
                        self.obs_start,
                        self.obs_stop)
        if t.size == 2:
            assert t[0] < t[1],\
            '{} shoulbd be before {}.'.format(t[0], t[1])
        self._time = t
        return
    

    @property
    def obs_start(self):
        """ First time record
        """
        return self.times[0]

    @property
    def obs_stop(self):
        """ Last time record
        """
        return self.times[-1]

    @property
    def d_start(self):
        """ Start of the selected numerical pointing
        """
        return self._num_beams['start'][self.db]

    @property
    def d_stop(self):
        """ End of the selected numerical pointing
        """
        return self._num_beams['stop'][self.db]

    @property
    def freq_min(self):
        """ Minimum observed frequency in MHz
        """
        freqs = self._meta['bea']['freqlist'][self.db]
        fmin = freqs[freqs > 0].min()
        return fmin * u.MHz

    @property
    def freq_max(self):
        """ Maximum observd frequency in MHz
        """
        f_max = self._meta['bea']['freqlist'][self.db].max()
        return f_max * u.MHz
    

    def get_snapshot(self, **kwargs):
        """ Generator that yield a XST snapshot (per time step) correctly formatted.

            TODO : remove the conj diagonal added
        """
        for key in self._keywords:
            if key in kwargs.keys():
                self.__setattr__(key, kwargs[key])

        nant = self._meta['ins']['nbMR'][0]
        snap = np.zeros(
            (
                nant, # MAs
                nant, # MAs
                4, # Cross polarizations
                self._xstsb[0].size # frequencies
            ),
            dtype=np.complex
            )
        count=0
        for it, t in enumerate(self.time):
            for isb, f in enumerate(self._xstsb[0]):
                # snap_sb is an array like:
                #       ma0_X ma0_Y ma1_X ma1_Y ma2_X
                # ma0_X  ...   ...   ...   ...   ...
                # ma0_Y  ...   ...   ...   ...   ...
                # ma1_X  ...   ...   ...   ...   ...
                # ma1_Y  ...   ...   ...   ...   ...
                # ma2_X  ...   ...   ...   ...   ...
                s_sb = np.zeros((nant*2, nant*2), dtype=np.complex)
                s_sb[np.tril_indices(n=nant*2, k=0)] = self._alldata[it, isb, :]

                snap[:, :, 0, isb] = s_sb[0::2, 0::2] + s_sb[0::2, 0::2].T.conj() # XX
                snap[:, :, 1, isb] = s_sb[0::2, 1::2] + s_sb[0::2, 1::2].T.conj() # XY
                snap[:, :, 2, isb] = s_sb[1::2, 0::2] + s_sb[1::2, 0::2].T.conj() # YX
                snap[:, :, 3, isb] = s_sb[1::2, 1::2] + s_sb[1::2, 1::2].T.conj() # YY
            yield snap

    def select(self, **kwargs):
        """ Select and load BST data.
            
            Parameters
            ----------

            ab : int
                Analog beam index.

            db : int
                Numerical beam index.

            freq : (float, array_like)
                Frequency selection in MHz.
                Single frequency:
                    `>>> freq=55`
                Frequency range selection:
                    `>>>freq=[20, 30]`

            time : (str or array_like)
                Time selection in ISO/ISOT format.
                Single time selection:
                    `>>> time='2019-08-07 12:00:00'`
                Time range selection:
                    `>>> time=['2019-08-07 12:00:00', '2019-08-07 13:00:00']`

            polar : str
                Polarization selection


            Returns
            -------

            data : SpecData
                Selected data are loaded and stored in the `data` attribute.
                This is a `SpecData` object.
        """
        for key in self._keywords:
            if key in kwargs.keys():
                self.__setattr__(key, kwargs[key])

        self.data = np.empty((
            self.time.size,
            self.ma1.size,
            self.ma2.size,
            self.polar.size,
            self.freq.size
            ), dtype=np.complex)

        for i, snap in enumerate(self.get_snapshot()):
            self.data[i] = snap[np.ix_(
                    self._ma1mask,
                    self._ma2mask,
                    self._pmask,
                    self._fmask,
                )]

        return

    def _read_xst(self, xst):
        """ Low-level XST reader, mainly used to parse beam informations.
        """

        # ------ Parse Analog Beams ------ #
        ana_start = np.array(
            self._meta['pan']['timestamp'].replace('Z', ''),
            dtype='datetime64[s]'
            )
        ana_stop = np.append(ana_start[1:],
            self.times[-1].astype('datetime64[s]'))
        self._ana_beams = {
                'name': np.arange(ana_start.size).astype('int16'),
                'index': self._meta['pan']['NoAnaBeam'],
                'start': ana_start,
                'stop': ana_stop,
                'duration': (ana_stop - ana_start).astype('timedelta64[s]'),
                'az': self._meta['pan']['AZ'],
                'el': self._meta['pan']['EL']
            }

        # ------ Parse Numeric Beams ------ #
        num_start = np.array(
            self._meta['pbe']['timestamp'].replace('Z', ''),
            dtype='datetime64[s]'
            )
        num_stop = np.append(num_start[1:],
            self.times[-1].astype('datetime64[s]'))
        self._num_beams = {
                'name': np.arange(num_start.size).astype('int16'),
                'index': self._meta['pbe']['NoBeam'],
                'toana': self._meta['bea']['NoAnaBeam']\
                    if self._meta['bea']['NoAnaBeam'].size==num_start.size\
                    else np.arange(num_start.size).astype('int16'),
                'start': num_start,
                'stop': num_stop,
                'duration': (num_stop - num_start),#.astype('timedelta64[s]'),
                'az': self._meta['pbe']['AZ'],
                'el': self._meta['pbe']['EL']
            }
# ============================================================= #

