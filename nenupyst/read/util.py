#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = [
    'NenuFAR_Obs',
    'NenuFAR',
    'SpecData'
]


import numpy as np 
import os
from astropy.io import fits
import astropy.units as u
from astropy.time import Time


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
    """ A class to handle dynamic spectrum data
    """

    def __init__(self, data, time, freq, **kwargs):
        self.time = time
        self.freq = freq
        self.data = data
        self.meta = kwargs


    def __repr__(self):
        return self.data.__repr__()


    def __and__(self, other):
        """ Concatenate two SpecData object in frequency
        """
        if not isinstance(other, SpecData):
            raise TypeError(
                'Trying to concatenate something else than SpecData'
                )
        if 'stokes' in self.meta.keys():
            if self.meta['stokes'] != other.meta['stokes']:
                raise ValueError(
                    'Inconsistent Stokes parameters'
                    )

        if self.freq.max() < other.freq.min():
            new_data = np.hstack((self.data, other.data))
            new_time = self.time
            new_freq = np.concatenate((self.freq, other.freq))
        else:
            new_data = np.hstack((other.data, self.data))
            new_time = self.time
            new_freq = np.concatenate((other.freq, self.freq))

        return SpecData(
            data=new_data,
            time=new_time,
            freq=new_freq,
            stokes=self.meta['stokes']
            )


    def __or__(self, other):
        """ Concatenate two SpecData in time
        """
        if not isinstance(other, SpecData):
            raise TypeError(
                'Trying to concatenate something else than SpecData'
                )
        if 'stokes' in self.meta.keys():
            if self.meta['stokes'] != other.meta['stokes']:
                raise ValueError(
                    'Inconsistent Stokes parameters'
                    )

        if self.time.max() < other.time.min():
            new_data = np.vstack((self.data, other.data))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_freq = self.freq
        else:
            new_data = np.vstack((other.data, self.data))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_freq = self.freq

        return SpecData(
            data=new_data,
            time=new_time,
            freq=new_freq,
            stokes=self.meta['stokes']
            )


    def __add__(self, other):
        """ Add two SpecData
        """
        if isinstance(other, SpecData):
            self._check_conformity(other)
            add = other.amp
        else:
            self._check_value(other)
            add = other 

        return SpecData(
            data=self.amp + add,
            time=self.time,
            freq=self.freq,
            stokes=self.meta['stokes']
            )


    def __sub__(self, other):
        """ Subtract two SpecData
        """
        if isinstance(other, SpecData):
            self._check_conformity(other)
            sub = other.amp
        else:
            self._check_value(other)
            sub = other 

        return SpecData(
            data=self.amp - sub,
            time=self.time,
            freq=self.freq,
            stokes=self.meta['stokes']
            )


    def __mul__(self, other):
        """ Multiply two SpecData
        """
        if isinstance(other, SpecData):
            self._check_conformity(other)
            mul = other.amp
        else:
            self._check_value(other)
            mul = other 

        return SpecData(
            data=self.amp * mul,
            time=self.time,
            freq=self.freq,
            stokes=self.meta['stokes']
            )


    def __truediv__(self, other):
        """ Divide two SpecData
        """
        if isinstance(other, SpecData):
            self._check_conformity(other)
            div = other.amp
        else:
            self._check_value(other)
            div = other 

        return SpecData(
            data=self.amp / div,
            time=self.time,
            freq=self.freq,
            stokes=self.meta['stokes']
            )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, d):
        ts, fs = d.shape
        assert self.time.size == ts,\
            'time axis inconsistent'
        assert self.freq.size == fs,\
            'frequency axis inconsistent'
        self._data = d
        return


    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, t):
        if not isinstance(t, Time):
            raise TypeError('Time object expected')
        self._time = t
        return


    @property
    def mjd(self):
        """ Return MJD dates
        """
        return self.time.mjd


    @property
    def jd(self):
        """ Return JD dates
        """
        return self.time.jd


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


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def fmean(self, freq1=None, freq2=None, method='mean'):
        """ Average over the frequency.
            
            Parameters
            ----------
            freq1 : float
                Lower frequency bound in MHz.

            freq2 : float
                Upper frequency bound in MHz.

            method : str
                Method used to average (either 'mean' or 'median')

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
        
        data = self.data[:, fmask]
        if clean:
            # Find out the noisiest profiles
            sigmas = np.std(data, axis=0)
            max_sig = np.percentile(sigmas, 90, axis=1)
            data = data[:, np.newaxis, sigmas < max_sig]

            # Smoothin' the time profiles
            from scipy.signal import savgol_filter
            tf = np.zeros(data.shape)
            for j in range(sigmas[sigmas < max_sig].size):
                tf[:, i, j] = savgol_filter(data[:, j], 201, 1)

            # Rescale everything not to bias the mean
            data = tf - (np.median(tf, axis=0) - np.median(tf))

        average = np.mean(data, axis=1)\
            if method == 'mean'\
            else np.median(data, axis=1)

        return SpecData(
            data=np.expand_dims(average, axis=1),
            time=self.time.copy(),
            freq=np.expand_dims(np.mean(self.freq[fmask]), axis=0),
            stokes=self.meta['stokes']
            )


    def frebin(self, bins):
        """
        """
        bins = int(bins)

        slices = np.linspace(
            0,
            self.freq.size,
            bins + 1,
            True
        ).astype(np.int)
        counts = np.diff(slices)
        return SpecData(
            data=np.expand_dims(np.add.reduceat(self.amp, slices[:-1]) / counts, axis=0),
            time=self.time.copy(),
            freq=np.add.reduceat(self.freq, slices[:-1]) / counts,
            stokes=self.meta['stokes']
            )


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
        average = np.mean(self.data[tmask, :], axis=0)\
            if method == 'mean'\
            else np.median(self.data[tmask, :], axis=0)
        return SpecData(
            data=np.expand_dims(average, axis=0),
            time=Time(np.array([tmasked[0] + dt/2.])),
            freq=self.freq.copy(),
            stokes=self.meta['stokes']
            )


    def background(self):
        """ Compute the median background
        """
        specf = self.fmean(method='median')
        spect = self.tmean(method='median')
        bkg = np.ones(self.amp.shape)
        bkg *= specf.amp[:, np.newaxis] * spect.amp[np.newaxis, :]
        return SpecData(
            data=bkg,
            time=self.time.copy(),
            freq=self.freq.copy(),
            stokes=self.meta['stokes']
            )


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
        if (self.data.shape[1] == 1) and (not isinstance(kernel, list)):
            kernel = [kernel, 1]
        filtered_data = np.zeros(self.data.shape)
        tf = medfilt(self.data[:, :], kernel)
        filtered_data[:, :] = tf
        return SpecData(
            data=filtered_data,
            time=self.time.copy(),
            freq=self.freq.copy()
            )


    def bg_remove(self):
        """
        """
        bg = self.background()
        return SpecData(
            data=self.amp,
            time=self.time,
            freq=self.freq,
            stokes=self.meta['stokes']
            ) - bg


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _check_conformity(self, other):
        """ Checks that other if of same type, same time, 
            frequency ans Stokes parameters than self
        """
        if self.meta['stokes'] != other.meta['stokes']:
            raise ValueError(
                'Different Stokes parameters'
                )

        if self.amp.shape != other.amp.shape:
            raise ValueError(
                'SpecData objects do not have the same dimensions'
                )

        if self.time != other.time:
            raise ValueError(
                'Not the same times'
                )

        if self.freq != other.freq:
            raise ValueError(
                'Not the same frequencies'
                )

        return


    def _check_value(self, other):
        """ Checks that other can be operated with self if
            other is not a SpecData object
        """
        if isinstance(other, np.ndarray):
            if other.shape != self.amp.shape:
                raise ValueError(
                    'Shape mismatch'
                )
        elif isinstance(other, (int, float)):
            pass
        else:
            raise Exception(
                'Operation unknown with {}'.format(type(other))
            )
        return
# ============================================================= #

