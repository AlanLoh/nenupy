#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *****
    SData
    *****

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'SData'
]


from astropy.time import Time
import astropy.units as u
import numpy as np


# ============================================================= #
# --------------------------- SData --------------------------- #
# ============================================================= #
class SData(object):
    """ A class to handle dynamic spectrum data
    """

    def __init__(self, data, time, freq, polar):
        self.time = time
        self.freq = freq
        self.data = data
        self.polar = polar


    def __repr__(self):
        return self.data.__repr__()


    def __and__(self, other):
        """ Concatenate two SData object in frequency
        """
        if not isinstance(other, SData):
            raise TypeError(
                'Trying to concatenate something else than SData'
                )
        if self.polar != other.polar:
            raise ValueError(
                'Inconsistent polar parameters'
                )

        if self.freq.max() < other.freq.min():
            new_data = np.hstack((self.data, other.data))
            new_time = self.time
            new_freq = np.concatenate((self.freq, other.freq))
        else:
            new_data = np.hstack((other.data, self.data))
            new_time = self.time
            new_freq = np.concatenate((other.freq, self.freq))

        return SData(
            data=new_data,
            time=new_time,
            freq=new_freq,
            polar=self.polar
            )


    def __or__(self, other):
        """ Concatenate two SData in time
        """
        if not isinstance(other, SData):
            raise TypeError(
                'Trying to concatenate something else than SData'
                )
        if self.polar != other.polar:
            raise ValueError(
                'Inconsistent polar parameters'
                )

        if self.time.max() < other.time.min():
            new_data = np.vstack((self.data, other.data))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_freq = self.freq
        else:
            new_data = np.vstack((other.data, self.data))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_freq = self.freq

        return SData(
            data=new_data,
            time=new_time,
            freq=new_freq,
            stokes=self.polar
            )


    def __add__(self, other):
        """ Add two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            add = other.amp
        else:
            self._check_value(other)
            add = other 

        return SData(
            data=self.amp + add,
            time=self.time,
            freq=self.freq,
            stokes=self.polar
            )


    def __sub__(self, other):
        """ Subtract two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            sub = other.amp
        else:
            self._check_value(other)
            sub = other 

        return SData(
            data=self.amp - sub,
            time=self.time,
            freq=self.freq,
            stokes=self.polar
            )


    def __mul__(self, other):
        """ Multiply two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            mul = other.amp
        else:
            self._check_value(other)
            mul = other 

        return SData(
            data=self.amp * mul,
            time=self.time,
            freq=self.freq,
            stokes=self.polar
            )


    def __truediv__(self, other):
        """ Divide two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            div = other.amp
        else:
            self._check_value(other)
            div = other 

        return SData(
            data=self.amp / div,
            time=self.time,
            freq=self.freq,
            stokes=self.polar
            )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, d):
        ts, fs, ps = d.shape
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
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, u.Quantity):
            raise TypeError('Quantity object expected')
        self._freq = f
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
    def datetime(self):
        """ Return datetime dates
        """
        return self.time.datetime


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
    

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _check_conformity(self, other):
        """ Checks that other if of same type, same time, 
            frequency ans Stokes parameters than self
        """
        if self.polar != other.polar:
            raise ValueError(
                'Different polarization parameters'
                )

        if self.amp.shape != other.amp.shape:
            raise ValueError(
                'SData objects do not have the same dimensions'
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
            other is not a SData object
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


