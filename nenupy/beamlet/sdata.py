#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *****
    SData
    *****

    :class:`~nenupy.beamlet.sdata.SData` is a class designed for
    embedding dynamic spectrum data sets. Many methods from the
    `nenupy` library return :class:`~nenupy.beamlet.sdata.SData`
    objects such as:

    * :meth:`~nenupy.beamlet.bstdata.BST_Data.select`
    * :meth:`~nenupy.crosslet.crosslet.Crosslet.beamform`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.time_profile`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.azel_transit`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.radec_transit`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.radec_tracking`

    It therefore provides a unified way of accessing, comparing
    and eventually plotting dynamic spectrum data sets throughout
    every `nenupy` manipulations. 

    :class:`~nenupy.beamlet.sdata.SData` are unit sensitive
    objets, therefore in addition it is worth importing
    :mod:`astropy.time` and :mod:`astropy.units` modules:

    >>> from nenupy.beamlet import SData
    >>> import numpy as np
    >>> from astropy.time import Time, TimeDelta
    >>> import astropy.units as u

    **Operations**

    All basic mathematical operations are enabled between
    :class:`~nenupy.beamlet.sdata.SData` instances (addition,
    subtraction, multiplication and division).

    In the following example, ``s1`` and ``s2`` are two identical
    :class:`~nenupy.beamlet.sdata.SData` instances with every
    data set to ``1.``. They define data taken at the same 
    times, frequencies and polarizations (``time``, ``freq`` and
    ``polar`` respectively):

    >>> dts = TimeDelta(np.arange(3), format='sec')
    >>> s1 = SData(
            data=np.ones((3, 5, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.ones(5) * u.MHz,
            polar=['NE']
        )
    >>> s2 = SData(
            data=np.ones((3, 5, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.ones(5) * u.MHz,
            polar=['NE']
        )

    Addition:

    >>> s_add = s1 + s2
    >>> s_add.amp
    array([[2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2.]])

    Subtraction:

    >>> s_sub = s1 - s2
    >>> s_add.amp
    array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])

    Multiplication:

    >>> s_mul = s1 * s2
    >>> s_add.amp
    array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])

    Division:

    >>> s_div = s1 / s2
    >>> s_add.amp
    array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])

    
    **Concatenation**
    
    Besides mathematical operations,
    :class:`~nenupy.beamlet.sdata.SData` objects may also be
    concatenated along the time or frequency axis for building up
    dataset purposes.
    
    In the following example, ``s1`` is the main instance, ``s2``
    has similar times but different frequencies and ``s3`` has
    similar frequencies but different times than ``s1``.

    >>> dts = TimeDelta(np.arange(2), format='sec')
    >>> s1 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
    >>> s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=(10+np.arange(3)) * u.MHz,
            polar=['NE']
        )
    >>> s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 13:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )

    Concatenation in frequency:

    >>> s_freq_concat = s1 & s2
    >>> s_freq_concat.shape
    (2, 6, 1)

    Concatenation in time:

    >>> s_time_concat = s1 | s3
    >>> s_time_concat.shape
    (4, 3, 1)

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
    """ A class to handle dynamic spectrum data.

        :param data: Dynamic spectrum data (time, freq, polar)
        :type data: :class:`~numpy.ndarray`
        :param freq: Times
        :type time: :class:`~astropy.time.Time`
        :param time: Frequencies
        :type freq: :class:`~astropy.units.Quantity`
        :param polar: Polarizations
        :type polar: `str`, `list` or :class:`~numpy.ndarray`
    """

    def __init__(self, data, time, freq, polar):
        self.time = time
        self.freq = freq
        self.polar = polar
        self.data = data


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
        if not all(self.time == other.time):
            raise ValueError(
                'Inconsistent time parameters'
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
        if not all(self.freq == other.freq):
            raise ValueError(
                'Inconsistent freq parameters'
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
            polar=self.polar
            )


    def __add__(self, other):
        """ Add two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            add = other.data
        else:
            self._check_value(other)
            add = other 

        return SData(
            data=self.data + add,
            time=self.time,
            freq=self.freq,
            polar=self.polar
            )


    def __sub__(self, other):
        """ Subtract two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            sub = other.data
        else:
            self._check_value(other)
            sub = other 

        return SData(
            data=self.data - sub,
            time=self.time,
            freq=self.freq,
            polar=self.polar
            )


    def __mul__(self, other):
        """ Multiply two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            mul = other.data
        else:
            self._check_value(other)
            mul = other 

        return SData(
            data=self.data * mul,
            time=self.time,
            freq=self.freq,
            polar=self.polar
            )


    def __truediv__(self, other):
        """ Divide two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            div = other.data
        else:
            self._check_value(other)
            div = other 

        return SData(
            data=self.data / div,
            time=self.time,
            freq=self.freq,
            polar=self.polar
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
        assert self.polar.size == ps,\
            'polar axis inconsistent'
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
    def polar(self):
        return self._polar
    @polar.setter
    def polar(self, p):
        if isinstance(p, list):
            p = np.array(p)
        if np.isscalar(p):
            p = np.array([p])
        self._polar = p
        return


    @property
    def mjd(self):
        """ Return MJD dates

            :getter: Times in Modified Julian Days
            
            :type: :class:`~numpy.ndarray`
        """
        return self.time.mjd


    @property
    def jd(self):
        """ Return JD dates

            :getter: Times in Julian days
            
            :type: :class:`~numpy.ndarray`
        """
        return self.time.jd


    @property
    def datetime(self):
        """ Return datetime dates

            :getter: Times in datetime format
            
            :type: :class:`~numpy.ndarray`
        """
        return self.time.datetime


    @property
    def amp(self):
        """ Linear amplitude of the data

            :getter: Data in linear scaling
            
            :type: :class:`~numpy.ndarray`
        """
        return self.data.squeeze()

    
    @property
    def db(self):
        """ Convert the amplitude in decibels

            :getter: Data in decibels
            
            :type: :class:`~numpy.ndarray`
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

        if self.data.shape != other.data.shape:
            raise ValueError(
                'SData objects do not have the same dimensions'
                )

        if all(self.time != other.time):
            raise ValueError(
                'Not the same times'
                )

        if all(self.freq != other.freq):
            raise ValueError(
                'Not the same frequencies'
                )

        return


    def _check_value(self, other):
        """ Checks that other can be operated with self if
            other is not a SData object
        """
        if isinstance(other, np.ndarray):
            if other.shape != self.data.shape:
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


