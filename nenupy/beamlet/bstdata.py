#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST_Data
    ********

    Object class :class:`.BST_Data` (from the :mod:`~nenupy.beamlet` module) 
    inherits from :class:`.Beamlet`.
    It is designed to read the so called *Beamformed Statistics Data* 
    or `BST <https://nenufar.obs-nancay.fr/en/astronomer/>`_ from the
    `NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_
    low-frequency radio-telescope.

    .. note::
        To read high-rate time-frequency data from the *UnDySPuTeD*
        NenuFAR backend, see `nenupytf <https://nenupytf.readthedocs.io/en/latest/>`_
        package.

    **Initialization**
    
    The :class:`.BST_Data` object needs to be provided with a path to
    a BST observation file (in `FITS <https://fits.gsfc.nasa.gov/>`_ format):

    >>> from nenupy.beamlet import BST_Data
    >>> bst = BST_Data(
            bstfile='/path/to/BST.fits'
        )

    **Data selection**
    
    Data selection is enabled thanks to dedicated keywords
    that set :class:`.BST_Data` attributes. They can be set
    directly or passed as keyword arguments to
    :func:`BST_Data.select`:

    * :attr:`BST_Data.abeam`: Analog beam index selection.
    * :attr:`BST_Data.dbeam`: Digital beam index selection.
    * :attr:`BST_Data.polar`: Polarization selection.
    * :attr:`BST_Data.timerange`: Time range selection.
    * :attr:`BST_Data.freqrange`: Frequency range selection.

    >>> from nenupy.beamlet import BST_Data
    >>> bst = BST_Data(
            bstfile='/path/to/BST.fits'
        )
    >>> data = bst.select(
            timerange=['2020-01-01 12:00:00', '2020-01-01 13:00:00'],
            freqrange=[55, 65],
            polar='NW',
            dbeam=0 
        ) 

    **Observation and current selection informations**

    Once initialized, a :class:`.BST_Data` contains many
    informations regarding the observation accessible via specific
    getters such as:

    * :attr:`BST_Data.abeams`: Available analog beal indices
    * :attr:`BST_Data.dbeams`: Available digital beam indices
    * :attr:`BST_Data.mas`: Mini-Arrays per per :attr:`BST_Data.abeam`
    * :attr:`BST_Data.marot`: Mini-Array rotations per :attr:`BST_Data.abeam`
    * :attr:`BST_Data.ants`: Antennas used per :attr:`BST_Data.abeam`
    * :attr:`BST_Data.beamlets`: Beamlet indices per :attr:`BST_Data.dbeam`
    * :attr:`BST_Data.freqs`: Available frequencies per :attr:`BST_Data.dbeam`
    * :attr:`BST_Data.freq`: Frequency selection
    * :attr:`BST_Data.time`: Time selection
    * :attr:`BST_Data.azana`: Azimuth pointed per :attr:`BST_Data.abeam`
    * :attr:`BST_Data.elana`: Elevation pointed per :attr:`BST_Data.abeam`
    * :attr:`BST_Data.utana`: Time range of analog pointings.
    * :attr:`BST_Data.azdig`: Azimuth pointed per :attr:`BST_Data.dbeam`
    * :attr:`BST_Data.eldig`: Elevation pointed per :attr:`BST_Data.dbeam`
    * :attr:`BST_Data.utdig`: Time range of digital pointings.

    Other informations linked to the base class 
    :class:`nenupy.beamlet.beamlet.Beamlet` can also be accessed.
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'BST_Data'
]


import numpy as np
from os.path import abspath, isfile
from astropy.time import Time
import astropy.units as u

from nenupy.beamlet import Beamlet
from nenupy.beamlet import SData

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- BST_Data -------------------------- #
# ============================================================= #
class BST_Data(Beamlet):
    """ Class to read *NenuFAR* BST data stored as FITS files.

        :param bstfile: Path to BST file.
        :type bstfile: str
    """

    def __init__(self, bstfile):
        super().__init__(
        )
        self.bstfile = bstfile

        # Selection attributes
        self.dbeam = 0
        self.polar = 'NW'
        self.timerange = [self.t_min, self.t_max]
        self.freqrange = [self.f_min, self.f_max]


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def bstfile(self):
        """ Path toward BST FITS file.
            
            :setter: BST file
            
            :getter: BST file
            
            :type: `str`

            :Example:
            
            >>> from nenupy.beamlet import BST_Data
            >>> bst = BST_Data(
                    bstfile='/path/to/BST.fits'
                )
            >>> bst.bstfile
            '/path/to/BST.fits'
        """
        return self._bstfile
    @bstfile.setter
    def bstfile(self, b):
        if not isinstance(b, str):
            raise TypeError(
                'String expected.'
                )
        b = abspath(b)
        if not isfile(b):
            raise FileNotFoundError(
                'Unable to find {}'.format(b)
                )
        self._bstfile = b
        self._load(self._bstfile)
        return


    @property
    def abeams(self):
        """ Analog beam indices recorded during the observation.
            Each analog beam defines a particular set of
            Mini-Arrays as well as a given pointing direction.
            
            :getter: Analog beam indices available
            
            :type: :class:`numpy.ndarray`
        """
        return np.arange(self.meta['obs']['nbAnaBeam'][0])


    @property
    def dbeams(self):
        """ Digital beam indices recorded during the observation.
            Each digital beam corresponds to a given pointing
            direction. They are also linked to corresponding
            :attr:`BST_Data.abeams`.
            
            :getter: Digital beam indices available
            
            :type: :class:`numpy.ndarray`
        """
        return np.arange(self.meta['obs']['nbBeam'][0])


    @property
    def mas(self):
        """ Mini-Arrays that have been used for a given analog
            beam (set with :attr:`BST_Data.abeam`).

            :getter: Mini-Arrays selected
            
            :type: :class:`numpy.ndarray`
        """
        infos = self.meta['ana']
        nma = infos['nbMRUsed'][self.abeam]
        return infos['MRList'][self.abeam][:nma]


    @property
    def marot(self):
        """ Rotations corresponding to Mini-Arrays listed in
            :attr:`BST_Data.mas`.

            :getter: Mini-Arrays rotations
            
            :type: :class:`astropy.units.Quantity`
        """
        return self.meta['ins']['rotation'][0][self.mas] * u.deg


    @property
    def ants(self):
        """ Antenna used within a Mini-Array. This is analog beam
            dependant (:attr:`BST_Data.abeam`).

            :getter: Antenna selected
            
            :type: :class:`numpy.ndarray`
        """
        infos = self.meta['ana']
        astr = infos['AntList'][self.abeam]
        return np.array(astr.strip('[]').split(',')).astype(int)


    @property
    def beamlets(self):
        """ Beamlet indices associated with a given digital beam
            (set with :attr:`BST_Data.dbeam`).

            :getter: Beamlet indices
            
            :type: :class:`numpy.ndarray`
        """
        infos = self.meta['bea']
        nbm = infos['nbBeamlet'][self.dbeam]
        return infos['BeamletList'][self.dbeam][:nbm]


    @property
    def freqs(self):
        """ Recorded frequencies for a given digital beam (set
            with :attr:`BST_Data.dbeam`). Converted to mid
            sub-band frequencies

            :getter: Available frequencies
            
            :type: :class:`astropy.units.Quantity`
        """
        infos = self.meta['bea']
        nbm = infos['nbBeamlet'][self.dbeam]
        hbw = 0.1953125/2
        return (infos['freqList'][self.dbeam][:nbm] - hbw)*u.MHz


    @property
    def freq(self):
        """ Current frequency selection made from setting
            :attr:`BST_Data.freqrange`.

            :getter: Frequency selection
            
            :type: :class:`astropy.units.Quantity`
        """
        if hasattr(self, '_freq_idx'):
            mask = np.isin(self.beamlets, self._freq_idx)
            return self.freqs[mask]
        else:
            return self.freqs
    

    @property
    def time(self):
        """ Current time selection made from setting
            :attr:`BST_Data.timerange`.

            :getter: Time selection
            
            :type: :class:`astropy.time.Time`
        """
        if hasattr(self, '_time_idx'):
            return self.times[self._time_idx]
        else:
            return self.times


    @property
    def azana(self):
        """ Pointed Azimuth for a given analog beam.
            
            :getter: Azimuth
            
            :type: :class:`astropy.units.Quantity`
        """
        info = self.meta['pan'][:-1]
        mask = info['noAnaBeam'] == self.abeam
        return info['AZ'][mask] * u.deg


    @property
    def elana(self):
        """ Pointed Elevation for a given analog beam.

            :getter: Elevation
            
            :type: :class:`astropy.units.Quantity`
        """
        info = self.meta['pan'][:-1]
        mask = info['noAnaBeam'] == self.abeam
        return info['EL'][mask] * u.deg


    @property
    def utana(self):
        """ This returns the time range of all pointings
            associated with a given :attr:`BST_Data.abeam`.
            The shape of the output array is:

            [[pointing_1_start, pointing_1_stop],
            [pointing_2_start, pointing_2_stop],
            ...]

            :getter: Time range of analog pointings.

            :type: :class:`astropy.time.Time`
        """
        lasttime = self.meta['pan']['timestamp'][-1]
        info = self.meta['pan'][:-1]
        mask = info['noAnaBeam'] == self.abeam
        ti = info['timestamp'][mask]
        tf = np.concatenate((ti[1:], [lasttime]))
        trange = np.vstack((ti, tf)).T
        return Time(trange)


    @property
    def azdig(self):
        """ Pointed Azimuth for a given digital beam.

            :getter: Azimuth
            
            :type: :class:`astropy.units.Quantity`
        """
        info = self.meta['pbe']
        mask = info['noBeam'] == self.dbeam
        return info['AZ'][mask] * u.deg


    @property
    def eldig(self):
        """ Pointed Elevation for a given digital beam.

            :getter: Elevation
            
            :type: :class:`astropy.units.Quantity`
        """
        info = self.meta['pbe']
        mask = info['noBeam'] == self.dbeam
        return info['EL'][mask] * u.deg


    @property
    def utdig(self):
        """ This returns the time range of all pointings
            associated with a given :attr:`BST_Data.dbeam`.
            The shape of the output array is:

            [[pointing_1_start, pointing_1_stop],
            [pointing_2_start, pointing_2_stop],
            ...]

            :getter: Time range of digital pointings.

            :type: :class:`astropy.time.Time`
        """
        lasttime = self.meta['pan']['timestamp'][-1]
        info = self.meta['pbe']
        mask = info['noBeam'] == self.dbeam
        ti = info['timestamp'][mask]
        tf = np.concatenate((ti[1:], [lasttime]))
        trange = np.vstack((ti, tf)).T
        return Time(trange)


    # Selection
    @property
    def abeam(self):
        """ Analog beam index selected for the available
            beams (see :attr:`BST_Data.abeams`).
            
            :setter: Analog beam index
            
            :getter: Analog beam index
            
            :type: `int`
        """
        return self._abeam
    @abeam.setter
    def abeam(self, a):
        if not np.issubdtype(type(a), np.integer):
            raise TypeError(
                'abeam index should be given as an integer'
            )
        if not a in self.abeams:
            raise IndexError(
                'abeam index not in abeams list {}'.format(
                    self.abeams
                )
            )
        self._abeam = a
        log.info(
            'AnaBeam {} selected'.format(a)
        )
        return


    @property
    def dbeam(self):
        """ Digital beam index selected for the available
            beams (see :attr:`BST_Data.dbeams`).
            
            :setter: Digital beam index
            
            :getter: Digital beam index
            
            :type: `int`
        """
        return self._dbeam
    @dbeam.setter
    def dbeam(self, d):
        if not np.issubdtype(type(d), np.integer):
            raise TypeError(
                'dbeam index should be given as an integer'
            )
        if not d in self.dbeams:
            raise IndexError(
                'dbeam index not in dbeams list {}'.format(
                    self.dbeams
                )
            )
        self._dbeam = d
        log.info(
            'DigiBeam {} selected'.format(d)
        )
        self.abeam = self.meta['bea']['NoAnaBeam'][d]
        return


    @property
    def polar(self):
        """ Polarization selection (between `'NW'` and `'NE'`).
            
            :setter: Polarization
            
            :getter: Polarization
            
            :type: `str`
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, str):
            raise TypeError(
                'polar should be a string'
            )
        p = p.upper()
        pols = self.meta['ins']['spol'][0]
        if p not in pols:
            raise ValueError(
                'polar should be in {}'.format(pols)
            )
        self._polar_idx = np.where(pols == p)[0]
        self._polar = p
        return


    @property
    def timerange(self):
        """ Time selection.

            :setter: Time range
            
            :getter: Time range
            
            :type: `str`, `list`, :class:`numpy.ndarray` or :class:`astropy.time.Time`

            :Example:
            
            Set time to the closest available from a given ISOT value:

            >>> bst.timerange = '2019-11-29T15:00:00'
            >>> bst.time                                                                                                         
            <Time object: scale='utc' format='jd' value=[2458817.125]>
            
            Select times falling between two values:
            
            >>> bst.timerange = ['2019-11-29T15:00:00', '2019-11-29T16:00:00']
            >>> bst.time
            <Time object: scale='utc' format='jd' value=[2458817.125 ... 2458817.16666667]>

            or, using :class:`astropy.time.Time`:

            >>> from astropy.time import Time
            >>> t0 = Time('2019-11-29T15:00:00')
            >>> t1 = Time('2019-11-29T16:00:00')
            >>> bst.timerange = [t0, t1]
            >>> bst.time
            <Time object: scale='utc' format='jd' value=[2458817.125 ... 2458817.16666667]>

            or:
            
            >>> from astropy.time import Time, TimeDelta
            >>> t0 = Time('2019-11-29T15:00:00')
            >>> t1 = t0 + TimeDelta(3600, format='sec')
            >>> bst.timerange = [t0, t1]
            >>> bst.time
            <Time object: scale='utc' format='jd' value=[2458817.125 ... 2458817.16666667]>
        """
        return self._timerange
    @timerange.setter
    def timerange(self, t):
        if not isinstance(t, Time):
            t = Time(t)
        if t.isscalar:
            dt_sec = (self.times - t).sec
            idx = [np.argmin(np.abs(dt_sec))]
        else:
            if len(t) != 2:
                raise ValueError(
                    'timerange should be of size 2'
                )
            idx = (self.times >= t[0]) & (self.times <= t[1])
            if not any(idx):
                log.warning(
                    (
                        'Empty time selection, time should fall '
                        'between {} and {}'.format(
                            self.t_min.isot,
                            self.t_max.isot
                        )
                    )
                )
        self._timerange = t
        self._time_idx = idx
        return


    @property
    def freqrange(self):
        """ Frequency selection.

            :setter: Frequency range
            
            :getter: Frequency range
            
            :type: `float`, `list`, :class:`numpy.ndarray` or :class:`astropy.units.Quantity`

            :Example:
            
            Set frequency to the closest available from 55 MHz:
            
            >>> bst.freqrange = 55
            >>> bst.freq
            <Quantity [54.98047] MHz>

            Select frequencies falling between two values:
            (assumed to be defined in MHz):
            
            >>> bst.freqrange = [55, 56]
            >>> bst.freq
            <Quantity [55.17578 , 55.371094, 55.566406, 55.76172 , 55.95703 ] MHz>

            or, using :class:`astropy.units.Quantity`: 

            >>> import astropy.units as u
            >>> import numpy as np
            >>> bst.freqrange = np.array([55e6, 56e6]) * u.Hz
            >>> bst.freq
            <Quantity [55.17578 , 55.371094, 55.566406, 55.76172 , 55.95703 ] MHz>
        """
        return self._freqrange
    @freqrange.setter
    def freqrange(self, f):
        if not isinstance(f, u.Quantity):
            f *= u.MHz
        else:
            f.to(u.MHz)
        if f.isscalar:
            idx = [np.argmin(np.abs(self.freqs - f))]
        else:
            if len(f) != 2:
                raise ValueError(
                    'freqrange should be of size 2'
                )
            idx = (self.freqs >= f[0]) & (self.freqs <= f[1])
            if not any(idx):
                log.warning(
                    (
                        'Empty freq selection, freq should fall '
                        'between {} and {}'.format(
                            self.f_min,
                            self.f_max
                        )
                    )
                )
        self._freqrange = f
        self._freq_idx = self.beamlets[idx]
        return
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def select(self, **kwargs):
        """ Select data from the BST observation file.

            :param freqrange:
                Frequency range (see :attr:`BST_Data.freqrange`).
            :type freqrange: `float`, `list`, :class:`numpy.ndarray`
                or :class:`astropy.units.Quantity`, optional
            :param timerange:
                Time range (see :attr:`BST_Data.timerange`).
            :type timerange: `str`, `list`, :class:`numpy.ndarray` or :class:`astropy.time.Time`
            :param dbeam:
                Digital beam index (see :attr:`BST_Data.dbeam`)
            :type dbeam: `int`, optional
            :param polar:
                Polarization (see :attr:`BST_Data.polar`)
            :type polar: `str`, optional

            :returns: Selected data
            :rtype: :class:`nenupy.beamlet.sdata.SData`
        """
        self._fill_attr(kwargs)
        data = self.data[
            np.ix_(
                self._time_idx,
                self._polar_idx,
                self._freq_idx
            )
        ]
        return SData(
            data=np.swapaxes(data, 1, 2),
            time=self.time,
            freq=self.freq,
            polar=self.polar
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _fill_attr(self, kwargs):
        """
        """
        keys = kwargs.keys()
        attributes = [
            'dbeam',
            'freqrange',
            'timerange',
            'polar'
        ]
        for key in keys:
            if key not in attributes:
                log.warning(
                    '{} not a valid attribute ({})'.format(
                        key,
                        attributes
                    )
                )
        for key in attributes:
            if hasattr(self, key) and (key not in keys):
                continue
            setattr(self, key, kwargs[key])
        return

# ============================================================= #

