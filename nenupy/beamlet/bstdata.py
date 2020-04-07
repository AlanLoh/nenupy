#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST Data
    ********
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

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- BST_Data -------------------------- #
# ============================================================= #
class BST_Data(Beamlet):
    """
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
        """ NenuFAR BST file
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
        """ Analog beam indices
        """
        return np.arange(self.meta['obs']['nbAnaBeam'][0])


    @property
    def dbeams(self):
        """ Digital beam indices
        """
        return np.arange(self.meta['obs']['nbBeam'][0])


    @property
    def mas(self):
        """ Mini-Array list used for a given analog beam
            CAN HAVE DIFFERENT MA FOR A GIVEN DIG BEAM
        """
        infos = self.meta['ana']
        nma = infos['nbMRUsed'][self.abeam]
        return infos['MRList'][self.abeam][:nma]


    @property
    def marot(self):
        """ Mini-Array rotations
        """
        return self.meta['ins']['rotation'][0][self.mas] * u.deg


    @property
    def ants(self):
        """ Antenna list used for a given analog beam
        """
        infos = self.meta['ana']
        astr = infos['AntList'][self.abeam]
        return np.array(astr.strip('[]').split(',')).astype(int)


    @property
    def beamlets(self):
        """ Beamlet list used for a given digital beam
        """
        infos = self.meta['bea']
        nbm = infos['nbBeamlet'][self.abeam]
        return infos['BeamletList'][self.dbeam][:nbm]


    @property
    def freqs(self):
        """ Frequency list used for a given digital beam
            This MAY be start of channel rather than mid-freq
        """
        infos = self.meta['bea']
        nbm = infos['nbBeamlet'][self.abeam]
        return infos['freqList'][self.dbeam][:nbm] * u.MHz


    @property
    def freq(self):
        """ Current frequency selection
        """
        if hasattr(self, '_freq_idx'):
            mask = np.isin(self.beamlets, self._freq_idx)
            return self.freqs[mask]
        else:
            return self.freqs
    

    @property
    def time(self):
        """ Current time selection
        """
        if hasattr(self, '_time_idx'):
            return self.times[self._time_idx]
        else:
            return self.times


    @property
    def azana(self):
        """ Pointed Azimuth for a given analog beam
        """
        info = self.meta['pan'][:-1]
        mask = info['noAnaBeam'] == self.abeam
        return info['AZ'][mask] * u.deg


    @property
    def elana(self):
        """ Pointed Elevation for a given analog beam
        """
        info = self.meta['pan'][:-1]
        mask = info['noAnaBeam'] == self.abeam
        return info['EL'][mask] * u.deg


    @property
    def utana(self):
        """ Time range of analog pointings
            Shape: (n_pointings, 2)
            2 = (start, stop)
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
        """ Pointed Azimuth for a given digital beam
        """
        info = self.meta['pbe']
        mask = info['noBeam'] == self.dbeam
        return info['AZ'][mask] * u.deg


    @property
    def eldig(self):
        """ Pointed Elevation for a given digital beam
        """
        info = self.meta['pbe']
        mask = info['noBeam'] == self.dbeam
        return info['EL'][mask] * u.deg


    @property
    def utdig(self):
        """ Time range of digital pointings
            Shape: (n_pointings, 2)
            2 = (start, stop)
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
        """
        """
        self._fill_attr(kwargs)
        data = self.data[
            np.ix_(
                self._time_idx,
                self._polar_idx,
                self._freq_idx
            )
        ]
        return data


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

