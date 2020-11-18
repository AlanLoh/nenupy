#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    SST Data
    ********

    Data reading
    ------------

    Single file
    ^^^^^^^^^^^^

    .. code-block:: python

        >>> from nenupy.beamlet import SST_Data
        >>> sst = SST_Data(filename='/path/to/XXX_SST.fits')


    .. code-block:: python

        >>> from nenupy.beamlet import SST_Data
        >>> sst = SST_Data(
                filename='/path/to/XXX_SST.fits',
                altazA='/path/to/XXX.altazA'
            )

    If the :attr:`~nenupy.beamlet.sstdata.SST_Data.altazA` attribute
    is filled with a valid ``'*.altazA'`` file, it returns an instance
    of :class:`~nenupy.observation.pointing.AnalogPointing` that
    is appropriate to handle the analog pointing(s) of the Mini-Arrays
    during the analyzed observation.

    Printing an :class:`~nenupy.beamlet.sstdata.SST_Data` instance
    allows to quickly display the associated files and main
    observation properties:

    .. code-block:: python

        >>> print(sst)
                    SST_Data instance
        SST File name(s):
            * /path/to/XXX_SST.fits
        altazA File name(s):
            * /path/to/XXX.altazA
        Time Range: 2020-02-16T08:00:00 -- 2020-02-16T10:00:00
        Frequency Range: 0.09765625 -- 99.90234375 MHz
        Mini-Arrays: array([ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
               52, 53, 54, 55], dtype=int16)

    List of SST files
    ^^^^^^^^^^^^^^^^^

    Combination of SST Observations
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    SST FITS files are most often split to only cover one or two hours
    of observations. In this regard, :class:`~nenupy.beamlet.sstdata.SST_Data`
    objects have been designed to allow for time concatenantion between
    several :class:`~nenupy.beamlet.sstdata.SST_Data` instances. The
    arithmetic operator *Addition* is used to fill this role.

    .. code-block:: python

        >>> from nenupy.beamlet import SST_Data
        
        >>> sst1 = SST_Data('/path/to/XXX_1_SST.fits')
        >>> sst2 = SST_Data('/path/to/XXX_2_SST.fits')
        
        >>> print(sst1.tMin, sst1.tMax)
        2017-04-26T00:00:00 2017-04-26T00:59:59
        
        >>> print(sst2.tMin, sst2.tMax)
        2017-04-26T01:00:00 2017-04-26T01:08:14

        >>> # Combine SST_Data instances with `+` operator
        >>> combined_sst = sst1 + sst2
        >>> print(combined_sst.tMin, combined_sst.tMax)
        2017-04-26T00:00:00 2017-04-26T01:08:14

        >>> # Combine SST_Data instances with `sum` (leads to the same result)
        >>> combined_sst = sum([sst1, sst2])
        >>> print(combined_sst.tMin, combined_sst.tMax)
        2017-04-26T00:00:00 2017-04-26T01:08:14

    The same idea governs the exclusion of one particular SST file. If
    an :class:`~nenupy.beamlet.sstdata.SST_Data` is composed of multiple
    observation files, the *Subtraction* operator allows to remove those
    associated with another :class:`~nenupy.beamlet.sstdata.SST_Data`
    instance. Considering the previous result stored in ``combined_sst``,
    which is the combination of two SST files, removing the first one
    is simply done by:

    .. code-block:: python

        >>> smaller_sst = combined_sst - sst1
        >>> print(smaller_sst.tMin, smaller_sst.tMax)
        2017-04-26T01:00:00 2017-04-26T01:08:14

    ``smaller_sst`` is therefore strictly identical to ``sst2`` in this example:

    .. code-block:: python

        >>> smaller_sst == sst2
        True

    Data analysis
    -------------
    
    Observation properties
    ^^^^^^^^^^^^^^^^^^^^^^

    Data selection
    ^^^^^^^^^^^^^^

    Plotting
    ^^^^^^^^

    SST_Data Reference
    ------------------

    Methods
    ^^^^^^^

    .. list-table:: Title
       :widths: 25 25 50
       :header-rows: 1

       * - Heading row 1, column 1
         - Heading row 1, column 2
         - Heading row 1, column 3
       * - Row 1, column 1
         -
         - Row 1, column 3
       * - Row 2, column 1
         - Row 2, column 2
         - Row 2, column 3


    Attributes
    ^^^^^^^^^^


"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'SST_Data'
]


from os.path import isfile, basename
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy.base import MiniArrays
from nenupy.beamlet import SData
from nenupy.observation import AnalogPointing

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- SST_Data -------------------------- #
# ============================================================= #
class SST_Data(object):
    """ Class to read *NenuFAR* SST data stored as FITS files.

        :param sstfile: Path to SST file.
        :type sstfile: `str` or `list`
        :param altazA: Path to ``*.altazA`` file, which describe
            the analogical pointing.
        :type altazA: `str` or `list`

        *Addition* and *Subtraction* are allowed between
        :class:`~nenupy.beamlet.sstdata.SST_Data` instances.

        :Example:
            >>> from nenupy.beamlet import SST_Data
            >>> sst1 = SST_Data('./File1_SST.fits')
            >>> sst2 = SST_Data('./File1_SST.fits')
            >>> newSST = sst1 + sst2

        .. versionadded:: 1.1.0

    """

    def __init__(self, filename, altazA=None, **kwargs):
        self._autoUpdate = kwargs.get('autoUpdate', True)
        self.obsProperties = {}
        self.filename = filename
        self.altazA = altazA
        self._freqRange = None
        self._timeRange = None
        self._ma = None
        self._polar = None


    def __str__(self):
        sstDescription = '\tSST_Data instance\n'
        sstDescription += 'SST File name(s):\n'
        for file in self.filename:
            sstDescription += '\t* {}\n'.format(file)
        sstDescription += 'altazA File name(s):\n'
        if self.altazA == []:
            sstDescription += '\t* None\n'
        else:
            for file in self.altazA.filename:
                sstDescription += '\t* {}\n'.format(file)
        sstDescription += 'Time Range: {} -- {}\n'.format(
            self.tMin.isot,
            self.tMax.isot
        )
        sstDescription += 'Frequency Range: {} -- {} MHz\n'.format(
            self.fMin.to(u.MHz).value,
            self.fMax.to(u.MHz).value
        )
        sstDescription += 'Mini-Arrays: {}'.format(
            self.mas
        )
        return sstDescription


    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('{} exepected.'.format(self.__class__))
        new = SST_Data(
            filename=self.filename + other.filename,
            autoUpdate=False
        )
        new.obsProperties = {**self.obsProperties, **other.obsProperties}
        new.altazA = self.altazA + other.altazA
        return new


    def __radd__(self, other):
        if other==0:
            return self
        else:
            return self.__add__(other)


    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('{} exepected.'.format(self.__class__))
        newFileList = self.filename.copy()
        newObsProperties = self.obsProperties.copy()
        for fileToRemove in list(map(basename, other.filename)):
            selfBaseNames = list(map(basename, newFileList))
            try:
                fileIndex = selfBaseNames.index(fileToRemove)
                removedFile = newFileList.pop(fileIndex)
            except ValueError:
                # SST file in other not in self
                pass
            try:
                fileKeys = list(newObsProperties.keys())
                baseKeys = list(map(basename, fileKeys))
                fileIndex = baseKeys.index(fileToRemove)
                del newObsProperties[fileKeys[fileIndex]]
            except (ValueError, KeyError):
                # SST file in other not in self
                pass
        new = SST_Data(
            filename=newFileList,
            autoUpdate=False
        )
        new.obsProperties = newObsProperties
        if (self.altazA==[]) or (other.altazA==[]):
            new.altazA = self.altazA
        else:
            new.altazA = self.altazA - other.altazA
        return new


    def __eq__(self, other):
        areEqual = False
        # SST FITS file comparison
        baseNames = list(map(basename, self.filename))
        sameFiles = all(
            [basename(fi) in baseNames for fi in other.filename]
        )
        # .altazA file comparison
        if (self.altazA==[]) and (other.altazA==[]):
            sameAltazA = True
        elif isinstance(self.altazA, AnalogPointing) and isinstance(other.altazA, AnalogPointing):
            selfBaseNames = list(map(basename, self.altazA.filename))
            otherBaseNames = list(map(basename, other.altazA.filename))
            sameAltazA = all(
                [fi in selfBaseNames for fi in otherBaseNames]
            )
        else:
            sameAltazA = False
        # Final comparison
        if sameFiles and sameAltazA:
            areEqual = True
        return areEqual


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def filename(self):
        """
            :setter: 

            :getter:

            :type: `list`
        """
        return sorted(self._filename)
    @filename.setter
    def filename(self, f):
        if not isinstance(f, list):
            f = [f]
        for fi in f:
            if not isfile(fi):
                raise FileNotFoundError(
                    'Unable to find {}'.format(fi)
                )
            self._fillObsDict(fi)
        self._filename = f


    @property
    def altazA(self):
        """
            :setter: 

            :getter:

            :type: `list`
        """
        return self._altazA
    @altazA.setter
    def altazA(self, a):
        if (a is None) or (a==[]):
            self._altazA = []
        else:
            self._altazA = AnalogPointing(filename=a)


    @property
    def timeRange(self):
        """
            :setter: 

            :getter:

            :type: `list` of :class:`~astropy.time.Time`
        """
        if self._timeRange is None:
            self._timeRange = Time([self.tMin, self.tMax])
        return self._timeRange
    @timeRange.setter
    def timeRange(self, tr):
        if tr is None:
            tr = [self.tMin, self.tMax]
        self._timeRange = self._intervalBoundaries(
            tr,
            Time,
            precision=0
        )


    @property
    def freqRange(self):
        """
            :setter: 

            :getter:

            :type: `list` of :class:`~astropy.units.Quantity`
        """
        if self._freqRange is None:
            self._freqRange = [self.fMin, self.fMax]
        return self._freqRange
    @freqRange.setter
    def freqRange(self, fr):
        if fr is None:
            fr = [self.fMin, self.fMax]
        self._freqRange = self._intervalBoundaries(
            fr,
            u.Quantity,
            unit='MHz'
        )


    @property
    def polar(self):
        """
            :setter: 

            :getter:

            :type: `str`
        """
        if self._polar is None:
            self._polar = self._getPropList('polars')[0][0]
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, str):
            raise TypeError(
                '`polar` should be a string.'
            )
        p = p.upper()
        if not all([p in ps for ps in self._getPropList('polars')]):
            raise IndexError(
                'Polarization `{}` unknown.'.format(p)
            )
        self._polar = p


    @property
    def ma(self):
        """
            :setter: 

            :getter:

            :type: `int`
        """
        if self._ma is None:
            self._ma = self._getPropList('MAs')[0].names[0]
        return self._ma
    @ma.setter
    def ma(self, m):
        if not isinstance(m, int):
            raise TypeError(
                '`ma` should be an integer.'
            )
        if not all([m in mas for mas in self._getPropList('MAs')]):
            raise IndexError(
                'Mini-Array `{}` not used in current observation.'.format(m)
            )
        self._ma = m


    @property
    def tMin(self):
        """
            :getter:

            :type: :class:`~astropy.time.Time`
        """
        return min(self._getPropList('tMin'))


    @property
    def tMax(self):
        """
            :getter:

            :type: :class:`~astropy.time.Time`
        """
        return max(self._getPropList('tMax'))


    @property
    def fMin(self):
        """
            :getter:

            :type: :class:`~astropy.units.Quantity`
        """
        return min(self._getPropList('fMin'))


    @property
    def fMax(self):
        """
            :getter:

            :type: :class:`~astropy.units.Quantity`
        """
        return max(self._getPropList('fMax'))


    @property
    def mas(self):
        """
            :getter:

            :type: :class:`~nenupy.base.MiniArrays`
        """
        return self._getPropList('MAs')[0]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def select(self, **kwargs):
        """
            :param freqRange:
            :type freqRange:
            :param timeRange:
            :type timeRange:
            :param ma:
            :type ma:
            :param polar:
            :type polar:

            :returns:
            :rtype: `~nenupy.beamlet.sdata.SData`
        """
        allowedSelection = [
            'freqRange',
            'timeRange',
            'ma',
            'polar'
        ]
        for key, val in kwargs.items():
            if key in allowedSelection:
                setattr(self, key, val)
        for key in allowedSelection:
            log.info(
                'Selection on {}={}'.format(
                    key,
                    getattr(self, key)
                )
            )

        sdata = None

        for sstfile in self.filename:

            if (self.timeRange[0] > self.obsProperties[sstfile]['tMax'])\
                or (self.timeRange[1] < self.obsProperties[sstfile]['tMin'])\
                or (self.freqRange[0] > self.obsProperties[sstfile]['fMax'])\
                or (self.freqRange[1] > self.obsProperties[sstfile]['fMax']):
                # File does not satisfy selection
                continue
            
            # Frequency selection
            freqs = self.obsProperties[sstfile]['freqs']
            if self.freqRange[0] == self.freqRange[1]:
                # Selection on closest value
                fMask = np.zeros(freqs.size, dtype=bool)
                fMask[np.argmin(np.abs(freqs - self.freqRange[0]))] = True
            else:
                # Selection based on the boundary values
                fMask = (freqs >= self.freqRange[0])\
                    & (freqs <= self.freqRange[1])

            # Time selection
            times = fits.getdata(sstfile, ext=7, memmap=True)['jd']
            if self.timeRange[0] == self.timeRange[1]:
                # Selection on closest value
                tMask = np.zeros(times.size, dtype=bool)
                tMask[np.argmin(np.abs(times - self.timeRange[0].jd))] = True
            else:
                # Selection based on the boundary values
                tMask = (times >= self.timeRange[0].jd)\
                    & (times <= self.timeRange[1].jd)
            
            # Polarization selection
            pMask = self.obsProperties[sstfile]['polars'] == self.polar
            
            data = fits.getdata(sstfile, ext=7, memmap=True)['data']
            sdataTemp = SData(
                data=np.swapaxes(data, 1, 3)[
                    np.ix_(tMask, fMask, pMask)
                ][..., self.ma], # MA selection
                time=Time(times[tMask], format='jd', precision=0),
                freq=freqs[fMask],
                polar=self.obsProperties[sstfile]['polars'][pMask]
            )

            if sdata is None:
                # Initialize the SData instance
                sdata = sdataTemp
            else:
                # Concatenate the data
                sdata = sdata | sdataTemp

        if sdata is None:
            log.warning(
                'Empty selection.'
            )
        return sdata


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _fillObsDict(self, filename):
        """
        """
        if not self._autoUpdate:
            return
        header = fits.getheader(filename)
        ins = fits.getdata(filename, ext=0, header=None)
        self.obsProperties[filename] = {
            'tMin': Time(header['DATE-OBS'] + 'T' + header['TIME-OBS'], precision=0),
            'tMax': Time(header['DATE-END'] + 'T' + header['TIME-END'], precision=0),
            'fMin': ins['frq'].min()*u.MHz,
            'fMax': ins['frq'].max()*u.MHz,
            'freqs': ins['frq'][0]*u.MHz,
            'polars': ins['spol'][0],
            'MAs': MiniArrays(ins['noMROn'][0])
        }


    def _getPropList(self, prop):
        """
        """
        obsFiles = self.obsProperties.keys()
        return [self.obsProperties[f][prop] for f in obsFiles]


    @staticmethod
    def _intervalBoundaries(bd, typ, **kwargs):
        """ For frequency or time
        """
        if typ is u.Quantity:
            bd = typ(bd, ndmin=1, **kwargs)
            bd = np.sort(bd)
        elif typ is Time:
            bd = typ(bd, **kwargs)
            if bd.ndim == 0:
                bd = typ([bd])
            bd = bd.sort()
        else:
            raise TypeError(
                '`_intervalBoundaries` only works with Time and Quantity objects.'
            )
        
        if bd.ndim > 1:
            raise IndexError(
                'Only dimension-1 arrays are understood.'
            )
        if bd.size > 2:
            raise IndexError(
                'Only length-2 arrays are allowed.'
            )
        elif bd.size == 1:
            bd = bd.insert(0, bd[0])
        
        return bd
# ============================================================= #

