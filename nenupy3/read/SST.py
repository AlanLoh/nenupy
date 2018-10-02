#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read NenuFAR SST data
        by A. Loh

TO DO:
    - read the parset if present
"""

import os
import sys
import glob
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.time import Time


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['SST']


class SST():
    def __init__(self, obsfile=None):
        self.obsfile = obsfile
        self.ma      = 0
        self.freq    = 50
        self.polar   = 'nw' 

    def __str__(self):
        toprint  = '\t=== Class SST of nenupy ===\n'
        toprint += '\tList of all current attributes:\n'
        for att in dir(self):
            avoid = ['t', 'd', 'f', 'maposition', 'marotation']
            if (not att.startswith('_')) & (not any(x.isupper() for x in att)) & (att not in avoid):
                toprint += "%s: %s\n"%(att, getattr(self, att))
        return toprint

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obsfile(self):
        """ SST observation file,
            obsfile set to 'fake' will produce a custom fake SST observation
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        if o == 'fake':
            self._createFakeObs()
            self._obsfile = o
            return
        
        if (o is None) or (o == '') or (os.path.isdir(o)):
            # Look at the current/specified directory
            if os.path.isdir(o):
                _opath = os.path.abspath(o)
                sstfiles = glob.glob( os.path.join(_opath, '*SST.fits') )
            else:
                sstfiles = glob.glob('*SST.fits')
            if len(sstfiles) == 0:
                raise IOError("\t=== No SST fits file in current directory, specify the file to read. ===")
            else:
                o = [os.path.abspath(f) for f in sstfiles]  
        elif isinstance(o, list):
            # List of files in input
            if all(os.path.isfile(f) for f in o):
                if all(f.endswith('SST.fits') for f in o):
                    o = [os.path.abspath(f) for f in o]
                else:
                    raise IOError("\t=== Multiple observation files (other than SST) are not understood - yet. ===")
            else:
                raise IOError("\t=== At least one file not found. ===")
        else:
            if not os.path.isfile(o):
                raise IOError("\t=== File {} not found. ===".format(os.path.abspath(o)))
            else:
                # The file has been correctly found
                o = [ os.path.abspath(o) ]
        self._obsfile = sorted(o)

        if not self._isSST():
            raise ValueError("\t=== Files might not all be SST observaitons ===")
        else:
            self._readSST()

        return

    @property
    def ma(self):
        """ Selected Mini-Array
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        if m is None:
            m = 0
        if not isinstance(m, int):
            try:
                m = int(m)
            except:
                print("\n\t=== WARNING: Mini-Array index {} not recognized ===".format(m))
                m = 0
        if m in self.miniarrays:
                self._ma = m
        else:
            print("\n\t=== WARNING: available Mini-Arrays are {} ===".format(self.miniarrays))
            self._ma = 0
        return

    @property
    def freq(self):
        """ Frequency selection in MHz
            Example:
            self.freq = 50
            self.freq = [20, 30]
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if isinstance(f, list):
            self._freq    = [0, 0]
            self._freq[0] = min(self._freqs, key=lambda x: np.abs(x-f[0]))
            self._freq[1] = min(self._freqs, key=lambda x: np.abs(x-f[1]))
        else:
            self._freq    = min(self._freqs, key=lambda x: np.abs(x-f))  
        return        

    @property
    def polar(self):
        """ Polarization selection ('NW' or 'NE')
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, str):
            self._polar = 'NW'
        elif p.upper() not in self._pols:
            print("\n\t=== WARNING: Polarization selection {} not recognized ===".format(p))
            self._polar = 'NW'
        else:
            self._polar = p.upper()

    @property
    def time(self):
        """ Time selection
        """
        return self._time
    @time.setter
    def time(self, t):
        if isinstance(t, list):
            try:
                if not isinstance(t[0], Time):
                    t[0] = Time(t[0])
                if not isinstance(t[1], Time):
                    t[1] = Time(t[1])
                self._time = t
            except:
                print("\n\t=== WARNING: Time syntax incorrect ===")
        else:
            if not isinstance(t, Time):
                try:
                    t = Time(t)
                    self._time = t
                except:
                    print("\n\t=== WARNING: Time syntax incorrect ===")
            return

    # ------ Specific getters ------ #
    @property
    def marotation(self):
        """ Mini-array rotation in degrees
        """
        self._marotation = self._marot[ self.ma ]
        return self._marotation

    @property
    def maposition(self):
        """ Lambert93 (x, y, z) Mini-Array position
        """
        self._maposition = self._mapos[ self.ma ]
        return self._maposition
    

    # ================================================================= #
    # =========================== Methods ============================= #
    def getData(self, **kwargs):
        """ Make the data selection
            Fill the attributes self.d (data), self.t (time), self.f (frequency)
        """
        self._evalkwargs(kwargs)

        # ------ load data only once ------ #
        if not hasattr(self, '_data'):
            self._timeall = []
            self._dataall = []
            for ssfile in sorted(self.obsfile):
                with fits.open(ssfile, mode='readonly', ignore_missing_end=True, memmap=True) as fitsfile:
                   self._timeall.append( fitsfile[7].data['jd']   ) 
                   self._dataall.append( fitsfile[7].data['data'] )
            self._timeall = Time( np.hstack(self._timeall), format='jd' )
            self._dataall  = np.vstack( self._dataall )

        # ------ polarization ------ #
        mask_pol = (self._pols == self.polar)

        # ------ frequency ------ #
        if isinstance(self.freq, list):
            mask_fre = (self._freqs > 0.) & (self._freqs >= self.freq[0]) & (self._freqs <= self.freq[-1])
        else:
            mask_fre = (self._freqs == min(self._freqs, key=lambda x: np.abs(x-self.freq)))

        # ------ time ------ #
        if isinstance(self.time, list):
            mask_tim = (self._timeall >= self.time[0]) & (self._timeall <= self.time[1])
        else:
            mask_tim = (self._timeall.mjd == min(self._timeall.mjd, key=lambda x: np.abs(x-self.time.mjd)) )
        
        # ------ mini-array ------ #
        mask_ma = (self.miniarrays == self.ma)
       
        # ------ selected data ------ #
        self.d = np.squeeze( self._dataall[ np.ix_(mask_tim, mask_ma, mask_pol, mask_fre) ] )
        self.t = self._timeall[ mask_tim ]
        self.f = self._freqs[ mask_fre ]
        return

    def plotData(self, **kwargs):
        """ Plot the data
        """
        self.getData(**kwargs)

        if self.f.size == 1:
            # ------ Light curve ------ #
            xtime = (self.t - self.t[0]).sec / 60
            plt.plot(xtime, self.d)
            plt.xlabel('Time (min since {})'.format(self.t[0].iso))
            plt.ylabel('Amplitude')
            plt.title('MA={}, f={:3.2f} MHz, pol={}'.format(self.ma, self.freq, self.polar))
            plt.show()
            plt.close('all')

        elif self.t.size == 1:
            # ------ Spectrum ------ #
            plt.plot(self.f, self.d)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')
            plt.title('MA={}, t={}, pol={}'.format(self.ma, self.time.iso, self.polar))
            plt.show()
            plt.close('all')

        elif (self.f.size > 1) & (self.t.size > 1):
            # ------ Dynamic spectrum ------ #
            xtime = (self.t - self.t[0]).sec / 60
            vmin, vmax = np.percentile(self.d, [5, 99]) 
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            normcb = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            spec   = ax.pcolormesh(xtime, self.f, self.d.T, cmap='bone', norm=normcb)
            ax.axis( [xtime.min(), xtime.max(), self.f.min(), self.f.max()] )
            plt.xlabel('Time (min since {})'.format(self.t[0].iso))
            plt.ylabel('Frequency (MHz)')
            plt.title('MA={}, pol={}'.format(self.ma, self.polar))
            plt.show()
            plt.close('all')

        else:
            raise ValueError("\n\t=== ERROR: Plot nature not understood ===")
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _isSST(self):
        """ Check that self.obsfile contain only SST observations
        """
        isSST = True
        for sstfile in self.obsfile:
            with fits.open(sstfile, mode='readonly', ignore_missing_end=True, memmap=True) as f:
                if f[0].header['OBJECT'] != 'subband Statistics':
                    isSST = False
                else:
                    pass
        return isSST

    def _readSST(self):
        """ Read SST fits files and fill the class attributes
        """
        self.obsname = os.path.basename( self.obsfile[0] )[:-9]
        
        headi = fits.getheader(self.obsfile[0], ext=0, ignore_missing_end=True, memmap=True)
        headf = fits.getheader(self.obsfile[-1], ext=0, ignore_missing_end=True, memmap=True)
        self.obstart  = Time( headi['DATE-OBS'] + 'T' + headi['TIME-OBS'] )
        self.obstop   = Time( headf['DATE-END'] + 'T' + headf['TIME-END'] )
        self.time     = [self.obstart.copy(), self.obstop.copy()]
        self.exposure = self.obstop - self.obstart

        setup_ins = fits.getdata(self.obsfile[0], ext=1, ignore_missing_end=True, memmap=True)
        self._freqs     = np.squeeze( setup_ins['frq'] )
        self.freqmin    = self._freqs.min()
        self.freqmax    = self._freqs.max()
        self.miniarrays = np.squeeze( setup_ins['noMROn'] )
        self._marot     = np.squeeze( setup_ins['rotation'] )
        self._mapos     = np.squeeze( setup_ins['noPosition'] )
        self._mapos     = self._mapos.reshape( int(self._mapos.size/3), 3 )
        self._pols      = np.squeeze( setup_ins['spol'] )
        return

    def _createFakeObs(self):
        """
        """
        return

    def _evalkwargs(self, kwargs):
        for key, value in kwargs.items():
            if   key == 'polar': self.polar = value
            elif key == 'freq':  self.freq  = value
            elif key == 'time':  self.time  = value
            elif key == 'ma':    self.ma    = value
            else:
                pass
        return

