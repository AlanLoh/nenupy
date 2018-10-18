#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read BST NenuFAR data
        by A. Loh
"""

import os
import glob
import sys
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.time import Time, TimeDelta

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = ['BST']


class BST():
    def __init__(self, obsfile):
        self.obsfile = obsfile
        self.abeam   = 0
        self.dbeam   = 0
        self.freq    = 50
        self.polar   = 'nw'

        self._attrlist = ['polar', 'freq', 'time', 'abeam', 'dbeam']

    def __str__(self):
        toprint  = '\t=== Class SST of nenupy ===\n'
        toprint += '\tList of all current attributes:\n'
        for att in dir(self):
            avoid = ['t', 'd', 'f', 'elana', 'azana', 'azdig', 'eldig', 'maposition', 'marotation', 'delays']
            if (not att.startswith('_')) & (not any(x.isupper() for x in att)) & (att not in avoid):
                toprint += "%s: %s\n"%(att, getattr(self, att))
        return toprint

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obsfile(self):
        """ BST observation file,
            obsfile set to 'fake' will produce a custom fake BST observation
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        if o == 'fake':
            self._createFakeObs()
            self._obsfile = o
            return
        
        if (o is None) or (o == '') or (isinstance(o, str)) and (not os.path.isfile(o)):
            # Look at the current/specified directory
            if o is None:
                o = ''
            if os.path.isdir(o):
                _opath = os.path.abspath(o)
                bstfiles = glob.glob( os.path.join(_opath, '*BST.fits') )
            else:
                bstfiles = glob.glob('*BST.fits')
            if len(bstfiles) == 0:
                raise IOError("\n\t=== No BST fits file in current directory, specify the file to read. ===")
            elif len(bstfiles) == 1:
                o = os.path.abspath(bstfiles[0])
            else:
                raise IOError("\n\t=== Multiple BST files are not handled yet ===")
        else:
            if not os.path.isfile(o):
                raise IOError("\t=== File {} not found. ===".format(os.path.abspath(o)))
            else:
                o = os.path.abspath(o) 
        self._obsfile = o

        if not self._isBST():
            raise ValueError("\t=== Files might not be a BST observaiton ===")
        else:
            self._readBST()

        return

    @property
    def abeam(self):
        """ Index of the selected analogic beam
        """
        return self._abeam
    @abeam.setter
    def abeam(self, a):
        if (a is None) or (a==-1):
            return
        elif a not in self.abeams:
            print("\n\t=== WARNING: available AnaBeams are {} ===".format(self.abeams))
            self._abeam = 0
        else:
            self._abeam = a
        self._dbeam = np.arange(self._digi2ana.size)[ self._digi2ana == self._abeam ][0]
        return

    @property
    def dbeam(self):
        """ Index of the selected numeric beam.
        """
        return self._dbeam
    @dbeam.setter
    def dbeam(self, b):
        if (b is None) or (b==-1):
            return
        elif b not in self.dbeams:
            print("\n\t=== WARNING: available DigiBeams are {} ===".format(self.dbeams))
            self._dbeam = 0
        else:
            self._dbeam = b
        freqs = self._freqs[self._dbeam]
        self.freqmin = freqs[freqs > 0.].min()
        self.freqmax = freqs[freqs > 0.].max()
        self._abeam  = self._digi2ana[self._dbeam] # change anabeam automatically
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
            if len(f) != 2:
                raise ValueError("\n\t=== 'freq' should be a size 2 array ===")
            elif not isinstance(f[0], (np.float32, np.float64, int, float)):
                raise ValueError("\n\t=== 'freq' Syntax error ===")
            elif not isinstance(f[1], (np.float32, np.float64, int, float)):
                raise ValueError("\n\t=== 'freq' Syntax error ===")
            else:
                self._freq = f
        elif isinstance(f, (np.float32, np.float64, int, float)):
            self._freq = f
        else:
            raise ValueError("\n\t=== 'freq' Syntax error ===")
        # if isinstance(f, list):
        #     self._freq    = [0, 0]
        #     self._freq[0] = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-f[0]))
        #     self._freq[1] = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-f[1]))
        # else:
        #     self._freq    = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-f))  
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
                except:
                    print("\n\t=== WARNING: Time syntax incorrect ===")
            self._time = t
            return

    # ------ Specific getters ------ #
    @property
    def ma(self):
        """ Mini-Arrays used
        """
        self._ma = self._malist[ self.abeam ]
        return self._ma

    @property
    def marotation(self):
        """ Mini-array rotation in degrees
        """
        self._marotation = self._marot[ np.searchsorted(self._maon, self.ma) ]
        return self._marotation

    @property
    def maposition(self):
        """ Lambert93 (x, y, z) Mini-Array positions
        """
        self._maposition = self._mapos[ np.searchsorted(self._maon, self.ma) ]
        return self._maposition

    @property
    def delays(self):
        """ Delays in nsec
        """
        self._delays = self._nsdelays[ self.ma ]
        return self._delays

    @property
    def azana(self):
        """ Azimuth during the analogic pointing self.abeam
        """
        az = self._azlistana[ self._pointana==self.abeam ]
        if az.size == 1:
            az = az[0]
        else:
            anatimes = self._pointanat[ self._pointana==self.abeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((anatimes >= self.time[0]) & (anatimes <= self.time[1]))
                az = az[tmask]
            else:
                az = az[ np.argmin(np.abs( (anatimes - self.time).sec )) ]
        return az 

    @property
    def elana(self):
        """ Elevation during the analogic pointing self.abeam
        """
        el = self._ellistana[ self._pointana==self.abeam ]
        if el.size == 1:
            el = el[0]
        else:
            anatimes = self._pointanat[ self._pointana==self.abeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((anatimes >= self.time[0]) & (anatimes <= self.time[1]))
                el = self._ellistana[tmask]
            else:
                el = self._ellistana[ np.argmin(np.abs( (anatimes - self.time).sec )) ]
        return el

    @property
    def azdig(self):
        """ Azimuth during the numeric pointing self.dbeam
        """
        az = self._azlistdig[ self._pointdig==self.dbeam ]
        if az.size == 1:
            az = az[0]
        else:
            digitimes = self._pointdigt[ self._pointdig==self.dbeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((digitimes >= self.time[0]) & (digitimes <= self.time[1]))
                az = az[tmask]
            else:
                az = az[ np.argmin(np.abs( (digitimes - self.time).sec )) ]
        return az

    @property
    def eldig(self):
        """ Elevation during the numeric pointing self.dbeam
        """ 
        el = self._ellistdig[ self._pointdig==self.dbeam ]
        if el.size == 1:
            el = el[0]
        else:
            digitimes = self._pointdigt[ self._pointdig==self.dbeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((digitimes >= self.time[0]) & (digitimes <= self.time[1]))
                el = el[tmask]
            else:
                el = el[ np.argmin(np.abs( (digitimes - self.time).sec )) ]
        return el


    # ================================================================= #
    # =========================== Methods ============================= #
    def getData(self, **kwargs):
        """ Make the data selection

            Parameters
            ----------
            kwargs : {freq, polar, time, abeam, dbeam}
                Keyword arguments

            Returns
            -------
            self.d : np.ndarray
                Data selected
            self.f : np.ndarray
                Frequency selected
            self.t : np.ndarray
                Time selected
        """
        self._evalkwargs(kwargs)

        # ------ load data only once ------ #
        if not hasattr(self, '_dataall'):
            with fits.open(self.obsfile, mode='readonly', ignore_missing_end=True, memmap=True) as fitsfile:
               self._timeall = Time( fitsfile[7].data['jd'], format='jd' ) 
               self._dataall = fitsfile[7].data['data'] 

        # ------ polarization ------ #
        mask_pol = (self._pols == self.polar)

        # ------ frequency ------ #
        if isinstance(self.freq, list):
            freqi = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-self.freq[0]))
            freqf = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-self.freq[1]))
            #mask_fre = (self._freqs[self.dbeam] > 0.) & (self._freqs[self.dbeam] >= self.freq[0]) & (self._freqs[self.dbeam] <= self.freq[-1])
            mask_fre = (self._freqs[self.dbeam] > 0.) & (self._freqs[self.dbeam] >= freqi) & (self._freqs[self.dbeam] <= freqf)
        else:
            mask_fre = (self._freqs[self.dbeam] == min(self._freqs[self.dbeam], key=lambda x: np.abs(x-self.freq)))
        mask_freq = np.roll(mask_fre, self._bletlist[self.dbeam][0])

        # ------ time ------ #
        digitimes = self._pointdigt[ self._pointdig==self.dbeam ]
        if digitimes.size == 1:
            digiti   = digitimes[0]
            digitf   = digitimes[0] + self._ddeltat[0]
        else:
            digiti   = digitimes[0]
            digitf   = digitimes[-1]
        mask_tim = (self._timeall >= digiti) & (self._timeall <= digitf)
        if isinstance(self.time, list):
            mask_tim = mask_tim & (self._timeall >= self.time[0]) & (self._timeall <= self.time[1])
        else:
            mask_tim = mask_tim & (self._timeall.mjd == min(self._timeall.mjd, key=lambda x: np.abs(x-self.time.mjd)) )
       
        # ------ selected data ------ #
        self.d = np.squeeze( self._dataall[ np.ix_(mask_tim, mask_pol, mask_freq) ] )
        self.t = self._timeall[ mask_tim ]
        self.f = self._freqs[self.dbeam][ mask_fre ]
        return

    def plotData(self, db=True, **kwargs):
        """ Plot the data
        """
        self.getData(**kwargs)

        plotkwargs = {key: value for (key, value) in kwargs.items() if key not in self._attrlist}

        if self.f.size == 1:
            # ------ Light curve ------ #
            xtime = (self.t - self.t[0]).sec / 60
            if db:
                plt.plot(xtime, 10.*np.log10(self.d), **plotkwargs)
                plt.ylabel('dB')
            else:
                plt.plot(xtime, self.d, **plotkwargs)
                plt.ylabel('Amplitude')
            plt.xlabel('Time (min since {})'.format(self.t[0].iso))
            plt.title('f={:3.2f} MHz, pol={}, abeam={}, dbeam={}'.format(self.f[0], self.polar, self.abeam, self.dbeam))

        elif self.t.size == 1:
            # ------ Spectrum ------ #
            if db:
                plt.plot(self.f, 10.*np.log10(self.d), **plotkwargs)
                plt.ylabel('dB')
            else:
                plt.plot(self.f, self.d, **plotkwargs)
                plt.ylabel('Amplitude')
            plt.xlabel('Frequency (MHz)')
            plt.title('t={}, pol={}, abeam={}, dbeam={}'.format(self.time.iso, self.polar, self.abeam, self.dbeam))

        elif (self.f.size > 1) & (self.t.size > 1):
            # ------ Dynamic spectrum ------ #
            xtime = (self.t - self.t[0]).sec / 60
            vmin, vmax = np.percentile(self.d, [5, 99])
            cmap = 'bone'
            for key, value in plotkwargs.items():
                if key == 'cmap': cmap = value
                if key == 'vmin': vmax = value
                if key == 'vmax': vmin = value 
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            normcb = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            spec   = ax.pcolormesh(xtime, self.f, self.d.T, cmap=cmap, norm=normcb)
            plt.colorbar(spec)
            ax.axis( [xtime.min(), xtime.max(), self.f.min(), self.f.max()] )
            plt.xlabel('Time (min since {})'.format(self.t[0].iso))
            plt.ylabel('Frequency (MHz)')
            plt.title('pol={}, abeam={}, dbeam={}'.format(self.polar, self.abeam, self.dbeam))

        else:
            raise ValueError("\n\t=== ERROR: Plot nature not understood ===")
        plt.show()
        plt.close('all')
        return

    def saveData(self, savefile=None, **kwargs):
        """ Save the data
        """
        if savefile is None:
            savefile = self.obsname + '_data.fits'
        else:
            if not savefile.endswith('.fits'):
                raise ValueError("\n\t=== It should be a FITS ===")
        self.getData(**kwargs)

        prihdr = fits.Header()
        prihdr.set('OBS', self.obsname)
        prihdr.set('FREQ', str(self.freq))
        prihdr.set('TIME', str(self.time))
        prihdr.set('POLAR', self.polar)
        #prihdr.set('MINI-ARR', str(self.ma))
        datahdu = fits.PrimaryHDU(self.d.T, header=prihdr)
        freqhdu = fits.BinTableHDU.from_columns( [fits.Column(name='frequency', format='D', array=self.f)] )
        timehdu = fits.BinTableHDU.from_columns( [fits.Column(name='mjd', format='D', array=self.t.mjd)] )
        hdulist = fits.HDUList([datahdu, freqhdu, timehdu])
        hdulist.writeto(savefile, overwrite=True)
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _isBST(self):
        """ Check that self.obsfile is a proper BST observation
        """
        isBST = True
        with fits.open(self.obsfile, mode='readonly', ignore_missing_end=True, memmap=True) as f:
            if f[0].header['OBJECT'] != 'beamlet Statistics':
                isBST = False
            else:
                pass
        return isBST

    def _readBST(self):
        """ Read BST fits files and fill the class attributes
        """
        self.obsname = os.path.basename( self.obsfile )[:-9]

        with fits.open(self.obsfile, mode='readonly', ignore_missing_end=True, memmap=True) as f:
            head      = f[0].header
            setup_ins = f[1].data
            setup_obs = f[2].data
            setup_ana = f[3].data
            setup_bea = f[4].data
            setup_pan = f[5].data
            setup_pbe = f[6].data

        self.obstart  = Time( head['DATE-OBS'] + 'T' + head['TIME-OBS'] )
        self.obstop   = Time( head['DATE-END'] + 'T' + head['TIME-END'] )
        self.time     = [self.obstart.copy(), self.obstop.copy()]
        self.exposure = self.obstop - self.obstart

        self._maon  = np.squeeze( setup_ins['noMROn'] )
        self._marot = np.squeeze( setup_ins['rotation'] )
        self._mapos = np.squeeze( setup_ins['noPosition'] )
        self._mapos = self._mapos.reshape( int(self._mapos.size/3), 3 )
        self._pols  = np.squeeze( setup_ins['spol'] )
        try:
            self._nsdelays = np.squeeze(setup_ins['delay'])[::2]#[ self._maon ]
        except:
            self._nsdelays = np.zeros( self._maon.max() )

        self.abeams     = setup_ana['NoAnaBeam']
        self._antlist   = np.array( [ np.array(eval(i)) - 1 for i in setup_ana['Antlist'] ])
        self._adeltat   = TimeDelta(setup_ana['Duration'], format='sec')
        self._malist    = np.array( [mrs[0:setup_ana['nbMRUsed'][i]] for i, mrs in enumerate(setup_ana['MRList'])] )

        self.dbeams     = setup_bea['noBeam']
        self._digi2ana  = setup_bea['NoAnaBeam']
        self._bletlist  = setup_bea['BeamletList']
        self._freqs     = setup_bea['freqList']
        self._ddeltat   = TimeDelta(setup_bea['Duration'], format='sec')

        self._pointana  = setup_pan['noAnaBeam']
        self._azlistana = setup_pan['AZ']
        self._ellistana = setup_pan['EL']
        self._pointanat = Time(np.array([setup_pan['timestamp'][self._pointana==i] for i in self.abeams]))
        
        self._pointdig  = setup_pbe['noBeam']
        self._azlistdig = setup_pbe['AZ']
        self._ellistdig = setup_pbe['EL']    
        self._pointdigt = Time(setup_pbe['timestamp'])
        return

    def _createFakeObs(self):
        return

    def _evalkwargs(self, kwargs):
        for key, value in kwargs.items():
            if   key == 'polar': self.polar = value
            elif key == 'freq':  self.freq  = value
            elif key == 'time':  self.time  = value
            elif key == 'abeam': self.abeam = value
            elif key == 'dbeam': self.dbeam = value
            else:
                pass
        return
    
