#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read NenuFAR SST data
        by A. Loh
"""

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = ['SST']


import os
import sys
import glob
import numpy as np

from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.time import Time, TimeDelta


class SST(object):
    def __init__(self, obsfile=None):
        self.obsfile = obsfile
        self.ma      = 0
        self.freq    = 50
        self.polar   = 'nw'
        self.abeam   = 0 

        self._attrlist = ['polar', 'freq', 'time', 'abeam', 'ma']

    def __str__(self):
        toprint  = '\t=== Class SST of nenupy ===\n'
        toprint += '\tList of all current attributes:\n'
        for att in dir(self):
            avoid = ['data', 'elana', 'azana', 'maposition', 'marotation', 'plot', 'save', 'select']
            if (not att.startswith('_')) & (not any(x.isupper() for x in att)) & (att not in avoid):
                toprint += "%s: %s\n"%(att, getattr(self, att))
        return toprint

    def __repr__(self):
        return '<SST object: obsfile={}>'.format(self.obsfile)

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
        
        if (o is None) or (o == '') or (isinstance(o, str)) and (not os.path.isfile(o)):
            # Look at the current/specified directory
            if o is None:
                o = ''
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
            raise ValueError("\t=== Files might not all be SST observations ===")
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

    # @property
    # def time(self):
    #     """ Time selection
    #     """
    #     return self._time
    # @time.setter
    # def time(self, t):
    #     if isinstance(t, list):
    #         try:
    #             if not isinstance(t[0], Time):
    #                 t[0] = Time(t[0])
    #             if not isinstance(t[1], Time):
    #                 t[1] = Time(t[1])
    #             self._time = t
    #         except:
    #             print("\n\t=== WARNING: Time syntax incorrect ===")
    #     else:
    #         if not isinstance(t, Time):
    #             try:
    #                 t = Time(t)
    #             except:
    #                 print("\n\t=== WARNING: Time syntax incorrect ===")
    #         self._time = t
    #         return

    @property
    def time(self):
        """ Time selection
        """
        return self._time
    @time.setter
    def time(self, t):
        if isinstance(t, list):
            if not isinstance(t[0], Time):
                t[0] = Time(t[0])
                
            if not isinstance(t[1], Time):
                t[1] = Time(t[1])
            assert self.obstart <= t[0] <= self.obstop, 'time[0] outside time range'
            assert self.obstart <= t[1] <= self.obstop, 'time[1] outside time range'
            assert t[0] < t[1], 'time[0] > time[1]!'
            self._time = t
        else:
            if not isinstance(t, Time):
                t = Time(t)
            assert self.obstart <= t <= self.obstop, 'time outside time range'
            self._time = t
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
        return

    # ------ Specific getters ------ #
    @property
    def marotation(self):
        """ Mini-array rotation in degrees
        """
        self._marotation = self._marot[ self.miniarrays == self.ma ][0]
        return self._marotation

    @property
    def maposition(self):
        """ Lambert93 (x, y, z) Mini-Array position
        """
        self._maposition = self._mapos[ self.miniarrays == self.ma ][0]
        return self._maposition
    
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
                # az = az[ np.argmin(np.abs( (anatimes - self.time).sec )) ]
                az = az[(anatimes <= self.time)][-1]
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
                # el = self._ellistana[tmask]
                el = el[tmask]
            else:
                # el = self._ellistana[ np.argmin(np.abs( (anatimes - self.time).sec )) ]
                el = el[(anatimes <= self.time)][-1]
        return el

    # ================================================================= #
    # =========================== Methods ============================= #
    def select(self, **kwargs):
        """ Make the data selection

            Parameters
            ----------
            kwargs : {freq, ma, polar, time}
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
        for ssfile in sorted(self.obsfile):
            with fits.open(ssfile, mode='readonly', ignore_missing_end=True, memmap=True) as fitsfile:
                all_time = fitsfile[7].data['jd']
                # ------ polarization ------ #
                mask_pol = (self._pols == self.polar)
                # ------ frequency ------ #
                if isinstance(self.freq, list):
                    mask_fre = (self._freqs > 0.) & (self._freqs >= self.freq[0]) & (self._freqs <= self.freq[-1])
                else:
                    mask_fre = (self._freqs == min(self._freqs, key=lambda x: np.abs(x-self.freq)))
                # ------ time ------ #
                time_obj = Time(all_time, format='jd')
                if isinstance(self.time, list):
                    mask_tim = (time_obj >= self.time[0]) & (time_obj <= self.time[1])
                else:
                    mask_tim = (time_obj.mjd == min(time_obj.mjd, key=lambda x: np.abs(x-self.time.mjd)) )
                # ------ mini-array ------ #
                mask_ma = (self.miniarrays == self.ma)
                if not 'this_data' in locals():
                    this_data = fitsfile[7].data['data'][np.ix_(mask_tim, mask_ma, mask_pol, mask_fre)]
                    this_time = all_time[mask_tim]
                else:
                    this_data = np.vstack( (this_data, fitsfile[7].data['data'][np.ix_(mask_tim, mask_ma, mask_pol, mask_fre)]))
                    this_time = np.hstack( (this_time, all_time[mask_tim]) )
        f = self._freqs[ mask_fre ]
        self.data = {'amp': np.squeeze(this_data), 'freq': f, 'time': Time(np.squeeze(this_time), format='jd')}
        return
    def select_v0(self, **kwargs):
        """ Make the data selection

            Parameters
            ----------
            kwargs : {freq, ma, polar, time}
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
        # self.d = np.squeeze( self._dataall[ np.ix_(mask_tim, mask_ma, mask_pol, mask_fre) ] )
        # self.t = self._timeall[ mask_tim ]
        # self.f = self._freqs[ mask_fre ]
        d = np.squeeze( self._dataall[ np.ix_(mask_tim, mask_ma, mask_pol, mask_fre) ] )
        t = self._timeall[ mask_tim ]
        f = self._freqs[ mask_fre ]
        self.data = {'amp': d, 'freq': f, 'time': t}
        return

    def plot(self, db=True, **kwargs):
        """ Plot the data
        """
        self.select(**kwargs)

        plotkwargs = {key: value for (key, value) in kwargs.items() if key not in self._attrlist}

        if self.data['freq'].size == 1:
            # ------ Light curve ------ #
            xtime = (self.data['time'] - self.data['time'][0]).sec / 60
            if db:
                plt.plot(xtime, 10.*np.log10(self.data['amp']), **plotkwargs)
                plt.ylabel('dB')
            else:
                plt.plot(xtime, self.data['amp'], **plotkwargs)
                plt.ylabel('Amplitude')
            plt.xlabel('Time (min since {})'.format(self.data['time'][0].iso))
            plt.title('MA={}, f={:3.2f} MHz, pol={}'.format(self.ma, self.freq, self.polar))

        elif self.data['time'].size == 1:
            # ------ Spectrum ------ #
            if db:
                plt.plot(self.data['freq'], 10.*np.log10(self.data['amp']), **plotkwargs)
                plt.ylabel('dB')
            else:
                plt.plot(self.data['freq'], self.data['amp'], **plotkwargs)
                plt.ylabel('Amplitude')
            plt.xlabel('Frequency (MHz)')
            plt.title('MA={}, t={}, pol={}'.format(self.ma, self.time.iso, self.polar))

        elif (self.data['freq'].size > 1) & (self.data['time'].size > 1):
            # ------ Dynamic spectrum ------ #
            xtime = (self.data['time'] - self.data['time'][0]).sec / 60
            vmin, vmax = np.percentile(self.data['amp'], [5, 99]) 
            cmap = 'bone'
            for key, value in plotkwargs.items():
                if key == 'cmap': cmap = value
                if key == 'vmin': vmax = value
                if key == 'vmax': vmin = value
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            normcb = LogNorm(vmin=vmin, vmax=vmax)
            spec   = ax.pcolormesh(xtime, self.data['freq'], self.data['amp'].T, cmap=cmap, norm=normcb)
            plt.colorbar(spec)
            ax.axis( [xtime.min(), xtime.max(), self.data['freq'].min(), self.data['freq'].max()] )
            plt.xlabel('Time (min since {})'.format(self.data['time'][0].iso))
            plt.ylabel('Frequency (MHz)')
            plt.title('MA={}, pol={}'.format(self.ma, self.polar))

        else:
            raise ValueError("\n\t=== ERROR: Plot nature not understood ===")
        plt.show()
        plt.close('all')
        return

    def save(self, savefile=None, **kwargs):
        """ Save the data
        """
        if savefile is None:
            savefile = self.obsname + '_data.fits'
        else:
            if not savefile.endswith('.fits'):
                raise ValueError("\n\t=== It should be a FITS ===")
        self.select(**kwargs)

        prihdr = fits.Header()
        prihdr.set('OBS', self.obsname)
        prihdr.set('FREQ', str(self.freq))
        prihdr.set('TIME', str(self.time))
        prihdr.set('POLAR', self.polar)
        prihdr.set('MINI-ARR', str(self.ma))
        datahdu = fits.PrimaryHDU(self.d.T, header=prihdr)
        freqhdu = fits.BinTableHDU.from_columns( [fits.Column(name='frequency', format='D', array=self.f)] )
        timehdu = fits.BinTableHDU.from_columns( [fits.Column(name='mjd', format='D', array=self.t.mjd)] )
        hdulist = fits.HDUList([datahdu, freqhdu, timehdu])
        hdulist.writeto(savefile, overwrite=True)
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _isSST(self):
        """ Check that self.obsfile contain only SST observations
        """
        isSST = True
        for sstfile in self.obsfile:
            with fits.open(sstfile, mode='readonly', ignore_missing_end=True, memmap=True) as f:
                if not 'OBJECT' in f[0].header.keys():
                    # This is an old fits
                    break
                if f[0].header['OBJECT'].lower() != 'subband statistics':
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
        # if headi['DATE-END'] == '':
        #     headi['DATE-END'] = headi['DATE-OBS']
        # if headf['DATE-END'] == '':
        #     headf['DATE-END'] = headf['DATE-OBS']
        if not hasattr(self, 'obstart'):
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

        self._readParset()

        # if self._pointanat.shape[1] != 1:
        #     self.type = 'tracking'
        # else:
        self.type = 'transit'
        return

    def _readParset(self):
        """ The SST Fits header is lacking some crucial informations
            It is thus needed to read the parset file
        """
        try:
            sstpath = os.path.dirname( self.obsfile[0] )
            parsetfile = glob.glob( os.path.join(sstpath, '{}_*.parset'.format(self.obsname.split('_')[0])) )[0]
            with open(parsetfile) as rf:
                parset = rf.read()
            parsedfile = parset.split('\n')

            self.abeams     = np.arange([int(i.split('=')[1]) for i in parsedfile if 'nrAnaBeams' in i][0])
            self._antlist   = np.array([ np.array(eval(i.split('=')[1]))-1 for i in sorted(parsedfile) if 'antList' in i ])
            self._adeltat   = TimeDelta( np.array([float(i.split('=')[1]) for i in sorted(parsedfile) if ('duration' in i) & ('AnaBeam' in i) ]), format='sec')
            self._pointana  = self.abeams.copy()
            self._azlistana = np.array([float(i.split('=')[1]) for i in sorted(parsedfile) if ('angle1' in i) & ('AnaBeam' in i) ])
            self._ellistana = np.array([float(i.split('=')[1]) for i in sorted(parsedfile) if ('angle2' in i) & ('AnaBeam' in i) ])
            self._pointanat = Time(np.array([i.split('=')[1] for i in sorted(parsedfile) if ('startTime' in i) & ('AnaBeam' in i) ]))
        except:
            # default values
            self.abeams     = np.zeros(1)
            self._antlist   = np.array([np.arange(19)])
            self._adeltat   = TimeDelta( np.array([24 * 3600.]), format='sec')
            self._pointana  = np.zeros(1)
            self._azlistana = np.array([180])
            self._ellistana = np.array([90])
            self._pointanat = Time(np.array([self.obstart.iso]))
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
            elif key == 'abeam': self.abeam = value
            else:
                pass
        return

