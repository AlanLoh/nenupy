#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Class to read BST NenuFAR data
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
__all__ = ['BST']


import os
import sys
import glob
import numpy as np

from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.time import Time, TimeDelta


class BST(object):
    def __init__(self, obsfile):
        self.obsfile = obsfile
        self.abeam   = 0
        self.dbeam   = 0
        self.freq    = 50
        self.polar   = 'nw'

        self._attrlist = ['polar', 'freq', 'time', 'abeam', 'dbeam']

    def __str__(self):
        toprint  = '\t=== Class BST of nenupy ===\n'
        toprint += '\tList of all current attributes:\n'
        for att in dir(self):
            avoid = ['data', 'elana', 'azana', 'azdig', 'eldig', 'maposition', 'marotation', 'delays', 'plot', 'save', 'select']
            if (not att.startswith('_')) & (not any(x.isupper() for x in att)) & (att not in avoid):
                toprint += "%s: %s\n"%(att, getattr(self, att))
        return toprint


    def __repr__(self):
        return '<BST object: obsfile={}>'.format(self.obsfile)

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
                return#o = ''
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
        halfsec = TimeDelta(0.5, format='sec')
        if isinstance(t, list):
            if not isinstance(t[0], Time):
                t[0] = Time(t[0])
                
            if not isinstance(t[1], Time):
                t[1] = Time(t[1])
            assert self.obstart-halfsec <= t[0] <= self.obstop+halfsec, 'time[0] outside time range'
            assert self.obstart-halfsec <= t[1] <= self.obstop+halfsec, 'time[1] outside time range'
            assert t[0] < t[1], 'time[0] > time[1]!'
            self._time = t
        else:
            if not isinstance(t, Time):
                t = Time(t)
            assert self.obstart-halfsec <= t <= self.obstop+halfsec, 'time outside time range'
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
            anatimes = self._pointanat[self.abeam][ self._pointana==self.abeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((anatimes >= self.time[0]) & (anatimes <= self.time[1]))
                az = az[tmask]
            else:
                # az = az[ np.argmin(np.abs( (anatimes - self.time).sec )) ] one should take the previous value not the closest!
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
            anatimes = self._pointanat[self.abeam][ self._pointana==self.abeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((anatimes >= self.time[0]) & (anatimes <= self.time[1]))
                # el = self._ellistana[tmask]
                el = el[tmask]
            else:
                # el = self._ellistana[ np.argmin(np.abs( (anatimes - self.time).sec )) ] one should take the previous value not the closest!
                el = el[(anatimes <= self.time)][-1]
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
                # az = az[ np.argmin(np.abs( (digitimes - self.time).sec )) ] one should take the previous value not the closest!
                az = az[(digitimes <= self.time)][-1]
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
                # el = el[ np.argmin(np.abs( (digitimes - self.time).sec )) ] # one should take the previous value not the closest!
                el = el[(digitimes <= self.time)][-1]
        return el

    @ property
    def angles(self):
        """ Pointed angle
        """
        if self.type == 'tracking':
            from nenupy.astro import toRadec
            digitimes = self._pointdigt[ self._pointdig==self.dbeam ]
            if isinstance(self.time, list):
                tmask = np.squeeze((digitimes >= self.time[0]) & (digitimes <= self.time[1]))
                tt = digitimes[tmask][0]
            else:
                tt = digitimes[(digitimes <= self.time)][-1]
            radec = toRadec(source=(self.azdig[0], self.eldig[0]),
                time=tt,
                loc='NenuFAR')
            return (radec.ra.deg, radec.dec.deg, 'J2000')
        else:
            self.type = 'transit'
            return (self.azdig, self.eldig, 'ALTAZ')


    # ================================================================= #
    # =========================== Methods ============================= #
    def select(self, **kwargs):
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
        d = np.squeeze( self._dataall[ np.ix_(mask_tim, mask_pol, mask_freq) ] )
        t = self._timeall[ mask_tim ]
        f = self._freqs[self.dbeam][ mask_fre ]
        self.data = {'amp': d, 'freq': f, 'time': t}
        return

    def plot(self, db=True, **kwargs):
        """ Plot the data

            .. note::
                This function uses the same keyword argument as :func:`select`.

            Args:
                * db : `bool`
                    We all know what foo does.

            Kwargs:
                * freq : `float` or `list`
                    The frequency in MHz
                
                * polar : `str`
                    The polarization (`'NE'` or `'NW'`)
                                
                * time : `str`
                    The time

                * abeam : `int`
                
                * dbeam : `int`
            
            Example:
                >>> plot(db=True, freq=50, polar='NE')
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
            plt.title('f={:3.2f} MHz, pol={}, abeam={}, dbeam={}'.format(self.data['freq'][0], self.polar, self.abeam, self.dbeam))

        elif self.data['time'].size == 1:
            # ------ Spectrum ------ #
            if db:
                plt.plot(self.data['freq'], 10.*np.log10(self.data['amp']), **plotkwargs)
                plt.ylabel('dB')
            else:
                plt.plot(self.data['freq'], self.data['amp'], **plotkwargs)
                plt.ylabel('Amplitude')
            plt.xlabel('Frequency (MHz)')
            plt.title('t={}, pol={}, abeam={}, dbeam={}'.format(self.time.iso, self.polar, self.abeam, self.dbeam))

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
            plt.title('pol={}, abeam={}, dbeam={}'.format(self.polar, self.abeam, self.dbeam))

        else:
            raise ValueError("\n\t=== ERROR: Plot nature not understood ===")

        if hasattr(self, 'src'):
            if self.type == 'transit':
                try:
                    from nenupy.astro import getTransit
                    transit = getTransit(source=self.src, time=self.obstart, loc='Nenufar', az=self.azdig)
                    transit = (transit - self.data['time'][0]).sec / 60.
                    plt.axvline(x=transit, color='black', linestyle='-.', linewidth=1)
                except:
                    pass

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
        #prihdr.set('MINI-ARR', str(self.ma))
        datahdu = fits.PrimaryHDU(self.d.T, header=prihdr)
        freqhdu = fits.BinTableHDU.from_columns( [fits.Column(name='frequency', format='D', array=self.data['freq'])] )
        timehdu = fits.BinTableHDU.from_columns( [fits.Column(name='mjd', format='D', array=self.data['time'].mjd)] )
        hdulist = fits.HDUList([datahdu, freqhdu, timehdu])
        hdulist.writeto(savefile, overwrite=True)
        return

    def mean(self, axis='freq'):
        """
        """
        if axis.lower() == 'freq':
            if self.data['freq'].size > 1:
                self.data['amp'] = np.mean(self.data['amp'], axis=1)
                self.data['freq'] = np.mean(self.data['freq'])
        elif axis.lower() == 'time':
            if self.data['time'].size > 1:
                self.data['amp'] = np.mean(self.data['amp'], axis=0)
                self.data['time'] = Time(np.mean(self.data['time'].mjd), format='mjd')
        else:
            print('Axis {} not in [freq, time]'.format(axis))
        return

    def from_transit(self, source, unit='sec'):
        """ Redefine the time with respect to the transit time
        """
        if self.type == 'transit':
            # try:
            from ..astro import getTransit
            transit = getTransit(source=source, time=self.obstart, loc='Nenufar', az=self.azdig)
            if unit.lower() == 'sec':
                self.data['time'] = (self.data['time'] - transit).sec
            elif unit.lower() == 'min':
                self.data['time'] = (self.data['time'] - transit).sec / 60.
            elif unit.lower() == 'hour':
                self.data['time'] = (self.data['time'] - transit).sec / 3600.
            else:
                print('Accepted values are [sec, min, hour]')
        #     except:
        #         pass
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

        try:
            self.name = setup_obs['name']
        except:
            self.name = ''

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
        # if (self._pointana.size > 1) & (np.unique(self._pointana).size == 1):
        #     # Multi-transit
        #     self.abeams = np.arange(self._pointana.size, dtype=int)
        #     self._pointana = np.arange(self._pointana.size, dtype=int)
        self._azlistana = setup_pan['AZ']
        self._ellistana = setup_pan['EL']
        self._pointanat = Time(np.array([setup_pan['timestamp'][self._pointana==i] for i in self.abeams]))
        
        self._pointdig  = setup_pbe['noBeam']
        # if (self._pointdig.size > 1) & (np.unique(self._pointdig).size == 1):
        #     # Multi-transit
        #     self._pointdig = np.arange(self._pointdig.size, dtype=int)
        #     self.dbeams = np.arange(self._pointdig.size, dtype=int)
        self._azlistdig = setup_pbe['AZ']
        self._ellistdig = setup_pbe['EL']
        self._pointdigt = Time(setup_pbe['timestamp'])

        # if self._pointanat.shape[1] != 1:
        if self._pointdig.size > self.dbeams.size:
            self.type = 'tracking'
        else:
            self.type = 'transit'
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
            elif key == 'src':   self.src   = value
            else:
                pass
        return
    
