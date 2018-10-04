#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read BST NenuFAR data
        by A. Loh
"""

import os
import sys
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
__all__ = ['BST']


class BST():
    def __init__(self, obsfile):
        self.obsfile = obsfile
        self.abeam   = 0
        self.dbeam   = 0
        self.freq    = 50
        self.polar   = 'nw'

    def __str__(self):
        toprint  = '\t=== Class SST of nenupy ===\n'
        toprint += '\tList of all current attributes:\n'
        for att in dir(self):
            avoid = ['t', 'd', 'f', 'elana', 'azana', 'azdig', 'eldig', 'maposition', 'marotation']
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
        
        if (o is None) or (o == '') or (os.path.isdir(o)):
            # Look at the current/specified directory
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
        self.freqmin = self._freqs[self._dbeam].min()
        self.freqmax = self._freqs[self._dbeam].max()
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
            self._freq    = [0, 0]
            self._freq[0] = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-f[0]))
            self._freq[1] = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-f[1]))
        else:
            self._freq    = min(self._freqs[self.dbeam], key=lambda x: np.abs(x-f))  
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
        self._marotation = self._marot[ self.miniarrays ]
        return self._marotation

    @property
    def maposition(self):
        """ Lambert93 (x, y, z) Mini-Array positions
        """
        self._maposition = self._mapos[ self.miniarrays ]
        return self._maposition

    @property
    def azana(self):
        """ Azimuth during the analogic pointing self.abeam
        """
        az = self._azlistana[ self._pointana==self.abeam ]
        if az.size == 1:
            az = az[0]
        return az 

    @property
    def elana(self):
        """ Elevation during the analogic pointing self.abeam
        """
        el = self._ellistana[ self._pointana==self.abeam ]
        if el.size == 1:
            el = el[0]
        return el

    @property
    def azdig(self):
        """ Azimuth during the numeric pointing self.dbeam
        """
        az = self._azlistdig[ self._pointdig==self.dbeam ]
        if az.size == 1:
            az = az[0]
        return az

    @property
    def eldig(self):
        """ Elevation during the numeric pointing self.dbeam
        """
        el = self._ellistdig[ self._pointdig==self.dbeam ]
        if el.size == 1:
            el = el[0]
        return el


    # ================================================================= #
    # =========================== Methods ============================= #
    def getData(self, **kwargs):
        """ Make the data selection
            Fill the attributes self.d (data), self.t (time), self.f (frequency)
        """
        self._evalkwargs(kwargs)

        # ------ load data only once ------ #
        if not hasattr(self, '_data'):
            with fits.open(self.obsfile, mode='readonly', ignore_missing_end=True, memmap=True) as fitsfile:
               self._timeall = Time( fitsfile[7].data['jd'], format='jd' ) 
               self._dataall = fitsfile[7].data['data'] 

        # ------ polarization ------ #
        mask_pol = (self._pols == self.polar)

        # ------ frequency ------ #
        if isinstance(self.freq, list):
            mask_fre = (self._freqs[self.dbeam] > 0.) & (self._freqs[self.dbeam] >= self.freq[0]) & (self._freqs[self.dbeam] <= self.freq[-1])
        else:
            mask_fre = (self._freqs[self.dbeam] == min(self._freqs[self.dbeam], key=lambda x: np.abs(x-self.freq)))
        mask_fre = np.roll(mask_fre, self._bletlist[self.dbeam][0])

        # ------ time ------ #
        if isinstance(self.time, list):
            mask_tim = (self._timeall >= self.time[0]) & (self._timeall <= self.time[1])
        else:
            mask_tim = (self._timeall.mjd == min(self._timeall.mjd, key=lambda x: np.abs(x-self.time.mjd)) )
       
        # ------ selected data ------ #
        self.d = np.squeeze( self._dataall[ np.ix_(mask_tim, mask_pol, mask_fre) ] )
        self.t = self._timeall[ mask_tim ]
        self.f = self._freqs[self.dbeam][ mask_fre ]
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
            plt.title('f={:3.2f} MHz, pol={}, abeam={}, dbeam={}'.format(self.freq, self.polar, self.abeam, self.dbeam))
            plt.show()
            plt.close('all')

        elif self.t.size == 1:
            # ------ Spectrum ------ #
            plt.plot(self.f, self.d)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')
            plt.title('t={}, pol={}, abeam={}, dbeam={}'.format(self.time.iso, self.polar, self.abeam, self.dbeam))
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
            plt.title('pol={}, abeam={}, dbeam={}'.format(self.polar, self.abeam, self.dbeam))
            plt.show()
            plt.close('all')

        else:
            raise ValueError("\n\t=== ERROR: Plot nature not understood ===")
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

        self.miniarrays = np.squeeze( setup_ins['noMROn'] )
        self._marot     = np.squeeze( setup_ins['rotation'] )
        self._mapos     = np.squeeze( setup_ins['noPosition'] )
        self._mapos     = self._mapos.reshape( int(self._mapos.size/3), 3 )
        self._pols      = np.squeeze( setup_ins['spol'] )
        try:
            self._delays = np.squeeze(setup_ins['delay'])[::2][ self.miniarrays ]
        except:
            self._delays = np.zeros( self.miniarrays.size )

        self.abeams     = setup_ana['NoAnaBeam']
        self._antlist   = np.array( [ np.array(eval(i)) - 1 for i in setup_ana['Antlist'] ])

        self.dbeams     = setup_bea['noBeam']
        self._digi2ana  = setup_bea['NoAnaBeam']
        self._bletlist  = setup_bea['BeamletList']
        self._freqs     = setup_bea['freqList']

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
    
