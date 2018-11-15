#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read XST NenuFAR data
        by A. Loh
"""

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['XST']


import os
import sys
import glob
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.time import Time, TimeDelta


class XST():
    def __init__(self, obsfile):
        self.obsfile = obsfile
        self.ma1     = 0
        self.ma2     = 0

    def __str__(self):
        toprint  = '\t=== Class XST of nenupy ===\n'
        toprint += '\tList of all current attributes:\n'
        for att in dir(self):
            avoid = ['d', 'maposition', 'marotation']
            if (not att.startswith('_')) & (not any(x.isupper() for x in att)) & (att not in avoid):
                toprint += "%s: %s\n"%(att, getattr(self, att))
        return toprint

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obsfile(self):
        """ XST observation file,
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        if (o is None) or (o == '') or (os.path.isdir(o)):
            if os.path.isdir(o):
                _opath = os.path.abspath(o)
                bstfiles = glob.glob( os.path.join(_opath, '*XST.fits') )
            else:
                bstfiles = glob.glob('*XST.fits')
            if len(bstfiles) == 0:
                raise IOError("\n\t=== No XST fits file in current directory, specify the file to read. ===")
            elif len(bstfiles) == 1:
                o = os.path.abspath(bstfiles[0])
            else:
                raise IOError("\n\t=== Multiple XST files are not handled yet ===")
        else:
            if not os.path.isfile(o):
                raise IOError("\t=== File {} not found. ===".format(os.path.abspath(o)))
            else:
                o = os.path.abspath(o) 
        self._obsfile = o

        if not self._isXST():
            raise ValueError("\t=== Files might not be XST observaiton ===")
        else:
            self._readXST()

        return

    @property
    def ma1(self):
        """ Selected Mini-Array number 1
        """
        return self._ma1
    @ma1.setter
    def ma1(self, m):
        if m is None:
            m = 0
        if not isinstance(m, int):
            try:
                m = int(m)
            except:
                print("\n\t=== WARNING: Mini-Array index {} not recognized ===".format(m))
                m = 0
        if m in self.miniarrays:
                self._ma1 = m
        else:
            print("\n\t=== WARNING: available Mini-Arrays are {} ===".format(self.miniarrays))
            self._ma1 = 0
        return

    @property
    def ma2(self):
        """ Selected Mini-Array number 2
        """
        return self._ma2
    @ma2.setter
    def ma2(self, m):
        if m is None:
            m = 0
        if not isinstance(m, int):
            try:
                m = int(m)
            except:
                print("\n\t=== WARNING: Mini-Array index {} not recognized ===".format(m))
                m = 0
        if m in self.miniarrays:
                self._ma2 = m
        else:
            print("\n\t=== WARNING: available Mini-Arrays are {} ===".format(self.miniarrays))
            self._ma2 = 0
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getData(self, **kwargs):
        """ Make the data selection
            Fill the attributes self.d (data)
        """
        self._evalkwargs(kwargs)
        baseline = np.sort( [self.ma1, self.ma2] )[::-1] 

        # ------ Indices ------
        # These formulas were defined to fit the table of detail_FITS_00_05.pdf
        MA1X_MA2X = baseline[0]*2*(baseline[0]*2+1)/2       + 2*baseline[1]
        MA1X_MA2Y = baseline[0]*2*(baseline[0]*2+1)/2+1     + 2*baseline[1] # it will be the same as next one for auto-corr
        MA1Y_MA2X = (baseline[0]*2+1)*(baseline[0]*2+2)/2   + 2*baseline[1]
        MA1Y_MA2Y = (baseline[0]*2+1)*(baseline[0]*2+2)/2+1 + 2*baseline[1]
        index     = np.array([MA1X_MA2X, MA1X_MA2Y, MA1Y_MA2X, MA1Y_MA2Y]).astype(int)
        data      = fits.getdata(self.obsfile, ext=7, memmap=True)['DATA'][:, :, index] # time, subband, correls
        # Auto-correlation
        if self.ma1 == self.ma2:
            # Index 1 (XY) is wrong for auto-correlation with the above algorithm,
            # doesn't matter, we replace the data by the conjugate of YX which exists.
            data[:, :, 1] = np.conjugate(data[:, :, 2])
        self.d = data
        return

    def plotCorrMat(self, polar='XX'):
        """ Plot a matrix of Cross correlations mean amplitudes
        """
        data = fits.getdata(self.obsfile, ext=7, memmap=True)['DATA'][:, 0, :] 
        data = np.mean(data, axis=0)

        try:
            pol = ['XX', 'XY', 'YX', 'YY'].index(polar.upper())
        except ValueError:
            raise

        amp = np.zeros( (self.allma.size, self.allma.size) )    

        for ant1 in self.allma:
            for ant2 in self.allma[ant1:]:
                baseline = np.sort( [ant1, ant2] )[::-1] # descending order
                # ------ Indices ------
                # These formulas were defined to fit the table of detail_FITS_00_05.pdf (it a kind of triangular series)
                MA1X_MA2X = baseline[0]*2*(baseline[0]*2+1)/2       + 2*baseline[1]
                MA1X_MA2Y = baseline[0]*2*(baseline[0]*2+1)/2+1     + 2*baseline[1] # it will be the same as next one for auto-corr
                MA1Y_MA2X = (baseline[0]*2+1)*(baseline[0]*2+2)/2   + 2*baseline[1]
                MA1Y_MA2Y = (baseline[0]*2+1)*(baseline[0]*2+2)/2+1 + 2*baseline[1]
                index     = np.array([MA1X_MA2X, MA1X_MA2Y, MA1Y_MA2X, MA1Y_MA2Y]).astype(int)
                d         = data[index]
                if ant1 == ant2:
                    # Index 1 (XY) is wrong for auto-correlations with the above algorithm,
                    # doesn't matter, we replace the data by the conjugate of YX which exists anyway.
                    d[1] = np.conjugate(d[2])
                if not (ant1 == ant2):
                    amp[ant1, ant2] = np.absolute(d[pol])
        amp[amp == 0] = np.nan

        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot."""
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('{} - {}'.format( os.path.basename(self.obsfile), polar.upper() ))
        ax.set_xlabel('MA')
        ax.set_ylabel('MA')
        im = plt.imshow(np.log10(amp), origin='lower', cmap='Spectral', interpolation='nearest')
        ax.grid(color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        # cb = fig.colorbar(im, ax=ax)
        # cb.set_label('log$_{10}$( Amplitude )')
        cb = add_colorbar(im)
        cb.set_label('log$_{10}$( Amplitude )')
        plt.show()
        return        

    def plotData(self, **kwargs):
        """ Plot
        """
        self.getData(**kwargs)
        xstamp   = np.absolute(self.d)
        xstphase = np.angle(self.d, deg=True)

        # ------ Amplitude ------ #
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        fig.suptitle(r'Amplitude (Baseline MA{}-MA{})'.format(self.ma1, self.ma2), fontsize=15)
        im1 = ax1.imshow(xstamp[:, :, 0].T, origin='lower', aspect='auto', interpolation='nearest')
        ax1.xaxis.set_visible(False)
        ax1.set_title(r'MA{}X-MA{}X'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        ax1.set_ylabel('Freq steps')
        im2 = ax2.imshow(xstamp[:, :, 1].T, origin='lower', aspect='auto', interpolation='nearest')
        ax2.xaxis.set_visible(False)
        ax2.set_title(r'MA{}X-MA{}Y'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        ax2.set_ylabel('Freq steps')
        im3 = ax3.imshow(xstamp[:, :, 2].T, origin='lower', aspect='auto', interpolation='nearest')
        ax3.xaxis.set_visible(False)
        ax3.set_title(r'MA{}Y-MA{}X'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)
        ax3.set_ylabel('Freq steps')
        im4 = ax4.imshow(xstamp[:, :, 3].T, origin='lower', aspect='auto', interpolation='nearest')
        ax4.set_xlabel('Time steps')
        ax4.set_title(r'MA{}Y-MA{}Y'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax)
        ax4.set_ylabel('Freq steps')
        plt.show()
        plt.close('')

        # ------ Phase ------ #
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        fig.suptitle(r'Phase (Baseline MA{}-MA{})'.format(self.ma1, self.ma2), fontsize=15)
        im1 = ax1.imshow(xstphase[:, :, 0].T, origin='lower', aspect='auto', interpolation='nearest')
        ax1.xaxis.set_visible(False)
        ax1.set_title(r'MA{}X-MA{}X'.format(self.ma1, self.ma2), fontsize=10)
        ax1.set_ylabel('Freq steps')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        im2 = ax2.imshow(xstphase[:, :, 1].T, origin='lower', aspect='auto', interpolation='nearest')
        ax2.xaxis.set_visible(False)
        ax2.set_title(r'MA{}X-MA{}Y'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        ax2.set_ylabel('Freq steps')
        im3 = ax3.imshow(xstphase[:, :, 2].T, origin='lower', aspect='auto', interpolation='nearest')
        ax3.xaxis.set_visible(False)
        ax3.set_title(r'MA{}Y-MA{}X'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)
        ax3.set_ylabel('Freq steps')
        im4 = ax4.imshow(xstphase[:, :, 3].T, origin='lower', aspect='auto', interpolation='nearest')
        ax4.set_xlabel('Time steps')
        ax4.set_title(r'MA{}Y-MA{}Y'.format(self.ma1, self.ma2), fontsize=10)
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax)
        ax4.set_ylabel('Freq steps')
        plt.show()
        plt.close('')
        return

    def convertMS(self):
        """ Convert the XST data into a Measurement Set
        """
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _isXST(self):
        """ Check that self.obsfile is a proper XST observation
        """
        isXST = True
        with fits.open(self.obsfile, mode='readonly', ignore_missing_end=True, memmap=True) as f:
            if f[0].header['OBJECT'] != 'crosscorrelation Statistics':
                isXST = False
            else:
                pass
        return isXST

    def _readXST(self):
        """ Read XST fits files and fill the class attributes
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

        self.allma      = np.squeeze( setup_ins['noMR'])
        self.miniarrays = np.squeeze( setup_ins['noMROn'] )
        self._marot     = np.squeeze( setup_ins['rotation'] )
        self._mapos     = np.squeeze( setup_ins['noPosition'] )
        self._mapos     = self._mapos.reshape( int(self._mapos.size/3), 3 )
        self._pols      = np.squeeze( setup_ins['spol'] )

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

    def _evalkwargs(self, kwargs):
        for key, value in kwargs.items():
            if   key == 'ma1': self.ma1  = value
            elif key == 'ma2': self.ma2  = value
            else:
                pass
        return


