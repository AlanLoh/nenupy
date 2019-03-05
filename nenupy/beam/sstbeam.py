#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to compute a SST NenuFAR beam
        by A. Loh
"""

import os
import sys
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
from astropy.io import fits

import healpy as hp
from astropy import constants as const

from . import PhasedArrayBeam
from .antenna import AntennaModel
from .antenna import miniarrays
from ..read import SST

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['SSTbeam', 'SSTbeamHPX']

class SSTbeam(object):
    """ 
    Class to handle SST beams
    
    Parameters
    ----------
    * sst : `SST` class, optional
        Instance of *SST class*, it refers to a particular observation
    * ma : integer, optional
        Mini-Array number (automatic by default)
    * marotation : float, optional
        Mini-Array rotation (automatic by default)
    * polar : str, optional
        Polarization, 'NE' or 'NW' ('NW' by default)
    * freq : float, optional
        Frequency in MHz (50 by default)
    * azana : float, optional
        Azimuth in degrees (180 by default)
    * elana : float, optional
        Elevation in degrees (90 by default)

    Returns
    -------
    beam : np.ndarray
        Normalized 2D array of shape (azimuth, elevation)

    Examples
    --------
    To plot a beam:
    
    >>> from nenupy.beam import SSTbeam
    >>> beam = SSTbeam(ma=12, azana=30, elana=45, freq=65, polar='NE')
    >>> beam.plotBeam()

    Work on the beam computed from a given SST observation:

    >>> from nenupy.read import SST
    >>> from nenupy.beam import SSTbeam
    >>> sst = SST('observation_SST.fits')
    >>> sst.freq = 45
    >>> sst.polar = 'ne'
    >>> beam = SSTbeam(sst)
    >>> beam.getBeam()
    """
    def __init__(self, sst=None, **kwargs):
        self.sst = sst
        self._evalkwargs(kwargs)
        
    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def sst(self):
        """ SST observation
        """
        return self._sst
    @sst.setter
    def sst(self, s):
        if isinstance(s, SST):
            self._sst       = s
            self.ma         = self._sst.ma
            self.marotation = self._sst.marotation
            self.freq       = self._sst.freq
            self.polar      = self._sst.polar
            self.azana      = self._sst.azana 
            self.elana      = self._sst.elana
        else:
            self._sst = s
        return

    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, (int, float, np.float32, np.float64)):
            raise TypeError("\n\t=== Attribute 'freq' should be a number ===")
        elif (f<=0.) or (f>100.):
            raise ValueError("\n\t=== Attribute 'freq' set to a value outside of NenuFAR's frequency range ===")
        else:
            self._freq = f
            return

    @property
    def polar(self):
        """ Polarization selection
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, (str)):
            raise TypeError("\n\t=== Attribute 'polar' should be a string ===")
        elif p.upper() not in ['NW', 'NE']:
            raise ValueError("\n\t=== Attribute 'polar' does not correspond to either 'NW' or 'NE' ===")
        else:
            self._polar = p.upper()
            return

    @property
    def ma(self):
        """ Miniarray selection
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        if self.sst is None:
            marecorded = miniarrays.ma[:, 0].astype(int)
        else:
            marecorded = self.sst.miniarrays
        if m is None:
            print("\n\t==== WARNING: ma is set by default ===")
            m = marecorded[0]
        if not isinstance(m, (int, np.integer) ):
            raise TypeError("\n\t=== Attribute 'ma' should be an integer ===")
        elif not m in marecorded:
            raise ValueError("\n\t=== Attribute 'ma' contains miniarray index not matching existing MAs (up to {}) ===".format(marecorded.max()))
        else:
            self._ma = m
            return

    @property
    def marotation(self):
        """ MA rotation selection
        """
        #if (self.sst is None) or (
        if isinstance(self._marotation, np.ndarray):
            return self._marotation[self.ma]
        else:
            return self._marotation
    @marotation.setter
    def marotation(self, r):
        if r is None:
            print("\n\t==== WARNING: MA rotation is set by default ===")
            self._marotation = miniarrays.ma[:, 1]
            return
        if not isinstance(r, (float, int, np.float32, np.float64, np.int16, np.int32, np.int64)):
            raise TypeError("\n\t=== Attribute 'marotation' should be a number ===")
        else:
            self._marotation = r
            return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getBeam(self):
        """ Get the SST beam
        """
        model = AntennaModel(design='nenufar', freq=self.freq, polar=self.polar)
        elevation = self._squintMA(self.elana)
        az, el = self._realPointing(self.azana, elevation)
        # # az   -= self.marotation + 90. # origin correction
        # az   -= 90.
        beam  = PhasedArrayBeam(p=self._antPos(), m=model, a=az, e=el)
        self.beam = beam.getBeam()
        # self.beam = np.roll(self.beam, int(1./beam.resol)*int(self.marotation + 90) , axis=1)
        # self.beam = np.roll(self.beam, int(1./beam.resol)*int(90) , axis=1)
        return

    def plot(self, store=None, **kwargs):
        """ Plot the SST Beam
        """
        self.getBeam()

        theta = np.linspace(0., 90., self.beam.shape[0])
        phi   = np.radians( np.linspace(0., 360., self.beam.shape[1]) )
        # ------ Plot ------ #
        fig = plt.figure(figsize=(18/2.54, 18/2.54))
        ax  = fig.add_subplot(111, projection='polar')
        normcb = mpl.colors.LogNorm(vmin=self.beam.max() * 1.e-4, vmax=self.beam.max())
        p = ax.pcolormesh(phi, theta, self.beam, norm=normcb, **kwargs)
        ax.grid(linestyle='-', linewidth=0.5, color='white', alpha=0.4)
        plt.setp(ax.get_yticklabels(), rotation='horizontal', color='white')
        g = lambda x,y: r'%d'%(90-x)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter( g ))
        if store is not None:
            plt.savefig(store, dpi=300, facecolor='none',
                edgecolor='none', transparent=True, bbox_inches='tight')
        else:
            plt.title('MA={}, pol={}, freq={:.2f}MHz, az={}, el={}'.format(self.ma, self.polar, self.freq, self.azana, self.elana))
            plt.show()
        plt.close('all')
        return

    def save(self, savefile=None):
        """ Save the beam
        """
        if savefile is None:
            savefile = 'beam.fits'
        else:
            if not savefile.endswith('.fits'):
                raise ValueError("\n\t=== It should be a FITS ===")
        self.getBeam()

        prihdr = fits.Header()
        prihdr.set('FREQ', str(self.freq))
        prihdr.set('POLAR', self.polar)
        prihdr.set('MINI-ARR', str(self.ma))
        prihdr.set('MA-ROT', str(self.marotation))
        datahdu = fits.PrimaryHDU( np.fliplr(self.beam).T, header=prihdr)
        hdulist = fits.HDUList([datahdu])
        hdulist.writeto(savefile, overwrite=True)
        return

    # ================================================================= #
    # =========================== Internal ============================ #
    def _antPos(self):
        """ Return the antenna position within a mini-array
        """
        antpos = miniarrays.antpos

        if self.marotation is not None:
            rot = np.radians( self.marotation )
            rotation = np.array([[ np.cos(rot),  np.sin(rot), 0],
                                 [-np.sin(rot),  np.cos(rot), 0],
                                 [ 0,            0,           1]])
            antpos = np.dot( antpos, rotation )
        return antpos

    def _realPointing(self, a, e):
        """ Find the real pointing direction from self.ra, self.dec
            Nenufar can only point towards directions on a 128 x 128 cells grid
            Therefore PZ computed for each -desired- direction at a resolution of 0.05deg
            the corresponding real observed direction.
        """ 
        modulepath = os.path.dirname( os.path.realpath(__file__) )
        thph = fits.getdata( os.path.join(modulepath, 'NenuFAR_thph.fits') )
        theta, phi = ( int((90.-e)/0.05 - 0.5), int(a/0.05 - 0.5)  )
        t, p = thph[:, theta, phi]
        return p, 90.-t

    def _squintMA(self, e):
        """ Compute the elevation to be pointed that take into account the squint if we want to observe ele
        """
        modulepath = os.path.dirname( os.path.realpath(__file__) )
        squint  = readsav( os.path.join(modulepath, 'squint_table.sav') )
        optfreq = 30
        indfreq = np.where(squint['freq']==optfreq)[0][0]
        newele  = interp1d(squint['elev_desiree'][indfreq,:], squint['elev_a_pointer'])(e)
        if newele < 20.: # squint is limited at 20 deg elevation
            newele = 20.
        return newele

    def _evalkwargs(self, kwargs):
        """
            Parameters
            ----------
            kwargs : dictionnary
                Dictionnary of keys and values to look from in order to fill the class attributes
        """
        allowed = ['ma', 'marotation', 'freq', 'polar', 'azana', 'elana']
        if not all([ki in allowed for ki in kwargs.keys()]):
            print("\n\t=== WARNING: unkwnown keywords, authorized: {} ===".format(allowed))
        if not hasattr(self, 'ma'):
            self.ma         = kwargs.get('ma', None)
        if not hasattr(self, 'marotation'):
            self.marotation = kwargs.get('marotation', None)
        if not hasattr(self, 'freq'):
            self.freq       = kwargs.get('freq', 50)
        if not hasattr(self, 'polar'):
            self.polar      = kwargs.get('polar', 'NW')
        if not hasattr(self, 'azana'):
            self.azana      = kwargs.get('azana', 180.)
        if not hasattr(self, 'elana'):
            self.elana      = kwargs.get('elana', 90.)
        return





















# ============================================================= #
# ============================================================= #
class SSTbeamHPX(object):
    """
    """

    def __init__(self, sst=None, **kwargs):
        self.sst = sst
        self._evalkwargs(kwargs)

    # ========================================================= #
    # ==================== Getter / Setter ==================== #
    @property
    def sst(self):
        """ SST observation
        """
        return self._sst
    @sst.setter
    def sst(self, s):
        if isinstance(s, SST):
            self._sst       = s
            self.ma         = self._sst.ma
            self.marotation = self._sst.marotation
            self.freq       = self._sst.freq
            self.polar      = self._sst.polar
            self.azana      = self._sst.azana 
            self.elana      = self._sst.elana
        else:
            self._sst = s
        return

    # --------------------------------------------------------- #
    @property
    def azana(self):
        """ Pointed azimuth in degrees
        """
        return self._azana
    @azana.setter
    def azana(self, a):
        self._azana = np.radians(a)
        return

    # --------------------------------------------------------- #
    @property
    def elana(self):
        """ Pointed elevation in degrees
        """
        return self._elana
    @elana.setter
    def elana(self, e):
        self._elana = np.radians(e)
        return

    # --------------------------------------------------------- #
    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, (int, float, np.float32, np.float64)):
            raise TypeError("\n\t=== Attribute 'freq' should be a number ===")
        elif (f<=0.) or (f>100.):
            raise ValueError("\n\t=== Attribute 'freq' set to a value outside of NenuFAR's frequency range ===")
        else:
            self._freq = f
            self.wavevec = 2 * np.pi * f * 1.e6 / const.c.value
            return

    # --------------------------------------------------------- #
    @property
    def polar(self):
        """ Polarization selection
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        if not isinstance(p, (str)):
            raise TypeError("\n\t=== Attribute 'polar' should be a string ===")
        elif p.upper() not in ['NW', 'NE']:
            raise ValueError("\n\t=== Attribute 'polar' does not correspond to either 'NW' or 'NE' ===")
        else:
            self._polar = p.upper()
            return

    # --------------------------------------------------------- #
    @property
    def ma(self):
        """ Miniarray selection
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        if self.sst is None:
            marecorded = miniarrays.ma[:, 0].astype(int)
        else:
            marecorded = self.sst.miniarrays
        if m is None:
            # print("\n\t==== WARNING: ma is set by default ===")
            m = marecorded[0]
        if not isinstance(m, (int, np.integer) ):
            raise TypeError("\n\t=== Attribute 'ma' should be an integer ===")
        elif not m in marecorded:
            raise ValueError("\n\t=== Attribute 'ma' contains miniarray index not matching existing MAs (up to {}) ===".format(marecorded.max()))
        else:
            self._ma = m
            return

    # --------------------------------------------------------- #
    @property
    def marotation(self):
        """ MA rotation selection
        """
        #if (self.sst is None) or (
        if isinstance(self._marotation, np.ndarray):
            return self._marotation[self.ma]
        else:
            return self._marotation
    @marotation.setter
    def marotation(self, r):
        if r is None:
            # print("\n\t==== WARNING: MA rotation is set by default ===")
            self._marotation = miniarrays.ma[:, 1]
            return
        if not isinstance(r, (float, int, np.float32, np.float64, np.int16, np.int32, np.int64)):
            raise TypeError("\n\t=== Attribute 'marotation' should be a number ===")
        else:
            self._marotation = r
            return

    @property
    def resolution(self):
        """ Angular resolution in degrees
        """
        return self._resolution
    @resolution.setter
    def resolution(self, r):
        # Find the nside corresponding to the desired resolution
        nsides = np.array([2**i for i in range(1, 12)])
        resol_deg = np.degrees(hp.nside2resol(nside=nsides, arcmin=False))
        idx = (np.abs(resol_deg - r)).argmin()
        self._nside = nsides[idx]
        self._npix = hp.nside2npix(self._nside)        
        self._resolution = resol_deg[idx]

        self._sky_grid()
        return

    # ========================================================= #
    def compute(self):
        """ Compute the phased array beam
        """
        # ------ Phase delay towards pointing ------ #
        elevation = self._squint(self.elana)
        az, el = self._real_pointing(self.azana, elevation)
        ux = np.cos(az) * np.cos(el)
        uy = np.sin(az) * np.cos(el)
        uz = np.cos(el)
        antpos = self._ant_pos()
        phix = antpos[:, 0] * ux[np.newaxis]
        phiy = antpos[:, 1] * uy[np.newaxis]
        phiz = antpos[:, 2] * uz[np.newaxis]
        phi  = self.wavevec * (phix + phiy + phiz)

        # ------ Phase reference ------ #
        _xp = np.cos(self._az) * np.cos(self._alt) 
        _yp = np.sin(self._az) * np.cos(self._alt) 
        _zp = np.sin(self._alt)
        phi0x = antpos[:, 0] * _xp[:, np.newaxis]
        phi0y = antpos[:, 1] * _yp[:, np.newaxis]
        phi0z = antpos[:, 2] * _zp[:, np.newaxis]
        phi0  = self.wavevec * (phi0x + phi0y + phi0z)
        
        phase = phi0[:, :] - phi[np.newaxis, :]

        # ------ e^(i Phi) ------ #
        eiphi = np.sum( np.exp(1j * phase), axis=1 )
        self.beam = np.real(eiphi * eiphi.conjugate())
        self.beam[self.below_horizon] = 0.
        self.beam *= self._nenufar_antenna()
 
        return

    # --------------------------------------------------------- #
    def plot(self):
        """
        """
        if not hasattr(self, 'beam'):
            self.compute()
        hp.orthview( np.log10(self.beam), rot=[0, 90, 90])
        hp.graticule()
        plt.show()
        plt.close('all')
        return

    # ========================================================= #
    # ========================================================= #
    def _sky_grid(self):
        """ Compute the sky grid
        """
        self._az, self._alt = np.radians(hp.pix2ang(nside=self._nside,
                                            ipix=np.arange(self._npix),
                                            lonlat=True,
                                            nest=False))
        self._az = self._az[::-1] # east is 90        
        # Remove local altitudes < 0deg
        self.below_horizon = (self._alt < 0.)
        return

    # --------------------------------------------------------- #
    def _ant_pos(self):
        """ Return the antenna position within a mini-array
        """
        antpos = miniarrays.antpos
        if self.marotation is not None:
            rot = np.radians( self.marotation )
            rotation = np.array([[ np.cos(rot), np.sin(rot), 0],
                                 [-np.sin(rot), np.cos(rot), 0],
                                 [ 0,           0,           1]])
            antpos = np.dot( antpos, rotation )
        return antpos

    def _nenufar_antenna(self):
        """ NenuFAR antenna gain
        """
        from scipy.interpolate import interp2d

        antpol = {'NE': 'NE_SW', 'NW': 'NW_SE'}
        pol    = antpol[self.polar]
        f1 = int( np.floor( self.freq/10. ) ) * 10
        f2 = int( np.ceil(  self.freq/10. ) ) * 10

        modulepath = os.path.dirname( os.path.realpath(__file__) )

        gain_inf = 10**( readsav( os.path.join(modulepath, 'antenna/LSS_GRID_AG_'+pol+'_'+str(f1)+'_astro.sav') )['ga']/10. )
        gain_sup = 10**( readsav( os.path.join(modulepath, 'antenna/LSS_GRID_AG_'+pol+'_'+str(f2)+'_astro.sav') )['ga']/10. )
        if f1 != f2:
            gain = gain_inf * (f2-self.kwargs['freq']) / 10. + gain_sup * (self.kwargs['freq']-f1) / 10.
        else:
            gain = gain_inf

        gaininterp = interp2d( np.linspace(0., 360., gain.shape[1]), 
             np.linspace(0., 90., gain.shape[0]), gain, kind='linear' )

        ant_gain = np.squeeze(list(map(gaininterp, np.degrees(self._az), np.degrees(self._alt))))

        # # WRITE THE FILE
        # import os
        # import numpy as np
        # import healpy as hp
        # from scipy.io.idl import readsav
        # from scipy.interpolate import interp2d
        # module = '/Users/aloh/Documents/Work/Scripts/Own_Packages/nenupy/nenupy/beam/antenna'
        # azgrid, elgrid = hp.pix2ang(nside=64,#1024,
        #     ipix=np.arange(hp.nside2npix(64)),
        #     lonlat=True,
        #     nest=False)
        # azgrid = np.radians(azgrid[::-1]) # east is 90
        # elgrid = np.radians(elgrid)
        # ant_gain = []
        # col_name = []
        # for pol in ['NW_SE', 'NE_SW']:
        #     for freq in [10, 20, 30, 40, 50, 60, 70, 80]:
        #         gain = 10**( readsav( os.path.join(module, 'LSS_GRID_AG_{}_{}_astro.sav'.format(pol, freq)) )['ga']/10.)
        #         gaininterp = interp2d( np.linspace(0., 360., gain.shape[1]), 
        #              np.linspace(0., 90., gain.shape[0]), gain, kind='linear' )
        #         ant_gain.append( np.squeeze(list(map(gaininterp, np.degrees(azgrid), np.degrees(elgrid)))) )
        #         col_name.append( '{}_{}'.format(pol[:2], freq) )
        # hp.write_map(filename="NenuFAR_Ant_Hpx.fits",
        #     m=ant_gain,
        #     column_names=col_name)
        
        return ant_gain/ant_gain.max()

    # --------------------------------------------------------- #
    def _real_pointing(self, a, e):
        """ Find the real pointing direction from self.ra, self.dec
            Nenufar can only point towards directions on a 128 x 128 cells grid
            Therefore PZ computed for each -desired- direction at a resolution of 0.05deg
            the corresponding real observed direction.
        """ 
        modulepath = os.path.dirname( os.path.realpath(__file__) )
        thph = fits.getdata( os.path.join(modulepath, 'NenuFAR_thph.fits') )
        theta, phi = ( int((90.-np.degrees(e))/0.05 - 0.5), int(np.degrees(a)/0.05 - 0.5)  )
        t, p = thph[:, theta, phi]
        return np.radians(p), np.radians(90.-t)

    # --------------------------------------------------------- #
    def _squint(self, e):
        """ Compute the elevation to be pointed that take into account
            the squint if we want to observe ele
        """
        modulepath = os.path.dirname( os.path.realpath(__file__) )
        squint  = readsav( os.path.join(modulepath, 'squint_table.sav') )
        optfreq = 30
        indfreq = np.where(squint['freq']==optfreq)[0][0]
        newele  = interp1d(squint['elev_desiree'][indfreq,:], squint['elev_a_pointer'])(np.degrees(e))
        if newele < 20.: # squint is limited at 20 deg elevation
            newele = 20.
        return np.radians(newele)

    # --------------------------------------------------------- #
    def _evalkwargs(self, kwargs):
        """
            Parameters
            ----------
            kwargs : dictionnary
                Dictionnary of keys and values to look from in order to fill the class attributes
        """
        allowed = ['ma', 'marotation', 'freq', 'polar', 'azana', 'elana', 'resolution']
        if not all([ki in allowed for ki in kwargs.keys()]):
            print("\n\t=== WARNING: unkwnown keywords, authorized: {} ===".format(allowed))
        if not hasattr(self, 'ma'):
            self.ma         = kwargs.get('ma', None)
        if not hasattr(self, 'marotation'):
            self.marotation = kwargs.get('marotation', None)
        if not hasattr(self, 'freq'):
            self.freq       = kwargs.get('freq', 50)
        if not hasattr(self, 'polar'):
            self.polar      = kwargs.get('polar', 'NW')
        if not hasattr(self, 'azana'):
            self.azana      = kwargs.get('azana', 180.)
        if not hasattr(self, 'elana'):
            self.elana      = kwargs.get('elana', 90.)
        if not hasattr(self, 'resolution'):
            self.resolution = kwargs.get('resolution', 0.2)
        return

