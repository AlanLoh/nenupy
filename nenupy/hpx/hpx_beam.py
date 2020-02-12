#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
"""

import os
import numpy as np
import healpy as hp

from nenupy.beam.antenna import miniarrays


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['Anabeam', 'Digibeam']


# ============================================================= #
# -------------------------- Anabeam -------------------------- #
# ============================================================= #
class Anabeam(object):
    """
    """
    
    def __init__(self, **kwargs):
        self.instrument_effect = True
        self.antpos_offset = 90 # angle to rotate
        self._ant_nec_mode = 'nec4'
        self._anakwargs(kwargs)

    # ========================================================= #
    # --------------------- Getter/Setter --------------------- #
    @property
    def freq(self):
        """ Frequency selection in MHz
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        assert 0. <= f <= 100.,\
            '{} MHz outside the frequency range'.format(f)
        self._freq = f
        self.kwave = 2 * np.pi * f * 1.e6 / 299792458
        return
    # --------------------------------------------------------- #
    @property
    def polar(self):
        """ Polarization selection
        """
        return self._polar
    @polar.setter
    def polar(self, p):
        assert p.upper() in ['NW', 'NE'],\
            'Polar {} not understood (NW or NE)'
        self._polar = p.upper()
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
        e = self._squint(e)
        self._elana = np.radians(e)
        return
    # --------------------------------------------------------- #
    @property
    def ma(self):
        """ Miniarray selection
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        marecorded = miniarrays.ma[:, 0].astype(int)
        assert m in marecorded,\
            'MA {} not recorded'.format(m)
        self._ma = m
        return
    # --------------------------------------------------------- #
    @property
    def marot(self):
        """ MA rotation selection
            The '0' seems to be shifted by 90 deg (conventions...)
        """
        return self._marot[self.ma]
    @marot.setter
    def marot(self, r):
        if r is None:
            self._marot = miniarrays.ma[:, 1] + self.antpos_offset 
        else:
            self._marot = np.repeat(r, miniarrays.ma[:, 1].size) + self.antpos_offset
        return
    # --------------------------------------------------------- #
    @property
    def resol(self):
        """ Angular resolution in degrees
        """
        return self._resol
    @resol.setter
    def resol(self, r):
        if r < 0.1:
            print('WARNING, high-res (<0.1deg) may result in long computational durations')
        # Find the nside corresponding to the desired resolution
        nsides = np.array([2**i for i in range(1, 12)])
        resol_rad = hp.nside2resol(nside=nsides, arcmin=False)
        resol_deg = np.degrees(resol_rad)
        idx = (np.abs(resol_deg - r)).argmin()
        if hasattr(self, 'resol'):
            if resol_deg[idx] == self.resol:
                # No need to recompute it
                return
        self._resol = resol_deg[idx]
        self._nside = nsides[idx]
        self._npix = hp.nside2npix(self._nside)
        self._skygrid()
        return


    # ========================================================= #
    # ------------------------ Methods ------------------------ #
    def get_anabeam(self):
        """
        """
        self._compute_anabeam()
        return

    # --------------------------------------------------------- #
    def get_azel_anabeam(self, az, el):
        """ Get the gain value at direction (az, el)
        """
        if not hasattr(self, 'anabeam'):
            self.get_anabeam()
        gain_idx = hp.ang2pix(theta=az, phi=el, nside=self._nside, lonlat=True)
        return self.anabeam[gain_idx]

    # --------------------------------------------------------- #
    def slice_anabeam(self, az=None, el=None, n=100):
        """ Get a slice of the analog beam.

            Parameters
            ----------
            az : float (default = None)
            el : float (default = None)
            n : int (default = 100)
                Number of points for the slice evaluation
        """
        n = int(n)
        if az is not None:
            if el is not None:
                raise Exception('Only one parameter should not be None between az and el')
            elevation = np.linspace(0, 90, n)
            azimuth = np.ones(n) * az
        elif el is not None:
            elevation = np.ones(n) * el
            azimuth = np.linspace(0, 360, n)
        else:
            raise Exception('Either az or el parameters should not be None')
        slice_pixels = hp.ang2pix(self._nside, theta=azimuth, phi=elevation, lonlat=True)
        return self.anabeam[slice_pixels]


    # ========================================================= #
    # ----------------------- Internal ------------------------ #
    def _skygrid(self):
        """ Compute the sky grid
        """
        self.azgrid, self.elgrid = hp.pix2ang(nside=self._nside,
                                    ipix=np.arange(self._npix),
                                    lonlat=True,
                                    nest=False)
        self.azgrid = np.radians(self.azgrid[::-1]) # east is 90
        self.elgrid = np.radians(self.elgrid)
        self._over_horizon = (self.elgrid < 0.)
        self.domega = hp.nside2resol(self._nside)**2. * np.cos(np.pi/2 - self.elgrid)
        self.domega[self._over_horizon] = 0.
        return
    # --------------------------------------------------------- #
    def _ant_pos(self):
        """ Return the antenna position within a mini-array
        """
        antpos = miniarrays.antpos
        if self.marot is not None:
            rot = np.radians( self.marot )
            rotation = np.array([[ np.cos(rot), np.sin(rot), 0],
                                 [-np.sin(rot), np.cos(rot), 0],
                                 [ 0,           0,           1]])
            antpos = np.dot( antpos, rotation )
        if self.ant is None:
            # use the 19 antennas
            return antpos
        else:
            self.ant = np.array(self.ant)
            return antpos[self.ant].reshape((self.ant.size, 3))
        return antpos
    # --------------------------------------------------------- #
    def _ant_gain(self):
        """
        """
        if hasattr(self, 'antgain'):
            if hp.npix2nside(self.antgain.size) == self._nside:
                # This has already been computed
                return self.antgain
        if self._ant_nec_mode == 'nec4':
            modulpath = os.path.dirname(os.path.realpath(__file__))
            antfile = os.path.join(modulpath, 'NenuFAR_Ant_NEC4_Hpx.fits')
            freqs = np.arange(10, 92.5, 2.5) 

            f1 = freqs[freqs <= self.freq].max()
            f2 = freqs[freqs >= self.freq].min()
            
            count = 0
            cols = {}
            for p in ['NW', 'NE']:
                for f in freqs:
                    cols['{}_{}'.format(p, f)] = count
                    count += 1

            antgain1 = hp.read_map(filename=antfile,
                hdu=1,
                field=cols['{}_{}'.format(self.polar, f1)],
                verbose=False,
                memmap=True)
            antgain2 = hp.read_map(filename=antfile,
                hdu=1,
                field=cols['{}_{}'.format(self.polar, f2)],
                verbose=False,
                memmap=True)

            if f1 != f2:
                antgain = antgain1 * (f2-self.freq) / 2.5 + antgain2 * (self.freq-f1) / 2.5
            else:
                antgain = antgain1
            self.antgain = hp.ud_grade(antgain, nside_out=self._nside)

        else:
            f1 = int( np.floor( self.freq/10. ) ) * 10
            f2 = int( np.ceil(  self.freq/10. ) ) * 10
            cols={'NW_10': 0,
                  'NW_20': 1,
                  'NW_30': 2,
                  'NW_40': 3,
                  'NW_50': 4,
                  'NW_60': 5,
                  'NW_70': 6,
                  'NW_80': 7,
                  'NE_10': 8,
                  'NE_20': 9,
                  'NE_30': 10,
                  'NE_40': 11,
                  'NE_50': 12,
                  'NE_60': 13,
                  'NE_70': 14,
                  'NE_80': 15}
            modulpath = os.path.dirname(os.path.realpath(__file__))
            antfile = os.path.join(modulpath, 'NenuFAR_Ant_Hpx.fits') 
            antgain1 = hp.read_map(filename=antfile,
                hdu=1,
                field=cols['{}_{}'.format(self.polar, f1)],
                verbose=False,
                memmap=True)
            antgain2 = hp.read_map(filename=antfile,
                hdu=1,
                field=cols['{}_{}'.format(self.polar, f2)],
                verbose=False,
                memmap=True)
            if f1 != f2:
                antgain = antgain1 * (f2-self.freq) / 10. + antgain2 * (self.freq-f1) / 10.
            else:
                antgain = antgain1
            self.antgain = hp.ud_grade(antgain, nside_out=self._nside)
        return self.antgain
    # --------------------------------------------------------- #
    def _real_pointing(self, azimuth, elevation):
        """ azimuth and elevation in radians
        """
        if self.instrument_effect:
            from astropy.io.fits import getdata
            ff = os.path.join(os.path.dirname(__file__), '../beam/NenuFAR_thph.fits')
            thph = getdata( ff )
            phi = int(np.degrees(azimuth)/0.05 - 0.5)  
            theta = int((90.-np.degrees(elevation))/0.05 - 0.5)
            t, p = thph[:, theta, phi]
            azimuth = np.radians(p)
            elevation = np.radians(90. - t)
        return azimuth, elevation
    # --------------------------------------------------------- #
    def _squint(self, elevation):
        """ elevation in degrees
        """
        if self.instrument_effect:
            from scipy.io.idl import readsav
            from scipy.interpolate import interp1d
            squint  = readsav(os.path.join(os.path.dirname(__file__), '../beam/squint_table.sav'))
            optfreq = 30
            indfreq = np.where(squint['freq']==optfreq)[0][0]
            newele  = interp1d(squint['elev_desiree'][indfreq,:], squint['elev_a_pointer'])(elevation)
            if newele < 20.: # squint is limited at 20 deg elevation
                newele = 20.
            elevation = newele
        return elevation    
    # --------------------------------------------------------- #
    def _compute_anabeam(self):
        """ Compute e^(i (phi0 - phi))
            Where: phi0 is the MA reference phase
                   phi is the phase towards the pointing
        """
        antpos = self._ant_pos()

        az, el = self._real_pointing(self.azana, self.elana)
        ux = np.cos(az) * np.cos(el)
        uy = np.sin(az) * np.cos(el)
        uz = np.sin(el)
        phix = antpos[:, 0] * ux[np.newaxis]
        phiy = antpos[:, 1] * uy[np.newaxis]
        phiz = antpos[:, 2] * uz[np.newaxis]
        phi  = phix + phiy + phiz

        # ------ Phase reference ------ #
        ux = np.cos(self.azgrid) * np.cos(self.elgrid) 
        uy = np.sin(self.azgrid) * np.cos(self.elgrid) 
        uz = np.sin(self.elgrid)
        phi0x = antpos[:, 0] * ux[:, np.newaxis]
        phi0y = antpos[:, 1] * uy[:, np.newaxis]
        phi0z = antpos[:, 2] * uz[:, np.newaxis]
        phi0  = phi0x + phi0y + phi0z
        
        dphase = self.kwave * (phi0[:, :] - phi[np.newaxis, :])

        # ------ e^(i Phi) ------ #
        eiphi = np.sum( np.exp(1j * dphase), axis=1 )

        beam = eiphi * eiphi.conjugate()# * self._ant_gain()
        # beam = self._ant_gain()
        beam = np.absolute(beam) * self._ant_gain()#np.real(beam)
        beam[self._over_horizon] = 0
        self.anabeam = beam# / beam.max()
        return
    # --------------------------------------------------------- #
    def _anakwargs(self, kwargs):
        """ Evaluate the keyword arguments and fill attributes
        """
        self.ma    = kwargs.get('ma', 0)
        self.marot = kwargs.get('marot', None)
        self.freq  = kwargs.get('freq', 50)
        self.polar = kwargs.get('polar', 'NW')
        self.azana = kwargs.get('azana', 180.)
        self.elana = kwargs.get('elana', 90.)
        self.resol = kwargs.get('resol', 0.9)
        self.ant   = kwargs.get('ant', None)
        return
# ============================================================= #
# ============================================================= #







# ============================================================= #
# ------------------------- Digibeam -------------------------- #
# ============================================================= #
class Digibeam(Anabeam):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._digikwargs(kwargs)

    # ========================================================= #
    # --------------------- Getter/Setter --------------------- #
    @property
    def miniarrays(self):
        return self._miniarrays
    @miniarrays.setter
    def miniarrays(self, m):
        marecorded = miniarrays.ma[:, 0].astype(int)
        if m is None:
            m = marecorded
        if not isinstance(m, (list, np.ndarray)):
            m = [m]
        assert all([mi in marecorded for mi in m]),\
            'Invalid MA index/indices.'
        self._miniarrays = np.array(m)
        return
    # --------------------------------------------------------- #
    @property
    def mapos(self):
        """ MA position selection [[x0,y0,z0], [x1,y1,...]]
        """
        return self._mapos[self.miniarrays]
    @mapos.setter
    def mapos(self, p):
        if p is None:
            self._mapos = miniarrays.ma[:, 2:5]
        else:
            self._mapos = np.array(p)
            assert self._mapos.shape[1] == 3,\
                'Positions should have 3 coordinates.'
            assert self._mapos.shape[0] == self.miniarrays.size,\
                'No of MAs not equal to No of positions.' 
        return
    # --------------------------------------------------------- #
    @property
    def azdig(self):
        """ Pointed azimuth in degrees
        """
        return self._azdig
    @azdig.setter
    def azdig(self, a):
        self._azdig = np.radians(a)
        return
    # --------------------------------------------------------- #
    @property
    def eldig(self):
        """ Pointed elevation in degrees
        """
        return self._eldig
    @eldig.setter
    def eldig(self, e):
        self._eldig = np.radians(e)
        return

    # ========================================================= #
    # ------------------------ Methods ------------------------ #
    def get_digibeam(self):
        """
        """
        # import pylab as plt

        self._compute_digibeam()

        # hp.cartview( np.log10(self.digibeam))
        # # hp.graticule()
        # plt.show()
        return self.digibeam

    # --------------------------------------------------------- #
    def get_azel_digibeam(self, az, el):
        """ Get the gain value at direction (az, el)
        """
        if not hasattr(self, 'digibeam'):
            self.get_digibeam()
        gain_idx = hp.ang2pix(theta=az, phi=el, nside=self._nside, lonlat=True)
        return self.digibeam[gain_idx]

    # --------------------------------------------------------- #
    def slice_digibeam(self, az=None, el=None, n=100):
        """ Get a slice of the numerical beam.

            Parameters
            ----------
            az : float (default = None)
            el : float (default = None)
            n : int (default = 100)
                Number of points for the slice evaluation
        """
        n = int(n)
        if az is not None:
            if el is not None:
                raise Exception('Only one parameter should not be None between az and el')
            elevation = np.linspace(0, 90, n)
            azimuth = np.ones(n) * az
        elif el is not None:
            elevation = np.ones(n) * el
            azimuth = np.linspace(0, 360, n)
        else:
            raise Exception('Either az or el parameters should not be None')
        slice_pixels = hp.ang2pix(self._nside, theta=azimuth, phi=elevation, lonlat=True)
        return self.digibeam[slice_pixels]

    # ========================================================= #
    # ----------------------- Internal ------------------------ #
    def _compute_digibeam(self):
        """
        """
        # Build the Mini-Array 'summed' response
        abeams = {}
        for ma in self.miniarrays:
            self.ma = ma
            if str(self.marot%60) not in abeams.keys():
                self._compute_anabeam()
                abeams[str(self.marot%60)] = self.anabeam.copy()
                if self.miniarrays.size == 1:
                    # Only take the Anabeam
                    self.digibeam = self.anabeam
                    return
            if not 'summed_mas' in locals():
                summed_mas = abeams[str(self.marot%60)]
            else:
                summed_mas += abeams[str(self.marot%60)]

        # ux = np.cos(self.azdig - np.pi/2) * np.cos(np.pi/2 - self.eldig)
        # uy = np.sin(self.azdig - np.pi/2) * np.cos(np.pi/2 - self.eldig)
        # uz = np.sin(self.eldig)
        uy = np.cos(self.azdig) * np.cos(self.eldig)
        ux = np.sin(self.azdig) * np.cos(self.eldig)
        uz = np.sin(self.eldig)
        phix = self.mapos[:, 0] * ux[np.newaxis]
        phiy = self.mapos[:, 1] * uy[np.newaxis]
        phiz = self.mapos[:, 2] * uz[np.newaxis]
        phi  = phix + phiy + phiz

        # ------ Phase reference ------ #
        # ux = np.cos(self.azgrid - np.pi/2) * np.cos(np.pi/2 - self.elgrid) 
        # uy = np.sin(self.azgrid - np.pi/2) * np.cos(np.pi/2 - self.elgrid) 
        # uz = np.sin(np.pi/2 - self.elgrid)
        uy = np.cos(self.azgrid) * np.cos(self.elgrid) 
        ux = np.sin(self.azgrid) * np.cos(self.elgrid) 
        uz = np.sin(self.elgrid)
        phi0x = self.mapos[:, 0] * ux[:, np.newaxis]
        phi0y = self.mapos[:, 1] * uy[:, np.newaxis]
        phi0z = self.mapos[:, 2] * uz[:, np.newaxis]
        phi0  = phi0x + phi0y + phi0z
        
        dphase = self.kwave * (phi0[:, :] - phi[np.newaxis, :])

        # ------ e^(i Phi) ------ #
        eiphi = np.sum( np.exp(1j * dphase), axis=1 )

        beam = eiphi * eiphi.conjugate() * summed_mas
        beam = np.absolute(beam)#np.real(beam)
        # beam[self._over_horizon] = 0.
        self.digibeam = beam# / beam.max()

        return
    # --------------------------------------------------------- #
    def _digikwargs(self, kwargs):
        """ Evaluate the keyword arguments and fill attributes
        """
        self.miniarrays = kwargs.get('miniarrays', None)
        self.mapos      = kwargs.get('mapos', None)
        self.azdig      = kwargs.get('azdig', 180.)
        self.eldig      = kwargs.get('eldig', 90.)
        return


# ============================================================= #
# ============================================================= #



