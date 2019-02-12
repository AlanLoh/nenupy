#! /usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
Class Simulate SST/BST time profiles
        by A. Loh


from nenupy.read import SST
from nenupy.beam import SSTbeam
from nenupy.skymodel import SkyModel
from nenupy.simulation import Transit
sst = SST('20170425_230000_SST.fits')
sst.select(freq=50)
beam = SSTbeam(sst)
sm = SkyModel()
sm.gsm2008(freq=sst.freq)
simul = Transit(obs=sst, beam=beam, skymodel=sm)
simul.dt = 300
simul.plotProfile()

from nenupy.read import BST
from nenupy.beam import BSTbeam
from nenupy.skymodel import SkyModel
from nenupy.simulation import Transit
bst = BST('20181018_195600_BST.fits')
bst.select(freq=50)
sm = SkyModel()
sm.gsm2008(freq=bst.freq)
beam = BSTbeam(bst)
simul = Transit(obs=bst, beam=beam, skymodel=sm)
simul.dt = 60
simul.plotProfile()

"""

import os
import sys
import numpy as np
# import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy import coordinates as coord

from ..read import SST, BST
from ..beam import SSTbeam, BSTbeam
from ..skymodel import SkyModel
from ..beam.antenna import miniarrays
from ..utils import ProgressBar

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['Transit', 'Tracking']


class Transit():
    def __init__(self, obs, beam, skymodel, start=None, stop=None):
        self.obs      = obs
        self.beam     = beam
        self.skymodel = skymodel
        self.dt       = TimeDelta(60, format='sec')
        self.start    = start
        self.stop     = stop

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obs(self):
        """ Observation (SST or BST)
        """
        return self._obs
    @obs.setter
    def obs(self, o):
        if not isinstance(o, (SST, BST)):
            raise AttributeError("\n\t=== obs must be either a SST or a BST class ===")
        else:
            assert o.type == 'transit', 'Observation doesnt look like a transit'

            if not hasattr(o, 'data'):
                # Select a default time-profile
                print("\n\t=== WARNING: default time profile ===")
                o.select(time=[o.obstart, o.obstop], freq=50, polar='nw')
            self._obs = o
        return

    @property
    def beam(self):
        """ Simulated beam
        """
        return self._beam
    @beam.setter
    def beam(self, b):
        if not isinstance(b, (SSTbeam, BSTbeam)):
            raise AttributeError("\n\t=== beam must be either a SSTbeam or a BSTbeam class ===")
        else:
            if not hasattr(b, 'beam'):
                # Compute the beam
                b.getBeam()
            self._beam = b
        return

    @property
    def skymodel(self):
        """ Sky model
        """
        return self._skymodel
    @skymodel.setter
    def skymodel(self, s):
        if not isinstance(s, SkyModel):
            raise AttributeError("\n\t=== skymodel must be a SkyModel class ===")
        else:
            self._skymodel = s
        return

    @property
    def dt(self):
        """ Simulation time resolution in seconds
        """
        return self._dt
    @dt.setter
    def dt(self, d):
        if not isinstance(d, TimeDelta):
            try:
                d = TimeDelta(d, format='sec')
            except:
                raise AttributeError("\n\t=== 'dt' format not understood ===")
        self._dt = d
        return

    @property
    def start(self):
        """ Simulation time start
        """
        return self._start
    @start.setter
    def start(self, t):
        if t is None:
            t = self.obs.data['time'][0]

        if not isinstance(t, Time):
            try:
                t = Time(t)
            except:
                raise AttributeError("\n\t=== 'start' format not understood ===")
        self._start = t
        return

    @property
    def stop(self):
        """ Simulation time stop
        """
        return self._stop
    @stop.setter
    def stop(self, t):
        if t is None:
            t = self.obs.data['time'][-1]

        if not isinstance(t, Time):
            try:
                t = Time(t)
            except:
                raise AttributeError("\n\t=== 'stop' format not understood ===")
        self._stop = t
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getProfile(self):
        """ Compute the time-profile simulation
        """
        # ------ Find number of steps ------ #
        nbtime  = int(np.ceil( (self.stop - self.start).sec / self.dt.sec ))
        self.dt = TimeDelta( (self.stop - self.start).sec / (nbtime-1), format='sec')
        
        self.f = self.obs.freq
        self.t = np.zeros( nbtime )
        self.d = np.zeros( nbtime ) 

        # ------ Load skymodel ------ #
        skymodel = self.skymodel.skymodel
        ramodel  = np.linspace(180., -180., skymodel.shape[1]) 
        decmodel = np.linspace(-90., 90., skymodel.shape[0]) 
        azimuth  = np.linspace(0., 360., skymodel.shape[1])
        zenitha  = 90. - np.linspace(0., 90., skymodel.shape[0]/2)
        agrid, zgrid = np.meshgrid( azimuth, zenitha )
        domega = np.radians(ramodel[0]-ramodel[1])**2. * np.cos(np.radians(zgrid))
        #domega /= domega.max()

        # ------ Fixed beam ------ #
        beam = np.flipud(self.beam.beam)
        bazi = np.linspace(0., 360., beam.shape[1])
        bele = np.linspace(0., 90.,  beam.shape[0])
        fb   = interp2d(bazi, bele, beam, kind='linear' )
        beam = fb( np.linspace(0, 360, skymodel.shape[1]), np.linspace(0, 90, skymodel.shape[0]/2) )

        # ------ Loop over time ------ #
        bar = ProgressBar(valmax=nbtime, title='Transit observation simulation')
        for i in range(nbtime):
            # ------ Rotate skymodel ------ #
            frame  = coord.AltAz(obstime=self.start + i*self.dt, location=miniarrays.nenufarloc)

            altaz  = coord.SkyCoord(agrid*u.deg, (90-zgrid)*u.deg, frame=frame)
            radec  = altaz.transform_to(coord.FK5(equinox='J2000'))
            ragrid, decgrid = radec.ra.deg, radec.dec.deg
            ragrid[ragrid > 180] -= 360 # RA is within -180 and 180
            xraind  = (ramodel[0] - ragrid)   / (ramodel[0]-ramodel[1])   #+ 0.5
            xdecind = (decgrid - decmodel[0]) / (decmodel[1]-decmodel[0]) #+ 0.5
            skysel  = skymodel[xdecind.astype(int), xraind.astype(int)]

            # ------ Integrate ------ #
            amp       = np.sum(skysel * beam * domega) # / (4 * np.pi) # domega is normalised
            self.d[i] = amp
            self.t[i] = (self.start + i*self.dt).mjd
            bar.update()
        return

    def plotProfile(self):
        """ Plot the simulation profile against the data
        """
        if not hasattr(self, 'd'):
            self.getProfile()
        tmask = (self.obs.data['time'] >= self.start) & (self.obs.data['time'] <= self.stop)
        scale = np.median(self.obs.data['amp'][tmask]) / np.median(self.d)
        plt.plot( (self.obs.data['time'][tmask] - self.start).sec/60., self.obs.data['amp'][tmask], label='Observation')
        plt.plot( (Time(self.t, format='mjd') - self.start).sec/60., self.d * scale, label='Simulation')
        plt.xlabel('Time (min since {})'.format(self.start.iso))
        plt.ylabel('Amplitude')
        plt.show()
        plt.close('all')
        return

    def saveProfile(self, savefile=None):
        """ Save the simulation
        """
        if savefile is None:
            savefile = self.obs.obsname + '_simulation.fits'
        else:
            if not savefile.endswith('.fits'):
                raise ValueError("\n\t=== It should be a FITS ===")
        if not hasattr(self, 'd'):
            self.getProfile()

        prihdr = fits.Header()
        prihdr.set('OBS', self.obs.obsname)
        prihdr.set('FREQ', str(self.obs.freq))
        prihdr.set('TIME', str(self.obs.time))
        prihdr.set('POLAR', self.polar)
        # prihdr.set('MINI-ARR', str(self.ma))
        datahdu = fits.PrimaryHDU(self.d.T, header=prihdr)
        freqhdu = fits.BinTableHDU.from_columns( [fits.Column(name='frequency', format='D', array=self.f)] )
        timehdu = fits.BinTableHDU.from_columns( [fits.Column(name='mjd', format='D', array=self.t)] )
        hdulist = fits.HDUList([datahdu, freqhdu, timehdu])
        hdulist.writeto(savefile, overwrite=True)
        return



class Tracking():
    def __init__(self, obs, skymodel, start=None, stop=None):
        self.obs      = obs
        self.skymodel = skymodel
        self.dt       = TimeDelta(60, format='sec')
        self.start    = start
        self.stop     = stop

    # ================================================================= #
    # ======================== Getter / Setter ======================== #
    @property
    def obs(self):
        """ Observation (SST or BST)
        """
        return self._obs
    @obs.setter
    def obs(self, o):
        if not isinstance(o, (SST, BST)):
            raise AttributeError("\n\t=== obs must be either a SST or a BST class ===")
        else:
            assert o.type == 'tracking', 'Observation doesnt look like a tracking'

            if not hasattr(o, 'd'):
                # Select a default time-profile
                print("\n\t=== WARNING: default time profile ===")
                o.select(time=[o.obstart, o.obstop], freq=50, polar='nw')
            self._obs = o
        return

    @property
    def skymodel(self):
        """ Sky model
        """
        return self._skymodel
    @skymodel.setter
    def skymodel(self, s):
        if not isinstance(s, SkyModel):
            raise AttributeError("\n\t=== skymodel must be a SkyModel class ===")
        else:
            self._skymodel = s
        return

    @property
    def dt(self):
        """ Simulation time resolution in seconds
        """
        return self._dt
    @dt.setter
    def dt(self, d):
        if not isinstance(d, TimeDelta):
            try:
                d = TimeDelta(d, format='sec')
            except:
                raise AttributeError("\n\t=== 'dt' format not understood ===")
        self._dt = d
        return

    @property
    def start(self):
        """ Simulation time start
        """
        return self._start
    @start.setter
    def start(self, t):
        if t is None:
            t = self.obs.data['time'][0]

        if not isinstance(t, Time):
            try:
                t = Time(t)
            except:
                raise AttributeError("\n\t=== 'start' format not understood ===")
        self._start = t
        return

    @property
    def stop(self):
        """ Simulation time stop
        """
        return self._stop
    @stop.setter
    def stop(self, t):
        if t is None:
            t = self.obs.data['time'][-1]

        if not isinstance(t, Time):
            try:
                t = Time(t)
            except:
                raise AttributeError("\n\t=== 'stop' format not understood ===")
        self._stop = t
        return

    # ================================================================= #
    # =========================== Methods ============================= #
    def getProfile(self):
        """ Compute the time-profile simulation
        """
        # ------ Find number of steps ------ #
        nbtime  = int(np.ceil( (self.stop - self.start).sec / self.dt.sec ))
        self.dt = TimeDelta( (self.stop - self.start).sec / (nbtime-1), format='sec')
        
        self.f = self.obs.freq
        self.t = np.zeros( nbtime )
        self.d = np.zeros( nbtime )
        tmp_coords = (None, None, None, None) 

        # ------ Load skymodel ------ #
        skymodel = self.skymodel.skymodel
        ramodel  = np.linspace(180., -180., skymodel.shape[1]) 
        decmodel = np.linspace(-90., 90., skymodel.shape[0]) 
        azimuth  = np.linspace(0., 360., skymodel.shape[1])
        zenitha  = 90. - np.linspace(0., 90., skymodel.shape[0]/2)
        agrid, zgrid = np.meshgrid( azimuth, zenitha )
        domega = np.radians(ramodel[0]-ramodel[1])**2. * np.cos(np.radians(zgrid))
        #domega /= domega.max()

        # ------ Loop over time ------ #
        bar = ProgressBar(valmax=nbtime, title='Tracking observation simulation')
        for i in range(nbtime):
            # ------ Re-compute the beam ------ #
            self.obs.time = self.start + i*self.dt
            if ((self.obs.azana != tmp_coords[0]) or
                (self.obs.elana != tmp_coords[1]) or
                (self.obs.azdig != tmp_coords[2]) or
                (self.obs.eldig != tmp_coords[3])):
                if isinstance(self.obs, SST):
                    pass
                elif isinstance(self.obs, BST):
                    obsbeam = BSTbeam(self.obs)
                    obsbeam.getBeam()
                beam = np.flipud(obsbeam.beam)
                bazi = np.linspace(0., 360., beam.shape[1])
                bele = np.linspace(0., 90.,  beam.shape[0])
                fb   = interp2d(bazi, bele, beam, kind='linear' )
                beam = fb( np.linspace(0, 360, skymodel.shape[1]), np.linspace(0, 90, skymodel.shape[0]/2) )
                tmp_coords = (self.obs.azana.copy(),
                              self.obs.elana.copy(),
                              self.obs.azdig.copy(),
                              self.obs.eldig.copy())

            # ------ Rotate skymodel ------ #
            frame  = coord.AltAz(obstime=self.start + i*self.dt, location=miniarrays.nenufarloc)

            altaz  = coord.SkyCoord(agrid*u.deg, (90-zgrid)*u.deg, frame=frame)
            radec  = altaz.transform_to(coord.FK5(equinox='J2000'))
            ragrid, decgrid = radec.ra.deg, radec.dec.deg
            ragrid[ragrid > 180] -= 360 # RA is within -180 and 180
            xraind  = (ramodel[0] - ragrid)   / (ramodel[0]-ramodel[1])   #+ 0.5
            xdecind = (decgrid - decmodel[0]) / (decmodel[1]-decmodel[0]) #+ 0.5
            skysel  = skymodel[xdecind.astype(int), xraind.astype(int)]

            # ------ Integrate ------ #
            amp       = np.sum(skysel * beam * domega) # / (4 * np.pi) # domega is normalised
            self.d[i] = amp
            self.t[i] = (self.start + i*self.dt).mjd
            bar.update()
        return

    def plotProfile(self, showdata=True):
        """ Plot the simulation profile against the data
        """
        if not hasattr(self, 'd'):
            self.getProfile()
        
        if showdata:
            tmask = (self.obs.data['time'] >= self.start) & (self.obs.data['time'] <= self.stop)
            scale = np.median(self.obs.data['amp'][tmask]) / np.median(self.d)
            plt.plot( (self.obs.data['time'][tmask] - self.start).sec/60., self.obs.data['amp'][tmask], label='Observation')
        else:
            scale = 1.
        plt.plot( (Time(self.t, format='mjd') - self.start).sec/60., self.d * scale, label='Simulation')
        plt.xlabel('Time (min since {})'.format(self.start.iso))
        plt.ylabel('Amplitude')
        plt.show()
        plt.close('all')
        return

    def saveProfile(self, savefile=None):
        """ Save the simulation
        """
        if savefile is None:
            savefile = self.obs.obsname + '_simulation.fits'
        else:
            if not savefile.endswith('.fits'):
                raise ValueError("\n\t=== It should be a FITS ===")
        if not hasattr(self, 'd'):
            self.getProfile()

        prihdr = fits.Header()
        prihdr.set('OBS', self.obs.obsname)
        prihdr.set('FREQ', str(self.obs.freq))
        prihdr.set('TIME', str(self.obs.time))
        prihdr.set('POLAR', self.polar)
        # prihdr.set('MINI-ARR', str(self.ma))
        datahdu = fits.PrimaryHDU(self.d.T, header=prihdr)
        freqhdu = fits.BinTableHDU.from_columns( [fits.Column(name='frequency', format='D', array=self.f)] )
        timehdu = fits.BinTableHDU.from_columns( [fits.Column(name='mjd', format='D', array=self.t)] )
        hdulist = fits.HDUList([datahdu, freqhdu, timehdu])
        hdulist.writeto(savefile, overwrite=True)
        return



