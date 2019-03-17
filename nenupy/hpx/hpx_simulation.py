#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
    Examples
    --------
        - Transit prediction (whole day, zenith)
        ```
        from nenupy.hpx import Transit
        t=Transit()
        t.predict(src=None, resol=1, freq=55, duration=86400, miniarrays=[10], azana=180, elana=90, azdig=180, eldig=90)
        ```

        - Transit prediction with a pointsource model
        ```
        from nenupy.hpx import Transit
        t=Transit()
        t.predict(src='Cyg A', resol=0.2, freq=60, duration=3600, skymodel='pointsource', ra=299.868, dec=40.734)
        ``` 

        - Tracking prediction of Cyg A using the GSM:
        ```
        from nenupy.hpx import Tracking
        t=Tracking()
        t.predict(src='Cyg A', duration=14400, dt=600, miniarrays=0, resol=1, freq=60)
        ```

        - Tracking prediction of Cyg A using a pointsource skymodel:
        ```
        from nenupy.hpx import Tracking
        t=Tracking()
        t.predict(src='Cyg A', time='2019-03-05 08:00:03', duration=7200, dt=600, miniarrays=0, resol=1, freq=60, skymodel='pointsource', ra=299.868, dec=40.734)
        ```
"""

# import os, sys
import numpy as np
import healpy as hp
from astropy.time import Time, TimeDelta

from nenupy.read import BST
from nenupy.hpx import Anabeam, Digibeam, Skymodel
from nenupy.astro import getTransit, getAltaz, getTime, getSrc
from nenupy.utils import ProgressBar

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['Simu', 'Transit', 'Tracking']




# ============================================================= #
# ---------------------------- Simu --------------------------- #
# ============================================================= #
class Simu(object):
    """
    """
    def __init__(self, **kwargs):
        self.dt = 60
        self.azana = 180
        self.elana = 90
        self.azdig = 180
        self.eldig = 90
        self.nomas = None
        self.freq  = 50
        self.polar = 'NW'
        self.resol = 0.2
        self.skymodel = 'gsm'
        self._simukwargs(kwargs)

    # ========================================================= #
    # --------------------- Getter/Setter --------------------- #
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
        self.nside = nsides[idx]
        return
    # --------------------------------------------------------- #
    @property
    def model(self):
        """ Skymodel object
        """
        m = Skymodel(nside=self.nside,
                     freq=self.freq)
        return m

    # ========================================================= #
    # ----------------------- Internal ------------------------ #
    def _compute_beam(self):
        """ Compute a numeric beam
        """
        self.db = Digibeam(azana=self.azana,
            azdig=self.azdig,
            elana=self.elana,
            eldig=self.eldig,
            miniarrays=self.nomas,
            freq=self.freq,
            polar=self.polar,
            resol=self.resol)
        self.beam = self.db.get_digibeam()
        return
    # --------------------------------------------------------- #
    def _transit_sky(self, **kwargs):
        """ Perform a sky rotation from `self.start` to `self.stop`
            with `dt` sec steps.
            Compute the -static- beam and integrate everything.
        """
        self.simulation = {'time': [],
                            'amp': []}
        sm = self.model
        
        nbtime  = int(np.ceil( (self.stop - self.start).sec / self.dt.sec ))
        self.dt = (self.stop - self.start) / (nbtime - 1)
        
        bar = ProgressBar(valmax=nbtime, title='Transit observation simulation')
        for i in range(nbtime):
            current = self.start + i*self.dt
            # Sky rotation:
            sky = sm.get_skymodel(time=current, model=self.skymodel, **kwargs)
            # Integrate the beam x sky
            integ = np.sum(self.beam * sky )#* self.db.domega ) #/ (4 * np.pi) #self.db.domega ) no need cause hpx?

            self.simulation['time'].append(current.mjd)
            self.simulation['amp'].append(integ)            
            bar.update()

        self.simulation['time'] = Time(np.array(self.simulation['time']),
            format='mjd')
        self.simulation['amp'] = np.array(self.simulation['amp'])
        return
    # --------------------------------------------------------- #
    def _tracking_sky(self, src=None, obs=None, **kwargs):
        """ Perform a sky rotation from `self.start` to `self.stop`
            with `dt` sec steps.
            For each time step, a new beam is computed toward `src`.
        """
        if obs is None:
            src = getSrc(src)

        self.simulation = {'time': [],
                            'amp': []}
        sm = self.model
        
        nbtime  = int(np.ceil( (self.stop - self.start).sec / self.dt.sec ))
        self.dt = (self.stop - self.start) / (nbtime - 1)

        bar = ProgressBar(valmax=nbtime, title='Tracking observation simulation')
        for i in range(nbtime):
            current = self.start + i*self.dt
            # Sky rotation:
            sky = sm.get_skymodel(time=current, model=self.skymodel, **kwargs)
            # Compute the beam
            if obs is None:
                altaz = getAltaz(source=src,
                    time=current,
                    loc='NenuFAR')
                self.azana = altaz.az.deg
                self.elana = altaz.alt.deg
                self.azdig = altaz.az.deg
                self.eldig = altaz.alt.deg
            else:
                obs.time = current
                self.azana = obs.azana
                self.elana = obs.elana
                self.azdig = obs.azdig
                self.eldig = obs.eldig
            self._compute_beam()
            # Integrate the beam x sky
            integ = np.sum(self.beam * sky) # / (4 * np.pi) # * self.db.domega)

            self.simulation['time'].append(current.mjd)
            self.simulation['amp'].append(integ)            
            bar.update()

        self.simulation['time'] = Time(np.array(self.simulation['time']),
            format='mjd')
        self.simulation['amp'] = np.array(self.simulation['amp'])
        return
    # --------------------------------------------------------- #
    def _simukwargs(self, kwargs):
        """
        """
        for key, value in kwargs.items():
            if   key == 'polar': self.polar = value
            elif key == 'freq': self.freq = value
            elif key == 'time': self.time = value
            elif key == 'azana': self.azana = value
            elif key == 'elana': self.elana = value
            elif key == 'azdig': self.azdig = value
            elif key == 'eldig': self.eldig = value
            elif key == 'miniarrays': self.nomas = value
            elif key == 'resol': self.resol = value
            elif key == 'dt': self.dt = value
            elif key == 'skymodel': self.skymodel = value
            else:
                pass
        return
# ============================================================= #
# ============================================================= #




# ============================================================= #
# -------------------------- Transit -------------------------- #
# ============================================================= #
class Transit(Simu):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ========================================================= #
    # ------------------------ Methods ------------------------ #
    def predict(self, src=None, az=180, time='now', duration=1800, **kwargs):
        """ Compute the next source `src` transit at azimuth `az`
            from `time` for `duration` sec around the transit.

            src: str or tuple
                Name of the source or RA/Dec coordinates 
        """
        self._simukwargs(kwargs)

        duration = TimeDelta(duration, format='sec')

        if src is not None:
            self.t_transit = getTransit(source=src,
                time=time,
                loc='NenuFAR',
                az=az)
            altaz = getAltaz(source=src,
                time=self.t_transit,
                loc='NenuFAR')
            self.azana = altaz.az.deg
            self.elana = altaz.alt.deg
            self.azdig = altaz.az.deg
            self.eldig = altaz.alt.deg

            self.start = self.t_transit - duration/2.
            self.stop = self.t_transit + duration/2.
        else:
            time = getTime(time)
            self.start = time
            self.stop = time + duration

        self._compute_beam()

        self._transit_sky(**kwargs)
        return
    # --------------------------------------------------------- #
    def from_bst(self, bst, dbeam=0, **kwargs):
        """ Do the simulaiton based on an existing BST observation
        """
        self._simukwargs(kwargs)

        if not isinstance(bst, BST):
            bst = BST(bst)
        assert bst.type == 'transit',\
            'Obs is not a transit.'
        bst.select(freq=self.freq, polar=self.polar, dbeam=dbeam)

        self.azana = bst.azana
        self.elana = bst.elana
        self.azdig = bst.azdig
        self.eldig = bst.eldig
        self.nomas = bst.ma
        self.freq = bst.data['freq'][0]
        self.polar = bst.polar
        self._compute_beam()

        self.start = bst.data['time'][0]
        self.stop = bst.data['time'][-1]

        self._transit_sky(**kwargs)
        return
    # --------------------------------------------------------- #
    def plot(self):
        import pylab as plt
        x = (self.simulation['time'] - self.simulation['time'][0]).sec / 60
        if hasattr(self, 't_transit'):
            plt.axvline(x=(self.t_transit - self.simulation['time'][0]).sec/60,
                linestyle='--',
                linewidth=0.8,
                color='black')
        plt.plot(x, 10 * np.log10(self.simulation['amp']))
        plt.xlabel('Time (min since {})'.format(self.simulation['time'][0].iso))
        plt.ylabel('dB')
        plt.show()
        plt.close('all')
        return
# ============================================================= #
# ============================================================= #





# ============================================================= #
# ------------------------- Tracking -------------------------- #
# ============================================================= #
class Tracking(Simu):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ========================================================= #
    # ------------------------ Methods ------------------------ #
    def predict(self, src=None, time='now', duration=1800, **kwargs):
        """ Compute the next source `src` transit at azimuth `az`
            from `time` for `duration` sec around the transit.

            src: str or tuple
                Name of the source or RA/Dec coordinates 
        """
        self._simukwargs(kwargs)

        duration = TimeDelta(duration, format='sec')
        self.start = getTime(time)
        self.stop = self.start + duration

        self._tracking_sky(src=src, **kwargs)
        return
    # --------------------------------------------------------- #
    def from_bst(self, bst, dbeam=0, **kwargs):
        """ Do the simulaiton based on an existing BST observation
        """
        self._simukwargs(kwargs)

        if not isinstance(bst, BST):
            bst = BST(bst)
        assert bst.type == 'tracking',\
            'Obs is not a tracking.'
        bst.select(freq=self.freq, polar=self.polar, dbeam=dbeam)

        self.azana = bst.azana
        self.elana = bst.elana
        self.azdig = bst.azdig
        self.eldig = bst.eldig
        self.nomas = bst.ma
        self.freq = bst.data['freq'][0]
        self.polar = bst.polar

        self.start = bst.data['time'][0]
        self.stop = bst.data['time'][-1]

        self._tracking_sky(obs=bst, **kwargs)
        return
    # --------------------------------------------------------- #
    def plot(self):
        import pylab as plt
        x = (self.simulation['time'] - self.simulation['time'][0]).sec / 60
        plt.plot(x, 10 * np.log10(self.simulation['amp']))
        plt.xlabel('Time (min since {})'.format(self.simulation['time'][0].iso))
        plt.ylabel('dB')
        plt.show()
        plt.close('all')
        return


# ============================================================= #
# ============================================================= #


