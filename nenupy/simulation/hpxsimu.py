#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ******************************
    NenuFAR Observation Simulation
    ******************************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HpxSimu'
]


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(func):
        """ Overdefine tqdm to return `range` or `enumerate`
        """
        return func
import numpy as np
import numexpr as ne
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS
from astropy.time import Time, TimeDelta

from nenupy.skymodel import HpxGSM
from nenupy.beam import HpxABeam, HpxDBeam
from nenupy.instru import nenufar_loc
from nenupy.beamlet import SData
from nenupy.beamlet import BST_Data

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# -------------------------- HpxSimu -------------------------- #
# ============================================================= #
class HpxSimu(object):
    """
    """

    def __init__(self, freq=50, resolution=1, **kwargs):
        self._gain = None
        self.freq = freq
        self.resolution = resolution
        self._kwargs = kwargs
        self._fill_attr(kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def ma(self):
        return self._ma
    @ma.setter
    def ma(self, m):
        if np.isscalar(m):
            self._gain = HpxABeam(
                resolution=self.resolution,
                **self._kwargs
            )
            log.info(
                'Simulation of analog pointing'
            )
        else:
            if len(m) == 1:
                self._gain = HpxABeam(
                    resolution=self.resolution,
                    **self._kwargs
                )
                log.info(
                    'Simulation of analog pointing'
                )
            else:
                self._gain = HpxDBeam(
                    resolution=self.resolution,
                    **self._kwargs
                )
                log.info(
                    'Simulation of digital pointing'
                )
        self._ma = m
        return


    @property
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, u.Quantity):
            f *= u.MHz
        self._freq = f
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, time, c1, c2, csys='altaz', figname=None, db=True, **kwargs):
        """ Overriding HpxSky plot()
        """
        az, el = self._to_altaz(
            c1=c1,
            c2=c2,
            times=time,
            csys=csys
        )
        # Instanciate the GSM
        gsm = HpxGSM(
            freq=self.freq,
            resolution=self.resolution
        )
        if 'azana' not in kwargs.keys():
            kwargs['azana'] = az[0]
        if 'elana' not in kwargs.keys():
            kwargs['elana'] = el[0]
        self._gain.beam(
            time=time,
            # azana=az[0],
            # elana=el[0],
            azdig=az[0],
            eldig=el[0],
            freq=self.freq,
            **kwargs
        )
        vmask = gsm._is_visible
        self._gain.skymap[vmask] *= gsm.skymap[vmask]
        self._gain.plot(
            figname=figname,
            db=db,
            **kwargs
        )
        return


    def time_profile(self, times, c1, c2, csys='altaz', **kwargs):
        """ Simulate a NenuFAR time-profile.

            :param times:
                Time (scalar or array) of the simulation
            :type times: :class:`astropy.time.Time` 
            :param c1:
                First coordinate (scalar or array) of the
                miniarray pointing (either RA or Az) in deg.
            :type c1: `np.ndarray` or `float`
            :param c2:
                Second coordinate (scalar or array) of the
                miniarray pointing (either Dec or Alt) in deg.
            :type c2: `np.ndarray` or `float`
            :param csys:
                Coordinate system used to understand `c1` and `c2`
                Either `'altaz'` or `'radec'`
                Default: `'altaz'`

            :returns: (times, amplitudes)
            :rtype: (:class:`astropy.time.Time`, `np.ndarray`)
        """
        amp_list = []
        time_list = []
        az, el = self._to_altaz(
            c1=c1,
            c2=c2,
            times=times,
            csys=csys
        )
        # Instanciate the GSM
        gsm = HpxGSM(
            freq=self.freq,
            resolution=self.resolution
        )
        # Loop over times
        for i, time in enumerate(tqdm(times)):
            if 'azana' not in kwargs.keys():
                kwargs['azana'] = az[i]
            if 'elana' not in kwargs.keys():
                kwargs['elana'] = el[i]
            self._gain.beam(
                time=time,
                # azana=az[i],
                # elana=el[i],
                azdig=az[i],
                eldig=el[i],
                freq=self.freq,
                **kwargs
            )
            # Rotate the HPX mask of the GSM
            gsm.time = time
            # Multiply and sum GMS and Beam
            vmask = gsm._is_visible
            gsmcut = gsm.skymap[vmask]
            beamcut = self._gain.skymap[vmask]
            amp_list.append(
                ne.evaluate(
                    'sum(gsmcut*beamcut)'
                )
            )
            time_list.append(time.mjd)
        return Time(time_list, format='mjd'), np.array(amp_list)


    def time_profile2(self, times, anadir, digdir=None, **kwargs):
        """
        """
        amp_list = []
        time_list = []
        # Make sure kwargs is emptied from some keywords
        for key in ['azana', 'elana', 'azdig', 'eldig']:
            try:
                del kwargs[key]
            except KeyError:
                pass
        # Instanciate the GSM
        gsm = HpxGSM(
            freq=self.freq,
            resolution=self.resolution
        )
        # Loop over times
        for i, time in enumerate(tqdm(times)):
            log.debug(
                'ana=({}; {}) dig=({}, {})'.format(
                   anadir[i].az.deg,
                   anadir[i].alt.deg,
                   None if digdir is None else digdir[i].az.deg,
                   None if digdir is None else digdir[i].alt.deg
                )
            )
            self._gain.beam(
                time=time,
                azana=anadir[i].az.deg,
                elana=anadir[i].alt.deg,
                azdig=None if digdir is None else digdir[i].az.deg,
                eldig=None if digdir is None else digdir[i].alt.deg,
                freq=self.freq,
                **kwargs
            )
            # Rotate the HPX mask of the GSM
            gsm.time = time
            # Multiply and sum GMS and Beam
            vmask = gsm._is_visible
            gsmcut = gsm.skymap[vmask]
            beamcut = self._gain.skymap[vmask]
            amp_list.append(
                ne.evaluate(
                    'sum(gsmcut*beamcut)'
                )
            )
            time_list.append(time.mjd)
        return SData(
            data=np.expand_dims(np.array(amp_list), axis=(1, 2)),
            time=Time(time_list, format='mjd'),
            freq=self.freq,
            polar=self._gain.polar
        )

    
    def azel_transit(self, az, el, t0, dt, duration, **kwargs):
        """ Simulate a transit at coordinates (`az`, `el`)
            at time `t0`.
            The beam is fixed in AltAz coordinates.
            
            :param az:
                Azimuth of transit in degrees
            :type az: `float`
            :param el:
                Elevation of transit in degrees
            :type el: `float`
            :param t0:
                Time of transit
            :type t0: :class:`astropy.time.Time` 
            :param dt:
                Time resolution of simulation
            :type dt: :class:`astropy.time.DeltaTime`
            :param duration:
                Duration of observation simulation
            :type duration: :class:`astropy.DeltaTime.Time`

            :returns: (times, amplitudes)
            :rtype: (:class:`astropy.time.Time`, `np.ndarray`)
        """
        tmin = t0 - duration/2
        tmax = t0 + duration/2
        times = tmin + dt * np.arange(int(duration/dt) + 1)
        az = np.ones(times.size) * az
        el = np.ones(times.size) * el
        return self.time_profile(
            times=times,
            c1=az,
            c2=el,
            csys='altaz',
            **kwargs
        )


    def radec_transit(self, ra, dec, t0, dt, duration, **kwargs):
        """ Simulate a transit at coordinates (`ra`, `dec`)
            at time `t0`.
            The beam is fixed in AltAz coordinates.

            :param ra:
                Right Ascension of transit in degrees
            :type ra: `float`
            :param dec:
                Declination of transit in degrees
            :type dec: `float`
            :param t0:
                Time of transit
            :type t0: :class:`astropy.time.Time` 
            :param dt:
                Time resolution of simulation
            :type dt: :class:`astropy.time.DeltaTime`
            :param duration:
                Duration of observation simulation
            :type duration: :class:`astropy.DeltaTime.Time`

            :returns: (times, amplitudes)
            :rtype: (:class:`astropy.time.Time`, `np.ndarray`)
        """
        az, el = self._to_altaz(
            c1=ra,
            c2=dec,
            times=t0,
            csys='radec'
        )
        return self.azel_transit(
            az=az[0],
            el=el[0],
            t0=t0,
            dt=dt,
            duration=duration,
            **kwargs
        )


    def radec_tracking(self, ra, dec, t0, dt, duration, **kwargs):
        """ Simulate a tracking at coordinates (`ra`, `dec`)
            at time `t0`.
            The beam is fixed in RADec coordinates.

            :param ra:
                Right Ascension of transit in degrees
            :type ra: `float`
            :param dec:
                Declination of transit in degrees
            :type dec: `float`
            :param t0:
                Start time of simulation
            :type t0: :class:`astropy.time.Time` 
            :param dt:
                Time resolution of simulation
            :type dt: :class:`astropy.time.DeltaTime`
            :param duration:
                Duration of observation simulation
            :type duration: :class:`astropy.DeltaTime.Time`

            :returns: (times, amplitudes)
            :rtype: (:class:`astropy.time.Time`, `np.ndarray`)
        """
        tmax = t0 + duration
        times = t0 + dt * np.arange(int(duration/dt) + 1)
        ra = np.ones(times.size) * ra
        dec = np.ones(times.size) * dec
        return self.time_profile(
            times=times,
            c1=ra,
            c2=dec,
            csys='radec',
            **kwargs
        )


    @classmethod
    def from_bst(cls, bstdata, resolution=1, dt=None, **kwargs):
        """
        """
        if not isinstance(bstdata, BST_Data):
            raise TypeError(
                'bstdata should be a BST_Data instance'
            )
        if bstdata.freq.size != 1:
            raise ValueError(
                (
                    'Multiple frequencies found, select only one'
                    'using e.g. BST_Data.select(freqrange=50)'
                )
            )
        if (dt is None) or (not isinstance(dt, TimeDelta)):
            raise TypeError(
                'dt should be a TimeDelta object'
            )
        # Prepare simulation properties
        kwargs['ma'] = bstdata.mas
        kwargs['polar'] = bstdata.polar
        times = bstdata.t_min + np.arange(
            int((bstdata.t_max - bstdata.t_min)/dt) + 1
        )*dt
        anaidx = np.searchsorted(
            bstdata.utana[:, 0],
            times,
            side='right'
        ) - 1
        digidx = np.searchsorted(
            bstdata.utdig[:, 0],
            times,
            side='right'
        ) - 1
        # Instanciate simulation object
        simu = cls(
            freq=bstdata.freq[0].value,
            resolution=resolution,
            **kwargs
        )
        anacoord = AltAz(
            az=bstdata.azana[anaidx],
            alt=bstdata.elana[anaidx],
            obstime=times,
            location=nenufar_loc
        )
        digcoord = AltAz(
            az=bstdata.azdig[digidx],
            alt=bstdata.eldig[digidx],
            obstime=times,
            location=nenufar_loc
        )
        return simu.time_profile2(
            times=times,
            anadir=anacoord,
            digdir=digcoord,
            **kwargs
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _to_altaz(self, c1, c2, times, csys):
        """
        """
        # Make sure to have corresponding dimensions
        elements = [times.mjd, c1, c2]
        if all(map(np.isscalar, elements)):
            time = [times]
            c1 = [c1]
            c2 = [c2]
        elif all(hasattr(i, '__len__') for i in elements):
            if len(set(len(i) for i in elements)) == 1:
                pass
            else:
                raise ValueError(
                    'Different lengths found'
                )
        else:
            raise ValueError(
                'times, c1 and c2 are mixed with scalars and lists'
            )
        # Convert coordinates
        if csys.lower() == 'radec':
            radec = ICRS(
                ra=c1*u.deg,
                dec=c2*u.deg
            )
            altaz_frame = AltAz(
                obstime=times,
                location=nenufar_loc
            )
            
            altaz = radec.transform_to(altaz_frame)
            az = altaz.az.deg
            el = altaz.alt.deg
        elif csys.lower() == 'altaz':
            az = c1
            el = c2
        else:
            raise ValueError(
                '{} not understood'.format(csys)
            )
        return az, el


    def _fill_attr(self, kwargs):
        """
        """
        def_vals = {
            'ma': 0,
            **kwargs
        } 
        for key, val in def_vals.items():
            if hasattr(self, key) and (key not in kwargs.keys()):
                continue
            setattr(self, key, val)
        return
# ============================================================= #
