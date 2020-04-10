#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Crosslet
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Crosslet'
]


import numpy as np
import astropy.units as un
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(func):
        """ Overdefine tqdm to return `range` or `enumerate`
        """
        return func

from nenupy.astro import wavelength, HpxSky, eq_zenith
from nenupy.instru import nenufar_loc, read_cal_table
from nenupy.beam import ma_pos
from nenupy.crosslet import UVW
from nenupy.beamlet.sdata import SData

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- Functions ------------------------- #
# ============================================================= #
try:
    from numba import jit
except ModuleNotFoundError:
    from functools import wraps
    def jit(**kwargs):
        """ Override numba.jit function by defining a decorator
            which does nothing but returning the decorated
            function.
        """
        def inner_function(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                function(*args, **kwargs)
            return wrapper
        return inner_function
    log.warning(
        'numba module not found, operations will take longer.'
    )

@jit(nopython=True, parallel=True, fastmath=True)
def ft_mul(x, y):
    return x * y

@jit(nopython=True, parallel=True, fastmath=True)
def ft_phase(ul, vm):
     return np.exp(
         2.j * np.pi *(ul + vm)
     )

@jit(nopython=True, parallel=True, fastmath=True)
def ft_sum(vis, exptf):
    return np.mean(
        vis * exptf,
    )
# ============================================================= #


# ============================================================= #
# ------------------------- Crosslet -------------------------- #
# ============================================================= #
class Crosslet(object):
    """
    """

    def __init__(self):
        self.freqs = None
        self.mas = None
        self.dt = None
        self.times = None
        self.vis = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def mas(self):
        return self._mas
    @mas.setter
    def mas(self, m):
        if m is not None:
            self._mas_idx = np.arange(m.size)
            self._ant1, self._ant2 = np.tril_indices(m.size, 0)
        self._mas = m
        return


    @property
    def xx(self):
        """ Cross correlation XX
        """
        xx = self.vis[..., self._get_cross_idx('X', 'X')]
        log.info(
            'XX loaded.'
        )
        return xx


    @property
    def yy(self):
        """ Cross correlation YY
        """
        yy = self.vis[..., self._get_cross_idx('Y', 'Y')]
        log.info(
            'YY loaded.'
        )
        return yy


    @property
    def yx(self):
        """ Cross correlation YX
        """
        yx = self.vis[..., self._get_cross_idx('Y', 'X')]
        log.info(
            'YX loaded.'
        )
        return yx


    @property
    def xy(self):
        """ Cross correlation XY
        """
        # Deal with lack of auto XY cross in XST-like data
        ma1, ma2 = np.tril_indices(self.mas.size, 0)
        auto = ma1 == ma2
        cross = ~auto
        _xy = np.zeros(
            (list(self.yx.shape[:-1]) + [ma1.size]),
            dtype=np.complex
        )
        _xy[..., auto] = self.yx[..., auto].conj()
        # Get XY correlations
        indices = self._get_cross_idx('X', 'Y')
        _xy[..., cross] = self.vis[..., indices]
        log.info(
            'XY loaded.'
        )
        return _xy


    @property
    def stokes_i(self):
        """ Stokes I
        """
        return 0.5*(self.xx + self.yy)


    @property
    def mask_auto(self):
        """
        """
        ma1, ma2 = np.tril_indices(self.mas.size, 0)
        auto = ma1 == ma2
        return ~auto
    


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    # def plot(self):
    #     """
    #     """
    #     mat = np.zeros(
    #         (self.mas.size, self.mas.size),
    #         dtype=np.float
    #     )
    #     mat[np.tril_indices(self.mas.size, 0)] = np.abs(self.xy[0, 0, :])
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.imshow(mat, origin='lower')
    #     return


    def beamform(self, az, el, pol='NW', ma=None, calibration='default'):
        """ Beamform the crosslets in the direction (az, el).
        """
        log.info(
            'Beamforming towards az={}, el={}, pol={}'.format(
                az,
                el,
                pol
            )
        )
        # Mini-Array selection
        if ma is None:
            ma = self.mas.copy()
        mas = self._mas_idx[np.isin(self.mas, ma)]
        # Calibration table
        if calibration.lower() == 'none':
            # No calibration
            cal = np.ones(
                (self.sb_idx.size, mas.size)
            )
        else:
            pol_idx = {'NW': [0], 'NE': [1]}
            cal = read_cal_table(
                calfile=calibration
            )
            cal = cal[np.ix_(
                self.sb_idx,
                mas,
                pol_idx[pol]
            )].squeeze()
        # Matrix of BSTs
        c = np.zeros(
            (
                self.times.size,
                self.freqs.size,
                mas.size,
                mas.size
            ),
            dtype=np.complex
        )
        # Matrix of phasings
        p = np.ones(
            (
                self.freqs.size,
                mas.size,
                mas.size
            ),
            dtype=np.complex
        )
        # Pointing direction
        if isinstance(az, un.Quantity):
            az = az.to(un.deg).value
        if isinstance(el, un.Quantity):
            el = el.to(un.deg).value
        az = np.radians(az)
        el = np.radians(el)
        u = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])
        # Polarization selection
        mask = np.isin(self._ant1, mas) & np.isin(self._ant2, mas)
        log.info(
            'Loading data...'
        )
        if pol.upper() == 'NW':
            cpol = self.xx[..., mask]
        else:
            cpol = self.yy[..., mask]
        log.info(
            'Data of shape {} loaded for beamforming'.format(
                cpol.shape
            )
        )
        # Put the Xcorr in a matrix
        trix, triy = np.tril_indices(mas.size, 0)
        c[:, :, trix, triy] = cpol
        c[:, :, triy, trix] = c[:, :, trix, triy].conj()
        # Calibrate the Xcorr with the caltable
        for fi in range(c.shape[1]):
            cal_i = np.expand_dims(cal[fi], axis=1)
            cal_i_h = np.expand_dims(cal[fi].T.conj(), axis=0)
            mul = np.matrix(cal_i) * np.matrix(cal_i_h)
            c[:, fi, ...] *= mul[np.newaxis, ...] 
        # Phase the Xcorr
        dphi = np.dot(
            ma_pos[self._ant1[mask]] - ma_pos[self._ant2[mask]],
            u
        )
        wavel = wavelength(self.freqs).value
        p[:, trix, triy] = np.exp(
            -2.j*np.pi/wavel[:, None] * dphi
        )
        p[:, triy, trix] = p[:, trix, triy].conj()
        data = np.sum((c * p).real, axis=(2, 3))

        return SData(
            data=np.expand_dims(data, axis=2),
            time=self.times,
            freq=self.freqs,
            polar=np.array([pol.upper()])
        )


    def image(self, resolution=1, fov=50):
        """
        """
        if not isinstance(fov, un.Quantity):
            fov *= un.deg
        f_idx = 0 # Frequency index
        # Sky preparation
        sky = HpxSky(resolution=resolution)
        exposure = self.times[-1] - self.times[0]
        sky.time = self.times[0] + exposure/2.
        sky._is_visible = sky._ho_coords.alt >= 90*un.deg - fov/2.
        phase_center = eq_zenith(sky.time)
        l, m, n = sky.lmn(phase_center=phase_center)
        # UVW coordinates
        uvw = UVW.from_tvdata(self)
        u = np.mean( # Mean in time
            uvw.uvw[..., 0],
            axis=0
        )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        v = np.mean(
            uvw.uvw[..., 1],
            axis=0
        )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        # Mulitply (u, v) by (l, m) and compute FT exp
        ul = ft_mul(
            x=np.tile(u, (l.size, 1)).T,
            y= np.tile(l, (u.size, 1))
        )
        vm = ft_mul(
            x=np.tile(v, (m.size, 1)).T,
            y=np.tile(m, (v.size, 1))
        )
        phase = ft_phase(ul, vm)
        # Phase visibilities
        vis = np.mean( # Mean in time
            self.stokes_i,
            axis=0
        )[f_idx, :][self.mask_auto]
        im = np.zeros(l.size)
        for i in tqdm(range(l.size)):
            im[i] = np.real(
                ft_sum(vis, phase[:, i])
            )
        sky.skymap[sky._is_visible] = im
        return sky


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_cross_idx(self, c1='X', c2='X'):
        """
        """
        corr = np.array(['X', 'Y']*self.mas.size)
        i_ant1, i_ant2 = np.tril_indices(self.mas.size*2, 0)
        corr_mask = (corr[i_ant1] == c1) & (corr[i_ant2] == c2)
        indices = np.arange(i_ant1.size)[corr_mask]
        return indices
# ============================================================= #

