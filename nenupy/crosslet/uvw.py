#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***
    UVW
    ***
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'UVW'
]


import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from pyproj import Transformer

from nenupy.instru import ma_info
from nenupy.astro import (
    lst,
    lha,
    toFK5,
    eq_zenith,
    wavelength
)

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ---------------------------- UVW ---------------------------- #
# ============================================================= #
class UVW(object):
    """
    """

    def __init__(self, times, mas, freqs=None):
        self.bsl_xyz = None
        self.times = times
        self.freqs = freqs
        self.mas = mas


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def mas(self):
        return self._mas
    @mas.setter
    def mas(self, m):
        if not isinstance(m, np.ndarray):
            raise TypeError(
                'mas should be a numpy array.'
            )
        ma_pos = np.array([a.tolist() for a in ma_info['pos']])
        available_mas = np.arange(ma_pos.shape[0])
        antpos = ma_pos[np.isin(available_mas, m)]
        # RGF93 to ITRF97
        # See http://epsg.io/?q=itrf97 to find correct EPSG
        t = Transformer.from_crs(
            crs_from='EPSG:2154', # RGF93
            crs_to='EPSG:4896'# ITRF2005
        )
        antpos[:, 0], antpos[:, 1], antpos[:, 2] = t.transform(
            xx=antpos[:, 0],
            yy=antpos[:, 1],
            zz=antpos[:, 2]
        )

        xyz = antpos[..., None]
        xyz = xyz[:, :, 0][:, None]
        # xyz = xyz - xyz.transpose(1, 0, 2)
        xyz = xyz.transpose(1, 0, 2) - xyz
        # self.bsl = xyz[np.triu_indices(m.size)]
        self.bsl = xyz[np.tril_indices(m.size)]
        self._mas = m
        return


    @property
    def times(self):
        """ Times at which the UVW must be computed.

            :setter: Times
            
            :getter: Times
            
            :type: :class:`~astropy.time.Time`
        """
        return self._times
    @times.setter
    def times(self, t):
        if not isinstance(t, Time):
            raise TypeError(
                'times must be a Time instance.'
            )
        self._times = t
        return    


    @property
    def freqs(self):
        return self._freqs
    @freqs.setter
    def freqs(self, f):
        if f is None:
            self._freqs = None
        else:
            if not isinstance(f, u.Quantity):
                f *= u.MHz
            if f.isscalar:
                f = np.array([f.value]) * u.MHz
            self._freqs = f
        return


    @property
    def uvw(self):
        """ UVW in meters.

            :getter: (times, baselines, UVW)
            
            :type: :class:`~numpy.ndarray`
        """
        if not hasattr(self, '_uvw'):
            raise Exception(
                'Run .compute() first.'
            )
        return self._uvw


    @property
    def uvw_wave(self):
        """ UVW in lambdas.

            :getter: (times, freqs, baselines, UVW)
            
            :type: :class:`~numpy.ndarray`
        """
        if not hasattr(self, '_uvw'):
            raise Exception(
                'Run .compute() first.'
            )
        if self.freqs is None:
            raise ValueError(
                'No frequency input, fill self.freqs.'
            )
        lamb = wavelength(self.freqs).value
        na = np.newaxis
        return self._uvw[:, na, :, :]/lamb[na, :, na, na]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self, phase_center=None):
        r""" Compute the UVW at a given ``phase_center`` for all
            the :attr:`~nenupy.crosslet.uvw.UVW.times` and baselines
            formed by :attr:`~nenupy.crosslet.uvw.UVW.mas`.

            :param phase_center: Observation phase center. If
                ``None``, local zenith is considered as phase
                center for all :attr:`~nenupy.crosslet.uvw.UVW.times`.
            :type phase_center: :class:`~astropy.coordinates.SkyCoord`

            UVW are computed such as:

            .. math::
                \pmatrix{
                    u \\
                    v \\
                    w
                } =
                \pmatrix{
                    \sin(h) & \cos(h) & 0\\
                    -\sin(\delta) \cos(h) & \sin(\delta) \sin(h) & \cos(\delta)\\
                    \cos(\delta)\cos(h) & -\cos(\delta) \sin(h) & \sin(\delta)
                }
                \pmatrix{
                    \Delta x\\
                    \Delta y\\
                    \Delta z
                }

            :math:`u`, :math:`v`, :math:`w` are in meters. :math:`h`
            is the hour angle (see :func:`~nenupy.astro.astro.lha`)
            at which the phase center is observed, :math:`\delta`
            is the phase center's declination, :math:`(\Delta x,
            \Delta y, \Delta z)` are the baselines projections
            with the convention of :math:`x` to the South, :math:`y`
            to the East and :math:`z` to :math:`\delta = 90` deg.

            Result of the computation are stored as a :class:`~numpy.ndarray`
            in :attr:`~nenupy.crosslet.uvw.UVW.uvw` whose shape is
            (times, cross-correlations, 3), 3 being :math:`(u, v, w)`.
            """
        # Phase center
        if phase_center is None:
            log.info(
                'UVW phase centered at local zenith.'
            )
            phase_center = eq_zenith(self.times)
        else:
            if not isinstance(phase_center, SkyCoord):
                raise TypeError(
                    'phase_center should be a SkyCoord object'
                )
            if phase_center.isscalar:
                ones = np.ones(self.times.size)
                ra_tab = ones * phase_center.ra
                dec_tab = ones * phase_center.dec
                phase_center = SkyCoord(ra_tab, dec_tab)
            else:
                if phase_center.size != self.times.size:
                    raise ValueError(
                        'Size of phase_center != times'
                    )
            log.info(
                'UVW phase centered at RA={}, Dec={}'.format(
                    phase_center.ra[0].deg,
                    phase_center.dec[0].deg
                )
            )
        # Hour angles
        lstTime = lst(
            time=self.times,
            kind='apparent'
        )
        phase_center = toFK5(
            skycoord=phase_center,
            time=self.times
        )
        ha = lha(
            lst=lstTime,
            skycoord=phase_center
        )

        # Transformations
        self._uvw = np.zeros(
            (
                self.times.size,
                self.bsl.shape[0],
                3
            )
        )
        xyz = np.array(self.bsl).T
        # rot = np.radians(-90) # x to the south, y to the east
        # rotation = np.array(
        #     [
        #         [ np.cos(rot), np.sin(rot), 0],
        #         [-np.sin(rot), np.cos(rot), 0],
        #         [ 0,           0,           1]
        #     ]
        # )
        for i in range(self.times.size):
            sr = np.sin(ha[i].rad)
            cr = np.cos(ha[i].rad)
            sd = np.sin(phase_center.dec[i].rad)
            cd = np.cos(phase_center.dec[i].rad)
            rot_uvw = np.array([
                [    sr,     cr,  0],
                [-sd*cr,  sd*sr, cd],
                [ cd*cr, -cd*sr, sd]
            ])
            # self.uvw[i, ...] = - np.dot(
            #     np.dot(rot_uvw, xyz).T,
            #     rotation
            # )
            self._uvw[i, ...] = - np.dot(rot_uvw, xyz).T
        return


    @classmethod
    def fromCrosslets(cls, crosslet):
        """
        """
        from nenupy.crosslet import XST_Data, TV_Data
        if isinstance(crosslet, (XST_Data, TV_Data)):
            pass
        elif isinstance(crosslet, str):
            if crosslet.endswith('.fits'):
                crosslet = XST_Data(crosslet)
            elif crosslet.endswith('.dat'):
                crosslet = TV_Data(crosslet)
        else:
            raise TypeError(
                'crosslet must be a XST_Data/TV_Data instance or XST/dat file.'
            )
        uvw = cls(
            times=crosslet.times,
            freqs=crosslet.freqs,
            mas=crosslet.mas
        )
        uvw.compute()
        return uvw


    # def rephase(self, ra, dec):
    #     """
    #     """
    #     self._uvw
    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #

# ============================================================= #

