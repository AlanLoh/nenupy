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
from pyproj import Transformer

from nenupy.instru import ma_info
from nenupy.astro import lha, eq_zenith, nenufar_loc

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ---------------------------- UVW ---------------------------- #
# ============================================================= #
class UVW(object):
    """
    """

    def __init__(self, times, freqs, mas):
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
                '`mas should be a numpy array.'
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


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self, phase_center=None):
        r"""
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
        ha = lha(
            time=self.times,
            ra=phase_center.ra.deg
        )
        # Transformations
        self.uvw = np.zeros(
            (
                self.times.size,
                self.bsl.shape[0],
                3
            )
        )
        xyz = np.array(self.bsl).T
        rot = np.radians(-90) # x to the south, y to the east
        rotation = np.array(
            [
                [ np.cos(rot), np.sin(rot), 0],
                [-np.sin(rot), np.cos(rot), 0],
                [ 0,           0,           1]
            ]
        )
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
            self.uvw[i, ...] = - np.dot(
                np.dot(rot_uvw, xyz).T,
                rotation
            )
            # self.uvw[i, ...] = np.dot(rot_uvw, xyz).T
        return


    @classmethod
    def from_tvdata(cls, tvdata):
        """
        """
        from nenupy.crosslet import TV_Data
        if not isinstance(tvdata, TV_Data):
            raise TypeError(
                'tvdata must be a TV_Data instance'
            )
        uvw = cls(
            times=tvdata.times,
            freqs=tvdata.freqs,
            mas=tvdata.mas
        )
        uvw.compute()
        return uvw


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #

# ============================================================= #

