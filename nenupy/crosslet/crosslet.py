#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Crosslet
    ********

    :class:`~nenupy.crosslet.crosslet.Crosslet` is the main class
    for both :class:`~nenupy.crosslet.xstdata.XST_Data` and
    :class:`~nenupy.crosslet.tvdata.TV_Data`, which inherit from
    it.

    This enables *beamforming* with the 
    :meth:`~nenupy.crosslet.crosslet.Crosslet.beamform` method
    (see also :ref:`tuto_beamforming` for a detailed tutorial),
    *imaging* with the
    :meth:`~nenupy.crosslet.crosslet.Crosslet.image` method
    and *near-field computing* with the
    :meth:`~nenupy.crosslet.crosslet.Crosslet.nearfield` method
    (see also :ref:`tuto_tv` for a detailed tutorial) from 
    cross-correlation statistics data.

    Any :class:`~nenupy.crosslet.crosslet.Crosslet` object can be
    sub-selected both in time (:attr:`~nenupy.crosslet.crosslet.Crosslet.timeRange`)
    and frequency (:attr:`~nenupy.crosslet.crosslet.Crosslet.freqRange`)
    prior to applying any operation, e.g.:

    .. code-block:: python
        :emphasize-lines: 9, 17

        >>> from nenupy.crosslet import XST_Data
        
        >>> xst = XST_Data('/path/to/XST.fits')
        
        >>> print(xst.freqMin, xst.freqMax)
        68.5546875 MHz 79.296875 MHz
        >>> xst.freqs
        [68.554688, 73.242188, 73.4375, ..., 79.296875] MHz
        >>> xst.freqRange = [73.1, 73.3]
        >>> xst.freqs
        [73.242188] MHz
        
        >>> print(xst.timeMin.isot, xst.timeMax.isot)
        2020-07-08T12:00:01 2020-07-08T12:00:30
        >>> xst.times
        array(['2020-07-08T12:00:01', '2020-07-08T12:00:02', ...])
        >>> xst.timeRange = ['2020-07-08T12:00:01', '2020-07-08T12:00:01.5']
        >>> xst.times
        array(['2020-07-08T12:00:01'])

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
import os
import astropy.units as un
from astropy.time import Time
from astropy.coordinates import SkyCoord
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(func):
        """ Overdefine tqdm to return `range` or `enumerate`
        """
        return func

from nenupy.astro import (
    wavelength,
    HpxSky,
    eq_zenith,
    toAltaz,
    l93_to_etrs,
    etrs_to_enu,
    getSource
)
from nenupy.instru import nenufar_loc, read_cal_table, ma_pos, getMAL93
from nenupy.crosslet import UVW, NearField, NenuFarTV
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
def ft_phase(ul, vm, wn):
    return np.exp(
        - 2.j * np.pi * (ul + vm + wn)
    )

@jit(nopython=True, parallel=True, fastmath=True)
def ft_sum(vis, exptf):
    return np.mean(
        vis * exptf,
    )

@jit(nopython=True, parallel=True, fastmath=True)
def ft_delay(delaylamb):
    return np.exp(
        2.j * np.pi * (delaylamb)
    )
# ============================================================= #


# ============================================================= #
# ------------------------- Crosslet -------------------------- #
# ============================================================= #
class Crosslet(object):
    """ :class:`~nenupy.crosslet.crosslet.Crosslet` class is not
        designed to be called directly but rather as a base class
        for both :class:`~nenupy.crosslet.xstdata.XST_Data` and
        :class:`~nenupy.crosslet.tvdata.TV_Data` (see their 
        related documentation for further instructions on how
        to load these data sets).
    """

    def __init__(self):
        self.freqs = None
        self.mas = None
        self.dt = None
        self.times = None
        self.vis = None
        self.phaseCenter = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def mas(self):
        """ Mini-Arrays used to get the cross-correlation
            statistics data.

            :setter: Mini-Array list
            
            :getter: Mini-Array list
            
            :type: :class:`~numpy.ndarray`
        """
        return self._mas
    @mas.setter
    def mas(self, m):
        if m is not None:
            self._mas_idx = np.arange(m.size)
            self._ant1, self._ant2 = np.tril_indices(m.size, 0)
        self._mas = m
        return


    @property
    def times(self):
        """ Times of the cross-correlation recorded.
            
            :setter: Array of selected times.
            
            :getter: Array of times.
            
            :type: :class:`~astropy.time.Time`
        """
        return self._times[self._tMask]
    @times.setter
    def times(self, t):
        if t is None:
            self._times = t
        else:
            if not isinstance(t, Time):
                raise TypeError(
                    'times should be an astropy Time instance.'
                )
            self.timeMin = t[0].copy()
            self.timeMax = t[-1].copy()
            self._times = t
            self.timeRange = [self.timeMin, self.timeMax]


    @property
    def freqs(self):
        """ Frequencies of the cross-correlation recorded.
            
            :setter: Array of selected frequencies.
            
            :getter: Array of frequencies.
            
            :type: :class:`~astropy.units.Quantity`
        """
        return self._freqs[self._fMask]
    @freqs.setter
    def freqs(self, f):
        if f is None:
            self._freqs = f
        else:
            if not isinstance(f, un.Quantity):
                raise TypeError(
                    'freqs should be an astropy Quantity instance.'
                )
            self.freqMin = f.min()
            self.freqMax = f.max()
            self._freqs = f
            self.freqRange = [self.freqMin.value, self.freqMax.value]


    @property
    def timeRange(self):
        """
        """
        return self._timeRange
    @timeRange.setter
    def timeRange(self, t):
        if t is None:
            t = [self.timeMin.isot, self.timeMax.isot]
        if not isinstance(t, Time):
            t = Time(t)
        if t.isscalar:
            dt_sec = (self._times - t).sec
            self._tMask = [np.argmin(np.abs(dt_sec))]
        else:
            if len(t) != 2:
                raise ValueError(
                    'timerange should be of size 2'
                )
            self._tMask = (self._times >= t[0]) & (self._times <= t[1])
            if not any(self._tMask):
                log.warning(
                    (
                        'Empty time selection, time should fall '
                        'between {} and {}'.format(
                            self.timeMin.isot,
                            self.timeMax.isot
                        )
                    )
                )
        self._timeRange = t


    @property
    def freqRange(self):
        """
        """
        return self._freqRange
    @freqRange.setter
    def freqRange(self, f):
        if f is None:
            f = np.array(
                [self.freqMin.value, self.freqMax.value]
            )*un.MHz
        if not isinstance(f, un.Quantity):
            f *= un.MHz
        else:
            f.to(un.MHz)
        if f.isscalar:
            self._fMask = [np.argmin(np.abs(self._freqs - f))]
        else:
            if len(f) != 2:
                raise ValueError(
                    'freqrange should be of size 2'
                )
            self._fMask = (self._freqs >= f[0]) & (self._freqs <= f[1])
            if not any(self._fMask):
                log.warning(
                    (
                        'Empty freq selection, freq should fall '
                        'between {} and {}'.format(
                            self.freqMin.min(),
                            self.freqMax.max()
                        )
                    )
                )
        self._freqRange = f
        return


    @property
    def phaseCenter(self):
        """
        """
        if self._phaseCenter is None:
            return eq_zenith(self.times)
        else:
            return self._phaseCenter
    @phaseCenter.setter
    def phaseCenter(self, pc):
        if pc is not None:
            if not isinstance(pc, SkyCoord):
                raise TypeError(
                    'phaseCenter should ba an astropy SkyCoord instance.'
                )
        self._phaseCenter = pc


    @property
    def xx(self):
        """ Extracts the XX polarization from the cross-correlation
            statistics data. Also refered to as ``'NW'``.
            Array is shaped like (time, frequencies, baselines).

            :getter: XX polarization
            
            :type: :class:`~numpy.ndarray`
        """
        xx = self.vis[
            np.ix_(
                self._tMask,
                self._fMask,
                self._get_cross_idx('X', 'X')
            )
        ]
        log.info(
            'XX loaded.'
        )
        return xx


    @property
    def yy(self):
        """ Extracts the YY polarization from the cross-correlation
            statistics data. Also refered to as ``'NE'``.
            Array is shaped like (time, frequencies, baselines).

            :getter: YY polarization
            
            :type: :class:`~numpy.ndarray`
        """
        yy = self.vis[
            np.ix_(
                self._tMask,
                self._fMask,
                self._get_cross_idx('Y', 'Y')
            )
        ]
        log.info(
            'YY loaded.'
        )
        return yy


    @property
    def yx(self):
        """ Extracts the YX polarization from the cross-correlation
            statistics data.
            Array is shaped like (time, frequencies, baselines).

            :getter: YX polarization
            
            :type: :class:`~numpy.ndarray`
        """
        yx = self.vis[
            np.ix_(
                self._tMask,
                self._fMask,
                self._get_cross_idx('Y', 'X')
            )
        ]
        log.info(
            'YX loaded.'
        )
        return yx


    @property
    def xy(self):
        """ Extracts the XY polarization from the cross-correlation
            statistics data. This polarization is not recorded by
            default for the NenuFAR auto-correlations. However
            this can be computed as it is the complex conjugate
            of :attr:`~nenupy.crosslet.crosslet.Crosslet.yx`
            auto-correlations.
            Array is shaped like (time, frequencies, baselines).

            :getter: XY polarization
            
            :type: :class:`~numpy.ndarray`
        """
        # Deal with lack of auto XY cross in XST-like data
        ma1, ma2 = np.tril_indices(self.mas.size, 0)
        auto = ma1 == ma2
        cross = ~auto
        yx = self.yx
        _xy = np.zeros(
            (list(yx.shape[:-1]) + [ma1.size]),
            dtype=np.complex
        )
        _xy[:, :, auto] = yx[:, :, auto].conj()
        # Get XY correlations
        _xy[:, :, cross] = self.vis[
            np.ix_(
                self._tMask,
                self._fMask,
                self._get_cross_idx('X', 'Y')
            )
        ]
        log.info(
            'XY loaded.'
        )
        return _xy


    @property
    def stokes_i(self):
        r""" Computes the stokes parameter I from the
            cross-correlation statistics data using
            :attr:`~nenupy.crosslet.crosslet.Crosslet.xx` and 
            :attr:`~nenupy.crosslet.crosslet.Crosslet.yy`.
            
            .. math::
                \mathbf{I}(t, \nu) = \frac{1}{2} \left[
                    \mathbf{X\overline{X}}(t, \nu) + \mathbf{Y\overline{Y}}(t, \nu)
                \right]

            Array is shaped like (time, frequencies, baselines).

            :getter: Stokes I data
            
            :type: :class:`~numpy.ndarray`
        """
        return 0.5*(self.xx + self.yy)


    @property
    def stokes_q(self):
        r""" Computes the stokes parameter Q from the
            cross-correlation statistics data using
            :attr:`~nenupy.crosslet.crosslet.Crosslet.xx` and 
            :attr:`~nenupy.crosslet.crosslet.Crosslet.yy`.
            
            .. math::
                \mathbf{I}(t, \nu) = \frac{1}{2} \left[
                    \mathbf{X\overline{X}}(t, \nu) - \mathbf{Y\overline{Y}}(t, \nu)
                \right]

            Array is shaped like (time, frequencies, baselines).

            :getter: Stokes Q data
            
            :type: :class:`~numpy.ndarray`
        """
        return 0.5*(self.xx - self.yy)


    @property
    def stokes_u(self):
        r""" Computes the stokes parameter U from the
            cross-correlation statistics data using
            :attr:`~nenupy.crosslet.crosslet.Crosslet.xy` and 
            :attr:`~nenupy.crosslet.crosslet.Crosslet.yx`.
            
            .. math::
                \mathbf{U}(t, \nu) = \frac{1}{2} \left[
                    \mathbf{X\overline{Y}}(t, \nu) + \mathbf{Y\overline{X}}(t, \nu)
                \right]

            Array is shaped like (time, frequencies, baselines).

            :getter: Stokes U data
            
            :type: :class:`~numpy.ndarray`
        """
        return 0.5*(self.xy + self.yx)


    @property
    def stokes_v(self):
        r""" Computes the stokes parameter V from the
            cross-correlation statistics data using
            :attr:`~nenupy.crosslet.crosslet.Crosslet.xy` and 
            :attr:`~nenupy.crosslet.crosslet.Crosslet.yx`.
            
            .. math::
                \mathbf{V}(t, \nu) = -\frac{i}{2} \left[
                    \mathbf{X\overline{Y}}(t, \nu) - \mathbf{Y\overline{X}}(t, \nu)
                \right]

            Array is shaped like (time, frequencies, baselines).

            :getter: Stokes V data
            
            :type: :class:`~numpy.ndarray`
        """
        return -0.5j * (self.xy - self.yx)


    @property
    def stokes_fl(self):
        r""" Computes the fractional linear polarization from
            the cross-correlation statistics Stokes parameters
            :attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_u`,
            :attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_q`,
            :attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_i`.

            .. math::
                \mathbf{P}(t, \nu) = \frac{ \sqrt{
                    \mathbf{U}^2(t, \nu) + \mathbf{Q}^2(t, \nu)
                } }{\mathbf{I}(t, \nu)}

            Array is shaped like (time, frequencies, baselines).

            :getter: Fractional linear polarization
            
            :type: :class:`~numpy.ndarray`
        """
        return np.sqrt(self.stokes_q**2 + self.stokes_u**2) / self.stokes_i


    @property
    def stokes_fv(self):
        r""" Computes the fractional circular polarization from
            the cross-correlation statistics Stokes parameters
            :attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_v`,
            :attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_i`.

            .. math::
                \mathbf{L}(t, \nu) = \frac{
                    \| \mathbf{V}(t, \nu) \|
                }{\mathbf{I}(t, \nu)}

            Array is shaped like (time, frequencies, baselines).

            :getter: Fractional circular polarization
            
            :type: :class:`~numpy.ndarray`
        """
        return np.abs(self.stokes_v)/self.stokes_i


    @property
    def mask_auto(self):
        """ Masks auto-correlations. In particular, this is used 
            to compute the image (see 
            :meth:`~nenupy.crosslet.crosslet.Crosslet.image`).

            :getter: Auto-correlation mask
            
            :type: :class:`~numpy.ndarray`
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
        r""" Converts cross correlation statistics data XST, 
            :math:`\mathbf{X}(t, \nu)`, in beamformed data BST,
            :math:`B(t, \nu)`, where :math:`t` and :math:`\nu` are
            the time and the frequency respectively.
            :math:`\mathbf{X}(t, \nu)` is a subset of XST data at
            the required polarization ``pol``.

            This is done for a given phasing direction in local
            sky coordinates :math:`\varphi` (azimuth, ``az``) and
            :math:`\theta` (elevation, ``el``), with a selection
            of Mini-Arrays ``ma`` (numbered :math:`a`).

            .. math::
                B (t, \nu) = \operatorname{Re} \left\{
                \sum
                    \left[
                        \underset{\scriptscriptstyle a \times 1}{\mathbf{C}}
                        \cdot
                        \underset{\scriptscriptstyle 1 \times a}{\mathbf{C}^{H}}
                    \right](\nu)
                    \cdot
                    \underset{\scriptscriptstyle a \times a}{\mathbf{X}} (t, \nu)
                    \cdot
                    \left[
                        \underset{\scriptscriptstyle a \times 1}{\mathbf{P}}
                        \cdot
                        \underset{\scriptscriptstyle 1 \times a}{\mathbf{P}^{H}}
                    \right](\nu)
                \right\}

            .. math::
                \rm{with} \quad
                \cases{
                    \mathbf{C}(\nu ) = e^{2 \pi i \nu \mathbf{ t }} \quad \rm{the~calibration~ file}\\
                    \mathbf{P} (\nu) = e^{-2 \pi i \frac{\nu}{c} (\mathbf{b} \cdot \mathbf{u})} \quad \rm{phasing}\\
                    \underset{\scriptscriptstyle a \times 3}{\mathbf{b}} = \mathbf{a}_{1} - \mathbf{a}_{2} \quad \rm{baseline~positions}\\
                    \underset{\scriptscriptstyle 3 \times 1}{\mathbf{u}} = \left[
                        \cos(\theta)\cos(\varphi),
                        \cos(\theta)\sin(\varphi),
                        \sin(\theta) \right]
                }

            :param az:
                Azimuth coordinate used for beamforming (default
                unit is degrees in `float` input).
            :type az: `float` or :class:`~astropy.units.Quantity`
            :param el:
                Elevation coordinate used for beamforming (default
                unit is degrees in `float` input).
            :type el: `float` or :class:`~astropy.units.Quantity`
            :param pol:
                Polarization (either ``'NW'`` or ``'NE'``.
            :type pol: `str`
            :param ma:
                Subset of Mini-Arrays (minimum 2) used for
                beamforming.
            :type ma: `list` or :class:`~numpy.ndarray`
            :param calibration:
                Antenna delay calibration file (i.e, :math:`\mathbf{C}`).
                If ``'none'``, no calibration is applied.
                If ``'default'``, the standard calibration file is
                used, otherwise the calibration file name should
                be given (see also :func:`~nenupy.instru.instru.read_cal_table`).
            :type calibration: `str`

            :returns: Beamformed data.
            :rtype: :class:`~nenupy.beamlet.sdata.SData`

            :Example:
                >>> from nenupy.crosslet import XST_Data
                >>> xst = XST_Data('20191129_141900_XST.fits')
                >>> bf = xst.beamform(
                        az=180,
                        el=90,
                        pol='NW',
                        ma=[17, 44],
                        calibration='default'
                    )

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
            cpol = self.xx[:, :, mask]
        else:
            cpol = self.yy[:, :, mask]
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
            mul = np.dot(cal_i, cal_i_h)
            c[:, fi, :, :] *= mul[np.newaxis, :, :]
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
        log.info(
            'Beamforming complete.'
        )
        return SData(
            data=np.expand_dims(data, axis=2),
            time=self.times,
            freq=self.freqs,
            polar=np.array([pol.upper()])
        )


    def image(self, resolution=1, fov=50, center=None, stokes='I'):
        r""" Converts NenuFAR-TV-like data sets containing
            visibilities (:math:`\mathcal{V}(u,v,w, \nu , t)`) into images
            :math:`I(l, m, \nu)` phase-centered at the local
            zenith while time averaging the visibilities.
            The Field of View ``fov`` argument defines the
            diameter angular size (zenith-centered) beyond which
            the image is not computed.
            
            .. math::
                I(l, m, \nu) = \int
                    \langle \mathcal{V}(u, v, w, \nu, t) \rangle_t e^{
                        -2 \pi i \frac{\nu}{c} \left[
                            \langle u(t) \rangle_t l + \langle v(t) \rangle_t m + \langle w(t) \rangle_t (n - 1)
                        \right]
                    }
                    \, du \, dv\, dw

            .. note::
                If re-phasing is selected towards a sky direction
                :math:`(\alpha, \delta)` (right-ascension and declination)
                other than the local zenith (i.e. ``center != None``),
                the cross-correlation data :math:`\mathcal{V}`
                and the original :math:`(u_{\rm zen},v_{\rm zen},w_{\rm zen})`
                coordinates are re-phased as follows:

                .. math::
                    \pmatrix{
                        u_{\alpha, \delta}(t)\\
                        v_{\alpha, \delta}(t)\\
                        w_{\alpha, \delta}(t)
                    } = \mathcal{R}(\alpha, \delta) \cdot
                    \mathcal{R}(\alpha_{\rm{zen}, t}, \delta_{\rm{zen}, t}) \cdot
                    \pmatrix{
                        u_{\rm zen}(t)\\
                        v_{\rm zen}(t)\\
                        w_{\rm zen}(t)
                    }

                .. math::
                    \Delta(t) = \mathcal{R}(\alpha_{\rm{zen}, t}, \delta_{\rm{zen}, t})^{\top} \cdot
                    \left[
                        \pmatrix{
                            \sin(\alpha_{\rm{zen}, t})\cos(\delta_{\rm{zen}, t})\\
                            \cos(\alpha_{\rm{zen}, t})\cos(\delta_{\rm{zen}, t})\\
                            \sin(\delta_{\rm{zen}, t})
                        }
                        -
                        \pmatrix{
                            \sin(\alpha)\cos(\delta)\\
                            \cos(\alpha)\cos(\delta)\\
                            \sin(\delta)
                        }
                    \right] \cdot
                    \pmatrix{
                        u_{\rm zen}(t)\\
                        v_{\rm zen}(t)\\
                        w_{\rm zen}(t)
                    }

                .. math::
                    \mathcal{V}_{\alpha, \delta}(u_{\alpha, \delta}, v_{\alpha, \delta}, w_{\alpha, \delta}, \nu, t) =
                     \mathcal{V}_{\rm zen} (u_{\rm zen}, v_{\rm zen}, w_{\rm zen}, \nu, t)  e^{
                        2 \pi i \frac{\nu}{c} \Delta(t)
                    }

                With the matrix :math:`\mathcal{R}` being:
                
                .. math::
                    \mathcal{R}(a, b) = \pmatrix{
                        \cos(a), -\sin(a), 0\\
                        -\sin(a)\sin(b), -\cos(a)\sin(b) \cos(b)\\
                        \sin(a)\cos(b), \cos(a)\cos(b), \sin(b)
                    }

            :param resolution:
                Resolution (in degrees if a `float` is given) of
                the HEALPix grid (passed to initialize the 
                :class:`~nenupy.astro.hpxsky.HpxSky` object).
            :type resolution: `float` or :class:`~astropy.units.Quantity`
            :param fov:
                Field of view diameter of the image (in degrees
                if a `float` is given).
            :type fov: `float` or :class:`~astropy.units.Quantity`
            :param center:
                If ``None``, the phase center is considered to be
                the local zenith and no re-phasing is applied.
                Otherwise, a re-phasing towards the new-phase
                center is performed. Default is ``None``.
            :type center: :class:`~astropy.coordinates.SkyCoord`
            :param stokes:
                Stokes parameter to compute, one of ``'I'``
                (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_i`),
                ``'Q'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_q`),
                ``'U'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_u`),
                ``'V'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_v`),
                ``'FRAC_L'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_fl`),
                ``'FRAC_V'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_fv`).
                Default is ``'I'``.
            :type stokes: `str`

            :returns: HEALPix sky object embedding the computed
                image.
            :rtype: :class:`~nenupy.crosslet.imageprod.NenuFarTV`

            :Example:
                >>> from nenupy.crosslet import TV_Data
                >>> import astropy.units as u
                >>> tv = TV_Data('20191204_132113_nenufarTV.dat')
                >>> im = tv.image(
                        resolution=0.2*u.deg,
                        fov=60*u.deg
                    )

            .. seealso::
                :class:`~nenupy.astro.hpxsky.HpxSky`,
                :class:`~nenupy.crosslet.imageprod.NenuFarTV`,
                :meth:`~nenupy.astro.hpxsky.HpxSky.lmn`,
                :meth:`~nenupy.crosslet.uvw.UVW.fromCrosslets`

            .. warning::
                This method is intended to be used for NenuFAR-TV
                data and relatively small XST datasets. It is not
                suited to long observations for which a MS
                conversion is required before using imaging
                dedicated softwares.

            .. versionadded:: 1.1.0

        """
        import dask.array as da
        from dask.diagnostics import ProgressBar

        if not isinstance(fov, un.Quantity):
            fov *= un.deg

        log.info(
            'Computing image over {} sub-bands and {} time steps.'.format(
                self.freqs.size,
                self.times.size
            )
        )

        exposure = self.times[-1] - self.times[0]

        uvw = UVW.fromCrosslets(self)
        uvw = uvw.uvw
        log.info(
            'UVW coordinates computed.'
        )

        if center is None:
            center = eq_zenith(self.times[0] + exposure/2.)
            center = SkyCoord(center.ra, center.dec)
            rotVis = np.ones((1, self.freqs.size, 1))
            log.info(
                'XST data and UVW are kept phased at the local zenith.'
            )
        else:
            rotVis, uvw = self._rephase(center, uvw)
            log.info(
                'XST data and UVW re-phased towards RA={}deg Dec={}deg.'.format(
                    center.ra.deg,
                    center.dec.deg
                )
            )

        # Sky preparation
        sky = NenuFarTV(
            resolution=resolution,
            time=self.times[0] + exposure/2.,
            stokes=stokes.upper(),
            meanFreq=np.mean(self.freqs),
            phaseCenter=center,
            fov=fov
        )

        sky._is_visible *= sky._eq_coords.separation(center) <= fov/2.

        l, m, n = sky.lmn(phase_center=center)

        log.info(
            'HEALPix image coordinates (l, m, n) of size ({}, 3) prepared.'.format(
                l.size
            )
        )

        # UVW coordinates
        chkImgSize = np.floor(l.size/os.cpu_count())
        chkVisSize = np.floor(self._ant1.size/os.cpu_count())
        l = da.from_array(
            l.astype(np.float32),
            chunks=chkImgSize
        )
        m = da.from_array(
            m.astype(np.float32),
            chunks=chkImgSize
        )
        n = da.from_array(
            n.astype(np.float32),
            chunks=chkImgSize
        )
        uvw = da.from_array(
            uvw.astype(np.float32),
            chunks=(self.times.size, chkVisSize, 3)
        )
        uvw = uvw[:, None, :, :]/wavelength(self.freqs).value[None, :, None, None]
        u = np.mean( # Mean in time
            uvw[..., 0],
            axis=0
        )[:, self.mask_auto]
        v = np.mean(
            uvw[..., 1],
            axis=0
        )[:, self.mask_auto]
        w = np.mean( # Mean in time
            uvw[..., 2],
            axis=0
        )[:, self.mask_auto]

        ul = u[:, :, None] * l[None, None, :]
        vm = v[:, :, None] * m[None, None, :]
        wn = w[:, :, None] * (n - 1)[None, None, :]

        phase = np.exp(
            - 2.j * np.pi * (ul + vm + wn)
        ) # (nfreqs, nvis, npix)

        # Phase visibilities
        visib = da.from_array(
            self._getStokes(stokes).astype(np.complex64),
            chunks=(self.times.size, 1, chkVisSize)
        )
        rotat = da.from_array(
            rotVis.astype(np.complex64),
            chunks=(self.times.size, 1, chkVisSize)
        )
        vis = np.mean( # Mean in time
            visib * rotat,
            axis=0
        )[:, self.mask_auto] # (nfreqs, nvis)
        
        # Make dirty image
        dirtyImage = np.nanmean(
            np.real(
                np.mean(
                    vis[:, :, None] * phase,
                    axis=0
                )
            ),
            axis=0
        )
        log.info(
            'Dirty image of Stokes {} computation...'.format(
                stokes.upper()
            )
        )
        with ProgressBar():
            sky.skymap[sky._is_visible] = dirtyImage.compute()
        return sky


    def nearfield(self, radius=400, npix=64, sources=[], stokes='I'):
        r""" Computes the Near-field image from the cross-correlation
            statistics data :math:`\mathcal{V}`.

            The distances between each Mini-Array :math:`{\rm MA}_i`
            and the ground positions :math:`Delta` is:

            .. math::
                d_{\rm{MA}_i} (x, y) = \sqrt{
                    ({\rm MA}_{i, x} - \Delta_x)^2 + ({\rm MA}_{i, y} - \Delta_y)^2 + \left( {\rm MA}_{i, z} - \sum_j \frac{{\rm MA}_{j, z}}{n_{\rm MA}} - 1 \right)^2
                } 

            Then, the near-field image :math:`n_f` can be retrieved
            as follows (:math:`k` and :math:`l` being two distinct
            Mini-Arrays):

            .. math::
                n_f (x, y) = \sum_{k, l} \left| \sum_{\nu} \langle \mathcal{V}_{\nu, k, l}(t) \rangle_t e^{2 \pi i \left( d_{{\rm MA}_k} - d_{{\rm MA}_l} \right) (x, y) \frac{\nu}{c}} \right|

            .. note::
                To simulate astrophysical source of brightness :math:`\mathcal{B}`
                footprint on the near-field, its visibility per baseline
                of Mini-Arrays :math:`k` and :math:`l` are computed as:

                .. math::
                    \mathcal{V}_{{\rm simu}, k, l} = \mathcal{B} e^{2 \pi i \left( \mathbf{r}_k - \mathbf{r}_l \right) \cdot \mathbf{u} \frac{\nu}{c}}

                with :math:`\mathbf{r}` the ENU position of the Mini-Arrays,
                :math:`\mathbf{u} = \left( \cos(\theta) \sin(\phi), \cos(\theta) \cos(\phi), sin(\theta) \right)`
                the ground projection vector (in East-North-Up coordinates),
                (:math:`\phi` and :math:`\theta` are the source horizontal
                coordinates azimuth and elevation respectively).
    

            :param radius:
                Radius in meters of the ground image. Default is 400m.
            :type radius: `float`
            :param npix:
                Number of pixels of the image size. Default is 64.
            :type npix: `int`
            :param sources:
                List of source names for which their near-field footprint
                may be computed. Only sources above 10 deg elevation
                will be considered.
            :type sources: `list`
            :param stokes:
                Stokes parameter to compute, one of ``'I'``
                (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_i`),
                ``'Q'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_q`),
                ``'U'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_u`),
                ``'V'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_v`),
                ``'FRAC_L'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_fl`),
                ``'FRAC_V'`` (:attr:`~nenupy.crosslet.crosslet.Crosslet.stokes_fv`).
                Default is ``'I'``.
            :type stokes: `str`

            :returns: Near-Field object. 
            :rtype: :class:`~nenupy.crosslet.imageprod.NearField`

            :Example:
                >>> from nenupy.crosslet import TV_Data
                >>> tv = TV_Data('20191204_132113_nenufarTV.dat')
                >>> nf = tv.nearfield(
                        radius=400,
                        npix=32,
                        sources=['Cyg A', 'Cas A', 'Vir A', 'Tau A'],
                        stokes='i'
                    )

            .. versionadded:: 1.1.0

        """
        import dask.array as da
        from dask.diagnostics import ProgressBar

        log.info(
            'Near-field over {} sub-bands and {} time steps (radius={}m, npix={}).'.format(
                self.freqs.size,
                self.times.size,
                radius,
                npix
            )
        )

        # Mini-Array positions in ENU coordinates
        mapos_l93 = getMAL93(self.mas)
        mapos_etrs = l93_to_etrs(mapos_l93)
        maposENU = etrs_to_enu(mapos_etrs)
        chkVisSize = np.floor(self._ant1[self.mask_auto].size/os.cpu_count())

        # Mean time of observation
        obsTime = self.times[0] + (self.times[-1] - self.times[0])/2.

        # Delays at the ground
        groundGranularity = np.linspace(-radius, radius, npix)
        posx, posy = np.meshgrid(groundGranularity, groundGranularity)
        posz = np.ones_like(posx) * (np.average(maposENU[:, 2]) + 1)
        groundGrid = np.stack((posx, posy, posz), axis=2)
        groundDistances = np.sqrt(
            np.sum(
                (maposENU[:, None, None, :] - groundGrid[None])**2,
                axis=-1
            )
        )
        gridDelays = groundDistances[self._ant1] - groundDistances[self._ant2]
        gridDelays = da.from_array(
            gridDelays[self.mask_auto].astype(np.float32),
            chunks=(
                chkVisSize,
                npix,
                npix
            )
        )

        log.info(
            'Computing near-field in Stokes {}...'.format(
                stokes.upper()
            )
        )
        # Compute the near-field image
        visData = np.mean(
            da.from_array(
                self._getStokes(stokes).astype(np.complex64),
                chunks=(
                    self.times.size,
                    self.freqs.size,
                    chkVisSize
                )
            ),
            axis=0
        )[:, self.mask_auto] # mean in time 
        nfImage = self._nearFieldImage(
            visData,
            gridDelays
        )

        # Simulate sources to get their imprint
        simuSources = {}
        for src in sources:
            radecSrc = getSource(src, time=obsTime)
            altazSrc = toAltaz(
                skycoord=radecSrc,
                time=obsTime,
                kind='fast'
            )
            if altazSrc.alt.deg <= 10:
                # Don't take into account sources too low
                log.info(
                    'Source {} elevation is {}<=10deg: not taken into account.'.format(
                        altazSrc.alt.deg,
                        src
                    )
                )
                continue
            log.info(
                'Simulating visibilities of {} in Stokes {}.'.format(
                    src,
                    stokes.upper(),
                )
            )
            # Projection from AltAz to ENU vector
            cosAz = np.cos(altazSrc.az.rad)
            sinAz = np.sin(altazSrc.az.rad)
            cosEl = np.cos(altazSrc.alt.rad)
            sinEl = np.sin(altazSrc.alt.rad)
            toENU = np.array(
                [cosEl*sinAz, cosEl*cosAz, sinEl]
            )
            srcDelays = np.matmul(
                maposENU[self._ant1] - maposENU[self._ant2],
                toENU
            )
            srcDelays = da.from_array(
                srcDelays[self.mask_auto, :].astype(np.float32),
                chunks=(
                    chkVisSize,
                    1
                )
            )
            # Simulate visibilities
            lamb = wavelength(self.freqs).value
            # srcVis = ft_delay(srcDelays/lamb)
            srcVis = np.exp(2.j * np.pi * (srcDelays/lamb))
            srcVis = np.swapaxes(srcVis, 1, 0)
            log.info(
                'Computing near-field imprint of {} in Stokes {}...'.format(
                    src,
                    stokes.upper(),
                )
            )
            simuSources[src] = self._nearFieldImage(
                srcVis,
                gridDelays
            )

        return NearField(
            nfImage=nfImage,
            antNames=self.mas,
            meanFreq=np.mean(self.freqs),
            obsTime=obsTime,
            simuSources=simuSources,
            radius=radius*un.m,
            stokes=stokes.upper()
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _getStokes(self, stokes):
        """
        """
        if stokes.upper() == 'I':
            stokesData = self.stokes_i
        elif stokes.upper() == 'Q':
            stokesData = self.stokes_q
        elif stokes.upper() == 'U':
            stokesData = self.stokes_u
        elif stokes.upper() == 'V':
            stokesData = self.stokes_v
        elif stokes.upper() == 'FRAC_L':
            stokesData = self.stokes_fl
        elif stokes.upper() == 'FRAC_V':
            stokesData = self.stokes_fv
        else:
            raise ValueError(
                'Unknown Stokes parameter `{}`'.format(stokes)
            )
        return stokesData


    def _get_cross_idx(self, c1='X', c2='X'):
        """
        """
        corr = np.array(['X', 'Y']*self.mas.size)
        i_ant1, i_ant2 = np.tril_indices(self.mas.size*2, 0)
        corr_mask = (corr[i_ant1] == c1) & (corr[i_ant2] == c2)
        indices = np.arange(i_ant1.size)[corr_mask]
        return indices


    def _rephase(self, newPhaseCenter, oldUVW):
        """
        """

        def rotMatrix(skycoord):
            """
            """
            raRad = skycoord.ra.rad
            decRad = skycoord.dec.rad

            if np.isscalar(raRad):
                raRad = np.array([raRad])
                decRad = np.array([decRad])

            cosRa = np.cos(raRad)
            sinRa = np.sin(raRad)
            cosDec = np.cos(decRad)
            sinDec = np.sin(decRad)

            return np.array([
                [cosRa, -sinRa, np.zeros(raRad.size)],
                [-sinRa*sinDec, -cosRa*sinDec, cosDec],
                [sinRa*cosDec, cosRa*cosDec, sinDec],
            ])

        # Transformation matrices
        phaseCenter2Origin = rotMatrix(self.phaseCenter) # (3, 3, ntimes)
        origin2NewPhaseCenter = rotMatrix(newPhaseCenter) # (3, 3, 1)
        totalTrans = np.matmul(
            np.transpose(
                origin2NewPhaseCenter,
                (2, 0, 1)
            ),
            phaseCenter2Origin
        ) # (3, 3, ntimes)
        rotUVW = np.matmul(
            np.expand_dims(
                (phaseCenter2Origin[2, :] - origin2NewPhaseCenter[2, :]).T,
                axis=1
            ),
            np.transpose(
                phaseCenter2Origin,
                (2, 1, 0)
            )
        ) # (ntimes, 1, 3)
        phase = np.matmul(
            rotUVW,
            np.transpose(oldUVW, (0, 2, 1))
        ) # (ntimes, 1, nvis)
        rotVis = np.exp(
            2.j * np.pi * phase / wavelength(self.freqs).value[None, :, None]
        ) # (ntimes, nfreqs, nvis)

        newUVW = np.matmul(
            oldUVW, # (ntimes, nvis, 3)
            np.transpose(totalTrans, (2, 0, 1))
        )

        return rotVis, newUVW


    def _nearFieldImage(self, vis, delays):
        """ vis = [freq, nant, nant]
        """
        from dask.diagnostics import ProgressBar

        assert self.freqs.size == vis.shape[0],\
            'Problem in visibility dimension {}'.format(vis.shape)

        nearfield = 0
        for i in range(self.freqs.size): 
            vi = vis[i][:, None, None]
            lamb = wavelength(self.freqs[i]).value
            # nearfield += vi * ft_delay(delays/lamb)
            nearfield += vi * np.exp(2.j * np.pi * (delays/lamb))

        nearfield /= self.freqs.size
        nearfield = np.nanmean(np.abs(nearfield), axis=0)

        with ProgressBar():
            nf = nearfield.compute()

        return nf
# ============================================================= #

