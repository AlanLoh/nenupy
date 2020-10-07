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
    (see also :ref:`tuto_beamforming` for a detailed tutorial)
    and *imaging* with the
    :meth:`~nenupy.crosslet.crosslet.Crosslet.image` method
    (see also :ref:`tuto_tv` for a detailed tutorial) from 
    cross-correlation statistics data.
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
from nenupy.instru import nenufar_loc, read_cal_table, ma_pos
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
        - 2.j * np.pi * (ul + vm)
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
    def xx(self):
        """ Extracts the XX polarization from the cross-correlation
            statistics data. Also refered to as ``'NW'``.
            Array is shaped like (time, frequencies, baselines).

            :getter: XX polarization
            
            :type: :class:`~numpy.ndarray`
        """
        xx = self.vis[:, :, self._get_cross_idx('X', 'X')]
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
        yy = self.vis[:, :, self._get_cross_idx('Y', 'Y')]
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
        yx = self.vis[:, :, self._get_cross_idx('Y', 'X')]
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
        _xy = np.zeros(
            (list(self.yx.shape[:-1]) + [ma1.size]),
            dtype=np.complex
        )
        _xy[:, :, auto] = self.yx[:, :, auto].conj()
        # Get XY correlations
        indices = self._get_cross_idx('X', 'Y')
        _xy[:, :, cross] = self.vis[:, :, indices]
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
                    \mathbf{XX}(t, \nu) + \mathbf{YY}(t, \nu)
                \right]

            Array is shaped like (time, frequencies, baselines).

            :getter: Stokes I data
            
            :type: :class:`~numpy.ndarray`
        """
        return 0.5*(self.xx + self.yy)


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


    def image(self, resolution=1, fov=50):
        r""" Converts NenuFAR-TV-like data sets containing
            visibilities (:math:`V(u,v,\nu , t)`) into images
            :math:`I(l, m, \nu)` phase-centered at the local
            zenith while time averaging the visibilities.
            The Field of View ``fov`` argument defines the
            diameter angular size (zenith-centered) above which
            the image is not computed.
            
            .. math::
                I(l, m, \nu) = \int
                    \langle V(u, v, \nu, t) \rangle_t e^{
                        2 \pi i \frac{\nu}{c} \left(
                            \langle u(t) \rangle_t l + \langle v(t) \rangle_t m
                        \right)
                    }
                    \, du \, dv

            :param resolution:
                Resoltion (in degrees if a `float` is given) of
                the HEALPix grid (passed to initialize the 
                :class:`~nenupy.astro.hpxsky.HpxSky` object).
            :type resolution: `float` or :class:`~astropy.units.Quantity`
            :param fov:
                Field of view diameter of the image (in degrees
                if a `float` is given).
            :type fov: `float` or :class:`~astropy.units.Quantity`

            :returns: HEALPix sky object embedding the computed
                image.
            :rtype: :class:`~nenupy.astro.hpxsky.HpxSky`

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
                :meth:`~nenupy.astro.hpxsky.HpxSky.lmn`,
                :meth:`~nenupy.crosslet.uvw.UVW.from_tvdata`

            .. warning::
                This method is intended to be used for NenuFAR-TV
                data and relatively small XST datasets. It is not
                suited to long observations for which a MS
                conversion is required before using imaging
                dedicated softwares.
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
            uvw.uvw[:, :, 0],
            axis=0
        )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        v = np.mean(
            uvw.uvw[:, :, 1],
            axis=0
        )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        w = np.mean( # Mean in time
            uvw.uvw[:, :, 2],
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


    def nearfield(self, resolution):
        """
        """
        return


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

