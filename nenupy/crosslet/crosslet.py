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


    def image(self, resolution=1, fov=50, center=None, fIndices=None):
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
        
        # f_idx = 0 # Frequency index
        if fIndices is None:
            fIndices = np.arange(self.freqs.size)
        else:
            if not isinstance(fIndices, np.ndarray):
                raise TypeError(
                    'fIndices sould be a numpy array'
                )
            if any(fIndices > 15):
                raise IndexError(
                    'Maximal authorized frequency index is 15'
                )
            if fIndices.size > 16:
                raise IndexError(
                    'Number of subbands is 16'
                )
        
        exposure = self.times[-1] - self.times[0]

        uvw = UVW.from_tvdata(self)
        uvw = uvw.uvw
        
        if center is None:
            center = eq_zenith(self.times[0] + exposure/2.)
            # center = SkyCoord([center.ra], [center.dec])
            center = SkyCoord(center.ra, center.dec)
            rotVis = np.ones((1, 16, 1))
        else:
            rotVis, uvw = self._rephase(center, uvw)

        # Sky preparation
        # sky = HpxSky(resolution=resolution)
        # sky.time = self.times[0] + exposure/2.
        sky = NenuFarTV(
            resolution=resolution,
            time=self.times[0] + exposure/2.,
            stokes='I',
            meanFreq=np.mean(self.freqs[fIndices])*un.MHz,
            phaseCenter=center,
            fov=fov
        )

        sky._is_visible *= sky._eq_coords.separation(center) <= fov/2.

        l, m, n = sky.lmn(phase_center=center)
        # # UVW coordinates
        # u = np.mean( # Mean in time
        #     uvw[:, :, 0],
        #     axis=0
        # )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        # v = np.mean(
        #     uvw[:, :, 1],
        #     axis=0
        # )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        # w = np.mean( # Mean in time
        #     uvw[:, :, 2],
        #     axis=0
        # )[self.mask_auto]/wavelength(self.freqs[f_idx]).value
        # # Mulitply (u, v) by (l, m) and compute FT exp
        # # ul = ft_mul(
        # #     x=np.tile(u, (l.size, 1)).T,
        # #     y= np.tile(l, (u.size, 1))
        # # )
        # # vm = ft_mul(
        # #     x=np.tile(v, (m.size, 1)).T,
        # #     y=np.tile(m, (v.size, 1))
        # # )
        # # wn = ft_mul(
        # #     x=np.tile(w, (n.size, 1)).T,
        # #     y=np.tile(n-1, (w.size, 1))
        # # )
        # ul = u[:, None] * l[None, :]
        # vm = v[:, None] * m[None, :]
        # wn = w[:, None] * (n - 1)[None, :]
        # phase = ft_phase(ul, vm, wn)
        # # Phase visibilities
        # vis = np.mean( # Mean in time
        #     self.stokes_i * rotVis,
        #     axis=0
        # )[f_idx, :][self.mask_auto]
        # im = np.zeros(l.size)
        # for i in tqdm(range(l.size)):
        #     im[i] = np.real(
        #         ft_sum(vis, phase[:, i])
        #     )
        # sky.skymap[sky._is_visible] = im
        # return sky

        import dask.array as da
        from dask.diagnostics import ProgressBar
        # UVW coordinates
        
        l = da.from_array(l.astype(np.float32))
        m = da.from_array(m.astype(np.float32))
        n = da.from_array(n.astype(np.float32))
        uvw = da.from_array(uvw.astype(np.float32))
        uvw = uvw[:, None, :, :]/wavelength(self.freqs[fIndices]).value[None, :, None, None]
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
        vis = np.mean( # Mean in time
            da.from_array(self.stokes_i[:, fIndices, :].astype(np.complex64)) * da.from_array(rotVis[:, fIndices, :].astype(np.complex64)),
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
        with ProgressBar():
            sky.skymap[sky._is_visible] = dirtyImage.compute()
        return sky


    def nearfield(self, radius=400, npix=64, sources=[], fIndices=None):
        """
        """
        # Frequency indices
        if fIndices is None:
            fIndices = np.arange(self.freqs.size)
        else:
            if not isinstance(fIndices, np.ndarray):
                raise TypeError(
                    'fIndices sould be a numpy array'
                )
            if any(fIndices > 15):
                raise IndexError(
                    'Maximal authorized frequency index is 15'
                )
            if fIndices.size > 16:
                raise IndexError(
                    'Number of subbands is 16'
                )

        # Mini-Array positions in ENU coordinates
        mapos_l93 = getMAL93(self.mas)
        mapos_etrs = l93_to_etrs(mapos_l93)
        maposENU = etrs_to_enu(mapos_etrs)

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

        # Compute the near-field image
        visData = np.mean(self.stokes_i[:, fIndices, :], axis=0) # mean in time
        nfImage = self._nearFieldImage(
            visData,
            gridDelays,
            fIndices
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
                continue
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
            # Simulate visibilities
            lamb = wavelength(self.freqs[fIndices]).value
            srcVis = ft_delay(srcDelays/lamb)
            srcVis = np.swapaxes(srcVis, 1, 0)
            simuSources[src] = self._nearFieldImage(
                srcVis,
                gridDelays,
                fIndices
            )

        return NearField(
            nfImage=nfImage,
            antNames=self.mas,
            meanFreq=np.mean(self.freqs[fIndices])*un.MHz,
            obsTime=obsTime,
            simuSources=simuSources,
            radius=radius*un.m
        )


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
        # self.phaseCenter = SkyCoord(
        #     ra=np.ones(self.times.size) * newPhaseCenter.ra,
        #     dec=np.ones(self.times.size) * newPhaseCenter.dec
        # )
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


    def _nearFieldImage(self, vis, delays, fIndices):
        """ vis = [freq, nant, nant]
        """
        assert self.freqs[fIndices].size == vis.shape[0],\
            'Problem in visibility dimension {}'.format(vis.shape)
        nearfield = 0
        for i in range(self.freqs[fIndices].size): 
            vi = vis[i][:, None, None]
            lamb = wavelength(self.freqs[fIndices][i])
            nearfield += vi * ft_delay(delays/lamb)
        nearfield /= self.freqs[fIndices].size
        nearfield = np.nanmean(np.abs(nearfield), axis=0)
        return nearfield
# ============================================================= #

