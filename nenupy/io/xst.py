#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    XST file
    ********

    .. inheritance-diagram:: nenupy.io.xst.XST nenupy.io.xst.NenufarTV nenupy.io.xst.TV_Image nenupy.io.xst.TV_Nearfield
        :parts: 3

"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2023, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = ["XST_Slice", "Crosslet", "XST", "TV_Image", "TV_Nearfield", "NenufarTV"]

from abc import ABC, abstractmethod
from typing import Tuple, List
import os
from itertools import islice
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz, Angle, ICRS
import astropy.units as u
from astropy.io import fits
from healpy.fitsfunc import write_map, read_map
from healpy.pixelfunc import mask_bad, nside2resol
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import LinearLocator
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dask.array as da
from dask.diagnostics import ProgressBar

import nenupy
from os.path import join, dirname
from nenupy.astro.target import FixedTarget, SolarSystemTarget
from nenupy.io.io_tools import StatisticsData, ST_Slice
from nenupy.astro import wavelength, altaz_to_radec, l93_to_etrs, etrs_to_enu
from nenupy.astro.uvw import compute_uvw
from nenupy.astro.sky import HpxSky
from nenupy.astro.pointing import Pointing
from nenupy.instru import (
    NenuFAR,
    MiniArray,
    read_cal_table,
    freq2sb,
    nenufar_miniarrays,
)
from nenupy import nenufar_position, DummyCtMgr

import logging

log = logging.getLogger(__name__)

# ============================================================= #
# --------------------------- Tools --------------------------- #
def _vis_rotation_matrix(skycoord: SkyCoord) -> np.ndarray:
    """Define the rotation matrix for visibility rephasing."""
    ra_rad = skycoord.ra.rad
    dec_rad = skycoord.dec.rad

    if np.isscalar(ra_rad):
        ra_rad = np.array([ra_rad])
        dec_rad = np.array([dec_rad])

    cos_ra = np.cos(ra_rad)
    sin_ra = np.sin(ra_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)

    return np.array(
        [
            [cos_ra, -sin_ra, np.zeros(ra_rad.size)],
            [-sin_ra * sin_dec, -cos_ra * sin_dec, cos_dec],
            [sin_ra * cos_dec, cos_ra * cos_dec, sin_dec],
        ]
    )

# ============================================================= #
# ------------------------- XST_Slice ------------------------- #
class XST_Slice:
    """Class to handle the result of selection upon XST-like data.

    See Also
    --------
    :ref:`xst_beamforming_doc`
    """

    def __init__(
        self,
        mini_arrays: np.ndarray,
        time: Time,
        frequency: u.Quantity,
        value: np.ndarray,
        phase_center: SkyCoord = None
    ):
        r"""Generate an instance of :class:`~nenupy.io.xst.XST_Slice`.

        Parameters
        ----------
        mini_arrays : :class:`~numpy.ndarray`
            NenuFAR Mini-Arrays involved in the data selection
        time : :class:`~astropy.time.Time`
            Time description of the selected dataset
        frequency : :class:`~astropy.units.Quantity`
            Frequency description of the selected dataset
        value : :class:`~numpy.ndarray`
            Data selection values, it should have the shape :math:`(n_{\rm \nu},\, n_{t},\, n_{\rm bl})` where :math:`n_{\rm \nu}`, :math:`n_{t}` and :math:`n_{\rm bl}=n_{\rm ant}(n_{\rm ant} - 1)/2 + n_{\rm ant}` are respectively the number of frequency and time samples, and the number of baselines
        phase_center : :class:`~astropy.coordinates.SkyCoord`
            Phase center of the visibility selection (in ICRS), by default `None` (i.e., zenith)
        """
        self.mini_arrays = mini_arrays
        self.time = time
        self.frequency = frequency
        self.value = value
        if phase_center is None:
            phase_center = SkyCoord(
                0*u.deg, 90*u.deg,
                frame=AltAz(
                    obstime=self.time[0] + (self.time[-1] - self.time[0]) / 2,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
        self.phase_center = phase_center

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot_correlaton_matrix(
        self,
        mask_autocorrelations: bool = False,
        figsize: Tuple[int, int] = (10, 10),
        decibel: bool = True,
        cmap: str = "YlGnBu",
        vmin: float = None,
        vmax: float = None,
        cbar_label: str = None,
        title: str = None,
        figname: str = None,
    ) -> None:
        r"""Plot the XST cross-correlation matrix.
        All visibilities are plotted against their Mini-Array indices.
        The absolute of their average over their available time and frequency samples is displayed.

        Parameters
        ----------
        mask_autocorrelations : `bool`, optional
            Mask the auto-correlation diagonal, by default `False`
        figsize : Tuple[`int`, `int`], optional
            Size of the figure, by default (10, 10)
        decibel : `bool`, optional
            Set the scale of the data displayed to dB (i.e., :math:`{\rm dB} = 10 \log_{10}({\rm data})`), by default True
        cmap : `str`, optional
            Color map used to represent the data, by default "YlGnBu"
        vmin : `float`, optional
            Minimal value displayed, by default `None` (i.e., overall minimal value)
        vmax : `float`, optional
            Maximal value displayed, by default `None` (i.e., overall maximal value)
        cbar_label : `str`, optional
            Label of the color bar, by default `None`
        title : `str`, optional
            Title of the plot, by default `None`
        figname : `str`, optional
            Name of the figure, if given the figure will be saved, by default `None`

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST

            >>> xst = XST("/path/to/XST.fits")
            >>> data = xst.get(...) # data selection that generate a XST_Slice
            >>> data.plot_correlaton_matrix()

        .. figure:: ../_images/io_images/xst_cross_matrix.png
            :width: 450
            :align: center

            Cross-correlation matrix, the Mini-Array #1 was flagged.
        """

        max_ma_index = self.mini_arrays.max() + 1
        all_mas = np.arange(max_ma_index)
        matrix = np.full([max_ma_index, max_ma_index], np.nan, "complex")
        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        for ma in all_mas:
            if ma not in self.mini_arrays:
                ma1[ma1 >= ma] += 1
                ma2[ma2 >= ma] += 1

        mask = None
        if mask_autocorrelations:
            mask = ma1 != ma2  # cross_correlation mask
        matrix[ma2[mask], ma1[mask]] = np.mean(self.value, axis=(0, 1))[mask]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        data = np.absolute(matrix)
        if decibel:
            data = 10 * np.log10(data)

        im = ax.pcolormesh(
            all_mas,
            all_mas,
            data,
            shading="nearest",
            cmap=cmap,
            vmin=np.nanmin(data) if vmin is None else vmin,
            vmax=np.nanmax(data) if vmax is None else vmax,
        )
        ax.set_xticks(all_mas[::2])
        ax.set_yticks(all_mas[::2])
        ax.grid(alpha=0.5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(
            cbar_label if cbar_label is not None else ("dB" if decibel else "Amp")
        )

        # Axis abels
        ax.set_xlabel(f"Mini-Array index")
        ax.set_ylabel(f"Mini-Array index")

        # Title
        if title is not None:
            ax.set_title(title)

        # Save or show the figure
        if (figname is None) or (figname == ""):
            plt.show()
        else:
            plt.savefig(figname, dpi=300, bbox_inches="tight", transparent=True)
            log.info(f"Figure '{figname}' saved.")

        plt.close("all")

    def rephase_visibilities(
        self, phase_center: SkyCoord, uvw: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rephase the XST visibilities towards a new phase center.
        By default, XST visibilities are phased at the local zenith.
        This method applies two rotations: one to reset the visibilities to the celestial frame origin and a second to phase towards the desired phase center.

        Parameters
        ----------
        phase_center : :class:`~astropy.coordinateds.SkyCoord`
            New phase center.
        uvw : :class:`~numpy.ndarray`
            UVW coordinates in meters of the input array, see also :func:`~nenupy.astro.uvw.compute_uvw`, by default `None` (i.e., the method automatically computes the cooridnates)

        Returns
        -------
        Tuple[:class:`~numpy.ndarray`, :class:`~numpy.ndarray`]
            Re-phased visibilities, rotated UVW coordinates

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST
            >>> from astropy.coordinates import SkyCoord

            >>> xst = XST("/path/to/XST.fits")
            >>> xx_data = xst.get(polarization="XX")
            >>> cyg_a = SkyCoord.from_name("Cyg A")
            >>> xx_data_rephased, uvw_rephased = xx_data.rephase_visibilities(cyg_a)
        """

        log.info(f"Rephasing the visibilities towards {phase_center}...")

        # Compute the zenith original phase center
        zenith = SkyCoord(
            np.zeros(self.time.size),
            np.ones(self.time.size) * 90,
            unit="deg",
            frame=AltAz(obstime=self.time, location=nenufar_position),
        )
        zenith_phase_center = altaz_to_radec(zenith)

        # Compute the UVW coordinates if they were not provided
        if uvw is None:
            uvw = compute_uvw(
                interferometer=NenuFAR()[self.mini_arrays],
                phase_center=None,  # will be zenith
                time=self.time,
            ).to_value(u.m)

        # Transformation matrices
        to_origin = _vis_rotation_matrix(zenith_phase_center)  # (3, 3, ntimes)
        to_new_center = _vis_rotation_matrix(phase_center)  # (3, 3, 1)
        total_transformation = np.matmul(
            np.transpose(to_new_center, (2, 0, 1)), to_origin
        )  # (3, 3, ntimes)
        rotUVW = np.matmul(
            np.expand_dims((to_origin[2, :] - to_new_center[2, :]).T, axis=1),
            np.transpose(to_origin, (2, 1, 0)),
        )  # (ntimes, 1, 3)
        phase = np.matmul(rotUVW, np.transpose(uvw, (0, 2, 1)))  # (ntimes, 1, nvis)
        rotate_visibilities = np.exp(
            2.0j
            * np.pi
            * phase
            / wavelength(self.frequency).to(u.m).value[None, :, None]
        )  # (ntimes, nfreqs, nvis)

        new_uvw = np.matmul(
            uvw, np.transpose(total_transformation, (2, 0, 1))  # (ntimes, nvis, 3)
        )

        return rotate_visibilities, new_uvw

    def make_image(
        self,
        resolution: u.Quantity = 1 * u.deg,
        fov_radius: u.Quantity = 25 * u.deg,
        phase_center: SkyCoord = None,
        stokes: str = "I",
    ) -> HpxSky:
        r"""Perform an inversion of the visibilities to get an image.

        A Discrete Fourier Transform is applied, based on the Fourier Transform relationship between the sky brightness distribution :math:`\mathcal{I}(l,m,n)` and the NenuFAR response (complex visibilities) :math:`\mathcal{V}(u,v,w)`:

        .. math::
            \mathcal{I}(l, m, n) = \int \left\langle \mathcal{V}(u, v, w) \right\rangle_{t, \nu} e^{-2\pi i (ul + vm + wn)}\,du\, dv\, dw ,

        where :math:`(u, v, w)` are coordinates expressed in :math:`\lambda` units, :math:`(l,m,n)` are image coordinates (taken as an HEALPix image by default using :class:`~nenupy.astro.sky.HpxSky`).

        Parameters
        ----------
        resolution : :class:`~astropy.units.Quantity`, optional
            Image HEALPix resolution (see :class:`~nenupy.astro.sky.HpxSky`), by default 1 degree
        fov_radius : :class:`~astropy.units.Quantity`, optional
            Field of View radius that will apply a selection on the image plane before computing the pixel values, by default 25 degrees
        phase_center : :class:`~astropy.coordinates.SkyCoord`, optional
            Center of the field of view, by default `None` (i.e., local zenith)
        stokes : `str`, optional
            Stokes parameter description of what contain the :attr:`~nenupy.io.xst.XST_Slice.value` that will be passed to :class:`~nenupy.astro.sky.HpxSky` (there is no computation dependency upon this paramater here), by default "I"

        Returns
        -------
        :class:`~nenupy.astro.sky.HpxSky`
            The image

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST
            >>> from astropy.coordinates import SkyCoord
            >>> import astropy.units as u

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> xx_data = xst.get(polarization="XX")
            >>> tau_a = SkyCoord.from_name("Tau A")
            >>> im = xx_data.make_image(
                    resolution=1*u.deg,
                    fov_radius=10*u.deg,
                    phase_center=tau_a,
                    stokes="XX"
                )

            >>> im[0, 0, 0].plot(center=tau_a, radius=5*u.deg)


        .. figure:: ../_images/io_images/make_image_xst.png
            :width: 450
            :align: center
        """

        exposure = self.time[-1] - self.time[0]

        # Compute XST UVW coordinates (zenith phased)
        uvw = compute_uvw(
            interferometer=NenuFAR()[self.mini_arrays],
            phase_center=None,  # will be zenith
            time=self.time,
        ).to_value(u.m)

        # Prepare visibilities rephasing
        if phase_center is None:
            phase_center = SkyCoord(
                0,
                90,
                unit="deg",
                frame=AltAz(
                    obstime=self.time[0] + exposure / 2, location=nenufar_position
                ),
            ).transform_to("icrs")
            rephase_matrix = 1.0
        else:
            rephase_matrix, uvw = self.rephase_visibilities(
                phase_center=phase_center, uvw=uvw
            )

        # Mask auto-correlations
        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        cross_mask = ma1 != ma2
        uvw = uvw[:, cross_mask, :]
        # Transform to lambda units
        wvl = wavelength(self.frequency).to(u.m).value
        uvw = uvw[:, None, :, :] / wvl[None, :, None, None]  # (t, f, bsl, 3)
        # Mean in time
        uvw = np.mean(uvw, axis=0)

        # Prepare the sky
        sky = HpxSky(
            resolution=resolution,
            time=self.time[0] + exposure / 2,
            frequency=np.mean(self.frequency),
            polarization=np.array([stokes]),
            value=np.nan,
        )

        # Compute LMN coordinates
        log.info("Preparing the sky...")
        image_mask = sky.visible_mask[0, 0, 0]
        image_mask *= sky.coordinates.separation(phase_center) <= fov_radius
        l, m, n = sky.compute_lmn(phase_center=phase_center, coordinate_mask=image_mask)
        lmn = np.array([l, m, (n - 1)], dtype=np.float32).T
        n_pix = l.size
        lmn = da.from_array(lmn, chunks=(np.floor(n_pix / os.cpu_count()), 3))

        # Transform to Dask array
        n_bsl = uvw.shape[1]
        n_freq = self.frequency.size
        n_pix = l.size
        uvw = da.from_array(
            uvw.astype(np.float32),
            chunks=(n_freq, max(1, np.floor(n_bsl / os.cpu_count())), 3),
        )

        # Compute the phase
        uvwlmn = np.sum(uvw[:, :, None, :] * lmn[None, None, :, :], axis=-1)
        phase = np.exp(-2j * np.pi * uvwlmn)  # (f, bsl, npix)

        # Rephase and average visibilites
        vis = np.mean(self.value * rephase_matrix, axis=0)[  # Mean in time
            ..., cross_mask
        ]  # (nfreqs, nvis)

        # Make dirty image
        dirty = np.nanmean(  # mean in baselines
            np.real(np.mean(vis[:, :, None] * phase, axis=0)), axis=0  # mean in freq
        )

        # Insert dirty image in Sky object
        log.info(
            f"Computing image (time: {self.time.size}, frequency: {self.frequency.size}, baselines: {vis.shape[1]}, pixels: {phase.shape[-1]})... "
        )
        with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
            sky.value[0, 0, 0, image_mask] = dirty.compute()

        return sky

    def make_nearfield(
        self, radius: u.Quantity = 400 * u.m, npix: int = 64, sources: List[int] = []
    ) -> Tuple[np.ndarray, dict]:
        r"""Computes the near-field image from the cross-correlation
        statistics data :math:`\mathcal{V}`.

        The distances between each Mini-Array :math:`{\rm MA}_i`
        and the ground positions :math:`\Delta` is:

        .. math::
            d_{\rm{MA}_i} (x, y) = \sqrt{
                ({\rm MA}_{i, x} - \Delta_x)^2 + ({\rm MA}_{i, y} - \Delta_y)^2 + \left( {\rm MA}_{i, z} - \sum_j \frac{{\rm MA}_{j, z}}{n_{\rm MA}} - 1 \right)^2
            }

        Then, the near-field image :math:`n_f` can be retrieved
        as follows (:math:`k` and :math:`l` being two distinct
        Mini-Arrays):

        .. math::
            n_f (x, y) = \sum_{k, l} \left| \sum_{\nu} \langle \mathcal{V}_{\nu, k, l}(t) \rangle_t e^{2 \pi i \left( d_{{\rm MA}_k} - d_{{\rm MA}_l} \right) (x, y) \frac{\nu}{c}} \right|

        Notes
        -----
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

        Parameters
        ----------
        radius : :class:`~astropy.units.Quantity`, optional
            Radius of the ground image, by default 400 m
        npix : `int`, optional
            Number of pixels of the image size, by default 64.
        sources : List[`int`], optional
            List of source names for which their near-field footprint may be computed (only sources above 10 deg elevation will be considered), source names are resolved using Simbad through :meth:`~astropy.coordinates.SkyCoord.from_name` within :class:`~nenupy.astro.target.FixedTarget`, Solar System object names are also valid (and called through :class:`~nenupy.astro.target.SolarSystemTarget`), by default []

        Returns
        -------
        Tuple[:class:`~numpy.ndarray`, `dict`]
            Tuple of near-field image (of shape ``(npix, npix)``) and a dictionnary containing all source footprints (of the same shapes)

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> xx_data = xst.get(polarization="XX")
            >>> nearfield, src_dict = xx_data.make_nearfield(
                    radius=400*u.m,
                    npix=64,
                    sources=["Tau A"]
                )

        One can further plot the nearfield map:

        .. code-block:: python

            >>> from nenupy.io.xst import TV_Nearfield

            >>> tv_nf = TV_Nearfield(
                    nearfield=nearfield,
                    source_imprints=src_dict,
                    npix=nearfield.shape[0],
                    time=xst.time[0],
                    frequency=xst.frequencies[0],
                    radius=400*u.m,
                    mini_arrays=xst.mini_arrays,
                    stokes="XX",
                )
            >>> tv_nf.save_png(figname="")

        .. figure:: ../_images/io_images/make_nearfield_xst.png
            :width: 450
            :align: center

        .. versionadded:: 1.1.0

        """

        def compute_nearfield_imprint(visibilities, phase):
            # Phase and average in frequency
            nearfield = np.mean(visibilities[..., None, None] * phase, axis=0)
            # Average in baselines
            nearfield = np.nanmean(np.abs(nearfield), axis=0)
            with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
                return nearfield.compute()

        # Mini-Array positions in ENU coordinates
        nenufar = NenuFAR()[self.mini_arrays]
        ma_etrs = l93_to_etrs(nenufar.antenna_positions)
        ma_enu = etrs_to_enu(ma_etrs)

        # Treat baselines
        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        cross_mask = ma1 != ma2

        # Mean time of observation
        obs_time = self.time[0] + (self.time[-1] - self.time[0]) / 2.0

        # Delays at the ground
        radius_m = radius.to(u.m).value
        ground_granularity = np.linspace(-radius_m, radius_m, npix)
        posx, posy = np.meshgrid(ground_granularity, ground_granularity)
        posz = np.ones_like(posx) * (np.average(ma_enu[:, 2]) + 1)
        ground_grid = np.stack((posx, posy, posz), axis=2)
        ground_distances = np.sqrt(
            np.sum((ma_enu[:, None, None, :] - ground_grid[None]) ** 2, axis=-1)
        )
        grid_delays = (
            ground_distances[ma1] - ground_distances[ma2]
        )  # (nvis, npix, npix)
        n_bsl = ma1[cross_mask].size
        grid_delays = da.from_array(
            grid_delays[cross_mask],
            chunks=(max(1, np.floor(n_bsl / os.cpu_count())), npix, npix),
        )

        # Mean in time the visibilities
        vis = np.mean(self.value, axis=0)[..., cross_mask]  # (nfreqs, nvis)
        vis = da.from_array(
            vis,
            chunks=(
                1,
                max(1, np.floor(n_bsl / os.cpu_count())),
            ),  # (self.frequency.size, np.floor(n_bsl/os.cpu_count()))
        )

        # Make the nearfield image
        log.info(
            f"Computing nearfield (time: {self.time.size}, frequency: {self.frequency.size}, baselines: {vis.shape[1]}, pixels: {posx.size})... "
        )
        wvl = wavelength(self.frequency).to(u.m).value
        phase = np.exp(
            2.0j * np.pi * (grid_delays[None, ...] / wvl[:, None, None, None])
        )
        log.debug("Computing the phase term...")
        with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
            phase = phase.compute()
        log.debug("Computing the nearf-field...")
        nearfield = compute_nearfield_imprint(vis, phase)

        # Compute nearfield imprints for other sources
        simu_sources = {}
        for src_name in sources:
            # Check that the source is visible
            if src_name.lower() in [
                "sun",
                "moon",
                "venus",
                "mars",
                "jupiter",
                "saturn",
                "uranus",
                "neptune",
            ]:
                src = SolarSystemTarget.from_name(name=src_name, time=obs_time)
            else:
                src = FixedTarget.from_name(name=src_name, time=obs_time)
            altaz = src.horizontal_coordinates  # [0]
            if altaz.alt.deg <= 10:
                log.debug(
                    f"{src_name}'s elevation {altaz[0].alt.deg}<=10deg, not considered for nearfield imprint."
                )
                continue

            # Projection from AltAz to ENU vector
            az_rad = altaz.az.rad
            el_rad = altaz.alt.rad
            cos_az = np.cos(az_rad)
            sin_az = np.sin(az_rad)
            cos_el = np.cos(el_rad)
            sin_el = np.sin(el_rad)
            to_enu = np.array([cos_el * sin_az, cos_el * cos_az, sin_el])
            # src_delays = np.matmul(
            #     ma_enu[ma1] - ma_enu[ma2],
            #     to_enu
            # )
            # src_delays = da.from_array(
            #     src_delays[cross_mask, :],
            #     chunks=((np.floor(n_bsl/os.cpu_count()), npix, npix), 1)
            # )

            ma1_enu = da.from_array(
                ma_enu[ma1[cross_mask]], chunks=max(1, np.floor(n_bsl / os.cpu_count()))
            )
            ma2_enu = da.from_array(
                ma_enu[ma2[cross_mask]], chunks=max(1, np.floor(n_bsl / os.cpu_count()))
            )
            src_delays = np.matmul(ma1_enu - ma2_enu, to_enu)

            # Simulate visibilities
            src_vis = np.exp(2.0j * np.pi * (src_delays / wvl))
            src_vis = np.swapaxes(src_vis, 1, 0)
            log.debug(f"Computing the nearf-field imprint of {src_name}...")
            simu_sources[src_name] = compute_nearfield_imprint(src_vis, phase)

        return nearfield, simu_sources


# ============================================================= #
# ------------------------- Crosslet -------------------------- #
class Crosslet(ABC):
    """Crosslet abstract class (both for XST and NenuFAR TV dat files)."""

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    @abstractmethod
    def phase_center(self) -> SkyCoord:
        """_summary_

        Returns
        -------
        SkyCoord
            _description_
        """
        return NotImplementedError

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(
        self,
        polarization: str = "XX",
        miniarray_selection: np.ndarray = None,
        frequency_selection: str = None,
        time_selection: str = None,
    ) -> XST_Slice:
        """Main data selection method.

        Parameters
        ----------
        polarization : `str`, optional
            Cross-correlation polarization selection (can either be "XX", "XY", "YX", "YY"), for Stokes parameters see :meth:`~nenupy.io.xst.Crosslet.get_stokes`, by default "XX"
        miniarray_selection : :class:`~numpy.ndarray`, optional
            Mini-Arrays selection, the visibilities will be filter to only take into account cross-correlations involving Mini-Arrays from the list provided, by default `None` (i.e., all available Mini-Arrays)
        frequency_selection : `str`, optional
            Frequency selection code (e.g., ">=40MHz & <51.1MHz"), by default `None` (i.e., all available frequencies)
        time_selection : `str`, optional
            Time selection code (e.g., "<2024-06-21T12:32:40"), by default `None` (i.e., all available time samples)

        Returns
        -------
        :class:`~nenupy.io.xst.XST_Slice`
            Data selection.
        
        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST
            >>> import numpy as np

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> xx_data = xst.get(
                    polarization="XX",
                    miniarray_selection=np.array([0, 2, 3]),
                    frequency_selection=">73MHz & <75MHz",
                    time_selection="<2020-02-19T19:00:00"
                )
            >>> xx_data.value.shape, xx_data.time.shape, xx_data.frequency.shape
            ((1, 7, 6), (1,), (7,))

        """
        mas = self._select_mini_arrays(miniarray_selection)
        frequency_mask = self._get_freq_mask(frequency_selection)
        time_mask = self._get_time_mask(time_selection)

        return XST_Slice(
            mini_arrays=mas,
            time=self.time[time_mask],
            frequency=self.frequencies[frequency_mask],
            value=self._get(
                frequency_selection=frequency_selection,
                time_selection=time_selection,
                polarization=polarization,
                mini_arrays=mas,
            ),
        )

    def rephase(self, phase_center: SkyCoord) -> XST_Slice:
        """_summary_

        Parameters
        ----------
        phase_center : SkyCoord
            _description_

        Returns
        -------
        XST_Slice
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        # Check the phase_center
        if not phase_center.isscalar:
            raise ValueError("phase_center must be scalar")

        log.info(f"Rephasing the visibilities towards {phase_center}...")

        # Normalize the current phase center
        if self._phase_center.isscalar or self._phase_center.size != self.time.size:
            if isinstance(self._phase_center.frame, AltAz):
                current_phase_center_altaz = SkyCoord(
                    np.ones(self.time.size) * self._phase_center.az.deg,
                    np.ones(self.time.size) * self._phase_center.alt.deg,
                    unit="deg",
                    frame=AltAz(
                        obstime=self.time,
                        location=nenufar_position
                    )
                )
                current_phase_center = altaz_to_radec(current_phase_center_altaz)
            else:
                current_phase_center = SkyCoord(
                    np.ones(self.time.size) * self._phase_center.ra.deg,
                    np.ones(self.time.size) * self._phase_center.dec.deg,
                    unit="deg"
                )
        else:
            current_phase_center = phase_center

        # Compute the UVW coordinates
        current_uvw = compute_uvw(
            interferometer=NenuFAR()[self.mini_arrays],
            phase_center=current_phase_center,
            time=self.time,
        ).to_value(u.m) # (times, nvis, 3)

        # Transformation matrices
        to_origin = _vis_rotation_matrix(current_phase_center)  # (3, 3, times)
        to_new_center = _vis_rotation_matrix(phase_center)  # (3, 3, 1)
        total_transformation = np.matmul(
            np.transpose(to_new_center, (2, 0, 1)), to_origin
        )  # (3, 3, ntimes)
        rotUVW = np.matmul(
            np.expand_dims((to_origin[2, :] - to_new_center[2, :]).T, axis=1),
            np.transpose(to_origin, (2, 1, 0)),
        )  # (ntimes, 1, 3)
        phase = np.matmul(rotUVW, np.transpose(current_uvw, (0, 2, 1)))  # (ntimes, 1, nvis)
        rotated_visibilities = np.exp(
            2.0j
            * np.pi
            * phase
            / wavelength(self.frequencies[0, :]).to_value(u.m)[None, :, None]
        )  # (ntimes, nfreqs, nvis)

        return XST_Slice(
            mini_arrays=self.mini_arrays,
            time=self.time,
            frequency=self.frequencies,
            value=rotated_visibilities,
            phase_center=phase_center
        )

    def get_stokes(
        self,
        stokes: str = "I",
        miniarray_selection: np.ndarray = None,
        frequency_selection: str = None,
        time_selection: str = None,
    ) -> XST_Slice:
        r"""Converts cross-correlation visibilities to Stokes parameter.
        Available Stokes parameters "I", "Q", "U", "V", "FL" or "FV" are respectively computed as follows:

        .. math::
            \begin{cases}
                \rm{I} = \frac{1}{2}(\rm{XX} + \rm{YY})\\
                \rm{Q} = \frac{1}{2}(\rm{XX} - \rm{YY})\\
                \rm{U} = \frac{1}{2}(\rm{XY} + \rm{YX})\\
                \rm{V} = \frac{-i}{2}(\rm{XY} - \rm{YX})\\
                \frac{\rm{L}}{\rm{I}} = \frac{\sqrt{\rm{Q}^2 + \rm{U}^2}}{\rm{I}}\\
                \frac{\rm{V}}{\rm{I}} = \frac{\rm{V}}{\rm{I}}\\
            \end{cases}

        Parameters
        ----------
        stokes : `str`, optional
            Stokes parameter to synthetize (can either be "I", "Q", "U", "V", "FL" or "FV"), by default "I"
        miniarray_selection : :class:`~numpy.ndarray`, optional
            Mini-Arrays selection, the visibilities will be filter to only take into account cross-correlations involving Mini-Arrays from the list provided, by default `None` (i.e., all available Mini-Arrays)
        frequency_selection : `str`, optional
            Frequency selection code (e.g., ">=40MHz & <51.1MHz"), by default `None` (i.e., all available frequencies)
        time_selection : `str`, optional
            Time selection code (e.g., "<2024-06-21T12:32:40"), by default `None` (i.e., all available time samples)

        Returns
        -------
        :class:`~nenupy.io.xst.XST_Slice`
            Data selection converted to desired Stokes parameter.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST
            >>> import numpy as np

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> u_data = xst.get_stokes(
                    polarization="U",
                    miniarray_selection=np.array([0, 2, 3]),
                    frequency_selection=">73MHz & <75MHz",
                    time_selection="<2020-02-19T19:00:00"
                )
            >>> u_data.value.shape, u_data.time.shape, u_data.frequency.shape
            ((1, 7, 6), (1,), (7,))

        """

        mas = self._select_mini_arrays(miniarray_selection)
        frequency_mask = self._get_freq_mask(frequency_selection)
        time_mask = self._get_time_mask(time_selection)

        stokes_parameters = {
            "I": {"cross": ["XX", "YY"], "compute": lambda xx, yy: 0.5 * (xx + yy)},
            "Q": {"cross": ["XX", "YY"], "compute": lambda xx, yy: 0.5 * (xx - yy)},
            "U": {"cross": ["XY", "YX"], "compute": lambda xy, yx: 0.5 * (xy + yx)},
            "V": {"cross": ["XY", "YX"], "compute": lambda xy, yx: -0.5j * (xy - yx)},
            "FL": {
                "cross": ["XX", "YY", "XY", "YX"],
                "compute": lambda xx, yy, xy, yx: np.sqrt(
                    (0.5 * (xx - yy)) ** 2 + (0.5 * (xy + yx)) ** 2
                )
                / (0.5 * (xx + yy)),
            },
            "FV": {
                "cross": ["XX", "YY", "XY", "YX"],
                "compute": lambda xx, yy, xy, yx: np.abs(-0.5j * (xy - yx))
                / (0.5 * (xx + yy)),
            },
        }

        try:
            selected_stokes = stokes_parameters[stokes]
        except KeyError:
            log.warning(f"Available polarizations are: {stokes_parameters.keys()}.")
            raise

        return XST_Slice(
            mini_arrays=mas,
            time=self.time[time_mask],
            frequency=self.frequencies[frequency_mask],
            value=selected_stokes["compute"](
                *map(
                    lambda pol: self._get(
                        frequency_selection=frequency_selection,
                        time_selection=time_selection,
                        polarization=pol,
                        mini_arrays=mas,
                    ),
                    selected_stokes["cross"],
                )
            ),
        )

    def get_beamform(
        self,
        pointing: Pointing,
        frequency_selection: str = None,
        time_selection: str = None,
        mini_arrays: np.ndarray = np.array([0, 1]),
        polarization: str = "NW",
        calibration: str = "default",
    ) -> ST_Slice:
        """Perform beamforming operation on XST-like NenuFAR visibilities.
        In a nutsheel, this method transforms XST into BST.

        Parameters
        ----------
        pointing : :class:`~nenupy.astro.pointing.Pointing`
            Pointing instance, describing where the beamforming must be applied across time
        frequency_selection : `str`, optional
            Frequency selection (see :meth:`~nenupy.io.xst.Crosslet.get`), by default `None`
        time_selection : `str`, optional
            Time selection (see :meth:`~nenupy.io.xst.Crosslet.get`), by default `None`
        mini_arrays : :class:`~numpy.ndarray`, optional
            Mini-Array selection (see :meth:`~nenupy.io.xst.Crosslet.get`), by default ``np.array([0, 1])``
        polarization : `str`, optional
            Polarization selection of the output data (can either be "NW" or "NE"), by default "NW"
        calibration : `str`, optional
            Residual delay calibration file to be used during beamforming, "none" does not apply any calibration, by default "default"

        Returns
        -------
        :class:`~nenupy.io.xst.XST_Slice`
            Beamformed data from XST-like visibilities.

        Raises
        ------
        IndexError
            Raised if the Mini Array selection is not valid.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.bst import BST, XST
            >>> from nenupy.astro.pointing import Pointing
            >>> bst = BST("20191129_141900_BST.fits")
            >>> xst = XST("20191129_141900_XST.fits")
            >>> bf_cal = xst.get_beamform(
                    pointing = Pointing.from_bst(bst, beam=0, analog=False),
                    mini_arrays=bst.mini_arrays,
                    calibration="default"
                )

        See Also
        --------
        :ref:`xst_beamforming_doc`
        """
        frequency_mask = self._get_freq_mask(frequency_selection)
        time_mask = self._get_time_mask(time_selection)

        # Select the mini-arrays cross correlations
        nenufar = NenuFAR()  # [self.mini_arrays]
        bf_nenufar = NenuFAR()[mini_arrays]
        ma_real_indices = np.array(
            [nenufar_miniarrays[name]["id"] for name in bf_nenufar.antenna_names]
        )
        if np.any(~np.isin(ma_real_indices, self.mini_arrays)):
            raise IndexError(
                f"Selected Mini-Arrays {mini_arrays} are outside possible values: {self.mini_arrays}."
            )
        ma_indices = np.arange(self.mini_arrays.size, dtype="int")[
            np.isin(self.mini_arrays, ma_real_indices)
        ]
        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        mask = np.isin(ma1, ma_indices) & np.isin(ma2, ma_indices)

        # Calibration table
        if calibration.lower() == "none":
            # No calibration
            cal = np.ones((self.frequencies[frequency_mask].size, ma_indices.size))
        else:
            pol_idx = {"NW": [0], "NE": [1]}
            cal = read_cal_table(calibration_file=calibration)
            cal = cal[
                np.ix_(
                    freq2sb(self.frequencies[frequency_mask]),
                    ma_real_indices,
                    pol_idx[polarization],
                )
            ].squeeze(axis=2)

        # Load and filter the data
        vis = self._get(
            frequency_selection=frequency_selection,
            time_selection=time_selection,
            polarization="XX" if polarization.upper() == "NW" else "YY",
        )[:, :, mask]

        # Insert the data in a matrix
        tri_x, tri_y = np.tril_indices(ma_indices.size, 0)
        vis_matrix = np.zeros(
            (
                self.time[time_mask].size,
                self.frequencies[frequency_mask].size,
                ma_indices.size,
                ma_indices.size,
            ),
            dtype=complex,
        )
        vis_matrix[:, :, tri_x, tri_y] = vis
        vis_matrix[:, :, tri_y, tri_x] = vis_matrix[:, :, tri_x, tri_y].conj()

        # Calibrate the Xcorr with the caltable
        for fi in range(vis_matrix.shape[1]):
            cal_i = np.expand_dims(cal[fi], axis=1)
            cal_i_h = np.expand_dims(cal[fi].T.conj(), axis=0)
            mul = np.dot(cal_i, cal_i_h)
            vis_matrix[:, fi, :, :] *= mul[np.newaxis, :, :]

        # Phase the visibilities towards the phase center
        phase = np.ones(
            (
                self.time[time_mask].size,
                self.frequencies[frequency_mask].size,
                ma_indices.size,
                ma_indices.size,
            ),
            dtype=complex,
        )
        altaz_pointing = pointing.horizontal_coordinates
        if altaz_pointing.size == 1:
            # Transit
            pass
        else:
            # Multiple pointings, get the correct value for all times
            altaz_pointing = pointing[self.time[time_mask]].horizontal_coordinates
        az = altaz_pointing.az.rad
        el = altaz_pointing.alt.rad
        ground_projection = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        rot = np.radians(-90)
        rotation = np.array(
            [[np.cos(rot), np.sin(rot), 0], [-np.sin(rot), np.cos(rot), 0], [0, 0, 1]]
        )
        ma1_pos = np.dot(nenufar.antenna_positions[ma1[mask]], rotation)
        ma2_pos = np.dot(nenufar.antenna_positions[ma2[mask]], rotation)
        dphi = np.dot(ma1_pos - ma2_pos, ground_projection).T
        wvl = wavelength(self.frequencies[frequency_mask]).to(u.m).value
        phase[:, :, tri_x, tri_y] = np.exp(
            -2.0j * np.pi / wvl[None, :, None] * dphi[:, None, :]
        )
        phase[:, :, tri_y, tri_x] = phase[:, :, tri_x, tri_y].conj().copy()
        data = np.sum((vis_matrix * phase).real, axis=(2, 3))

        return ST_Slice(
            time=self.time[time_mask],
            frequency=self.frequencies[frequency_mask],
            value=data.squeeze()
        )

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @abstractmethod
    def _parse_frequency_condition(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_time_condition(self):
        raise NotImplementedError

    def _select_mini_arrays(self, mini_arrays):
        """ """
        if mini_arrays is None:
            mini_arrays = self.mini_arrays
        if np.any(~np.isin(mini_arrays, self.mini_arrays)):
            raise IndexError(
                f"Selected Mini-Arrays {mini_arrays} are outside possible values: {self.mini_arrays}."
            )
        return mini_arrays

    def _get_freq_mask(self, frequency_selection=None):
        """ """
        # Frequency selection
        frequencies = self.frequencies
        if frequency_selection is None:
            frequency_selection = f">={frequencies.min()} & <= {frequencies.max()}"
        frequency_mask = self._parse_frequency_condition(frequency_selection)(
            frequencies
        )
        if not np.any(frequency_mask):
            log.warning(
                "Empty frequency selection, input values should fall "
                f"between {frequencies.min()} and {frequencies.max()}, "
                f"i.e.: '>={frequencies.min()} & <= {frequencies.max()}'"
            )
        return frequency_mask

    def _get_time_mask(self, time_selection=None):
        """ """
        # Time selection
        if time_selection is None:
            time_selection = f">={self.time[0].isot} & <= {self.time[-1].isot}"
        time_mask = self._parse_time_condition(time_selection)(self.time)
        if not np.any(time_mask):
            log.warning(
                "Empty time selection, input values should fall "
                f"between {self.time[0].isot} and {self.time[-1].isot}, "
                f"i.e.: '>={self.time[0].isot} & <= {self.time[-1].isot}'"
            )
        return time_mask

    def _get_cross_idx(self, c1="X", c2="X", mini_arrays=None):
        """Retrieves visibilities indices for the given cross polarizations"""

        mini_arrays_size = self.mini_arrays.size

        # Mini-arrays selection
        ma_indices = np.arange(mini_arrays_size, dtype="int")[
            np.isin(self.mini_arrays, mini_arrays)
        ]

        # Polarization array
        corr = np.array(["X", "Y"] * mini_arrays_size)
        i_ant1, i_ant2 = np.tril_indices(mini_arrays_size * 2, 0)

        # Define polarization and mini-arrays masks
        corr_mask = (corr[i_ant1] == c1) & (corr[i_ant2] == c2)
        ma_mask = np.isin(i_ant1 // 2, ma_indices) & np.isin(i_ant2 // 2, ma_indices)

        indices = np.arange(i_ant1.size)[corr_mask & ma_mask]

        return indices

    def _get(
        self,
        frequency_selection: str = None,
        time_selection: str = None,
        polarization: str = "XX",
        mini_arrays: np.ndarray = None,
    ) -> np.ndarray:
        """ """
        # Polarization selection
        allowed_polarizations = ["XX", "XY", "YX", "YY"]
        if polarization not in allowed_polarizations:
            raise ValueError(
                f"'polarization' argument must be one of the following: {allowed_polarizations}."
            )

        # Frequency selection
        frequency_mask = self._get_freq_mask(frequency_selection)

        # Time selection
        time_mask = self._get_time_mask(time_selection)

        # Combined mask
        time_freq_mask = time_mask[:, None] * frequency_mask

        # Final shape
        if mini_arrays is None:
            mini_arrays = self.mini_arrays
        ma1, ma2 = np.tril_indices(mini_arrays.size, 0)
        # final_shape = (np.sum(time_mask), np.sum(frequency_mask), ma1.size)
        final_shape = (
            np.any(time_freq_mask, axis=1).sum(),
            np.any(time_freq_mask, axis=0).sum(),
            ma1.size,
        )

        if polarization == "XY":
            # Deal with lack of auto XY cross in XST-like data
            auto_mask = ma1 == ma2
            cross_mask = ~auto_mask

            data_tf_selected = self.data[time_freq_mask]
            # Deal with dimension cut in case of True over an entire axis
            if final_shape[0] == 1:
                data_tf_selected = np.expand_dims(data_tf_selected, axis=0)
            if final_shape[1] == 1:
                data_tf_selected = np.expand_dims(data_tf_selected, axis=1)

            yx = data_tf_selected[:, :, self._get_cross_idx("Y", "X", mini_arrays)]

            _xy = np.zeros((list(yx.shape[:-1]) + [ma1.size]), dtype=complex)
            _xy[:, :, auto_mask] = yx[:, :, auto_mask].conj()

            # Get XY correlations
            _xy[:, :, cross_mask] = data_tf_selected[
                :, :, self._get_cross_idx("X", "Y", mini_arrays)
            ]
            return _xy.reshape(final_shape)
        else:
            return self.data[time_freq_mask[:, :]][
                :, self._get_cross_idx(*list(polarization), mini_arrays)
            ].reshape(final_shape)
            # return self.data[
            #     np.ix_(
            #         time_mask,
            #         frequency_mask,
            #         self._get_cross_idx(*list(polarization), mini_arrays)
            #     )
            # ].reshape(final_shape)


# ============================================================= #
# ---------------------------- XST ---------------------------- #
class XST(StatisticsData, Crosslet):
    """Crosslet STatistics reading class.

    See Also
    --------
    :ref:`xst_reading_doc`
    """

    def __init__(self, file_name):
        super().__init__(file_name=file_name)

        self._phase_center = SkyCoord(
            0*u.deg, 90*u.deg,
            frame=AltAz(
                obstime=self.time[0] + (self.time[-1] - self.time[0]) / 2,
                location=nenufar_position
            )
        )

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def phase_center(self) -> SkyCoord:
        """_summary_

        Returns
        -------
        SkyCoord
            _description_
        """
        return self._phase_center

    @property
    def mini_arrays(self):
        """Retrieves the list of Mini-Arrays used to get the cross-correlations.

        :getter: Mini-Arrays list.

        :type: :class:`~numpy.ndarray`
        """
        return self._meta_data["ins"]["noMROn"][0]


# ============================================================= #
# ------------------------- TV_Image -------------------------- #
class TV_Image:
    """Class to store and display NenuFAR TV image."""

    def __init__(
        self, tv_image: HpxSky, analog_pointing: SkyCoord, fov_radius: u.Quantity
    ):
        """Produce an instance of :class:`~nenupy.io.xst.TV_Image`.

        Parameters
        ----------
        tv_image : :class:`~nenupy.astro.sky.HpxSky`
            The celestial image in HEALPix format, such as returned by :meth:`~nenupy.io.xst.Crosslet.make_image` for instance
        analog_pointing : :class:`~astropy.coordinates.SkyCoord`
            Celestial coordinates of the mean Mini-Array pointing in effect during the data acquisition, only serves as image display center
        fov_radius : :class:`~astropy.units.Quantity`
            Radius of the image field of view

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 16

            >>> from nenupy.io.xst import XST, TV_Image
            >>> from nenupy.astro import radec_to_altaz
            >>> from astropy.coordinates import SkyCoord
            >>> import astropy.units as u

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> i_data = xst.get_stokes(stokes="I")
            >>> tau_a = SkyCoord.from_name("Tau A")
            >>> im = i_data.make_image(
                    resolution=1*u.deg,
                    fov_radius=10*u.deg,
                    phase_center=tau_a,
                    stokes="I"
                )
            >>> pointing = radec_to_altaz(tau_a, xst.time[0])
            >>> tv_im = TV_Image(im, pointing, 10*u.deg)

        """
        self.tv_image = tv_image
        self.analog_pointing = analog_pointing
        self.fov_radius = fov_radius

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def analog_pointing(self) -> SkyCoord:
        """Analog pointing representing the image center in Alt-Az frame. 

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`
            The celestial coordinates in Alt-Az frame
        
        Raises
        -----
        ValueError
            Raised if the celestial coordinates frame is different from :class:`~astropy.coordinates.AltAz`.
        """
        return self._analog_pointing
    @analog_pointing.setter
    def analog_pointing(self, coord: SkyCoord):
        if not isinstance(coord.frame, AltAz):
            raise ValueError(f"analog_pointing pointing frame should be 'AltAz', got '{coord.frame}' instead.")
        self._analog_pointing = coord

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_fits(cls, file_name: str):
        """Read the content of a FITS file and return an instance of :class:`~nenupy.io.xst.TV_Image`.

        Parameters
        ----------
        file_name : `str`
            FITS file path storing the NenuFAR-TV image (such as produced by :meth:`~nenupy.io.xst.TV_Image.save_fits`)

        Returns
        -------
        :class:`~nenupy.io.xst.TV_Image`
            NenuFAR-TV image loaded
        
        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import TV_Image

            >>> tv_im = TV_Image.from_fits(".../my_nenufar_tv_image.fits")

        """
        header = fits.getheader(file_name, ext=1)

        # Load the image
        image = read_map(file_name, dtype=None, partial="PARTIAL" in header["OBJECT"])
        # Fill NaNs
        if "PARTIAL" in header["OBJECT"]:
            image[mask_bad(image)] = np.nan

        # Re-create the sky
        sky = HpxSky(
            resolution=Angle(
                angle=nside2resol(header["NSIDE"], arcmin=True), unit=u.arcmin
            ),
            time=Time(header["OBSTIME"]),
            frequency=header["FREQ"] * u.MHz,
            polarization=np.array([header["STOKES"]]),
            value=image.reshape((1, 1, 1, image.size)),
        )

        return cls(
            tv_image=sky,
            analog_pointing=SkyCoord(
                header["AZANA"] * u.deg,
                header["ELANA"] * u.deg,
                frame=AltAz(obstime=Time(header["OBSTIME"]), location=nenufar_position),
            ),
            fov_radius=header["FOV"] * u.deg / 2,
        )

    def save_fits(self, file_name: str, partial: bool = True) -> None:
        """Store the NenuFAR-TV image in a FITS file.
        The HEALPix FITS file is written using :func:`~healpy.fitsfunc.write_map`.

        Parameters
        ----------
        file_name : `str`
            Name of the FITS file to be written
        partial : `bool`, optional
            If `True`, the HEALPix map is only written where the sky isn't masked to save space (argument directly passed to :func:`~healpy.fitsfunc.write_map`), by default `True`

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 17

            >>> from nenupy.io.xst import XST, TV_Image
            >>> from astropy.coordinates import SkyCoord
            >>> import astropy.units as u
            >>> from nenupy.astro import radec_to_altaz

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> i_data = xst.get_stokes(stokes="I")
            >>> tau_a = SkyCoord.from_name("Tau A")
            >>> im = i_data.make_image(
                    resolution=1*u.deg,
                    fov_radius=10*u.deg,
                    phase_center=tau_a,
                    stokes="I"
                )
            >>> pointing = radec_to_altaz(tau_a, xst.time[0])
            >>> tv_im = TV_Image(im, pointing, 10*u.deg)
            >>> tv_im.save_fits(".../nenufar-tv.fits")

        """

        phase_center_eq = altaz_to_radec(self.analog_pointing)

        header = [
            ("software", "nenupy"),
            ("version", nenupy.__version__),
            ("contact", nenupy.__email__),
            ("azana", self.analog_pointing.az.deg),
            ("elana", self.analog_pointing.alt.deg),
            ("freq", self.tv_image.frequency[0].to(u.MHz).value),
            ("obstime", self.tv_image.time[0].isot),
            ("fov", self.fov_radius.to(u.deg).value * 2),
            ("pc_ra", phase_center_eq.ra.deg),
            ("pc_dec", phase_center_eq.dec.deg),
            ("stokes", self.tv_image.polarization[0]),
        ]

        map2write = self.tv_image.value[0, 0, 0].copy()
        write_map(
            filename=file_name,
            m=map2write,
            nest=False,
            coord="C",
            overwrite=True,
            dtype=self.tv_image.value.dtype,
            extra_header=header,
            partial=partial,
        )
        log.info(
            "HEALPix image of {} cells (nside={}) saved in '{}'.".format(
                map2write.size, self.tv_image.nside, file_name
            )
        )

    def save_png(
        self,
        figname: str = "",
        beam_contours: bool = True,
        show_sources: bool = True,
        **kwargs,
    ) -> None:
        """Plot the NenuFAR-TV image.
        Most of the arguments are passed to :meth:`~nenupy.astro.sky.SkySliceBase.plot`.

        Parameters
        ----------
        figname : `str`, optional
            If a name is provided, the figure will be saved, by default ""
        beam_contours : `bool`, optional
            Display the simulated analog beam contours, by default `True`
        show_sources : `bool`, optional
            Show the positions of (bright) radio sources listed in ``nenufar_tv_sources.json``, by default `True`

        Example
        -------
        .. code-block:: python
            :emphasize-lines: 16,17

            >>> from nenupy.io.xst import XST, TV_Image
            >>> from astropy.coordinates import SkyCoord
            >>> import astropy.units as u
            >>> from nenupy.astro import radec_to_altaz

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> i_data = xst.get_stokes(stokes="I")
            >>> tau_a = SkyCoord.from_name("Tau A")
            >>> im = i_data.make_image(
                    resolution=1*u.deg,
                    fov_radius=10*u.deg,
                    phase_center=tau_a,
                    stokes="I"
                )
            >>> pointing = radec_to_altaz(tau_a, xst.time[0])
            >>> tv_im = TV_Image(im, pointing, 10*u.deg)
            >>> tv_im.save_png()

        .. figure:: ../_images/io_images/tv_image.png
            :width: 450
            :align: center
        """
        image_center = altaz_to_radec(
            SkyCoord(
                self.analog_pointing.az,
                self.analog_pointing.alt,
                frame=AltAz(obstime=self.tv_image.time[0], location=nenufar_position),
            )
        )

        if show_sources:
            src_names = []
            src_position = []

            with open(join(dirname(__file__), "nenufar_tv_sources.json")) as src_file:
                sources = json.load(src_file)

            for name in sources["FixedSources"]:
                src = FixedTarget.from_name(name, time=self.tv_image.time[0])
                if src.coordinates.separation(image_center) <= 0.8 * self.fov_radius:
                    src_names.append(name)
                    src_position.append(src.coordinates)
            for name in sources["SolarSystemSources"]:
                src = SolarSystemTarget.from_name(name, time=self.tv_image.time[0])
                if src.coordinates.separation(image_center) <= 0.8 * self.fov_radius:
                    src_names.append(name)
                    src_position.append(src.coordinates)

            if len(src_position) != 0:
                kwargs["text"] = (SkyCoord(src_position), src_names, "white")

        if beam_contours:
            # Simulate the array factor
            ma = MiniArray()
            af_sky = ma.array_factor(
                sky=HpxSky(
                    resolution=0.2 * u.deg,
                    time=self.tv_image.time[0],
                    frequency=self.tv_image.frequency[0],
                ),
                pointing=Pointing(coordinates=image_center, time=self.tv_image.time[0]),
            )
            # Normalize the array factor
            af = af_sky[0, 0, 0].compute()
            af_normalized = af / af.max()
            kwargs["contour"] = (af_normalized, np.arange(0.5, 1, 0.2), "copper")

        # Plot
        self.tv_image[0, 0, 0].plot(
            center=image_center,
            radius=self.fov_radius - 2.5 * u.deg,
            figname=figname,
            colorbar_label=f"Stokes {self.tv_image.polarization[0]}",
            **kwargs,
        )


# ============================================================= #
# ----------------------- TV_Nearfield ------------------------ #
class TV_Nearfield:
    """Class to handle NenuFAR TV nearfield storage and display."""

    def __init__(
        self,
        nearfield: np.ndarray,
        source_imprints: dict,
        npix: int,
        time: Time,
        frequency: u.Quantity,
        radius: u.Quantity,
        mini_arrays: np.ndarray,
        stokes: str,
    ):
        """Produce an instance of :class:`~nenupy.io.xst.TV_Nearfield`.

        Parameters
        ----------
        nearfield : :class:`~numpy.ndarray`
            The Near-field map to display, image must be square
        source_imprints : `dict`
            Dictionnary of Near-field imprints from radio sources, each imprint must be of same shape as the main near-field image
        npix : `int`
            Side pixel width of the nearfield image
        time : :class:`~astropy.time.Time`
            Average time of the near-field data acquisition
        frequency : :class:`~astropy.units.Quantity`
            Average frequency of the near-field data acquisition
        radius : :class:`~astropy.units.Quantity`
            Ground radius of projected near-field
        mini_arrays : :class:`~numpy.ndarray`
            Mini-Arrays indices used to compute the near-field
        stokes : `str`
            Stokes parameter represented
        
        Raises
        ------
        ValueError
            Raised if nearfield is not square, or source_imprints do not match nearfield, or npix does not match the nearfield dimensions

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST
            >>> from nenupy.io.xst import TV_Nearfield
            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> xx_data = xst.get(polarization="XX")
            >>> nearfield, src_dict = xx_data.make_nearfield(
                    radius=400*u.m,
                    npix=64,
                    sources=["Tau A"]
                )
            >>> tv_nf = TV_Nearfield(
                    nearfield=nearfield,
                    source_imprints=src_dict,
                    npix=nearfield.shape[0],
                    time=xst.time[0],
                    frequency=xst.frequencies[0],
                    radius=400*u.m,
                    mini_arrays=xst.mini_arrays,
                    stokes="XX",
                )

        """
        if nearfield.shape[0] != nearfield.shape[1]:
            raise ValueError(f"Nearfield (of dimensions {nearfield.shape}) is not square.")
        self.nearfield = nearfield
        for source, imprint in source_imprints.items():
            if imprint.shape != nearfield.shape:
                raise ValueError(f"{source} imprint's dimension {imprint.shape} does not match nearfield shape {nearfield.shape}.")
        self.source_imprints = source_imprints
        if npix != nearfield.shape[0]:
            raise ValueError(f"npix={npix} does not match nearfield dimensions {nearfield.shape}.")
        self.npix = npix
        self.time = time
        self.frequency = frequency
        self.radius = radius
        self.mini_arrays = mini_arrays
        self.stokes = stokes

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_fits(cls, file_name: str):
        """Load a nearfield previously stored in a FITS file.

        Parameters
        ----------
        file_name : `str`
            Path to the FITS file containing a near-field
            image (whose format is such as created by the
            :meth:`~nenupy.io.xst.TV_Nearfield.save_fits`
            method).

        Returns
        -------
        :class:`~nenupy.io.xst.TV_Nearfield`
            Instance of :class:`~nenupy.io.xst.TV_Nearfield`.

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import TV_Nearfield
            >>> nf = TV_Nearfield.from_fits("/path/to/nearfield.fits")

        """
        
        reserved_names = ["PRIMARY", "NEAR-FIELD", "MINI-ARRAYS"]
        hdus = fits.open(file_name)
        nf_header = hdus["NEAR-FIELD"].header
        nf = cls(
            nearfield=hdus["NEAR-FIELD"].data,
            mini_arrays=hdus["MINI-ARRAYS"].data,
            npix=nf_header["NAXIS1"],
            frequency=nf_header["FREQUENC"] * u.MHz,
            time=Time(nf_header["DATE-OBS"]),
            source_imprints={
                hdu.header["SOURCE"]: hdu.data
                for hdu in hdus
                if hdu.name not in reserved_names
            },
            radius=nf_header["RADIUS"] * u.m,
            stokes=nf_header["STOKES"],
        )
        return nf

    def save_fits(self, file_name: str) -> None:
        """Save a near-field made from NenuFAR-TV data as a FITS file.

        Parameters
        ----------
        file_name : `str`
            Name of the file to save

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import NenufarTV
            >>> tv = NenufarTV("20191204_132113_nenufarTV.dat")
            >>> nf_object = tv.compute_nearfield_tv(sources=["Cyg A", "Cas A", "Sun"])
            >>> nf_object.save_fits(file_name="/path/to/nearfield.fits")

        """

        # Header
        prim_header = fits.Header()
        prim_header["OBSERVER"] = "NenuFAR Team"
        prim_header["AUTHOR"] = f"nenupy {nenupy.__version__}"
        prim_header["DATE"] = Time.now().isot
        prim_header["INSTRUME"] = "XST"
        prim_header["OBSERVER"] = "NenuFAR-TV"
        prim_header[
            "ORIGIN"
        ] = "Station de Radioastronomie de Nancay, LESIA, Observatoire de Paris"
        prim_header[
            "REFERENC"
        ] = "Alan Loh and the NenuFAR team, nenupy, 2020 (DOI: 10.5281/zenodo.3775196.)"
        prim_header["TELESCOP"] = "NenuFAR"

        prim_hdu = fits.PrimaryHDU(header=prim_header)

        # Near-Field
        nf_header = fits.Header()
        nf_header["NAXIS"] = 2
        nf_header["NAXIS1"] = self.npix
        nf_header["NAXIS2"] = self.npix
        nf_header["DATE-OBS"] = (self.time.isot, "Mean observation UTC date")
        nf_header["DATAMIN"] = self.nearfield.min()
        nf_header["DATAMAX"] = self.nearfield.max()
        nf_header["FREQUENC"] = (
            self.frequency.to_value(u.MHz),
            "Mean observing frequency in MHz.",
        )
        nf_header["STOKES"] = self.stokes.upper()
        nf_header["DESCRIPT"] = "Near-Field image."
        nf_header["RADIUS"] = (
            self.radius.to_value(u.m),
            "Radius of the ground (in m).",
        )

        nf_hdu = fits.ImageHDU(data=self.nearfield, header=nf_header, name="Near-Field")

        # Mini-Arrays
        ant_header = fits.Header()
        ant_header["DESCRIPT"] = "Mini-Array names"
        ant_hdu = fits.ImageHDU(
            data=self.mini_arrays, header=ant_header, name="Mini-Arrays"
        )

        # HDU list
        hduList = fits.HDUList([prim_hdu, nf_hdu, ant_hdu])

        for src in self.source_imprints:
            hdu_name = src.replace(" ", "_")
            src_header = fits.Header()
            src_header["NAXIS"] = 2
            src_header["NAXIS1"] = self.npix
            src_header["NAXIS2"] = self.npix
            src_header["DATE-OBS"] = (self.time.isot, "Mean observation UTC date")
            src_header["DATAMIN"] = self.source_imprints[src].min()
            src_header["DATAMAX"] = self.source_imprints[src].max()
            src_header["FREQUENC"] = (
                self.frequency.to_value(u.MHz),
                "Mean observing frequency in MHz.",
            )
            src_header["STOKES"] = self.stokes.upper()
            src_header["SOURCE"] = (src, "Name of the source imprint on the near-field")
            src_header["DESCRIPT"] = "Normalized sky source imprint on the near-field."
            src_header["RADIUS"] = (
                self.radius.to_value(u.m),
                "Radius of the ground (in m).",
            )
            src_hdu = fits.ImageHDU(
                data=self.source_imprints[src], name=hdu_name, header=src_header
            )
            hduList.append(src_hdu)

        hduList.writeto(file_name, overwrite=True)

        log.info(f"NearField saved in {file_name}.")

    def save_png(
        self,
        figname: str = "",
        fig: mpl.figure.Figure = None,
        ax: mpl.axes.Axes = None,
        figsize: Tuple[int, int] = (10, 10),
        decibel: bool = True,
        cmap: str = "YlGnBu_r",
        vmin: float = None,
        vmax: float = None,
        cbar_label: str = None,
        title: str = None,
    ) -> None:
        r"""Display the near-field image.

        Parameters
        ----------
        figname : `str`, optional
            Name of the figure, if given the figure will be saved, by default ""
        fig : :class:`~matplotlib.figure.Figure`, optional
            `matplotlib` figure instance, by default `None`
        ax : :class:`~matplotlib.axes.Axes`, optional
            `matplotlib` ax instance, by default `None`
        figsize : Tuple[`int`, `int`], optional
            Size of the figure, by default (10, 10)
        decibel : `bool`, optional
            Set the scale of the data displayed to dB (i.e., :math:`{\rm dB} = 10 \log_{10}({\rm data})`), by default True
        cmap : `str`, optional
            Color map used to represent the data, by default "YlGnBu_r"
        vmin : `float`, optional
            Minimal value displayed, by default `None` (i.e., overall minimal value)
        vmax : `float`, optional
            Maximal value displayed, by default `None` (i.e., overall maximal value)
        cbar_label : `str`, optional
            Label of the color bar, by default `None` (i.e., automatic label)
        title : `str`, optional
            Title of the plot, by default `None` (i.e., automatic label)
        
        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import XST, TV_Nearfield

            >>> xst = XST(".../nenupy/tests/test_data/XST.fits")
            >>> xx_data = xst.get(polarization="XX")
            >>> nearfield, src_dict = xx_data.make_nearfield(
                    radius=400*u.m,
                    npix=64,
                    sources=["Tau A"]
                )
            >>> tv_nf = TV_Nearfield(
                    nearfield=nearfield,
                    source_imprints=src_dict,
                    npix=nearfield.shape[0],
                    time=xst.time[0],
                    frequency=xst.frequencies[0],
                    radius=400*u.m,
                    mini_arrays=xst.mini_arrays,
                    stokes="XX",
                )
            >>> tv_nf.save_png(figname="")

        .. figure:: ../_images/io_images/make_nearfield_xst.png
            :width: 450
            :align: center

        """

        radius = self.radius.to_value(u.m)

        # Mini-Array positions in ENU coordinates
        nenufar = NenuFAR()[self.mini_arrays]
        ma_etrs = l93_to_etrs(nenufar.antenna_positions)
        ma_enu = etrs_to_enu(ma_etrs)

        # Plot the nearfield
        if fig is None:
            fig = plt.figure(figsize=figsize)

        if ax is None:
            ax = fig.add_subplot()

        nf_image = 10 * np.log10(self.nearfield) if decibel else self.nearfield
        nf_image_min = np.nanmin(nf_image) if vmin is None else vmin
        nf_image_max = np.nanmax(nf_image) if vmax is None else vmax

        ax.imshow(
            np.flipud(nf_image),  # This needs to be understood...
            cmap=cmap,
            extent=[-radius, radius, -radius, radius],
            zorder=0,
            vmin=nf_image_min,
            vmax=nf_image_max,
        )

        # Colorbar
        cax = inset_axes(
            ax,
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cb = ColorbarBase(
            cax,
            cmap=mpl.colormaps[cmap],
            orientation="vertical",
            norm=Normalize(
                vmin=nf_image_min,
                vmax=nf_image_max,
            ),
            ticks=LinearLocator(),
            format="%.2f",
        )
        cb.solids.set_edgecolor("face")
        cb.set_label(
            (f"dB (Stokes {self.stokes})" if decibel else f"amp (Stokes {self.stokes})")
            if cbar_label is None
            else cbar_label
        )

        # Show the contour of the simulated source imprints
        ground_granularity = np.linspace(-radius, radius, self.npix)
        posx, posy = np.meshgrid(ground_granularity, ground_granularity)
        dist = np.sqrt(posx**2 + posy**2)
        border_min = 0.1 * self.npix
        border_max = self.npix - 0.1 * self.npix
        for src in self.source_imprints.keys():
            # Normalize the imprint
            imprint = self.source_imprints[src]
            imprint /= imprint.max()
            # Plot the contours
            ax.contour(
                imprint,
                np.arange(0.8, 1, 0.04),
                cmap="copper",
                alpha=0.5,
                extent=[-radius, radius, -radius, radius],
                zorder=5,
            )
            # Find the maximum of the emission
            max_y, max_x = np.unravel_index(imprint.argmax(), imprint.shape)
            # If maximum outside the plot, recenter it
            if (
                (max_x <= border_min)
                or (max_y <= border_min)
                or (max_x >= border_max)
                or (max_y >= border_max)
            ):
                dist[dist <= np.median(dist)] = 0
                max_y, max_x = np.unravel_index(
                    ((1 - dist / dist.max()) * imprint).argmax(), imprint.shape
                )
            # Show the source name associated to the imprint
            ax.text(
                ground_granularity[max_x],
                ground_granularity[max_y],
                f" {src}",
                color="#b35900",
                fontweight="bold",
                va="center",
                ha="center",
                zorder=30,
            )

        # NenuFAR mini-array positions
        ax.scatter(ma_enu[:, 0], ma_enu[:, 1], 20, color="black", zorder=10)
        for i in range(ma_enu.shape[0]):
            ax.text(
                ma_enu[i, 0],
                ma_enu[i, 1],
                f" {self.mini_arrays[i]}",
                color="black",
                zorder=10,
            )
        # ax.scatter(
        #     building_enu[:, 0],
        #     building_enu[:, 1],
        #     20,
        #     color="tab:red",#'tab:orange',
        #     zorder=10
        # )

        # Plot axis labels
        ax.set_xlabel(r"$\Delta x$ (m)")
        ax.set_ylabel(r"$\Delta y$ (m)")
        ax.set_title(
            f"{np.mean(self.frequency.to_value(u.MHz)):.3f} MHz -- {self.time.isot}"
            if title is None
            else title
        )

        # Save or show the figure
        if (figname is None) or (figname == ""):
            plt.show()
        else:
            plt.savefig(figname, dpi=300, bbox_inches="tight", transparent=True)
            log.info(f"Figure '{figname}' saved.")
        plt.close("all")


# ============================================================= #
# ------------------------- NenufarTV ------------------------- #
class NenufarTV(StatisticsData, Crosslet):
    """Class to load and process NenuFAR TV data.
        Results of such operations can be seen live on `NenuFAR-TV <https://nenufar.obs-nancay.fr/nenufar-tv/>`_ where the current view of the Sky observed by NenuFAR is displayed.
    """

    def __init__(self, file_name: str):
        """Generate an instance of :class:`~nenupy.io.xst.NenufarTV`

        Parameters
        ----------
        file_name : `str`
            NenuFAR-TV data file (must end with .dat)
        
        Raises
        ------
        AssertionError
            Raised if the data file is not correctly formated
        """
        self.file_name = file_name
        self.mini_arrays = None
        self.time = None
        self.dt = None
        self.frequencies = None
        self.data = None
        self._load_tv_data()

        self._phase_center = SkyCoord(
            0*u.deg, 90*u.deg,
            frame=AltAz(
                obstime=self.time[0] + (self.time[-1] - self.time[0]) / 2,
                location=nenufar_position
            )
        )

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute_nenufar_tv(self, analog_pointing_file: str = None, fov_radius: u.Quantity = 27 * u.deg, resolution: u.Quantity = 0.5 * u.deg, stokes: str = "I") -> TV_Image:
        """Compute the NenuFAR-TV image.

        Parameters
        ----------
        analog_pointing_file : `str`, optional
            NenuFAR analog pointing file (as read by :meth:`~nenupy.astro.pointing.from_file`), by default `None` (i.e., a zenithal pointing is assumed)
        fov_radius : :class:`~astropy.units.Quantity`, optional
            Radius of the image field of view (passed to :meth:`~nenupy.io.xst.XST_Slice.make_image`), by default 27 degrees
        resolution : :class:`~astropy.units.Quantity`, optional
            Resolution of the image (passed to :meth:`~nenupy.io.xst.XST_Slice.make_image`), by default 0.5 degree
        stokes : `str`, optional
            Stokes parameter to display (passed to :meth:`~nenupy.io.xst.Crosslet.get_stokes`), by default "I"

        Returns
        -------
        :class:`~nenupy.io.xst.TV_Image`
            The NenuFAR-TV image
        
        Example
        -------
        .. code-block:: python
            
            >>> from nenupy.io.xst import NenufarTV

            >>> tv = NenufarTV("20191204_132113_nenufarTV.dat")
            >>> tv_obj = tv.compute_nenufar_tv()

        """

        obs_time = self.time[0] + (self.time[-1] - self.time[0]) / 2

        if analog_pointing_file is None:
            phase_center_altaz = SkyCoord(
                0,
                90,
                unit="deg",
                frame=AltAz(obstime=obs_time, location=nenufar_position),
            )
        else:
            pointing = Pointing.from_file(
                analog_pointing_file, include_corrections=False
            )[obs_time.reshape((1,))]
            phase_center_altaz = pointing.custom_ho_coordinates[0]

        data = self.get_stokes(stokes)
        sky_image = data.make_image(
            resolution=resolution,
            fov_radius=fov_radius,
            phase_center=altaz_to_radec(phase_center_altaz),
        )

        return TV_Image(
            tv_image=sky_image,
            analog_pointing=phase_center_altaz,
            fov_radius=fov_radius,
        )

    def compute_nearfield_tv(self, sources: list = [], stokes: str = "I", radius: u.Quantity = 400 * u.m, npix: int = 64) -> TV_Nearfield:
        """Compute the near-field from NenuFAR-TV data.

        Parameters
        ----------
        sources : `list`, optional
            List of celestial sources for which a near-field imprint will be computed (passed to :meth:`~nenupy.io.xst.XST_Slice.make_nearfield`), by default []
        stokes : `str`, optional
            Stokes parameter to display (passed to :meth:`~nenupy.io.xst.Crosslet.get_stokes`), by default "I"
        radius : :class:`~astropy.units.Quantity`, optional
            Ground radius on which the near-field projection is computed (passed to :meth:`~nenupy.io.xst.XST_Slice.make_nearfield`), by default 400 meters
        npix : `int`, optional
            Number of pixels of the image size (passed to :meth:`~nenupy.io.xst.XST_Slice.make_nearfield`), by default 64.

        Returns
        -------
        :class:`~nenupy.io.xst.TV_Nearfield`
            The near-field

        Example
        -------
        .. code-block:: python

            >>> from nenupy.io.xst import NenufarTV

            >>> tv = NenufarTV("20191204_132113_nenufarTV.dat")
            >>> nf_obj = tv.compute_nearfield_tv(
                    sources=["Cyg A", "Cas A", "Vir A", "Tau A", "Sun"],
                    npix=64
                )
        """

        obs_time = self.time[0] + (self.time[-1] - self.time[0]) / 2

        data = self.get_stokes(stokes)
        nf, src_imprints = data.make_nearfield(
            radius=radius, npix=npix, sources=sources
        )

        return TV_Nearfield(
            nearfield=nf,
            source_imprints=src_imprints,
            npix=npix,
            time=obs_time,
            frequency=np.mean(self.frequencies),
            radius=radius,
            mini_arrays=data.mini_arrays,
            stokes=stokes,
        )

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load_tv_data(self) -> None:
        """Load the data contained in a NenuFAR-TV binary file.
        """
        # Extract the ASCII header (5 first lines)
        with open(self.file_name, "rb") as f:
            header = list(islice(f, 0, 5))
        assert header[0] == b"HeaderStart\n", "Wrong header start"
        assert header[-1] == b"HeaderStop\n", "Wrong header stop"
        header = [s.decode("utf-8") for s in header]
        hd_size = sum([len(s) for s in header])

        # Parse informations into Crosslet attributes
        keys = ["frequencies", "mini_arrays", "dt"]
        search = ["Freq.List", "Mr.List", "accumulation"]
        types = ["float64", "int", "int"]
        for key, word, typ in zip(keys, search, types):
            unit = u.MHz if key == "freqs" else 1
            for h in header:
                if word in h:
                    setattr(
                        self,
                        key,
                        np.array(h.split("=")[1].split(","), dtype=typ) * unit,
                    )

        # Deduce the dtype for decoding
        n_ma = self.mini_arrays.size
        n_sb = self.frequencies.size
        dtype = np.dtype(
            [("jd", "float64"), ("data", "complex64", (n_sb, n_ma * n_ma * 2 + n_ma))]
        )

        # Decoding the binary file
        tmp = np.memmap(filename=self.file_name, dtype="int8", mode="r", offset=hd_size)
        decoded = tmp.view(dtype)

        self.dt = TimeDelta(self.dt, format="sec")
        self.frequencies *= u.MHz
        self.data = decoded["data"] / self.dt.sec
        self.time = Time(decoded["jd"], format="jd", precision=0)
