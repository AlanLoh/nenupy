#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST file
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "XST"
]

from abc import ABC
import os
from itertools import islice
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz, Angle
import astropy.units as u
from astropy.io import fits
from healpy.fitsfunc import write_map, read_map
from healpy.pixelfunc import mask_bad, nside2resol
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import LinearLocator
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dask.array as da
from dask.diagnostics import ProgressBar

import nenupy
from os.path import join, dirname
from nenupy.astro.target import FixedTarget, SolarSystemTarget
from nenupy.io.io_tools import StatisticsData
from nenupy.io.bst import BST_Slice
from nenupy.astro import wavelength, altaz_to_radec, l93_to_etrs, etrs_to_enu
from nenupy.astro.uvw import compute_uvw
from nenupy.astro.sky import HpxSky
from nenupy.astro.pointing import Pointing
from nenupy.instru import NenuFAR, MiniArray, read_cal_table, freq2sb, nenufar_miniarrays
from nenupy import nenufar_position, DummyCtMgr

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- XST_Slice ------------------------- #
# ============================================================= #
class XST_Slice:
    """ """

    def __init__(self, mini_arrays, time, frequency, value):
        self.mini_arrays = mini_arrays
        self.time = time
        self.frequency = frequency
        self.value = value


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot_correlaton_matrix(self, mask_autocorrelations: bool = False, **kwargs):
        """
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
            mask = ma1 != ma2 # cross_correlation mask
        matrix[ma2[mask], ma1[mask]] = np.mean(self.value, axis=(0, 1))[mask]

        fig = plt.figure(figsize=kwargs.get("figsize", (10, 10)))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        data = np.absolute(matrix)
        if kwargs.get("decibel", True):
            data = 10*np.log10(data)

        im = ax.pcolormesh(
            all_mas,
            all_mas,
            data,
            shading="nearest",
            cmap=kwargs.get("cmap", "YlGnBu"),
            vmin=kwargs.get("vmin", np.nanmin(data)),
            vmax=kwargs.get("vmax", np.nanmax(data))
        )
        ax.set_xticks(all_mas[::2])
        ax.set_yticks(all_mas[::2])
        ax.grid(alpha=0.5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(kwargs.get("colorbar_label", "dB" if kwargs.get("decibel", True) else "Amp"))
        
        # Axis abels
        ax.set_xlabel(f"Mini-Array index")
        ax.set_ylabel(f"Mini-Array index")

        # Title
        ax.set_title(kwargs.get("title", ""))

        # Save or show the figure
        figname = kwargs.get("figname", "")
        if figname != "":
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches="tight",
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close("all")


    def rephase_visibilities(self, phase_center, uvw):
        """ """

        # Compute the zenith original phase center
        zenith = SkyCoord(
            np.zeros(self.time.size),
            np.ones(self.time.size)*90,
            unit="deg",
            frame=AltAz(
                obstime=self.time,
                location=nenufar_position
            )
        )
        zenith_phase_center = altaz_to_radec(zenith)

        # Define the rotation matrix
        def rotation_matrix(skycoord):
            """
            """
            ra_rad = skycoord.ra.rad
            dec_rad = skycoord.dec.rad

            if np.isscalar(ra_rad):
                ra_rad = np.array([ra_rad])
                dec_rad = np.array([dec_rad])

            cos_ra = np.cos(ra_rad)
            sin_ra = np.sin(ra_rad)
            cos_dec = np.cos(dec_rad)
            sin_dec = np.sin(dec_rad)

            return np.array([
                [cos_ra, -sin_ra, np.zeros(ra_rad.size)],
                [-sin_ra*sin_dec, -cos_ra*sin_dec, cos_dec],
                [sin_ra*cos_dec, cos_ra*cos_dec, sin_dec],
            ])

        # Transformation matrices
        to_origin = rotation_matrix(zenith_phase_center) # (3, 3, ntimes)
        to_new_center = rotation_matrix(phase_center) # (3, 3, 1)
        total_transformation = np.matmul(
            np.transpose(
                to_new_center,
                (2, 0, 1)
            ),
            to_origin
        ) # (3, 3, ntimes)
        rotUVW = np.matmul(
            np.expand_dims(
                (to_origin[2, :] - to_new_center[2, :]).T,
                axis=1
            ),
            np.transpose(
                to_origin,
                (2, 1, 0)
            )
        ) # (ntimes, 1, 3)
        phase = np.matmul(
            rotUVW,
            np.transpose(uvw, (0, 2, 1))
        ) # (ntimes, 1, nvis)
        rotate_visibilities = np.exp(
            2.j*np.pi*phase/wavelength(self.frequency).to(u.m).value[None, :, None]
        ) # (ntimes, nfreqs, nvis)

        new_uvw = np.matmul(
            uvw, # (ntimes, nvis, 3)
            np.transpose(total_transformation, (2, 0, 1))
        )

        return rotate_visibilities, new_uvw


    def make_image(self,
            resolution: u.Quantity = 1*u.deg,
            fov_radius: u.Quantity = 25*u.deg,
            phase_center: SkyCoord = None,
            stokes: str = "I"
        ):
        """
            :Example:

                xst = XST("XST.fits")
                data = xst.get_stokes("I")
                sky = data.make_image(
                    resolution=0.5*u.deg,
                    fov_radius=27*u.deg,
                    phase_center=SkyCoord(277.382, 48.746, unit="deg")
                )
                sky[0, 0, 0].plot(
                    center=SkyCoord(277.382, 48.746, unit="deg"),
                    radius=24.5*u.deg
                )

        """
        exposure = self.time[-1] - self.time[0]

        # Compute XST UVW coordinates (zenith phased)
        uvw = compute_uvw(
            interferometer=NenuFAR()[self.mini_arrays],
            phase_center=None, # will be zenith
            time=self.time,
        )

        # Prepare visibilities rephasing
        rephase_matrix, uvw = self.rephase_visibilities(
            phase_center=phase_center,
            uvw=uvw
        )

        # Mask auto-correlations
        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        cross_mask = ma1 != ma2
        uvw = uvw[:, cross_mask, :]
        # Transform to lambda units
        wvl = wavelength(self.frequency).to(u.m).value
        uvw = uvw[:, None, :, :]/wvl[None, :, None, None] # (t, f, bsl, 3)
        # Mean in time
        uvw = np.mean(uvw, axis=0)

        # Prepare the sky
        sky = HpxSky(
            resolution=resolution,
            time=self.time[0] + exposure/2,
            frequency=np.mean(self.frequency),
            polarization=np.array([stokes]),
            value=np.nan
        )

        # Compute LMN coordinates
        image_mask = sky.visible_mask[0, 0, 0]
        image_mask *= sky.coordinates.separation(phase_center) <= fov_radius
        l, m, n = sky.compute_lmn(
            phase_center=phase_center,
            coordinate_mask=image_mask
        )
        lmn = np.array([l, m, (n - 1)], dtype=np.float32).T
        n_pix = l.size
        lmn = da.from_array(
            lmn,
            chunks=(np.floor(n_pix/os.cpu_count()), 3)
        )

        # Transform to Dask array
        n_bsl = uvw.shape[1]
        n_freq = self.frequency.size
        n_pix = l.size
        uvw = da.from_array(
            uvw.astype(np.float32),
            chunks=(n_freq, np.floor(n_bsl/os.cpu_count()), 3)
        )

        # Compute the phase
        uvwlmn = np.sum(uvw[:, :, None, :] * lmn[None, None, :, :], axis=-1)
        phase = np.exp( -2j * np.pi * uvwlmn ) # (f, bsl, npix)

        # Rephase and average visibilites
        vis = np.mean( # Mean in time
            self.value * rephase_matrix,
            axis=0
        )[..., cross_mask] # (nfreqs, nvis)

        # Make dirty image
        dirty = np.nanmean( # mean in baselines
            np.real(
                np.mean( # mean in freq
                    vis[:, :, None] * phase,
                    axis=0
                )
            ),
            axis=0
        )

        # Insert dirty image in Sky object
        log.info(
            f"Computing image (time: {self.time.size}, frequency: {self.frequency.size}, baselines: {vis.shape[1]}, pixels: {phase.shape[-1]})... "
        )
        with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
            sky.value[0, 0, 0, image_mask] = dirty.compute()

        return sky


    def make_nearfield(self,
            radius: u.Quantity = 400*u.m,
            npix: int = 64,
            sources: list = []
        ):
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
                Radius of the ground image. Default is ``400m``.
            :type radius:
                :class:`~astropy.units.Quantity`
            :param npix:
                Number of pixels of the image size. Default is ``64``.
            :type npix:
                `int`
            :param sources:
                List of source names for which their near-field footprint
                may be computed. Only sources above 10 deg elevation
                will be considered.
            :type sources:
                `list`

            :returns:
                Tuple of near-field image and a dictionnary 
                containing all source footprints. 
            :rtype:
                `tuple`(:class:`~numpy.ndarray`, `dict`)

            :Example:

                from nenupy.io.xst import XST
                xst = XST("xst_file.fits")
                nearfield, src_dict = xst.make_nearfield(sources=["Cas A", "Sun"])

            .. versionadded:: 1.1.0

        """

        def compute_nearfield_imprint(visibilities, phase):
            # Phase and average in frequency
            nearfield = np.mean(
                visibilities[..., None, None] * phase,
                axis=0
            )
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
        obs_time = self.time[0] + (self.time[-1] - self.time[0])/2.

        # Delays at the ground
        radius_m = radius.to(u.m).value
        ground_granularity = np.linspace(-radius_m, radius_m, npix)
        posx, posy = np.meshgrid(ground_granularity, ground_granularity)
        posz = np.ones_like(posx) * (np.average(ma_enu[:, 2]) + 1)
        ground_grid = np.stack((posx, posy, posz), axis=2)
        ground_distances = np.sqrt(
            np.sum(
                (ma_enu[:, None, None, :] - ground_grid[None])**2,
                axis=-1
            )
        )
        grid_delays = ground_distances[ma1] - ground_distances[ma2] # (nvis, npix, npix)
        n_bsl = ma1[cross_mask].size
        grid_delays = da.from_array(
            grid_delays[cross_mask],
            chunks=(np.floor(n_bsl/os.cpu_count()), npix, npix)
        )
    
        # Mean in time the visibilities
        vis = np.mean(
            self.value,
            axis=0
        )[..., cross_mask] # (nfreqs, nvis)
        vis = da.from_array(
            vis,
            chunks=(1, np.floor(n_bsl/os.cpu_count()))#(self.frequency.size, np.floor(n_bsl/os.cpu_count()))
        )

        # Make the nearfield image
        log.info(
            f"Computing nearfield (time: {self.time.size}, frequency: {self.frequency.size}, baselines: {vis.shape[1]}, pixels: {posx.size})... "
        )
        wvl = wavelength(self.frequency).to(u.m).value
        phase = np.exp(2.j * np.pi * (grid_delays[None, ...]/wvl[:, None, None, None]))
        log.debug("Computing the phase term...")
        with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
            phase = phase.compute()
        log.debug("Computing the nearf-field...")
        nearfield = compute_nearfield_imprint(vis, phase)

        # Compute nearfield imprints for other sources
        simu_sources = {}
        for src_name in sources:

            # Check that the source is visible
            if src_name.lower() in ["sun", "moon", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]:
                src = SolarSystemTarget.from_name(name=src_name, time=obs_time)
            else:
                src = FixedTarget.from_name(name=src_name, time=obs_time)
            altaz = src.horizontal_coordinates#[0]
            if altaz.alt.deg <= 10:
                log.debug(f"{src_name}'s elevation {altaz[0].alt.deg}<=10deg, not considered for nearfield imprint.")
                continue

            # Projection from AltAz to ENU vector
            az_rad = altaz.az.rad
            el_rad = altaz.alt.rad
            cos_az = np.cos(az_rad)
            sin_az = np.sin(az_rad)
            cos_el = np.cos(el_rad)
            sin_el = np.sin(el_rad)
            to_enu = np.array(
                [cos_el*sin_az, cos_el*cos_az, sin_el]
            )
            # src_delays = np.matmul(
            #     ma_enu[ma1] - ma_enu[ma2],
            #     to_enu
            # )
            # src_delays = da.from_array(
            #     src_delays[cross_mask, :],
            #     chunks=((np.floor(n_bsl/os.cpu_count()), npix, npix), 1)
            # )
            
            ma1_enu = da.from_array(
                ma_enu[ma1[cross_mask]],
                chunks=np.floor(n_bsl/os.cpu_count())
            )
            ma2_enu = da.from_array(
                ma_enu[ma2[cross_mask]],
                chunks=np.floor(n_bsl/os.cpu_count())
            )
            src_delays = np.matmul(
                ma1_enu - ma2_enu,
                to_enu
            )

            # Simulate visibilities
            src_vis = np.exp(2.j * np.pi * (src_delays/wvl))
            src_vis = np.swapaxes(src_vis, 1, 0)
            log.debug(f"Computing the nearf-field imprint of {src_name}...")
            simu_sources[src_name] = compute_nearfield_imprint(src_vis, phase)

        return nearfield, simu_sources
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- Crosslet -------------------------- #
# ============================================================= #
class Crosslet(ABC):
    """ """

    # def __init__(self,
    #         mini_arrays: np.ndarray,
    #         frequency: u.Quantity,
    #         time: Time,
    #         visibilities: np.ndarray
    #     ):
    #     self.mini_arrays = mini_arrays
    #     self.frequency = frequency
    #     self.time = time
    #     self.visibilities = visibilities
    
    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self,
            frequency_selection: str = None,
            time_selection: str = None,
            polarization: str = "XX",
        ):
        """ """
        # Polarization selection
        allowed_polarizations = ["XX", "XY", "YX", "YY"]
        if polarization not in allowed_polarizations:
            raise ValueError(
                f"'polarization' argument must be equal to one of the following: {allowed_polarizations}."
            )

        # Frequency selection
        frequency_mask = self._get_freq_mask(frequency_selection)

        # Time selection
        time_mask = self._get_time_mask(time_selection)

        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        auto_mask = ma1 == ma2
        cross_mask = ~auto_mask

        if polarization == "XY":
            # Deal with lack of auto XY cross in XST-like data
            yx = self.data[
                np.ix_(
                    time_mask,
                    frequency_mask,
                    self._get_cross_idx("Y", "X")
                )
            ]
            _xy = np.zeros(
                (list(yx.shape[:-1]) + [ma1.size]),
                dtype=np.complex
            )
            _xy[:, :, auto_mask] = yx[:, :, auto_mask].conj()
            # Get XY correlations
            _xy[:, :, cross_mask] = self.data[
                np.ix_(
                    time_mask,
                    frequency_mask,
                    self._get_cross_idx("X", "Y")
                )
            ]
            return _xy
        else:
            return self.data[
                np.ix_(
                    time_mask,
                    frequency_mask,
                    self._get_cross_idx(*list(polarization))
                )
            ]


    def get_stokes(self,
            stokes: str = "I",
            frequency_selection: str = None,
            time_selection: str = None
        ):
        """ """
        frequency_mask = self._get_freq_mask(frequency_selection)
        time_mask = self._get_time_mask(time_selection)

        stokes_parameters = {
            "I": {
                "cross": ["XX", "YY"],
                "compute": lambda xx, yy: 0.5*(xx + yy)
            },
            "Q": {
                "cross": ["XX", "YY"],
                "compute": lambda xx, yy: 0.5*(xx - yy)
            },
            "U": {
                "cross": ["XY", "YX"],
                "compute": lambda xy, yx: 0.5*(xy + yx)
            },
            "V": {
                "cross": ["XY", "YX"],
                "compute": lambda xy, yx: -0.5j*(xy - yx)
            },
            "FL": {
                "cross": ["XX", "YY", "XY", "YX"],
                "compute": lambda xx, yy, xy, yx: np.sqrt((0.5*(xx - yy))**2 + (0.5*(xy + yx))**2) / (0.5*(xx + yy))
            },
            "FV": {
                "cross": ["XX", "YY", "XY", "YX"],
                "compute": lambda xx, yy, xy, yx: np.abs(-0.5j*(xy - yx))/(0.5*(xx + yy))
            }
        }

        try:
            selected_stokes = stokes_parameters[stokes]
        except KeyError:
            log.warning(f"Available polarizations are: {stokes_parameters.keys()}.")

        return XST_Slice(
            mini_arrays=self.mini_arrays,
            time=self.time[time_mask],
            frequency=self.frequencies[frequency_mask],
            value=selected_stokes["compute"](
                *map(
                    lambda pol: self.get(
                        frequency_selection=frequency_selection,
                        time_selection=time_selection,
                        polarization=pol
                    ),
                    selected_stokes["cross"]
                )
            )
        )


    def get_beamform(self,
            pointing: Pointing,
            frequency_selection: str = None,
            time_selection: str = None,
            mini_arrays: np.ndarray = np.array([0, 1]),
            polarization: str = "NW",
            calibration: str = "default"
        ):
        """
            :Example:

                from nenupy.io.bst import BST, XST
                bst = BST("20191129_141900_BST.fits")
                xst = XST("20191129_141900_XST.fits")
                bf_cal = xst.get_beamform(
                    pointing = Pointing.from_bst(bst, beam=0, analog=False),
                    mini_arrays=bst.mini_arrays,
                    calibration="default"
                )

        """
        frequency_mask = self._get_freq_mask(frequency_selection)
        time_mask = self._get_time_mask(time_selection)

        # Select the mini-arrays cross correlations
        nenufar = NenuFAR()#[self.mini_arrays]
        bf_nenufar = NenuFAR()[mini_arrays]
        ma_real_indices = np.array([nenufar_miniarrays[name]["id"] for name in bf_nenufar.antenna_names])
        if np.any( ~np.isin(ma_real_indices, self.mini_arrays) ):
            raise IndexError(
                f"Selected Mini-Arrays {mini_arrays} are outside possible values: {self.mini_arrays}."
            )
        ma_indices = np.arange(self.mini_arrays.size, dtype="int")[np.isin(self.mini_arrays, ma_real_indices)]
        ma1, ma2 = np.tril_indices(self.mini_arrays.size, 0)
        mask = np.isin(ma1, ma_indices) & np.isin(ma2, ma_indices)

        # Calibration table
        if calibration.lower() == "none":
            # No calibration
            cal = np.ones(
                (self.frequencies[frequency_mask].size, ma_indices.size)
            )
        else:
            pol_idx = {"NW": [0], "NE": [1]}
            cal = read_cal_table(
                calibration_file=calibration
            )
            cal = cal[np.ix_(
                freq2sb(self.frequencies[frequency_mask]),
                ma_real_indices,
                pol_idx[polarization]
            )].squeeze(axis=2)

        # Load and filter the data
        vis = self.get(
            frequency_selection=frequency_selection,
            time_selection=time_selection,
            polarization= "XX" if polarization.upper() == "NW" else "YY",
        )[:, :, mask]

        # Insert the data in a matrix
        tri_x, tri_y = np.tril_indices(ma_indices.size, 0)
        vis_matrix = np.zeros(
            (
                self.time[time_mask].size,
                self.frequencies[frequency_mask].size,
                ma_indices.size,
                ma_indices.size
            ),
            dtype=np.complex
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
                ma_indices.size
            ),
            dtype=np.complex
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
        ground_projection = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])
        rot = np.radians(-90)
        rotation = np.array(
            [
                [ np.cos(rot), np.sin(rot), 0],
                [-np.sin(rot), np.cos(rot), 0],
                [ 0,           0,           1]
            ]
        )
        ma1_pos = np.dot(
            nenufar.antenna_positions[ma1[mask]],
            rotation
        )
        ma2_pos = np.dot(
            nenufar.antenna_positions[ma2[mask]],
            rotation
        )
        dphi = np.dot(
            ma1_pos - ma2_pos,
            ground_projection
        ).T
        wvl = wavelength(self.frequencies[frequency_mask]).to(u.m).value
        phase[:, :, tri_x, tri_y] = np.exp(
            -2.j*np.pi/wvl[None, :, None] * dphi[:, None, :]
        )
        phase[:, :, tri_y, tri_x] = phase[:, :, tri_x, tri_y].conj().copy()
        data = np.sum((vis_matrix * phase).real, axis=(2, 3))

        return BST_Slice(
            time=self.time[time_mask],
            frequency=self.frequencies[frequency_mask],
            value=data.squeeze()
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_freq_mask(self, frequency_selection=None):
        """ """
        # Frequency selection
        frequencies = self.frequencies
        if frequency_selection is None:
            frequency_selection = f">={frequencies.min()} & <= {frequencies.max()}"
        frequency_mask = self._parse_frequency_condition(frequency_selection)(frequencies)
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


    def _get_cross_idx(self, c1='X', c2='X'):
        """ Retrieves visibilities indices for the given cross polarizations
        """
        mini_arrays_size = self.mini_arrays.size
        corr = np.array(['X', 'Y']*mini_arrays_size)
        i_ant1, i_ant2 = np.tril_indices(mini_arrays_size*2, 0)
        corr_mask = (corr[i_ant1] == c1) & (corr[i_ant2] == c2)
        indices = np.arange(i_ant1.size)[corr_mask]
        return indices
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------------- XST ---------------------------- #
# ============================================================= #
class XST(StatisticsData, Crosslet):
    """ """

    def __init__(self, file_name):
        super().__init__(file_name=file_name)
        self.mini_arrays = self._meta_data['ins']['noMROn'][0]

# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- TV_Image -------------------------- #
# ============================================================= #
class TV_Image:
    """ """

    def __init__(self,
            tv_image: HpxSky,
            analog_pointing: SkyCoord,
            fov_radius: u.Quantity
    ):
        self.tv_image = tv_image
        self.analog_pointing = analog_pointing
        self.fov_radius = fov_radius


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_fits(cls, file_name):
        """ """
        header = fits.getheader(file_name, ext=1)
        
        # Load the image
        image = read_map(
            file_name,
            dtype=None,
            partial="PARTIAL" in header["OBJECT"]
        )
        # Fill NaNs
        if "PARTIAL" in header["OBJECT"]:
            image[mask_bad(image)] = np.nan

        # Recreate the sky
        sky = HpxSky(
            resolution=Angle(
                angle=nside2resol(
                    header["NSIDE"],
                    arcmin=True
                ),
                unit=u.arcmin
            ),
            time=Time(header["OBSTIME"]),
            frequency=header["FREQ"]*u.MHz,
            polarization=np.array([header["STOKES"]]),
            value=image.reshape((1, 1, 1, image.size))
        )

        return cls(
            tv_image=sky,
            analog_pointing=SkyCoord(
                header["AZANA"]*u.deg,
                header["ELANA"]*u.deg,
                frame=AltAz(
                    obstime=Time(header["OBSTIME"]),
                    location=nenufar_position
                )
            ),
            fov_radius=header["FOV"]*u.deg/2
        )


    def save_fits(self, file_name: str, partial: bool = True):
        """ """
        
        phase_center_eq = altaz_to_radec(self.analog_pointing)

        header = [
            ("software", 'nenupy'),
            ("version", nenupy.__version__),
            ("contact", nenupy.__email__),
            ("azana", self.analog_pointing.az.deg),
            ("elana", self.analog_pointing.alt.deg),
            ("freq", self.tv_image.frequency[0].to(u.MHz).value),
            ("obstime", self.tv_image.time[0].isot),
            ("fov", self.fov_radius.to(u.deg).value * 2),
            ("pc_ra", phase_center_eq.ra.deg),
            ("pc_dec", phase_center_eq.dec.deg),
            ("stokes", self.tv_image.polarization[0])
        ]

        map2write = self.tv_image.value[0, 0, 0].copy()
        write_map(
            filename=file_name,
            m=map2write,
            nest=False,
            coord='C',
            overwrite=True,
            dtype=self.tv_image.value.dtype,
            extra_header=header,
            partial=partial
        )
        log.info(
            'HEALPix image of {} cells (nside={}) saved in `{}`.'.format(
                map2write.size,
                self.tv_image.nside,
                file_name
            )
        )


    def save_png(self, figname: str, beam_contours: bool = True, show_sources: bool = True):
        """ """
        image_center = altaz_to_radec(
            SkyCoord(
                self.analog_pointing.az,
                self.analog_pointing.alt,
                frame=AltAz(
                    obstime=self.tv_image.time[0],
                    location=nenufar_position
                )
            )
        )

        kwargs = {}

        if show_sources:
            src_names = []
            src_position = []

            with open(join(dirname(__file__), "nenufar_tv_sources.json")) as src_file:
                sources = json.load(src_file)

            for name in sources["FixedSources"]:
                src = FixedTarget.from_name(name, time=self.tv_image.time[0])
                if src.coordinates.separation(image_center) <= 0.8*self.fov_radius:
                    src_names.append(name)
                    src_position.append(src.coordinates)
            for name in sources["SolarSystemSources"]:
                src = SolarSystemTarget.from_name(name, time=self.tv_image.time[0])
                if src.coordinates.separation(image_center) <= 0.8*self.fov_radius:
                    src_names.append(name)
                    src_position.append(src.coordinates)

            if len(src_position) != 0:
                kwargs["text"] = (SkyCoord(src_position), src_names, "white")

        if beam_contours:
            # Simulate the array factor
            ma = MiniArray()
            af_sky = ma.array_factor(
                sky=HpxSky(
                    resolution=0.2*u.deg,
                    time=self.tv_image.time[0],
                    frequency=self.tv_image.frequency[0]
                ),
                pointing=Pointing(
                    coordinates=image_center,
                    time=self.tv_image.time[0]
                )
            )
            # Normalize the array factor
            af = af_sky[0, 0, 0].compute()
            af_normalized = af/af.max()
            kwargs["contour"] = (af_normalized, np.arange(0.5, 1, 0.2), "copper")

        # Plot
        self.tv_image[0, 0, 0].plot(
            center=image_center,
            radius=self.fov_radius - 2.5*u.deg,
            figname=figname,
            colorbar_label=f"Stokes {self.tv_image.polarization[0]}",
            **kwargs
        )
        return
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- TV_Nearfield ------------------------ #
# ============================================================= #
class TV_Nearfield:
    """ """

    def __init__(self,
        nearfield: np.ndarray,
        source_imprints: dict,
        npix: int,
        time: Time,
        frequency: u.Quantity,
        radius: u.Quantity,
        mini_arrays: np.ndarray,
        stokes: str
    ):
        self.nearfield = nearfield
        self.source_imprints = source_imprints
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
        """ Loads a nearfield previously stored in a FITS file.

            :param file_name:
                Path to the FITS file containing a near-field
                image (whose format is such as created by the
                :meth:`~nenupy.io.xst.TV_Nearfield.save_fits`
                method).
            :type file_name:
                `str`

            :returns:
                Instance of :class:`~nenupy.io.xst.TV_Nearfield`.
            :rtype:
                :class:`~nenupy.io.xst.TV_Nearfield`

            :Example:

                from nenupy.io.xst import TV_Nearfield
                nf = TV_Nearfield.from_fits("/path/to/nearfield.fits")

        """
        reserved_names = [
            "PRIMARY",
            "NEAR-FIELD",
            "MINI-ARRAYS"
        ]
        hdus = fits.open(file_name)
        nf_header = hdus["NEAR-FIELD"].header
        nf = cls(
            nearfield=hdus["NEAR-FIELD"].data,
            mini_arrays=hdus["MINI-ARRAYS"].data,
            npix=nf_header["NAXIS1"],
            frequency=nf_header["FREQUENC"]*u.MHz,
            time=Time(nf_header["DATE-OBS"]),
            source_imprints={
                hdu.header["SOURCE"]: hdu.data for hdu in hdus if hdu.name not in reserved_names
            },
            radius=nf_header["RADIUS"]*u.m,
            stokes=nf_header["STOKES"]
        )
        return nf


    def save_fits(self, file_name: str):
        """ Saves a nearfield made from NenuFAR TV data as a FITS file.

            :param file_name:
                Name of the file to save.
            :type file_name:
                `str`

            :Example:

                from nenupy.io.xst import NenufarTV
                tv = NenufarTV("20191204_132113_nenufarTV.dat")
                nf_object = tv.compute_nearfield_tv(sources=["Cyg A", "Cas A", "Sun"])
                nf_object.save_fits(file_name="/path/to/nearfield.fits")

        """

        # Header
        prim_header = fits.Header()
        prim_header["OBSERVER"] = "NenuFAR Team"
        prim_header["AUTHOR"] = f"nenupy {nenupy.__version__}"
        prim_header["DATE"] = Time.now().isot
        prim_header["INSTRUME"] = "XST"
        prim_header["OBSERVER"] = "NenuFAR-TV"
        prim_header["ORIGIN"] = "Station de Radioastronomie de Nancay, LESIA, Observatoire de Paris"
        prim_header["REFERENC"] = "Alan Loh and the NenuFAR team, nenupy, 2020 (DOI: 10.5281/zenodo.3775196.)"
        prim_header["TELESCOP"] = "NenuFAR"

        prim_hdu = fits.PrimaryHDU(
            header=prim_header
        )

        # Near-Field
        nf_header = fits.Header()
        nf_header["NAXIS"] = 2
        nf_header["NAXIS1"] = self.npix
        nf_header["NAXIS2"] = self.npix
        nf_header["DATE-OBS"] = (
            self.time.isot,
            "Mean observation UTC date"
        )
        nf_header["DATAMIN"] = self.nearfield.min()
        nf_header["DATAMAX"] = self.nearfield.max()
        nf_header["FREQUENC"] = (
            self.frequency.to(u.MHz).value,
            "Mean observing frequency in MHz."
        )
        nf_header["STOKES"] = self.stokes.upper()
        nf_header["DESCRIPT"] = "Near-Field image."
        nf_header["RADIUS"] = (
            self.radius.to(u.m).value,
            "Radius of the ground (in m)."
        )

        nf_hdu = fits.ImageHDU(
            data=self.nearfield,
            header=nf_header,
            name="Near-Field"
        )
        
        # Mini-Arrays
        ant_header = fits.Header()
        ant_header["DESCRIPT"] = "Mini-Array names"
        ant_hdu = fits.ImageHDU(
            data=self.mini_arrays,
            header=ant_header,
            name="Mini-Arrays"
        )

        # HDU list
        hduList = fits.HDUList(
            [
                prim_hdu,
                nf_hdu,
                ant_hdu
            ]
        )

        for src in self.source_imprints:
            hdu_name = src.replace(' ', '_')
            src_header = fits.Header()
            src_header["NAXIS"] = 2
            src_header["NAXIS1"] = self.npix
            src_header["NAXIS2"] = self.npix
            src_header["DATE-OBS"] = (
                self.time.isot,
                "Mean observation UTC date"
            )
            src_header["DATAMIN"] = self.source_imprints[src].min()
            src_header["DATAMAX"] = self.source_imprints[src].max()
            src_header["FREQUENC"] = (
                self.frequency.to(u.MHz).value,
                "Mean observing frequency in MHz."
            )
            src_header["STOKES"] = self.stokes.upper()
            src_header["SOURCE"] = (
                src,
                "Name of the source imprint on the near-field"
            )
            src_header["DESCRIPT"] = "Normalized sky source imprint on the near-field."
            src_header["RADIUS"] = (
                self.radius.to(u.m).value,
                "Radius of the ground (in m)."
            )
            src_hdu = fits.ImageHDU(
                data=self.source_imprints[src],
                name=hdu_name,
                header=src_header
            )
            hduList.append(src_hdu)

        hduList.writeto(file_name, overwrite=True)

        log.info(f"NearField saved in {file_name}.") 


    def save_png(self, figname: str = "", **kwargs):
        """ """
        radius = self.radius.to(u.m).value
        colormap = kwargs.get("cmap", "YlGnBu_r")

        # Mini-Array positions in ENU coordinates
        nenufar = NenuFAR()[self.mini_arrays]
        ma_etrs = l93_to_etrs(nenufar.antenna_positions)
        ma_enu = etrs_to_enu(ma_etrs)

        # Plot the nearfield
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 10)))
        nf_image_db = 10*np.log10(self.nearfield)
        ax.imshow(
            np.flipud(nf_image_db), # This needs to be understood...
            cmap=colormap,
            extent=[-radius, radius, -radius, radius],
            zorder=0,
            vmin=kwargs.get("vmin", np.min(nf_image_db)),
            vmax=kwargs.get("vmax", np.max(nf_image_db))
        )

        # Colorbar
        cax = inset_axes(ax,
           width="5%",
           height="100%",
           loc="lower left",
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax.transAxes,
           borderpad=0,
           )
        cb = ColorbarBase(
            cax,
            cmap=get_cmap(name=colormap),
            orientation="vertical",
            norm=Normalize(
                vmin=kwargs.get("vmin", np.min(nf_image_db)),
                vmax=kwargs.get("vmax", np.max(nf_image_db))
            ),
            ticks=LinearLocator(),
            format='%.2f'
        )
        cb.solids.set_edgecolor("face")
        cb.set_label(f"dB (Stokes {self.stokes})")
    
        # Show the contour of the simulated source imprints
        ground_granularity = np.linspace(-radius, radius, self.npix)
        posx, posy = np.meshgrid(ground_granularity, ground_granularity)
        dist = np.sqrt(posx**2 + posy**2)
        border_min = 0.1*self.npix
        border_max = self.npix - 0.1*self.npix
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
                zorder=5
            )
            # Find the maximum of the emission
            max_y, max_x = np.unravel_index(
                imprint.argmax(),
                imprint.shape
            )
            # If maximum outside the plot, recenter it
            if (max_x <= border_min) or (max_y <= border_min) or (max_x >= border_max) or (max_y >= border_max):
                dist[dist<=np.median(dist)] = 0
                max_y, max_x = np.unravel_index(
                    ((1 - dist/dist.max())*imprint).argmax(),
                    imprint.shape
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
                zorder=30
            )
        
        # NenuFAR mini-array positions
        ax.scatter(
            ma_enu[:, 0],
            ma_enu[:, 1],
            20,
            color='black',
            zorder=10
        )
        for i in range(ma_enu.shape[0]):
            ax.text(
                ma_enu[i, 0],
                ma_enu[i, 1],
                f" {self.mini_arrays[i]}",
                color="black",
                zorder=10
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
            f"{np.mean(self.frequency.to(u.MHz).value):.3f} MHz -- {self.time.isot}"
        )

        # Save or show the figure
        if figname != "":
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches="tight",
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close("all")
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- NenufarTV ------------------------- #
# ============================================================= #
class NenufarTV(StatisticsData, Crosslet):
    """ """

    def __init__(self, file_name):
        self.file_name = file_name
        self.mini_arrays = None
        self.time = None
        self.dt = None
        self.frequencies = None
        self.data = None
        self.load_tv_data()


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute_nenufar_tv(self, analog_pointing_file: str = None, **kwargs):
        """ """

        obs_time = self.time[0] + (self.time[-1] - self.time[0])/2
        fov_radius = kwargs.get("fov_radius", 27*u.deg)
        resolution = kwargs.get("resolution", 0.5*u.deg)
        stokes = kwargs.get("stokes", "I")

        if analog_pointing_file is None:
            phase_center_altaz = SkyCoord(
                0, 90, unit="deg",
                frame=AltAz(
                    obstime=obs_time,
                    location=nenufar_position
                )
            )
        else:
            pointing = Pointing.from_file(
                analog_pointing_file,
                include_corrections=False
            )[obs_time.reshape((1,))]
            phase_center_altaz = pointing.custom_ho_coordinates[0]

        data = self.get_stokes(stokes)
        sky_image = data.make_image(
            resolution=resolution,
            fov_radius=fov_radius,
            phase_center=altaz_to_radec(phase_center_altaz)
        )

        return TV_Image(
            tv_image=sky_image,
            analog_pointing=phase_center_altaz,
            fov_radius=fov_radius,
        )


    def compute_nearfield_tv(self, sources: list = [], **kwargs):
        """ 
            :Example:

                from nenupy.io.xst import NenufarTV
                tv = NenufarTV("20191204_132113_nenufarTV.dat")
                nf_object = tv.compute_nearfield_tv(
                    sources=["Cyg A", "Cas A", "Vir A", "Tau A", "Sun"],
                    npix=64
                )

        """

        obs_time = self.time[0] + (self.time[-1] - self.time[0])/2
        stokes = kwargs.get("stokes", "I")
        radius = kwargs.get("radius", 400*u.m)
        npix = kwargs.get("npix", 64)

        data = self.get_stokes(stokes)
        nf, src_imprints = data.make_nearfield(
            radius=radius,
            npix=npix,
            sources=sources
        )

        return TV_Nearfield(
            nearfield=nf,
            source_imprints=src_imprints,
            npix=npix,
            time=obs_time,
            frequency=np.mean(self.frequencies),
            radius=radius,
            mini_arrays=data.mini_arrays,
            stokes=stokes
        )

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def load_tv_data(self):
        """ """
        # Extract the ASCII header (5 first lines)
        with open(self.file_name, 'rb') as f:
            header = list(islice(f, 0, 5))
        assert header[0] == b'HeaderStart\n',\
            'Wrong header start'
        assert header[-1] == b'HeaderStop\n',\
            'Wrong header stop'
        header = [s.decode('utf-8') for s in header]
        hd_size = sum([len(s) for s in header])

        # Parse informations into Crosslet attributes
        keys = ["frequencies", "mini_arrays", 'dt']
        search = ["Freq.List", "Mr.List", "accumulation"]
        types = ["float64", "int", "int"]
        for key, word, typ in zip(keys, search, types):
            unit = u.MHz if key == 'freqs' else 1
            for h in header:
                if word in h:
                    setattr(
                        self,
                        key,
                        np.array(
                            h.split('=')[1].split(','),
                            dtype=typ
                        )*unit
                    )

        # Deduce the dtype for decoding
        n_ma = self.mini_arrays.size
        n_sb = self.frequencies.size
        dtype = np.dtype(
            [('jd', 'float64'),
            ('data', 'complex64', (n_sb, n_ma*n_ma*2 + n_ma))]
            )

        # Decoding the binary file
        tmp = np.memmap(
            filename=self.file_name,
            dtype='int8',
            mode='r',
            offset=hd_size
            )
        decoded = tmp.view(dtype)

        self.dt = TimeDelta(self.dt, format='sec')
        self.frequencies *= u.MHz
        self.data = decoded['data'] / self.dt.sec
        self.time = Time(decoded['jd'], format='jd', precision=0)
# ============================================================= #
# ============================================================= #

