#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ************
    UVW Coverage
    ************
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2022, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "compute_uvw"
]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.modeling import models, fitting
import astropy.units as u

from nenupy.instru.interferometer import Interferometer
from nenupy.astro import hour_angle, altaz_to_radec, wavelength
from nenupy import nenufar_position

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------ compute_uvw ------------------------ #
# ============================================================= #
def compute_uvw(
        interferometer: Interferometer,
        phase_center: SkyCoord = None,
        time: Time = Time.now(),
        observer: EarthLocation = nenufar_position
    ) -> u.Quantity:
    """ """

    # Get the baselines in ITRF coordinates
    baselines_itrf = interferometer.baselines.bsl
    xyz = baselines_itrf[np.tril_indices(interferometer.size)].T
    #xyz = np.array(baselines_itrf).T

    log.info(f"Computing UVW (time steps: {time.size}, baselines: {xyz.shape[0]})...")

    # Select zenith phase center if nothing is provided
    if phase_center is None:
        log.debug("Default zenith phase center selected.")
        zenith = SkyCoord(
            np.zeros(time.size),
            np.ones(time.size)*90,
            unit="deg",
            frame=AltAz(
                obstime=time,
                location=observer
            )
        )
        phase_center = altaz_to_radec(zenith)
    center_dec_rad = phase_center.dec.rad
    if np.isscalar(center_dec_rad):
        center_dec_rad = np.repeat(center_dec_rad, time.size)

    # Compute the hour angle of the phase center
    lha = hour_angle(
        radec=phase_center,
        time=time,
        observer=observer,
        fast_compute=True
    )
    lha_rad = lha.rad

    # Force the time to be an array    
    if np.isscalar(lha_rad):
        lha_rad = np.array([lha_rad])
    if time.isscalar:
        time = time.reshape((1,))

    # celestial transformation
    # lat_rad = interferometer.position.lat.rad
    # cl = np.cos(lat_rad)
    # sl = np.sin(lat_rad)
    # transfo = np.array([   
    #     [0, -sl, cl],
    #     [1,   0,  0],
    #     [0,  cl, sl]
    # ])
    # xyz = np.dot(np.moveaxis(transfo, -1, 0), xyz)

    # Compute UVW projection
    sr = np.sin(lha_rad)
    cr = np.cos(lha_rad)
    sd = np.sin(center_dec_rad)
    cd = np.cos(center_dec_rad)
    rot_uvw = np.array([
        [    sr,     cr,  np.zeros(time.size)],
        [-sd*cr,  sd*sr,                   cd],
        [ cd*cr, -cd*sr,                   sd]
    ])

    # Project the baselines in the UVW frame
    uvw = - np.dot(np.moveaxis(rot_uvw, -1, 0), xyz)

    return np.moveaxis(uvw, -1, 1) * u.m
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ compute_uvw ------------------------ #
# ============================================================= #
class UV_Coverage:
    """ """

    def __init__(self, uvw: u.Quantity) -> None:
        
        # Getting the number of antennas used to compute the UVW
        n_baselines = uvw.shape[-2]
        # Solving n(n-1)/2 + n = n_baselines
        delta = 1 + 8*n_baselines # 2nd degree discriminant
        n_antennas = int( (-1 + np.sqrt(delta))/2 ) # only positive solution
        ant_1, ant_2 = np.tril_indices(n_antennas, 0)

        # Add the negative part (while not including auto-correlations twice)
        self.uvw = np.concatenate(
            (uvw, uvw[:, ant_1 != ant_2, :]*np.array([-1, -1, 1])),
            axis=1
        ) # shape (time, bsl, 3)


    @property
    def uv_distance(self):
        """ """
        return np.linalg.norm(self.uvw, axis=-1)


    @classmethod
    def compute(cls,
            interferometer: Interferometer,
            phase_center: SkyCoord = None,
            time: Time = Time.now(),
            observer: EarthLocation = nenufar_position
        ):
        """ """
        uvw = compute_uvw(
            interferometer=interferometer,
            phase_center=phase_center,
            time=time,
            observer=observer
        )
        return cls(uvw=uvw)


    def radial_profile(self,
            frequency: u.Quantity = None,
            plot: bool = True,
            **kwargs
        ):
        """ Plot the radial cut on the UV distribution

            kwargs
                step_width
                log
                figsize
                title

        """

        if frequency is not None:
            wavel = wavelength(frequency)
            if wavel.isscalar:
                wavel = wavel.reshape((1,))
            x_label = r"uv distance ($\lambda$)"
            uu = (self.uvw[..., 0][:, None, :] / wavel[None, :, None]).to(u.dimensionless_unscaled)
            vv = (self.uvw[..., 1][:, None, :] / wavel[None, :, None]).to(u.dimensionless_unscaled)
        else:
            x_label = "uv distance (m)"
            uu = self.uvw[..., 0].to(u.m).value
            vv = self.uvw[..., 1].to(u.m).value

        # Only work on the upper part of the plot since its symmetrical
        positive_v = vv > 0.
        uu = uu[positive_v]
        vv = vv[positive_v]
        uv_distance = np.sqrt(uu**2 + vv**2)

        average_distance = []
        density = []
        step_width = kwargs.get("step_width", 10) # lambda unit
        uv_distances_probed = np.arange(
            uv_distance.min(),
            uv_distance.max() + step_width,
            step_width
        )
        for min_dist, max_dist in zip(uv_distances_probed[:-1], uv_distances_probed[1:]):
            mask = (uv_distance>=min_dist) & (uv_distance<=max_dist)
            avg_dist = np.mean([min_dist, max_dist])
            average_distance.append( avg_dist )
            density.append( uu[mask].size / avg_dist )
        density = np.array(density)/np.max(density) # normalize
        average_distance = np.array(average_distance)

        # Fit
        # Ignore first points for the Gaussian fit
        start_index = np.argmax(density)

        # Perform a Gaussian fit
        gaussian_init = models.Gaussian1D(
            amplitude=1.,
            mean=0,
            stddev=0.1 * max(average_distance),#0.68 * max(average_distance),
            bounds={'mean': (0., 0.)}
            )
        fit_gaussian = fitting.LevMarLSQFitter()
        gaussian = fit_gaussian(gaussian_init, average_distance[start_index:], density[start_index:])
        gstd = gaussian.stddev.value

        if plot:
            fig = plt.figure(figsize=kwargs.get("figsize", (10, 5)))
            ax = fig.add_subplot(1, 1, 1)

            ax.bar(
                average_distance,
                height=density,
                width=step_width,
                edgecolor="black",
                log=kwargs.get("log", False),
                linewidth=0.5
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel("Density")
            ax.set_title(kwargs.get("title", ""))
            y_limits = ax.get_ylim()

            # Plot the fit
            x = np.linspace(min(average_distance), max(average_distance), 100)
            ax.plot(
                x,
                gaussian(x),
                linestyle=':',
                color='black',
                linewidth=2,
                label="Gaussian fit"
            )
            plt.legend()
            ax.set_ylim(y_limits)

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

        return average_distance, density, np.std((density, gaussian(average_distance)), axis=0)


    def azimuthal_profile(self,
            frequency: u.Quantity = None,
            plot: bool = True,
            **kwargs
        ):
        """ Plot the azimuthal cut ont the UV distribution
        """

        if frequency is not None:
            wavel = wavelength(frequency)
            if wavel.isscalar:
                wavel = wavel.reshape((1,))
            uu = (self.uvw[..., 0][:, None, :] / wavel[None, :, None]).to(u.dimensionless_unscaled)
            vv = (self.uvw[..., 1][:, None, :] / wavel[None, :, None]).to(u.dimensionless_unscaled)
        else:
            uu = self.uvw[..., 0].to(u.m).value
            vv = self.uvw[..., 1].to(u.m).value        

        # Only work on the upper part of the plot since its symmetrical
        positive_v = vv > 0.
        uu = uu[positive_v]
        vv = vv[positive_v]
        uv_distance = np.sqrt(uu**2 + vv**2)
        uv_ang = np.degrees( np.arccos(uu/uv_distance) )

        angle = []
        density = []
        step_angle = kwargs.get("step_angle", 5)
        angles_probed = np.arange(
            0,
            180 + step_angle,
            step_angle
            )
        for min_ang, max_ang in zip(angles_probed[:-1], angles_probed[1:]):
            mask = (uv_ang>=min_ang) & (uv_ang<=max_ang)
            angle.append( np.mean([min_ang, max_ang]) )
            density.append( uu[mask].size )
        density = np.array(density)/np.max(density)
        mean_value = np.mean(density)
        angle = np.array(angle)

        if plot:
            fig = plt.figure(figsize=kwargs.get("figsize", (10, 5)))
            ax = fig.add_subplot(1, 1, 1)

            ax.bar(
                angle,
                height=density,
                width=step_angle,
                edgecolor="black",
                log=kwargs.get("log", False),
                linewidth=0.5
            )

            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Density")
            ax.set_title(kwargs.get("title", ""))
            
            ax.axhline(
                mean_value,
                linestyle=":",
                color="black",
                linewidth=2,
                label="Median"
            )
            plt.legend()

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

        return angle, density, np.std((density, mean_value), axis=0)


    def psf(self,
            frequency: u.Quantity = None,
            grid_size: int = 1000,
            plot: bool = True,
            **kwargs
        ):
        """ """

        if frequency is not None:
            wavel = wavelength(frequency)
            if wavel.isscalar:
                wavel = wavel.reshape((1,))
            x_label = r"uv distance ($\lambda$)"
            uu = (self.uvw[..., 0][:, None, :] / wavel[None, :, None]).to(u.dimensionless_unscaled)
            vv = (self.uvw[..., 1][:, None, :] / wavel[None, :, None]).to(u.dimensionless_unscaled)
        else:
            x_label = "uv distance (m)"
            uu = self.uvw[..., 0].to(u.m).value
            vv = self.uvw[..., 1].to(u.m).value
        

        # Remove auto-correlations
        auto_corr = (uu == 0.) * (vv == 0.)

        # Grid the UV coverage
        uv_grid, x_edges, y_edges = np.histogram2d(
            uu[..., ~auto_corr].ravel(),
            vv[..., ~auto_corr].ravel(),
            bins=grid_size
        )

        # Fourrier transform
        psf = np.fft.fft2(uv_grid)
        psf = np.fft.fftshift(psf)

        # Plot
        if plot:
            fig = plt.figure(figsize=kwargs.get("figsize", (10, 10)))
            ax = fig.add_subplot(1, 1, 1)

            abs_psf = np.abs(psf)
            im = ax.imshow(
                abs_psf,
                origin="lower",
                cmap=kwargs.get("cmap", "Blues"),
                interpolation="nearest",
                norm=LogNorm(vmin=abs_psf.min(), vmax=abs_psf.max()) if kwargs.get("log", False) else None
            )

            # Colorbar
            cax = inset_axes(
                ax,
                width='3%',
                height='100%',
                loc='lower left',
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cb = fig.colorbar(im, cax=cax)
            cb.solids.set_edgecolor("face")
            cb.set_label(kwargs.get("colorbar_label", ""))

        return psf


    def plot(self, frequency: u.Quantity = None, **kwargs) -> None:
        """
            kwargs
                figsize
                gridsize
                cmap
                overlay_scatter
                colorbar_label
                title
                figname
        """

        # Either plot raw (u, v) in meters, or, if the frequency is specified,
        # plots (u, v) in lambda units.
        if frequency is not None:
            wavel = wavelength(frequency)
            if wavel.isscalar:
                wavel = wavel.reshape((1,))
            x_label = r"u ($\lambda$)"
            y_label = r"v ($\lambda$)"
            uu = self.uvw[..., 0][:, None, :] / wavel[None, :, None]
            vv = self.uvw[..., 1][:, None, :] / wavel[None, :, None]
        else:
            x_label = "u (m)"
            y_label = "v (m)"
            uu = self.uvw[..., 0]
            vv = self.uvw[..., 1]

        # Scale the haxagon grid, with respect to the data, in order to not
        # get deformed hexagons (since the plot is scaled equally in x and y).
        uv_min = min((uu.min().value, vv.min().value))
        uv_max = max((uu.max().value, vv.max().value))
        uv_width = np.abs(uv_min - uv_max)
        ax_min = uv_min - 0.05 * uv_width
        ax_max = uv_max + 0.05 * uv_width

        # Plot the hexagon bins
        fig = plt.figure(figsize=kwargs.get("figsize", (10, 10)))
        ax = fig.add_subplot(1, 1, 1)
        hexb = ax.hexbin(
            uu.value,
            vv.value,
            gridsize=kwargs.get("gridsize", 70),
            bins=None,
            xscale="linear",
            yscale="linear",
            extent=(ax_min, ax_max, ax_min, ax_max),
            cmap=kwargs.get("cmap", "Blues"),
            edgecolors="face",
            mincnt=1,
            norm=LogNorm()#vmin=Z.min(), vmax=Z.max()),
        )

        if kwargs.get("overlay_scatter", False):
            ax.scatter(uu.value, vv.value, 2, alpha=0.5)

        # Colorbar
        cax = inset_axes(
            ax,
            width='3%',
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(hexb, cax=cax)
        cb.solids.set_edgecolor("face")
        cb.set_label(kwargs.get("colorbar_label", "Density"))

        # ax.set_facecolor("0.8")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect('equal', adjustable='datalim')
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


    @staticmethod
    def _prepare_plot(uvw: u.Quantity, frequency: u.Quantity) -> u.Quantity:
        """ """
        return
# ============================================================= #
# ============================================================= #

