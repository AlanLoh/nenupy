#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "Sky",
    "HpxSky"
]


import numpy as np
import copy
import logging
log = logging.getLogger(__name__)

from astropy.coordinates import SkyCoord, EarthLocation, ICRS
from astropy.time import Time
import astropy.units as u
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy.wcs import WCS
from reproject import reproject_from_healpix

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import get_cmap
from matplotlib.ticker import LinearLocator
from matplotlib.colors import Normalize

try:
    import healpy.pixelfunc as hpx
except ImportError:
    log.warning("Unable to load 'healpy', some functionalities may not be working.")
    hpx = None

from nenupy import nenufar_position
from nenupy.astro2.astro_tools import AstroObject, radec_to_altaz


# ============================================================= #
# ---------------------------- Sky ---------------------------- #
# ============================================================= #
class Sky(AstroObject):
    """  """

    def __init__(self,
            coordinates: SkyCoord,
            time: Time = Time.now(),
            frequency: u.Quantity = 50*u.MHz,
            value: np.ndarray = np.array([0]),
            observer: EarthLocation = nenufar_position
        ):
        self.coordinates = coordinates
        self.time = time
        self.frequency = frequency
        self.value = value
        self.observer = observer


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def value(self):
        """ """
        return self._value
    @value.setter
    def value(self, v):
        if v.dtype < np.float64:
            v = v.astype(np.float64)
        self._value = v


    @property
    def visible_sky(self):
        """ """
        altaz = radec_to_altaz(
            radec=self.coordinates,
            time=self.time,
            observer=self.observer,
            fast_compute=True
        )
        return altaz.alt.deg > 0


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, **kwargs):
        """
        """
        # Parsing the keyword arguments
        resolution = kwargs.get("resolution", 1*u.deg)
        figname = kwargs.get("figname", None)
        cmap = kwargs.get("cmap", "YlGnBu_r")
        figsize = kwargs.get("figsize", (15, 10))
        center = kwargs.get("center", SkyCoord(0*u.deg, 0*u.deg))
        radius = kwargs.get("radius", None)
        ticks_color = kwargs.get("ticks_color", "0.9")
        colorbar_label = kwargs.get("colorbar_label", "Colorbar")
        title = kwargs.get("title", "")
        visible_sky = kwargs.get("only_visible", True)
        decibel = kwargs.get("decibel", False)

        # Initialize figure
        wcs, shape = self._compute_wcs(
            center=center,
            resolution=getattr(self, "resolution", resolution),
            radius=radius
        )
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(
            projection=wcs,
            frame_class=EllipticalFrame
        )

        # Get the data projected on fullsky
        data = self._fullsky_projection(
            wcs=wcs,
            shape=shape,
            display_visible_sky=visible_sky
        )

        # Scale the data in decibel
        if decibel:
            data = 10 * np.log10(data)

        vmin = kwargs.get("vmin", np.nanmin(data))
        vmax = kwargs.get("vmax", np.nanmax(data))

        # Plot the data
        im = ax.imshow(
            data,
            origin="lower",
            interpolation="none",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        # Define ax ticks
        ax.coords.grid(color=ticks_color, alpha=0.5)
        path_effects=[patheffects.withStroke(linewidth=3, foreground='black')]

        ra_axis = ax.coords[0]
        dec_axis = ax.coords[1]
        ra_axis.set_ticks_visible(False)
        ra_axis.set_ticklabel(color=ticks_color, path_effects=path_effects)
        ra_axis.set_axislabel("RA", color=ticks_color, path_effects=path_effects)
        ra_axis.set_major_formatter("d")
        
        ra_axis.set_ticks(number=12)
        dec_axis.set_ticks_visible(False)
        dec_axis.set_axislabel("Dec", minpad=2)
        dec_axis.set_major_formatter("d")
        dec_axis.set_ticks(number=10)

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
        cb = ColorbarBase(
            cax,
            cmap=get_cmap(name=cmap),
            orientation='vertical',
            norm=Normalize(
                vmin=vmin,
                vmax=vmax
            ),
            ticks=LinearLocator()
        )
        cb.solids.set_edgecolor("face")
        cb.set_label(colorbar_label)
        cb.formatter.set_powerlimits((0, 0))

        # Other
        im.set_clip_path(ax.coords.frame.patch)
        ax.set_title(title, pad=20)

        # Save or show
        if figname is None:
            plt.show()
        elif figname.lower() == 'return':
            return fig, ax
        else:
            fig.savefig(
                figname,
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
        plt.close('all')
    
    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _compute_wcs(center: SkyCoord, resolution: u.Quantity, radius: u.Quantity):
        """ """
        dangle = 0.675
        scale = int(dangle/resolution.to(u.deg).value)
        #scale = int(resolution.to(u.deg).value/dangle)
        scale = 1 if scale <= 1 else scale
        ra_dim = 480*scale
        dec_dim = 240*scale
        if radius is not None:
            resol = dangle/scale
            ra_dim = int(2 * radius.to(u.deg).value / resol)
            dec_dim = ra_dim
            #raauto = False
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [ra_dim/2 + 0.5, dec_dim/2 + 0.5]
        wcs.wcs.cdelt = np.array([-dangle/scale, dangle/scale])
        wcs.wcs.crval = [center.ra.deg, center.dec.deg]
        wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']

        return wcs, (ra_dim, dec_dim)


    def _fullsky_projection(self, wcs: WCS, shape: tuple, display_visible_sky: bool):
        """ """
        x, y = wcs.world_to_pixel(self.coordinates)

        data = np.zeros(shape, dtype=np.float64)
        data[:, :] = np.nan
        weights = np.zeros(shape, dtype=int)

        x_int = np.floor(x).astype(int)
        x_in_image = (x_int >= 0) & (x_int < shape[0])
        y_int = np.floor(y).astype(int)
        y_in_image = (y_int >= 0) & (y_int < shape[1])
        in_image_mask = x_in_image & y_in_image
        x_int = x_int[in_image_mask]
        y_int = y_int[in_image_mask]

        values = copy.deepcopy(self.value)
        if display_visible_sky:
            values[~self.visible_sky] = np.nan
        values = values[in_image_mask]

        #data.mask[(x_int, y_int)] = False
        data[(x_int, y_int)] = 0.
        np.add.at(weights, (x_int, y_int), 1)
        weights[weights<0.5] = 1.
        np.add.at(data, (x_int, y_int), values)
        data[(x_int, y_int)] /= weights[(x_int, y_int)]

        #data = np.ma.masked_array(
        #    data,
        #    mask=np.ones(shape, dtype=bool),
        #    fill_value=np.nan
        #)

        return data.T
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------------- HpxSky --------------------------- #
# ============================================================= #
class HpxSky(Sky):
    """ """

    def __init__(self,
            resolution: u.Quantity = 1*u.deg,
            time: Time = Time.now(),
            frequency: u.Quantity = 50*u.MHz,
            value: np.ndarray = np.array([0]),
            observer: EarthLocation = nenufar_position
        ):

        if hpx is None:
            log.error(
                f"Unable to create an instance of {self.__qualname__} since 'healpy' does not work."
            )

        self.nside, self.resolution = self._resol2nside(resolution=resolution)

        # Construct the Healpix coordinates map
        ra, dec = hpx.pix2ang(
            nside=self.nside,
            ipix=np.arange(
                hpx.nside2npix(self.nside),
                dtype=np.int64
            ),
            lonlat=True,
            nest=False
        )

        super().__init__(
            coordinates=SkyCoord(ra, dec, unit="deg"),
            time=time,
            frequency=frequency,
            value=value,
            observer=observer
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _resol2nside(resolution: u.Quantity):
        """ Returns the HEALPix nside and effective resolution. """
        
        # Get all nsides for all HEALPix oders
        healpix_nsides = hpx.order2nside(np.arange(30))

        # Convert them into angular resolutions
        available_resolutions = hpx.nside2resol(
            healpix_nsides,
            arcmin=True
        )*u.arcmin

        # Find the index of the closest matching HEALPix resolution
        order_index = np.argmin(
            np.abs(available_resolutions - resolution)
        )

        # Retrieve the corresponding nside and reoslution
        nside = healpix_nsides[order_index]
        effective_resolution = available_resolutions[order_index]

        return nside, effective_resolution
    

    @staticmethod
    def _compute_wcs(center: SkyCoord, resolution: u.Quantity, radius: u.Quantity = None):
        """ """
        dangle = 0.675
        scale = int(dangle/resolution.to(u.deg).value)
        scale = 1 if scale <= 1 else scale
        ra_dim = 480*scale
        dec_dim = 240*scale
        if radius is not None:
            resol = dangle/scale
            ra_dim = int(2 * radius.to(u.deg).value / resol)
            dec_dim = ra_dim
            #raauto = False
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [ra_dim/2 + 0.5, dec_dim/2 + 0.5]
        wcs.wcs.cdelt = np.array([-dangle/scale, dangle/scale])
        wcs.wcs.crval = [center.ra.deg, center.dec.deg]
        wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']

        return wcs, (dec_dim, ra_dim)


    def _fullsky_projection(self, wcs: WCS, shape: tuple, display_visible_sky: bool):
        """ """
        values = copy.deepcopy(self.value)
        if display_visible_sky:
            values[~self.visible_sky] = np.nan

        array, _ = reproject_from_healpix(
            (values, ICRS()),
            wcs,
            nested=False,
            shape_out=shape
        )
        return array
# ============================================================= #
# ============================================================= #
