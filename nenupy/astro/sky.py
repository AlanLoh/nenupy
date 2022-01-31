#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***
    Sky
    ***
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "SkySliceBase",
    "SkySlice",
    "HpxSkySlice",
    "Sky",
    "HpxSky"
]


import numpy as np
import copy
from typing import Union
import logging
log = logging.getLogger(__name__)

from astropy.coordinates import SkyCoord, EarthLocation, ICRS, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy.wcs import WCS
from reproject import reproject_from_healpix
import dask.array as da
from dask.diagnostics import ProgressBar

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

from nenupy import nenufar_position, DummyCtMgr
from nenupy.astro.astro_tools import AstroObject


# ============================================================= #
# ----------------------- SkySliceBase ------------------------ #
# ============================================================= #
class SkySliceBase(AstroObject):
    """ """

    def __init__(self,
            coordinates: SkyCoord,
            frequency: u.Quantity,
            time: Time,
            polarization: Union[str, float, int],
            value: np.ndarray,
            observer: EarthLocation = nenufar_position
        ):
        self.coordinates = coordinates
        self.time = time
        self.frequency = frequency
        self.polarization = polarization
        self.observer = observer
        self.value = value


    @property
    def visible_sky(self):
        """ """
        altaz = self.horizontal_coordinates
        return altaz.alt.deg > 0


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, **kwargs):
        r""" Display the selected content of the :attr:`~nenupy.astro.sky.Sky.value`
            attribute belonging to a :class:`~nenupy.astro.sky.Sky` instance as
            a celestial map in equatorial coordinates.

            This method is available on a :class:`~nenupy.astro.sky.SkySlice` instance,
            resulting from a selection upon a :class:`~nenupy.astro.sky.Sky` instance
            (using the indexing operator).

            Several parameters, listed below, can be tuned to adapt the plot
            to the user requirements:

            .. rubric:: Data display keywords

            :param center:
                Coordinates of the celestial map to be displayed
                at the center of the image.
                Default is ``(RA=0deg, Dec=0deg)``.
            :type center:
                :class:`~astropy.coordinates.SkyCoord`
            :param radius:
                Angular radius from the center of the image above
                which the plot should be cropped.
                Default is ``None`` (i.e., full sky image).
            :type radius:
                :class:`~astropy.units.Quantity`
            :param resolution:
                Set the pixel resolution. The upper threshold is 0.775 deg,
                any value above that does not affect the figure appearence.
                Default is ``astropy.units.Quantity(1, unit="deg")``.
            :type resolution:
                :class:`~astropy.units.Quantity`
            :param only_visible:
                If set to ``True`` only the sky above the horizon is displayed.
                Setting this parameter to ``False`` does not really make sense
                for :class:`~nenupy.astro.sky.Sky` instances representing antenna
                response for instance.
                Default is ``True``.
            :type only_visible:
                `bool`
            :param decibel:
                If set to ``True``, the data values are displayed at the decibel scale,
                i.e., :math:`10 \log( \rm{data} )`. 
                Default is ``False``.
            :type decibel:
                `bool`

            .. rubric:: Overplot keywords

            :param scatter:
                Add a scatter plot (as defined in `matplotlib.pyplot.scatter`).
                Expected syntax is ``(<SkyCoord>, <marker_size>, <color>)``.
                Default is ``None`` (i.e., no scatter overplot).
            :type scatter:
                `tuple`
            :param text:
                Add a text overlay (as defined in `matplotlib.pyplot.text`).
                Expected syntax is ``(<SkyCoord>, <[text]>, <color>)``.
                Default is ``None`` (i.e., no text overplot).
            :type text:
                `tuple`
            :param contour:
                Add a contour plot (as defined in `matplotlib.pyplot.contour`).
                Expected syntax is ``(<numpy.ndarray>, <[levels]>, <colormap>)``.
                Default is ``None`` (i.e., no contour overplot).
            :type contour:
                `tuple`

            .. rubric:: Plotting layout keywords
            
            :param altaz_overlay:
                If set to ``True``, the horizontal coordinates grid is overplotted
                in addition to the equatorial one.
                Default is ``False``.
            :type altaz_overlay:
                `bool`
            :param cmap:
                Color map applied while representing the data (see 
                `Matplotlib colormaps <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_).
                Default is ``"YlGnBu_r"``.
            :type cmap:
                `str`
            :param show_colorbar:
                Show or not the color bar.
                Default is ``True``.
            :type show_colorbar:
                `bool`
            :param colorbar_label:
                Set the label of the color bar.
                Default is ``""``.
            :type colorbar_label:
                `str`
            :param figname:
                Name of the file (absolute or relative path) to save the figure.
                If set to ``"return"``, the method returns the `tuple` ``(fig, ax)``
                (as defined by `matplotlib <https://matplotlib.org/>`_).
                Default is ``None`` (i.e., only show the figure).
            :type figname:
                `str`
            :param figsize:
                Set the figure size.
                Default is ``(15, 10)``.
            :type figsize:
                `tuple`
            :param ticks_color:
                Set the color of the equatorial grid and the Right Ascension ticks.
                Default is ``"0.9"`` (grey).
            :type ticks_color:
                `str`
            :param title:
                Set the figure title.
                Default is ``"<time>, <frequency>"``.
            :type title:
                `str`
            
        """
        # Parsing the keyword arguments
        resolution = kwargs.get("resolution", 1*u.deg)
        figname = kwargs.get("figname", None)
        cmap = kwargs.get("cmap", "YlGnBu_r")
        figsize = kwargs.get("figsize", (15, 10))
        center = kwargs.get("center", SkyCoord(0*u.deg, 0*u.deg))
        radius = kwargs.get("radius", None)
        ticks_color = kwargs.get("ticks_color", "0.9")
        colorbar_label = kwargs.get("colorbar_label", "")
        title = kwargs.get("title", f"{self.time.isot.split('.')[0]}, {self.frequency:.2f}")
        visible_sky = kwargs.get("only_visible", True)
        decibel = kwargs.get("decibel", False)
        altaz_overlay = kwargs.get("altaz_overlay", False)

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
            interpolation="quadric",
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
        ra_axis.set_ticklabel_visible(True)
        ra_axis.set_ticklabel(color=ticks_color, exclude_overlapping=True, path_effects=path_effects)
        ra_axis.set_axislabel("RA", color=ticks_color, path_effects=path_effects)
        ra_axis.set_major_formatter("d")
        
        ra_axis.set_ticks(number=12)
        dec_axis.set_ticks_visible(False)
        dec_axis.set_ticklabel_visible(True)
        dec_axis.set_axislabel("Dec", minpad=2)
        dec_axis.set_major_formatter("d")
        dec_axis.set_ticks(number=10)

        if altaz_overlay:
            frame = AltAz(obstime=self.time, location=self.observer)
            overlay = ax.get_coords_overlay(frame)
            overlay.grid(color="tab:orange", alpha=0.5)
            az_axis = overlay[0]
            alt_axis = overlay[1]
            az_axis.set_axislabel("Azimuth", color=ticks_color, path_effects=path_effects)
            az_axis.set_ticks_visible(False)
            az_axis.set_ticklabel_visible(True)
            az_axis.set_ticklabel(color=ticks_color, path_effects=path_effects)
            az_axis.set_major_formatter("d")
            az_axis.set_ticks(number=12)
            alt_axis.set_axislabel("Elevation")
            alt_axis.set_ticks_visible(False)
            alt_axis.set_ticklabel_visible(True)
            alt_axis.set_major_formatter("d")
            alt_axis.set_ticks(number=10)

            # Add NSEW points
            nesw_labels = np.array(["N", "E", "S", "W"])
            nesw = SkyCoord(
                np.array([0, 90, 180, 270]),
                np.array([0, 0, 0, 0]),
                unit="deg",
                frame=frame
            ).transform_to(ICRS)
            for label, coord in zip(nesw_labels, nesw):
                ax.text(
                    x=coord.ra.deg,
                    y=coord.dec.deg,
                    s=label,
                    color="tab:orange",
                    transform=ax.get_transform("world"),
                    path_effects=path_effects,
                    verticalalignment="center",
                    horizontalalignment="center",
                    clip_on=True
                )

        # Colorbar
        if kwargs.get("show_colorbar", True):
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

        # Overplot
        # if kwargs.get("circle", None) is not None:
        #     from matplotlib.patches import Circle
        #     frame = AltAz(obstime=self.time, location=self.observer)
        #     c = Circle(
        #         (0, 75),
        #         20,
        #         edgecolor='yellow',
        #         linewidth=5,
        #         facecolor='none',
        #         #transform=ax.get_transform('world')
        #         #transform=ax.get_transform('fk5')
        #         transform=ax.get_transform(frame)
        #     )
        #     ax.add_patch(c)
        if kwargs.get("moc", None) is not None:
            # In order fo that to work; I had to comment #axis_viewport.set(ax, wcs)
            # from add_patches_to_mpl_axe() in mocpy/moc/plot/fill.py
            # OR re-set the limits (done here)
            try:
                frame = AltAz(obstime=self.time, location=self.observer)
                xlimits = ax.get_xlim()
                ylimits = ax.get_ylim()
                mocs = kwargs["moc"] if isinstance(kwargs["moc"], list) else [kwargs["moc"]]
                for moc, color in zip(mocs, ["tab:red", "tab:green"]):
                    moc.fill(
                        ax=ax,
                        wcs=wcs,
                        alpha=0.5,
                        fill=True,
                        color=color,
                        linewidth=0,
                    )
                ax.set_xlim(xlimits)
                ax.set_ylim(ylimits)
            except AttributeError:
                log.warning("A 'MOC' object, generated from mocpy is expected.")
                raise

        if kwargs.get("altaz_moc", None) is not None:
            xlimits = ax.get_xlim()
            ylimits = ax.get_ylim()
            altaz = self.horizontal_coordinates
            mask = kwargs["altaz_moc"].contains(altaz.az, altaz.alt)
            ax.scatter(
                x=self.coordinates[mask].ra.deg,
                y=self.coordinates[mask].dec.deg,
                s=0.1,#[marker_size]*coords.size,
                facecolor="red",
                edgecolor=None,
                alpha=0.5,
                transform=ax.get_transform("world")
            )
            ax.set_xlim(xlimits)
            ax.set_ylim(ylimits)

        if kwargs.get("scatter", None) is not None:
            parameters = kwargs["scatter"]
            if len(parameters) != 3:
                raise ValueError(
                    "'scatter' syntax should be: (<SkyCoord>, <size>, <color>)"
                )
            coords = parameters[0]
            if coords.isscalar:
                coords = coords.reshape((1,))
            marker_size = parameters[1]
            marker_color = parameters[2]
            ax.scatter(
                x=coords.ra.deg,
                y=coords.dec.deg,
                s=[marker_size]*coords.size,
                color=marker_color,
                transform=ax.get_transform("world")
            )

        if kwargs.get("text", None) is not None:
            parameters = kwargs["text"]
            if len(parameters) != 3:
                raise ValueError(
                    "'text' syntax should be: (<SkyCoord>, <[text]>, <color>)"
                )
            coords = parameters[0]
            if coords.isscalar:
                coords = coords.reshape((1,))
            text = parameters[1]
            text_color = parameters[2]

            for i in range(coords.size):
                ax.text(
                    x=coords[i].ra.deg,
                    y=coords[i].dec.deg,
                    s=text[i],
                    color=text_color,
                    transform=ax.get_transform("world"),
                    clip_on=True
                )

        if kwargs.get("contour", None) is not None:
            parameters = kwargs["contour"]
            data = parameters[0]
            if len(parameters) != 3:
                raise ValueError(
                    "'contour' syntax should be: (<numpy.ndarray>, <[levels]>, <colormap>)"
                )
            contour, _ = reproject_from_healpix(
                (data, ICRS()),
                wcs,
                nested=False,
                shape_out=shape#(ndec, nra)
            )
            ax.contour(
                contour,
                levels=parameters[1],
                cmap=parameters[2],
            )

        # Other
        im.set_clip_path(ax.coords.frame.patch)
        ax.set_title(title, pad=20)

        # Save or show
        if figname is None:
            plt.show()
        elif figname.lower() == "return":
            return fig, ax
        else:
            fig.savefig(
                figname,
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
        plt.close('all')
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- SkySlice -------------------------- #
# ============================================================= #
class SkySlice(SkySliceBase):
    """ """

    def __init__(self,
            coordinates: SkyCoord,
            frequency: u.Quantity,
            time: Time,
            polarization: Union[str, float, int],
            value: np.ndarray,
            observer: EarthLocation = nenufar_position
        ):
        super().__init__(
            coordinates=coordinates,
            time=time,
            frequency=frequency,
            polarization=polarization,
            observer=observer,
            value=value
        )


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

        if isinstance(values, da.Array):
            with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
                values = values.compute()

        data[(x_int, y_int)] = 0.
        np.add.at(weights, (x_int, y_int), 1)
        weights[weights<0.5] = 1.
        np.add.at(data, (x_int, y_int), values)
        data[(x_int, y_int)] /= weights[(x_int, y_int)]

        return data.T
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ HpxSkySlice ------------------------ #
# ============================================================= #
class HpxSkySlice(SkySliceBase):
    """ """

    def __init__(self,
            coordinates: SkyCoord,
            frequency: u.Quantity,
            time: Time,
            polarization: Union[str, float, int],
            value: np.ndarray,
            observer: EarthLocation = nenufar_position
        ):
        super().__init__(
            coordinates=coordinates,
            time=time,
            frequency=frequency,
            polarization=polarization,
            observer=observer,
            value=value
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
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
        
        if isinstance(values, da.Array):
            with ProgressBar() if log.getEffectiveLevel() <= logging.INFO else DummyCtMgr():
                values = values.compute()

        with np.errstate(invalid='ignore'):
            # Ignore the invalid value in bilinear_interpolation (astropy-healpix)
            array, _ = reproject_from_healpix(
                (values, ICRS()),
                wcs,
                nested=False,
                shape_out=shape
            )
        return array
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------------- Sky ---------------------------- #
# ============================================================= #
class Sky(AstroObject):
    """  """

    def __init__(self,
            coordinates: SkyCoord,
            time: Time = Time.now(),
            frequency: u.Quantity = 50*u.MHz,
            polarization: np.ndarray = np.array([0]),
            value: Union[float, np.ndarray] = 0.,
            observer: EarthLocation = nenufar_position
        ):
        self.coordinates = coordinates
        self.time = time
        self.frequency = frequency
        self.polarization = polarization
        self.observer = observer
        self.value = value
    

    def __str__(self):
        text = (
            f"{self.__class__} instance\n"
            f"value: {self.shape}\n"
            f"\t* time: {self.time.shape}\n"
            f"\t* frequency: {self.frequency.shape}\n"
            f"\t* polarization: {self.polarization.shape}\n"
            f"\t* coordinates: {self.coordinates.shape}\n"
        )
        return text


    def __truediv__(self, other):
        if isinstance(other, Sky):
            self.value /= other.value
        else:
            self.value /= other
        return self
    

    def __mul__(self, other):
        new_sky = copy.copy(self)
        if isinstance(other, Sky):
            new_sky.value *= other.value
        else:
            new_sky.value *= other
        return new_sky


    def __getitem__(self, n):
        """ """
        val = self.value[n]
        if val.ndim != 1:
            raise IndexError(
                "<class 'HpxSky'>: wrong index selection on <arg "
                f"'value'> of shape {self.value.shape} (time, "
                "frequency, healpix_cells). A 1D array is "
                "expected as a result of the selection."
            )
        return SkySlice(
            coordinates=self.coordinates,
            value=val,
            time=self.time[n[0]],
            frequency=self.frequency[n[1]],
            polarization=self.polarization[n[2]],
            observer=self.observer
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def value(self):
        """ """
        return self._value
    @value.setter
    def value(self, v):
        expected_shape = (
            self.time.size,
            self.frequency.size,
            self.polarization.size,
            self.coordinates.size
        )

        if np.isscalar(v):
            v *= np.ones(expected_shape)
        else:
            if v.shape != expected_shape:
                raise ValueError(f"Shape incorrect, expected {expected_shape}, got {v.shape}.")
        
        if v.dtype < np.float64:
            v = v.astype(np.float64)

        self._value = v

    
    @property
    def time(self):
        """ """
        return self._time
    @time.setter
    def time(self, t):
        if t.isscalar:
            t = t.reshape((1,))
        self._time = t


    @property
    def frequency(self):
        """ """
        return self._frequency
    @frequency.setter
    def frequency(self, f):
        if f.isscalar:
            f = f.reshape((1,))
        self._frequency = f


    @property
    def polarization(self):
        """ """
        return self._polarization
    @polarization.setter
    def polarization(self, p):
        if np.ndim(p) == 0:
            p = np.array([p])
        self._polarization = p


    @property
    def shape(self):
        """ """
        return self.value.shape
    # @property
    # def visible_sky(self):
    #     """ """
    #     altaz = self.horizontal_coordinates
    #     return altaz.alt.deg > 0


    @property
    def visible_mask(self):
        """ """
        mask = self.horizontal_coordinates.alt.deg > 0
        mask = np.expand_dims(mask, (1, 2)) # add frequency and polarization
        mask = np.repeat(mask, self.frequency.size, axis=1)
        mask = np.repeat(mask, self.polarization.size, axis=2)
        return mask

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute_lmn(self, phase_center: SkyCoord, coordinate_mask: np.ndarray = None):
        r""" (l, m, n) image domain coordinates computed from 
            HEALPix equatorial coordinates (in Right-Ascension
            :math:`\alpha` and Declination :math:`\delta`, see
            :attr:`~nenupy.astro.sky.Sky.coordinates`) with
            respect to the ``phase_center`` (of equatorial 
            coordinates :math:`\alpha_0`, :math:`\delta_0`).

            .. math::
                \cases{
                    l = \cos(\delta) \sin( \Delta \alpha)\\
                    m = \sin(\delta) \cos(\delta_0) - \cos(\delta) \sin(\delta_0) \cos(\Delta \alpha)\\
                    n = \sqrt{ 1 - l^2 - m^2 }
                }

            where :math:`\Delta \alpha = \alpha - \alpha_0`.

            :param phase_center:
                Image phase center.
            :type phase_center:
                :class:`~astropy.coordinates.SkyCoord`
            :param coordinate_mask:
                Mask applied to coordinates before computing (l,m,n) values.
            :type coordinate_mask:
                :class:`~numpy.ndarray`

            :returns: (l, m, n)
            :rtype: `tuple` of 3 :class:`~numpy.ndarray`
        """
        ra = self.coordinates[coordinate_mask].ra.rad
        dec = self.coordinates[coordinate_mask].dec.rad
        ra_0 = phase_center.ra.rad
        dec_0 = phase_center.dec.rad
        ra_delta = ra - ra_0
        # ra_delta = ra_0 - ra
        l = np.cos(dec)*np.sin(ra_delta)
        m = np.sin(dec)*np.cos(dec_0) -\
            np.cos(dec)*np.sin(dec_0)*np.cos(ra_delta)
        n = np.sqrt(1 - l**2 - m**2)
        return l, m, n


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #

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
            polarization: np.ndarray = np.array([0]),
            value: Union[float, np.ndarray] = 0.,
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
            polarization=polarization,
            value=value,
            observer=observer
        )


    def __getitem__(self, n):
        """ """
        val = self.value[n]
        if val.ndim != 1:
            raise IndexError(
                "<class 'HpxSky'>: wrong index selection on <arg "
                f"'value'> of shape {self.value.shape} (time, "
                "frequency, healpix_cells). A 1D array is "
                "expected as a result of the selection."
            )
        return HpxSkySlice(
            coordinates=self.coordinates,
            value=val,
            time=self.time[n[0]],
            frequency=self.frequency[n[1]],
            polarization=self.polarization[n[2]],
            observer=self.observer
        )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def shaped_like(cls, other):
        """ """
        if not isinstance(other, HpxSky):
            raise TypeError(
                f"{cls.__class__} instance expected."
            )
        return cls(
            resolution=other.resolution,
            time=other.time,
            frequency=other.frequency,
            polarization=other.polarization,
            observer=other.observer
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
# ============================================================= #
# ============================================================= #
