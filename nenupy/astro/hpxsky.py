#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***********
    HEALPix Sky
    ***********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HpxSky'
]


import numpy as np
from astropy import units as u
from astropy.coordinates import (
    Angle,
    EarthLocation,
    SkyCoord,
    AltAz,
    ICRS
)
from astropy.wcs import WCS
from astropy.time import Time
from healpy.pixelfunc import (
    nside2resol,
    order2nside,
    pix2ang,
    ang2pix,
    nside2npix
)
from healpy.rotator import Rotator

from nenupy.instru import nenufar_loc, HiddenPrints


# ============================================================= #
# -------------------------- HpxSky --------------------------- #
# ============================================================= #
class HpxSky(object):
    """
    """

    def __init__(self, resolution=1):
        self.visible_sky = True
        self.nside = None
        self.pixidx = None
        self.resolution = resolution
        self._is_visible = np.ones(nside2npix(self.nside), dtype=bool)
        self.skymap = np.zeros(nside2npix(self.nside))


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def resolution(self):
        """ Angular resolution of the HEALPix grid defined as the
            mean spacing between pixels.
            See for e.g. `HEALPix <https://lambda.gsfc.nasa.gov/toolbox/tb_pixelcoords.cfm>`_
            
            Default value is `1`.
            
            :setter: :class:`astropy.units.Quantity` that can be 
                converted as an angle or :class:`astropy.coordinates.Angle` 
                or a `float` which will be understood as degrees.
            
            :getter: resolution
            
            :type: :class:`astropy.coordinates.Angle`

            :Example:
            
            >>> from nenupysim.astro import HpxSky
            >>> sky = HpxSky(resolution=0.5)
        """
        return self._resolution
    @resolution.setter
    def resolution(self, r):
        if isinstance(r, u.Quantity):
            r = Angle(r)
        elif isinstance(r, Angle):
            pass
        else:
            r = Angle(r, unit='deg')

        if r < 0.05 * u.deg:
            raise ValueError(
                'Resolutions < 0.05 deg make slow computations'
            )

        self._resolution = r
        
        self._get_nside()

        self._get_eq_coords()
        return


    @property
    def time(self):
        """ UTC time at which horizontal coordinates should be 
            computed.
            
            :setter: :class:`astropy.time.Time` or `str` able
                to be parsed as a time.
            
            :getter: time
            
            :type: :class:`astropy.time.Time`
        """
        return self._time
    @time.setter
    def time(self, t):
        if t is None:
            t = Time.now()
        self._time = Time(t)

        # Update Az-Alt coordinates
        altaz_frame = AltAz(
            obstime=self._time,
            location=nenufar_loc
        )
        self._ho_coords = self._eq_coords.transform_to(altaz_frame)
        self._is_visible = self._ho_coords.alt.deg >= 0.
        return


    @property
    def skymap(self):
        if self.visible_sky:
            mask = np.ones(self._skymap.size, dtype=bool)
            mask[self._is_visible] = False
            return np.ma.masked_array(
                self._skymap,
                mask=mask,
                fill_value=-1.6375e+30
            )
        else:
            return self._skymap
    @skymap.setter
    def skymap(self, s):
        self._skymap = s
        return


    @property
    def visible_sky(self):
        """
        """
        return self._visible_sky
    @visible_sky.setter
    def visible_sky(self, v):
        if not isinstance(v, bool):
            raise TypeError(
                'visible_sky should be a boolean'
            )
        self._visible_sky = v
        return
    

    @property
    def eq_coords(self):
        """ Equatorial coordinates of the HEALPix sky.

            :getter: (RA, Dec) coordinates
            
            :type: :class:`astropy.coordinates.SkyCoord`
        """
        if self.visible_sky:
            return self._eq_coords[self._is_visible]
        else:
            return self._eq_coords


    @property
    def ho_coords(self):
        """ Horizontal coordinates of the HEALPix sky.

            :getter: (Alt, Az) coordinates

            :type: :class:`astropy.coordinates.SkyCoord`
        """
        if self.visible_sky:
            return self._ho_coords[self._is_visible]
        else:
            return self._ho_coords


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def rotate_to(self, position):
        """ Rotate the sky to the RA Dec `position`

            :param position:
                Equatorial position
            :type position: :class:`astropy.coordinates.SkyCoord`
        """
        rot = Rotator(
            deg=True,
            rot=[position.ra.deg, position.dec.deg],
            coord=['C', 'C']
        )
        sky = rot.rotate_map_alms(self.skymap)
        return sky


    def lmn(self, phase_center):
        """ (l, m, n) image domain coordinates

            :param phase_center:
                Phase center of image
            :type phase_center: :class:`astropy.coordinates.SkyCoord`
        """
        ra = self.eq_coords.ra.rad
        dec = self.eq_coords.dec.rad
        ra_0 = phase_center.ra.rad
        dec_0 = phase_center.dec.rad
        ra_delta = ra - ra_0
        l = np.cos(dec)*np.sin(ra_delta)
        m = np.sin(dec)*np.cos(dec_0) -\
            np.cos(dec)*np.sin(dec_0)*np.cos(ra_delta)
        n = np.sqrt(1 - l**2 - m**2)
        return l, m, n


    def radec_value(self, ra=None, dec=None, n=100):
        """ Get the `skymap` values at `az`, `al` coordinates.

            :param ra:
                RA in equatorial coordinates (degrees)
                Default: None
            :type ra: `float` or `np.ndarray`
            :param dec:
                Declination in equatorial coordinates (degrees)
                Default: None
            :type dec: `float` or `np.ndarray`
            :param n:
                Number of points to evaluate if one coordinate is `None`
                Default: 100
            :type n: `int`
        """
        if (ra is not None) and (dec is not None):
            pass
        elif (ra is not None) and (dec is None):
            ra = np.ones(n) * ra
            dec = np.linspace(0, 90, n)
        elif (ra is None) and (dec is not None):
            ra = np.linspace(0, 360, n)
            dec = np.ones(n) * dec
        else:
            raise Exception(
                'Give at least one coordinate'
            )
        # Get the indices
        indices = ang2pix(
            theta=ra,
            phi=dec,
            nside=self.nside,
            lonlat=True
        )
        return self.skymap[indices]


    def azel_value(self, az=None, el=None, n=100):
        """ Get the `skymap` values at `az`, `al` coordinates.

            :param az:
                Azimuth in horizontal coordinates (degrees)
                Default: None
            :type az: `float` or `np.ndarray`
            :param el:
                Elevation in horizontal coordinates (degrees)
                Default: None
            :type el: `float` or `np.ndarray`
            :param n:
                Number of points to evaluate if one coordinate is `None`
                Default: 100
            :type n: `int`
        """
        if (az is not None) and (el is not None):
            pass
        elif (az is not None) and (el is None):
            az = np.ones(n) * az
            el = np.linspace(0, 90, n)
        elif (az is None) and (el is not None):
            az = np.linspace(0, 360, n)
            el = np.ones(n) * el
        else:
            raise Exception(
                'Give at least one coordinate'
            )
        # Transform to RADEC
        altaz = AltAz(
            az=az*u.deg,
            alt=el*u.deg,
            location=nenufar_loc,
            obstime=self.time
        )
        radec = altaz.transform_to(ICRS)
        # Get the indices
        indices = ang2pix(
            theta=radec.ra.deg,
            phi=radec.dec.deg,
            nside=self.nside,
            lonlat=True
        )
        return self.skymap[indices]


    def plot(self, figname=None, db=True, **kwargs):
        """
            Possible kwargs:
                cmap
                vmin
                vmax
                tickscol
                title
                cblabel
                grid
                center
                size
        """
        # Lot of imports for this one...
        from reproject import reproject_from_healpix
        from astropy.coordinates import ICRS
        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.ticker import LinearLocator
        from matplotlib.colors import Normalize
        from matplotlib.cm import get_cmap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Cutout?
        raauto = True
        if 'center' not in kwargs.keys():
            kwargs['center'] = SkyCoord(0.*u.deg, 0.*u.deg)

        # Preparing WCS projection
        dangle = 0.675
        scale = int(dangle/self.resolution.deg)
        scale = 1 if scale <= 1 else scale
        nra = 480*scale
        ndec = 240*scale
        if 'size' in kwargs.keys():
            resol = dangle/scale
            nra = int(kwargs['size'].to(u.deg).value / resol)
            ndec = nra
            raauto = False

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [nra/2 + 0.5, ndec/2 + 0.5]
        wcs.wcs.cdelt = np.array([-dangle/scale, dangle/scale])
        wcs.wcs.crval = [kwargs['center'].ra.deg, kwargs['center'].dec.deg]
        wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']

        # Make an array out of HEALPix representation
        skymap = self.skymap.copy()
        skymap[~self._is_visible] = np.nan
        array, fp = reproject_from_healpix(
            (skymap, ICRS()),
            wcs,
            nested=False,
            shape_out=(ndec, nra)            
        )

        # Decibel or linear?
        if db:
            data = 10 * np.log10(array)
            cblabel = 'dB'
        else:
            data = array
            cblabel = 'Amp'
        mask = ~np.isnan(data) * ~np.isinf(data)

        # Make sure everything is correctly set up
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'YlGnBu_r'
        if 'vmin' not in kwargs.keys():
            kwargs['vmin'] = np.min(data[mask])
        if 'vmax' not in kwargs.keys():
            kwargs['vmax'] = np.max(data[mask])
        if 'tickscol' not in kwargs.keys():
            kwargs['tickscol'] = 'black'
        if 'title' not in kwargs.keys():
            kwargs['title'] = None
        if 'cblabel' not in kwargs.keys():
            kwargs['cblabel'] = cblabel
        if 'grid' not in kwargs.keys():
            kwargs['grid'] = True
        if 'figsize' not in kwargs.keys():
            kwargs['figsize'] = (15, 10)

        # Initialize figure
        fig = plt.figure(figsize=kwargs['figsize'])
        ax = plt.subplot(
            projection=wcs,
            frame_class=EllipticalFrame
        )

        # Full sky
        im = ax.imshow(
            data,
            origin='lower',
            cmap=kwargs['cmap'],
            vmin=kwargs['vmin'],
            vmax=kwargs['vmax']
        )
        axra = ax.coords[0]
        axdec = ax.coords[1]
        if kwargs['grid']:
            ax.coords.grid(color=kwargs['tickscol'], alpha=0.5)    
            axra.set_ticks_visible(False)
            axra.set_ticklabel(color=kwargs['tickscol'])
            axra.set_axislabel('RA', color=kwargs['tickscol'])
            axra.set_major_formatter('d')
            if raauto:
                axra.set_ticks([0, 45, 90, 135, 225, 270, 315]*u.degree)
            else:
                axra.set_ticks(number=10)
            axdec.set_ticks_visible(False)
            axdec.set_axislabel('Dec')
            axdec.set_major_formatter('d')
            axdec.set_ticks(number=10)
        else:
            axra.set_ticks_visible(False)
            axdec.set_ticks_visible(False)
            axra.set_ticklabel_visible(False)
            axdec.set_ticklabel_visible(False)
        im.set_clip_path(ax.coords.frame.patch)
        ax.set_title(kwargs['title'], pad=25)

        # Colorbar
        cax = inset_axes(ax,
           width='3%',
           height='100%',
           loc='lower left',
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax.transAxes,
           borderpad=0,
           )
        cb = ColorbarBase(
            cax,
            cmap=get_cmap(name=kwargs['cmap']),
            orientation='vertical',
            norm=Normalize(
                vmin=kwargs['vmin'],
                vmax=kwargs['vmax']
            ),
            ticks=LinearLocator()
        )
        cb.solids.set_edgecolor('face')
        cb.set_label(kwargs['cblabel'])
        cb.formatter.set_powerlimits((0, 0))

        # Save or show
        if figname is None:
            plt.show()
        else:
            fig.savefig(
                figname,
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
        plt.close('all')
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_nside(self):
        """ Compute the nearest HEALPix nside given an angular
            resolution.
        """
        orders = np.arange(30)
        nsides = order2nside(orders)
        resols = Angle(
            angle=nside2resol(
                nsides,
                arcmin=True
            ),
            unit=u.arcmin
        )
        order_idx = np.argmin( np.abs(resols - self._resolution))
        self.nside = nsides[order_idx]
        self.pixidx = np.arange(nside2npix(self.nside))
        return


    def _get_eq_coords(self):
        """ Compute the equatorial and horizontal coordinates
            for the HEALPix sky 
        """
        ra, dec = pix2ang(
            nside=self.nside,
            ipix=self.pixidx,
            lonlat=True,
            nest=False
        )
        ra *= u.deg
        dec *= u.deg
        self._eq_coords = SkyCoord(ra, dec, frame='icrs')
        return
# ============================================================= #
