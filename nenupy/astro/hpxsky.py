#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***********
    HEALPix Sky
    ***********

    :class:`~nenupy.astro.hpxsky.HpxSky` class is designed to
    handle `HEALPix <https://healpix.jpl.nasa.gov/>`_ sky map
    representation and comes with many attributes and methods
    to ease pixel selection and plotting on sky coordinates.

    .. seealso::
        `healpy documentation <https://healpy.readthedocs.io/en/latest/>`_
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
from healpy.fitsfunc import write_map

import nenupy
from nenupy.instru import nenufar_loc

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# -------------------------- HpxSky --------------------------- #
# ============================================================= #
class HpxSky(object):
    """ Base class for all ``nenupy`` HEALPix sky representation
        related objects.

        .. seealso::
            :class:`~nenupy.simulation.hpxsimu.HpxSimu`,
            :class:`~nenupy.skymodel.hpxgsm.HpxGSM`,
            :class:`~nenupy.skymodel.hpxlofar.HpxLOFAR`,
            :class:`~nenupy.beam.hpxbeam.HpxABeam`,
            :class:`~nenupy.beam.hpxbeam.HpxDBeam`,
            :meth:`~nenupy.crosslet.crosslet.Crosslet.image`,
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
            See for e.g. `HEALPix <https://lambda.gsfc.nasa.gov/toolbox/tb_pixelcoords.cfm>`_.
            
            Default value is `1`.
            
            :setter: :class:`~astropy.units.Quantity` that can be 
                converted as an angle or :class:`~astropy.coordinates.Angle` 
                or a `float` which will be understood as degrees.
            
            :getter: resolution
            
            :type: :class:`~astropy.coordinates.Angle`

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

        self._resolution = Angle(
            angle=nside2resol(
                self.nside,
                arcmin=True
            ),
            unit=u.arcmin
        )

        self._get_eq_coords()
        return


    @property
    def time(self):
        """ UTC time at which horizontal coordinates should be 
            computed.
            
            :setter: :class:`~astropy.time.Time` or `str` able
                to be parsed as a time
            
            :getter: time
            
            :type: :class:`~astropy.time.Time`
        """
        if not hasattr(self, '_time'):
            self._time = Time.now()
        return self._time
    @time.setter
    def time(self, t):
        if t is None:
            t = Time.now()
        self._time = Time(t) if not isinstance(t, Time) else t

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
        """ Map of the sky in HEALPix RING representation.
            If :attr:`~nenupy.astro.hpxsky.HpxSky.visible_sky` is
            `True`, pixels belonging to the sky portion invisible
            at NenuFAR's location at :attr:`~nenupy.astro.hpxsky.HpxSky.time`
            are masked.

            :setter: Sky map
            
            :getter: Sky map
            
            :type: :class:`~numpy.ma.core.MaskedArray`
        """
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
        """ Mask or not the invisible sky from NenuFAR's location
            (see :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`).

            :setter: Mask invisible sky?
            
            :getter: Mask invisible sky?
            
            :type: `bool`
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
            
            :type: :class:`~astropy.coordinates.SkyCoord`
        """
        if self.visible_sky:
            return self._eq_coords[self._is_visible]
        else:
            return self._eq_coords


    @property
    def ho_coords(self):
        """ Horizontal coordinates of the HEALPix sky computed
            at NenuFAR's location and at
            time :attr:`~nenupy.astro.hpxsky.HpxSky.time`.

            :getter: (Alt, Az) coordinates

            :type: :class:`~astropy.coordinates.SkyCoord`
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
            :type position: :class:`~astropy.coordinates.SkyCoord`
        """
        rot = Rotator(
            deg=True,
            rot=[position.ra.deg, position.dec.deg],
            coord=['C', 'C']
        )
        sky = rot.rotate_map_alms(self.skymap)
        return sky


    def lmn(self, phase_center):
        r""" (l, m, n) image domain coordinates computed from 
            HEALPix equatorial coordinates (in Right-Ascension
            :math:`\alpha` and Declination :math:`\delta`, see
            :attr:`~nenupy.astro.hpxsky.HpxSky.eq_coords`) with
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
                Phase center of image
            :type phase_center: :class:`~astropy.coordinates.SkyCoord`

            :returns: (l, m, n)
            :rtype: `tuple` of 3 :class:`~numpy.ndarray`
        """
        ra = self.eq_coords.ra.rad
        dec = self.eq_coords.dec.rad
        ra_0 = phase_center.ra.rad
        dec_0 = phase_center.dec.rad
        ra_delta = ra - ra_0
        # ra_delta = ra_0 - ra
        l = np.cos(dec)*np.sin(ra_delta)
        m = np.sin(dec)*np.cos(dec_0) -\
            np.cos(dec)*np.sin(dec_0)*np.cos(ra_delta)
        n = np.sqrt(1 - l**2 - m**2)
        return l, m, n


    def radec_value(self, ra=None, dec=None, n=100):
        """ Get the :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`
            values at ``ra``, ``dec`` coordinates.

            :param ra:
                RA in equatorial coordinates (in degrees if `float`)
                Default: None
            :type ra: `float`, :class:`~numpy.ndarray`, or :class:`~astropy.units.Quantity`
            :param dec:
                Declination in equatorial coordinates (in degrees if `float`)
                Default: None
            :type dec: `float`, :class:`~numpy.ndarray`, or :class:`~astropy.units.Quantity`
            :param n:
                Number of points to evaluate if one coordinate is `None`
                Default: 100
            :type n: `int`

            :returns: Sky map values at ``ra``, ``dec``
            :rtype: :class:`~numpy.ndarray`
        """
        if (ra is not None) and (dec is not None):
            pass
        elif (ra is not None) and (dec is None):
            if isinstance(ra, u.Quantity):
                ra = ra.to(u.deg).value
            ra = np.ones(n) * ra
            dec = np.linspace(0, 90, n)
        elif (ra is None) and (dec is not None):
            if isinstance(dec, u.Quantity):
                dec = dec.to(u.deg).value
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
        """ Get the :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`
            values at ``az``, ``el`` coordinates.

            :param az:
                Azimuth in horizontal coordinates (degrees)
                Default: None
            :type az: `float`, :class:`~numpy.ndarray`, or :class:`~astropy.units.Quantity`
            :param el:
                Elevation in horizontal coordinates (degrees)
                Default: None
            :type el: `float`, :class:`~numpy.ndarray`, or :class:`~astropy.units.Quantity`
            :param n:
                Number of points to evaluate if one coordinate is `None`
                Default: 100
            :type n: `int`

            :returns: Sky map values at ``az``, ``el``
            :rtype: :class:`~numpy.ndarray`
        """
        if (az is not None) and (el is not None):
            pass
        elif (az is not None) and (el is None):
            if isinstance(az, u.Quantity):
                az = az.to(u.deg).value
            az = np.ones(n) * az
            el = np.linspace(0, 90, n)
        elif (az is None) and (el is not None):
            if isinstance(el, u.Quantity):
                el = el.to(u.deg).value
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
        """ Plot the HEALPix :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`
            on an equatorial grid with a Elliptical frame.

            :param figname:
                Figure name, if ``None`` (default value), the
                figure is not saved.
            :type figname: `str`
            :param db:
                Sacle the data in decibel units. Default is
                ``True``.
            :type db: `bool`
            :param cmap:
                Name of the colormap. Default is ``'YlGnBu_r'``.
            :type cmap: `str`
            :param vmin:
                Minimum value to scale the figure. Default is
                min(:attr:`~nenupy.astro.hpxsky.HpxSky.skymap`).
            :type vmin: `float`
            :param vmax:
                Maximum value to scale the figure. Default is
                max(:attr:`~nenupy.astro.hpxsky.HpxSky.skymap`).
            :type vmax: `float`
            :param tickscol:
                Color of the RA ticks. Default is ``'black'``.
            :type tickscol: `str`
            :param title:
                Title of the plot. Default is ``None``.
            :type title: `str`
            :param cblabel:
                Colorbar label. Default is ``'Amp'`` if ``db=False``
                of ``'dB'`` if ``db=True``.
            :type cblabel: `str`
            :param grid:
                Show the equatorial grid.
            :type grid: `bool`
            :param cbar:
                Plot a colorbar.
            :type cbar: `bool`
            :param center:
                Center of the plot. Default is
                ``SkyCoord(0.*u.deg, 0.*u.deg)``.
            :type center: :class:`~astropy.coordinates.SkyCoord`
            :param size:
                Diameter of the cutout. Default is whole sky.
            :type size: `float` or :class:`~astropy.units.Quantity`
            :param figsize:
                Figure size in inches. Default is ``(15, 10)``.
            :type figsize: `tuple`
            :param indices:
                Default is ``None``. If not, a scatter plot is
                made on the desired HEALPix indices:
                ``(indices, size, color)``.
            :type indices: `tuple`
            :param scatter:
                Default is ``None``. If not, a scatter plot is
                made on the desired equatorial coordinates:
                ``(ra (deg), dec (deg), size, color)``.
            :type scatter: `tuple`
            :param text:
                Default is ``None``. If not, text is overplotted
                on the desired equatorial coordinates:
                ``(ra (deg), dec (deg), text, color)``.
            :type text: `tuple`
            :param curve:
                Default is ``None``. If not, a curve plot is
                made on the desired equatorial coordinates:
                ``(ra (deg), dec (deg), linestyle, color)``.
            :type curve: `tuple`
            :param contour: ...
            :type contour: ...

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
        if not isinstance(kwargs['center'], SkyCoord):
            raise TypeError(
                'center must be a SkyCoord object.'
            )

        # Preparing WCS projection
        dangle = 0.675
        scale = int(dangle/self.resolution.deg)
        scale = 1 if scale <= 1 else scale
        nra = 480*scale
        ndec = 240*scale
        if 'size' in kwargs.keys():
            if isinstance(kwargs['size'], u.Quantity):
                kwargs['size'] = kwargs['size'].to(u.deg).value
            resol = dangle/scale
            nra = int(kwargs['size'] / resol)
            ndec = nra
            raauto = False

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [nra/2 + 0.5, ndec/2 + 0.5]
        wcs.wcs.cdelt = np.array([-dangle/scale, dangle/scale])
        wcs.wcs.crval = [kwargs['center'].ra.deg, kwargs['center'].dec.deg]
        wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']

        # Make an array out of HEALPix representation
        skymap = self.skymap.copy()
        if self.visible_sky:
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
        elif kwargs['vmin'] is None:
            kwargs['vmin'] = np.min(data[mask])
        else:
            pass
        if 'vmax' not in kwargs.keys():
            kwargs['vmax'] = np.max(data[mask])
        elif kwargs['vmax'] is None:
            kwargs['vmax'] = np.max(data[mask])
        else:
            pass
        if 'tickscol' not in kwargs.keys():
            kwargs['tickscol'] = 'black'
        if 'title' not in kwargs.keys():
            kwargs['title'] = None
        if 'cblabel' not in kwargs.keys():
            kwargs['cblabel'] = cblabel
        if 'grid' not in kwargs.keys():
            kwargs['grid'] = True
        if 'cbar' not in kwargs.keys():
            kwargs['cbar'] = True
        if 'figsize' not in kwargs.keys():
            kwargs['figsize'] = (15, 10)
        if 'indices' not in kwargs.keys():
            kwargs['indices'] = None
        if 'scatter' not in kwargs.keys():
            kwargs['scatter'] = None
        if 'curve' not in kwargs.keys():
            kwargs['curve'] = None
        if 'contour' not in kwargs.keys():
            kwargs['contour'] = None
        if 'text' not in kwargs.keys():
            kwargs['text'] = None


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
            interpolation='none',
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

        # Overplot
        if kwargs['indices'] is not None:
            ax.scatter(
                x=self.eq_coords[kwargs['indices'][0]].ra.deg,
                y=self.eq_coords[kwargs['indices'][0]].dec.deg,
                s=[kwargs['indices'][1]]*len(kwargs['indices'][0]),
                color=kwargs['indices'][2],
                transform=ax.get_transform('world')
            )
        if kwargs['scatter'] is not None:
            ax.scatter(
                x=kwargs['scatter'][0],
                y=kwargs['scatter'][1],
                s=[kwargs['scatter'][2]]*len(kwargs['scatter'][0]),
                color=kwargs['scatter'][3],
                transform=ax.get_transform('world')
            )
        if kwargs['curve'] is not None:
            ax.plot(
                kwargs['curve'][0],
                kwargs['curve'][1],
                linestyle=kwargs['curve'][2],
                color=kwargs['curve'][3],
                transform=ax.get_transform('world')
            )
        if kwargs['text'] is not None:
            for i in range(len(kwargs['text'][0])):
                ax.text(
                    x=kwargs['text'][0][i],
                    y=kwargs['text'][1][i],
                    s=kwargs['text'][2][i],
                    color=kwargs['text'][3],
                    transform=ax.get_transform('world')
                )
        if kwargs['contour'] is not None:
            contour, fp = reproject_from_healpix(
                (kwargs['contour'][0], ICRS()),
                wcs,
                nested=False,
                shape_out=(ndec, nra)
            )
            ax.contour(
                contour,
                levels=kwargs['contour'][1],
                cmap=kwargs['contour'][2],
            )

        im.set_clip_path(ax.coords.frame.patch)
        ax.set_title(kwargs['title'], pad=25)

        # Colorbar
        if kwargs['cbar']:
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
        return


    def save(self, filename, header=None, partial=False):
        """ Save the :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`
            to a FITS file, using
            :func:`~healpy.fitsfunc.write_map`. The masked values
            of :attr:`~nenupy.astro.hpxsky.HpxSky.skymap` (i.e.,
            below horizon pixels) are converted to NaN values.

            :param filename:
                Name of the FITS file to save. If the name
                already exists, the file is overwritten.
            :type filename: `str`
            :param header:
                Extra records to add the the FITS header. The
                syntax is ``[('rec1', 10), ('rec2', 'test')]``. 
            :type header: `list`
        """
        hd = [
            # ('fillval', self.skymap.fill_value),
            ('software', 'nenupy'),
            ('version', nenupy.__version__),
            ('contact', nenupy.__email__)
        ]
        if header is not None:
            if not isinstance(header, list):
                raise TypeError(
                    'header must be a list'
                )
            for hd_i in header:
                if not isinstance(hd_i, tuple):
                    raise TypeError(
                        'header element should be tuple'
                    )
                if len(hd_i) != 2:
                    raise IndexError(
                        'header element should be of length 2'
                    )
                if not isinstance(hd_i[0], str):
                    raise TypeError(
                        'First value of header element should be string'
                    )
                hd.append(hd_i)
        map2write = self.skymap.data.copy()
        map2write[~self._is_visible] = np.nan
        write_map(
            filename=filename,
            m=map2write,
            nest=False,
            coord='C',
            overwrite=True,
            dtype=self.skymap.dtype,
            extra_header=hd,
            partial=partial
        )
        log.info(
            'HEALPix image of {} cells (nside={}) saved in `{}`.'.format(
                map2write.size,
                self.nside,
                filename
            )
        )
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_nside(self):
        """ Compute the nearest HEALPix nside given an angular
            resolution
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
        order_idx = np.argmin(np.abs(resols - self._resolution))
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

