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
from nenupysim.instru import nenufar_loc


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

