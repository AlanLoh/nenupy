#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **********************
    Point source sky model
    **********************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'extrapol_flux',
    'get_point_sources'
]


from os.path import dirname, abspath, join
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

import logging
log = logging.getLogger(__name__)


skymodel_file = join(
    dirname(abspath(__file__)),
    'lfsky.fits'
)


# ============================================================= #
# ----------------------- extrapol_flux ----------------------- #
# ============================================================= #
def extrapol_flux(freq, rflux, rfreq, index):
    """ Given the ``rflux`` in Jy at the ``rfreq`` in MHz,
        and the spectral index ``index``, extrapolate the ``flux``
        at ``freq`` MHz.

        :param freq:
            Frequency at which the flux extrapolation should be
            done. If `float`, assumed to be given in MHz. 
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param rflux:
            Reference flux of the source. If `float` assumed to
            be in Jy.
        :type rflux: `float` or :class:`~astropy.units.Quantity`
        :param rfreq:
            Reference frequency at which ``rflux`` was measured.
            If `float`, assumed to be given in MHz.
        :type rfreq: `float` or :class:`~astropy.units.Quantity`
        :param index:
            Spectral index of the power-law source model.
        :type index: `float`

        :returns: Extrapolated flux in Jy.
        :rtype: :class:`~astropy.units.Quantity`
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    else:
        freq = freq.to(u.MHz)
    if not isinstance(rflux, u.Quantity):
        rflux *= u.Jy
    else:
        rflux = rflux.to(u.Jy)
    if not isinstance(rfreq, u.Quantity):
        rfreq *= u.MHz
    else:
        rfreq = rfreq.to(u.MHz)

    return (rflux * (freq/rfreq)**index)
# ============================================================= #


# ============================================================= #
# ----------------------- extrapol_flux ----------------------- #
# ============================================================= #
def get_point_sources(freq, center, radius):
    """ Retrieve point sources coordinates and flux from the
        model gathered from the `LOFAR <https://lcs165.lofar.eu/>`_
        database. This returns sources spread over the ``center``
        sky coordinates at a given ``radius`` as well as their
        extrapolated fluxes at the desired frequency ``freq``.

        :param freq:
            Frequency at which the flux extrapolation should be
            done. If `float`, assumed to be given in MHz. 
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param center:
            Sky coordinates of the center fo the field to look
            for point-sources. 
        :type center: :class:`~astropy.coordinates.SkyCoord`
        :param radius:
            Radius from the ``center`` to search for sources. If
            `float`, assumed to be given in degrees.
        :type radius: `float` or :class:`~astropy.units.Quantity`

        :returns:
            Point source coordinates and their associated fluxes
            (:class:`~astropy.coordinates.SkyCoord`, `~numpy.ndarray`)
        :rtype: `tuple`
    """
    if not isinstance(center, SkyCoord):
        raise TypeError(
            'center must be a SkyCoord object.'
        )
    if not isinstance(radius, u.Quantity):
        radius *= u.deg
    else:
        radius = radius.to(u.deg)
    model_table = Table.read(skymodel_file)
    pointsrc = SkyCoord(
        ra=model_table['ra']*u.deg,
        dec=model_table['dec']*u.deg,
    )
    separations = center.separation(pointsrc)
    srcmask = separations <= radius
    fluxes = extrapol_flux(
        freq=freq,
        rflux=model_table[srcmask]['flux']*u.Jy,
        rfreq=model_table[srcmask]['rfreq']*u.Hz,
        index=model_table[srcmask]['index']
    )
    return pointsrc[srcmask], fluxes
# ============================================================= #

