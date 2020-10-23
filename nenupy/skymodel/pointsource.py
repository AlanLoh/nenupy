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
    'get_point_sources',
    'LofarSkymodel'
]


from os.path import dirname, abspath, join
from astropy.table import Table, QTable
import astropy.units as u
from astropy.coordinates import SkyCoord
import urllib
import numpy as np

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


# ============================================================= #
# ----------------------- LofarSkymodel ----------------------- #
# ============================================================= #
class LofarSkymodel(object):
    """
    """

    def __init__(self, center, radius, cutoff):
        self.center = center
        self.radius = radius
        self.cutoff = cutoff
        self.table = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def center(self):
        """ Sky coordinates center of the search field.
        """
        return self._center
    @center.setter
    def center(self, cen):
        if not isinstance(cen, SkyCoord):
            raise TypeError(
                'center should be an astropy SkyCoord instance'
            )
        self._center = cen


    @property
    def radius(self):
        """ Search radius
        """
        return self._radius.to(u.deg)
    @radius.setter
    def radius(self, r):
        if not isinstance(r, u.Quantity):
            raise TypeError(
                'radius should be an astropy Quantity (deg or equivalent) instance'
            )
        self._radius = r


    @property
    def cutoff(self):
        """
        """
        return self._cutoff.to(u.Jy)
    @cutoff.setter
    def cutoff(self, cut):
        if not isinstance(cut, u.Quantity):
            raise TypeError(
                'cutoff should be an astropy Quantity (Jy or equivalent) instance'
            )
        self._cutoff = cut


    @property
    def gsmURL(self):
        """ URL to query
        """
        return (
            'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?'
            'coord={},{}&'
            'radius={}&'
            'unit=deg&'
            'cutoff={}'.format(
                self.center.ra.deg,
                self.center.dec.deg,
                self.radius.value,
                self.cutoff.value
            )
        )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def getTable(self):
        """
        """
        self._buildTable()


    def getSkymodel(self, filename):
        """
        """
        with open(filename, 'w') as wfile:
            for line in self._getUrlResult():
                wfile.write(line + '\n')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _buildTable(self):
        """
        """
        skymodel = self._getUrlResult()
        rows = []
        for srcModel in self._parseUrlResult(skymodel):
            rows.append(
                self._parseSourceModel(srcModel)
            )
        self.table = QTable(
            rows=rows,
            names=(
                'name',
                'position',
                'i',
                'q',
                'u',
                'v',
                'sindex'
            )
        )


    @staticmethod
    def _parseSourceModel(sourceModel):
        """
        """
        name, typ, ra, dec, i, q, su, v, reff, sidx = sourceModel
        return (
            name,
            SkyCoord(
                ra + dec.replace('.', ':', 2),
                unit=(u.hourangle, u.deg)
            ),
            float(i if i.strip() else '0') * u.Jy,
            float(q if q.strip() else '0') * u.Jy,
            float(su if su.strip() else '0') * u.Jy,
            float(v if v.strip() else '0') * u.Jy,
            np.array(
                sidx.split('[')[1].split(']')[0].split(',')
            ).astype(np.float32)
        )


    @staticmethod
    def _parseUrlResult(lines):
        """
        """
        for line in lines:
            if line.startswith(('FORMAT', '#')):
                continue
            if ('POINT' in line) or ('GAUSSIAN' in line):
                yield line.split(',', 9)
            else:
                continue


    def _getUrlResult(self):
        """
        """
        urlOpen = urllib.request.urlopen(self.gsmURL)
        lines = urlOpen.read().decode('utf-8').split('\n')
        return lines
# ============================================================= #

