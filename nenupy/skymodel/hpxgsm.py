#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ***********
    HEALPix GSM
    ***********

    The :class:`~nenupy.skymodel.hpxgsm.HpxGSM` is a wrapper
    around `PyGSM <https://github.com/telegraphic/PyGSM>`_ 
    (`Oliveira-Costa et al., 2008 <https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2966.2008.13376.x>`_)
    taking advantage of the :class:`~nenupy.astro.hpxsky.HpxSky`
    HEALPix sky representation handler. This can be used to 
    simulate NenuFAR observations (see :ref:`tuto_simu_ref`
    tutorial).
    
    Usage is fairly simple, only requiring the setting of the
    frequency at which the sky model should be produced (``freq``)
    and the resolution ``resolution`` to tesselate the HEALPix
    grid.

    >>> from nenupy.skymodel import HpxGSM
    >>> gsm = HpxGSM(freq=50, resolution=0.2)
    >>> gsm.plot()
    
    .. image:: ./_images/gsm.png
      :width: 800

    If the :attr:`~nenupy.astro.hpxsky.HpxSky.time` attribute is
    set and :attr:`~nenupy.astro.hpxsky.HpxSky.visible_sky` is
    ``True``, the :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`
    HEALPix array is masked in order to only display pixels above
    the horizon with respect to NenuFAR Earth location:

    >>> from nenupy.skymodel import HpxGSM
    >>> gsm = HpxGSM(freq=50, resolution=0.2)
    >>> gsm.time = '2020-04-01 12:00:00'
    >>> gsm.visible_sky = True
    >>> gsm.plot()
    
    .. image:: ./_images/gsm_visible.png
      :width: 800

    Sky model cutouts can be achieved using :func:`~nenupy.astro.hpxsky.HpxSky.plot`
    arguments ``center`` (that takes :class:`~astropy.coordinates.SkyCoord`
    object) and ``size`` (angular diameter of the cutout):

    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> cyga = SkyCoord(299.868*u.deg, 40.734*u.deg)
    >>> from nenupy.skymodel import HpxGSM
    >>> gsm = HpxGSM(freq=50*u.MHz, resolution=0.1*u.deg)
    >>> gsm.plot(center=cyga, size=30*u.deg)
    
    .. image:: ./_images/gsm_cyga.png
      :width: 800

    .. seealso::
        :class:`~nenupy.astro.hpxsky.HpxSky`

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HpxGSM'
]


from pygsm import GlobalSkyModel
from healpy.rotator import Rotator
from healpy.pixelfunc import ud_grade
import astropy.units as u

from nenupy.astro import HpxSky
from nenupy.instru import _HiddenPrints


# ============================================================= #
# -------------------------- HpxGSM --------------------------- #
# ============================================================= #
class HpxGSM(HpxSky):
    r""" This class inherits from :class:`~nenupy.astro.hpxsky.HpxSky`
        and aims at wrapping around pygsm to display the
        low-frequency sky model in HEALPix representation in sky
        coordinates relevant to the NenuFAR array.

        :param freq:
            Frequency at which the GSM should be displayed (in
            MHz if `float`).
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param resolution:
            Desired sky model resolution. The best matching
            available HEALPix angular resolution is set. Due to
            weight files availability (e.g. 
            `healpy weights <https://healpy.github.io/healpy-data/>`_),
            only ``resolution`` :math:`\leq` 1.8323 degrees (or
            nside :math:`\geq` 32) are allowed).
        :type resolution: `float` or :class:`~astropy.units.Quantity`

    """

    def __init__(self, freq=50, resolution=1):
        if isinstance(resolution, u.Quantity):
            resolution = resolution.to(u.deg).value
        if resolution > 1.9:
            raise ValueError(
                'resolution must be <= 1.83 deg'
            )
        super().__init__(
            resolution=resolution
        )
        self.freq = freq 
        
        gsm_map = self._load_gsm(self.freq)
        gsm_map = self._resize(gsm_map, self.nside)
        gsm_map = self._to_celestial(gsm_map)
        self.skymap = gsm_map


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _load_gsm(freq):
        """
        """
        gsm = GlobalSkyModel(freq_unit='MHz')
        gsmap = gsm.generate(freq)
        return gsmap


    @staticmethod
    def _to_celestial(sky):
        """ Convert the GSM from naitve Galactic to Equatorial
            coordinates.
        """
        rot = Rotator(
            deg=True,
            rot=[0, 0],
            coord=['G', 'C']
        )
        with _HiddenPrints():
            sky = rot.rotate_map_alms(sky)
        return sky


    @staticmethod
    def _resize(sky, nside):
        """ Resize the GSM to match tjhe desired resolution
        """
        sky = ud_grade(
            map_in=sky,
            nside_out=nside
        )
        return sky
# ============================================================= #

