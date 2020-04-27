#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ****
    Beam
    ****
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Beam',
    'ABeam',
    'DBeam'
]


import numpy as np
import astropy.units as u
from astropy.coordinates import ICRS
from astropy.time import Time
from healpy.pixelfunc import get_interp_val

from nenupy.instru import (
    nenufar_ant_gain,
    desquint_elevation,
    analog_pointing,
    ma_antpos,
    ma_info,
    ma_pos
)
from nenupy.astro import (
    wavelength,
    to_altaz,
    ho_coord
)

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# --------------------------- Beam ---------------------------- #
# ============================================================= #
class Beam(object):
    """
    """

    def __init__(self, freq, polar):
        self.freq = freq
        self.polar = polar


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, u.Quantity):
            f *= u.MHz
        if not f.isscalar:
            raise ValueError(
                'Only scalar frequency allowed'
            )
        self._freq = f
        log.info(
            'Frequency sets at {}'.format(self._freq)
        )
        return


    @property
    def polar(self):
        return self._polar
    @polar.setter
    def polar(self, p):
        if not np.isscalar(p):
            raise ValueError(
                'Only scalar polar allowed'
            )
        allowed = ['NW', 'NE']
        if p.upper() not in allowed:
            raise ValueError(
                'Polarization {} not in {}'.format(
                    p,
                    allowed
                )
            )
        self._polar = p.upper()
        log.info(
            'Polarization sets at {}'.format(self._polar)
        )
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def ant_gain(self, coords, time):
        """ Returns NenuFAR antenna gain values interpolated at
            coordinates ``coords`` at time ``time``.

            :param coords:
            :type coords: :class:`~astropy.coordinates.ICRS`
            :param time:
            :type time: :class:`~astropy.time.Time`

            :returns: NenuFAR normalized antenna gain
            :rtype: `~numpy.ndarray`

            :Example:
                To get back the HEALPix ant gain:
                
                >>> from nenupy.beam import ABeam
                >>> import numpy as np
                >>> import astropy.units as u
                >>> from astropy.coordinates import ICRS
                >>> from astropy.time import Time
                >>> from healpy.pixelfunc import nside2npix, pix2ang
                >>> from healpy.visufunc import mollview
                >>> b = Beam(
                        freq=50,
                        polar='NE',
                    )
                >>> npix = nside2npix(nside=32)
                >>> ra, dec = pix2ang(
                        nside=32,
                        ipix=np.arange(npix),
                        lonlat=True
                    )
                >>> gain = b.ant_gain(
                        coords=ICRS(
                            ra=ra*u.deg,
                            dec=dec*u.deg
                        ),
                        time=Time.now()
                    )
                >>> mollview(gain)

        """
        if not isinstance(coords, ICRS):
            raise TypeError(
                'coords should be ICRS object'
            )
        if not isinstance(time, Time):
            raise TypeError(
                'time should be Time object'
            )
        hpxgain = nenufar_ant_gain(
            freq=self.freq,
            polar=self.polar,
            nside=32,
            time=time
        )
        log.info(
            'NenuFAR HEALPix antenna gain loaded.'
        )
        vals = get_interp_val(
            m=hpxgain,
            theta=coords.ra.deg,
            phi=coords.dec.deg,
            lonlat=True
        )
        log.info(
            'NenuFAR antenna gain values interpolated.'
        )
        return vals


    def array_factor(self, phase_center, coords, antpos):
        """
        """
        def get_phi(az, el, antpos):
            """ az, el in radians
            """
            xyz_proj = np.array(
                [
                    np.cos(az) * np.cos(el),
                    np.sin(az) * np.cos(el),
                    np.sin(el)
                ]
            )
            antennas = np.array(antpos)
            phi = np.dot(antennas, xyz_proj)
            return phi
        phi0 = get_phi(
            az=[phase_center.az.rad],
            el=[phase_center.alt.rad],
            antpos=antpos
        )
        phi_grid = get_phi(
            az=coords.az.rad,
            el=coords.alt.rad,
            antpos=antpos
        )
        delay = phi_grid - phi0
        coeff = 2j * np.pi / wavelength(self.freq).value
        af = np.sum(np.exp(coeff*delay), axis=0)
        return np.real(af * af.conjugate())
    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #


# ============================================================= #
# --------------------------- Beam ---------------------------- #
# ============================================================= #
class ABeam(Beam):
    """
    """

    def __init__(self, freq, polar, azana, elana, ma=0):
        super().__init__(
            freq=freq,
            polar=polar
        )
        self.azana = azana
        self.elana = elana
        self.ma = ma
        self.squint_freq = 30*u.MHz
        self.beamsquint = True


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def azana(self):
        """
        """
        return self._azana
    @azana.setter
    def azana(self, a):
        if not isinstance(a, u.Quantity):
            a *= u.deg
        self._azana = a
        log.info(
            'Desired azimuth: {}'.format(self._azana)
        )
        return


    @property
    def elana(self):
        """
        """
        return self._elana
    @elana.setter
    def elana(self, e):
        if not isinstance(e, u.Quantity):
            e *= u.deg
        self._elana = e
        log.info(
            'Desired elevation: {}'.format(self._elana)
        )
        return


    @property
    def squint_freq(self):
        return self._squint_freq
    @squint_freq.setter
    def squint_freq(self, f):
        if not isinstance(f, u.Quantity):
            f *= u.MHz
        if not f.isscalar:
            raise ValueError(
                'Only scalar squint_freq allowed'
            )
        self._squint_freq = f
        log.info(
            'Squint frequency sets at {}'.format(f)
        )
        return


    @property
    def beamsquint(self):
        return self._beamsquint
    @beamsquint.setter
    def beamsquint(self, b):
        if not isinstance(b, bool):
            raise TypeError(
                'beamsquint should be a boolean'
            )
        self._beamsquint = b
        if b:
            log.info(
                'Beam squint correction activated.'
            )
        else:
            log.info(
                'No beam squint correction.'
            )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def beam_values(self, coords, time):
        """
            :param coords:
            :type coords: :class:`~astropy.coordinates.ICRS`
            :param time:
            :type time: :class:`~astropy.time.Time`
        """
        if not isinstance(coords, ICRS):
            raise TypeError(
                'coords should be ICRS object'
            )
        if not isinstance(time, Time):
            raise TypeError(
                'time should be Time object'
            )
        # Real pointing
        if self.beamsquint:
            el = desquint_elevation(
                elevation=self.elana,
                opt_freq=self.squint_freq
            )
        else:
            el = self.elana.copy()
        az, el = analog_pointing(self.azana, el)
        log.info(
            'Effective pointing=({}, {})'.format(
                az,
                el
            )
        )
        # Array factor
        phase_center = ho_coord(
                az=az,
                alt=el,
                time=time
        )
        altazcoords = to_altaz(
            radec=coords,
            time=time
        )  
        arrfac = self.array_factor(
            phase_center=phase_center,
            coords=altazcoords,
            antpos=ma_antpos(
                rot=ma_info['rot'][ma_info['ma'] == self.ma][0]
            )
        )
        # Antenna Gain
        antgain = self.ant_gain(
            coords=coords,
            time=time
        )
        anagain = arrfac * antgain
        log.info(
            'Anabeam computed for {} pixels.'.format(
                anagain.size
            )
        )
        return anagain

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #


# ============================================================= #
# --------------------------- Beam ---------------------------- #
# ============================================================= #
class DBeam(Beam):
    """
    """

    def __init__(self, freq, polar, azana, elana, azdig, eldig):
        super().__init__(
            freq=freq,
            polar=polar
        )

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #

