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
from astropy.coordinates import ICRS, SkyCoord, AltAz
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
    toAltaz,
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
            :type coords: :class:`~astropy.coordinates.ICRS` or 
                :class:`~astropy.coordinates.SkyCoord`
            :param time:
            :type time: :class:`~astropy.time.Time`

            :returns: NenuFAR normalized antenna gain
            :rtype: `~numpy.ndarray`

            :Example:
                To get back the HEALPix ant gain:
                
                >>> from nenupy.beam import Beam
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
        if not isinstance(coords, (ICRS, SkyCoord)):
            raise TypeError(
                'coords should be ICRS or SkyCoord object'
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
        if not (isinstance(phase_center, AltAz) or hasattr(phase_center, 'altaz')):
            raise TypeError(
                'phase_center should be an AltAz instance'
            )
        if not (isinstance(coords, AltAz) or hasattr(coords, 'altaz')):
            raise TypeError(
                'coords should be an AltAz instance'
            )
        if not isinstance(antpos, np.ndarray):
            raise TypeError(
                'antpos should be an np.ndarray instance'
            )
        if antpos.shape[1] != 3:
            raise IndexError(
                'antpos should have 2nd dimension = 3 (x, y, z)'
            )
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
            'Desired analog azimuth: {}'.format(self._azana)
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
            'Desired analog elevation: {}'.format(self._elana)
        )
        return


    @property
    def ma(self):
        return self._ma
    @ma.setter
    def ma(self, m):
        if not isinstance(m, (int, np.integer)):
            raise TypeError(
                'ma should be integer'
            )
        max_ma_name = ma_info['ma'].size - 1
        if m > max_ma_name:
            raise ValueError(
                'select a MA name <= {}'.format(
                    max_ma_name
                )
            )
        self._ma = m
        self._rot = ma_info['rot'][ma_info['ma'] == m][0]
        log.info(
            'MA {} selected (rotation {} mod 60 deg)'.format(
                m,
                self._rot%60
            )
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
            :type coords: :class:`~astropy.coordinates.ICRS` or 
                :class:`~astropy.coordinates.SkyCoord`
            :param time:
            :type time: :class:`~astropy.time.Time`
        """
        if not isinstance(coords, (ICRS, SkyCoord)):
            raise TypeError(
                'coords should be ICRS or SkyCoord object'
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
            'Effective analog pointing=({}, {})'.format(
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
        altazcoords = toAltaz(
            skycoord=coords,
            time=time
        )  
        arrfac = self.array_factor(
            phase_center=phase_center,
            coords=altazcoords,
            antpos=ma_antpos(
                rot=self._rot
            )
        )
        # Antenna Gain
        antgain = self.ant_gain(
            coords=coords,
            time=time
        )
        anagain = arrfac * antgain
        log.info(
            'Anabeam (rot {}) computed for {} pixels.'.format(
                self._rot%60,
                anagain.size
            )
        )
        return anagain
# ============================================================= #


# ============================================================= #
# --------------------------- Beam ---------------------------- #
# ============================================================= #
class DBeam(Beam):
    """
    """

    def __init__(self, freq, polar, azdig, eldig, ma,
        azana=None, elana=None, squint_freq=30, beamsquint=True):
        super().__init__(
            freq=freq,
            polar=polar
        )
        self.azdig = azdig
        self.eldig = eldig
        self.ma = ma
        self.azana = azdig if azana is None else azana
        self.elana = eldig if elana is None else elana
        self.squint_freq = squint_freq
        self.beamsquint = beamsquint


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def azdig(self):
        """
        """
        return self._azdig
    @azdig.setter
    def azdig(self, a):
        if not isinstance(a, u.Quantity):
            a *= u.deg
        self._azdig = a
        log.info(
            'Digital azimuth: {}'.format(self._azdig)
        )
        return


    @property
    def eldig(self):
        """
        """
        return self._eldig
    @eldig.setter
    def eldig(self, e):
        if not isinstance(e, u.Quantity):
            e *= u.deg
        self._eldig = e
        log.info(
            'Digital elevation: {}'.format(self._eldig)
        )
        return


    @property
    def ma(self):
        return self._ma
    @ma.setter
    def ma(self, m):
        if isinstance(m, list):
            m = np.array(m)
        if np.isscalar(m):
            raise ValueError(
                'ma should at list be of length 2'
            )
        if not np.isin(m, ma_info['ma']).all():
            raise ValueError(
                'Some MA names are > {}'.format(
                    ma_info['ma'].max()
                )
            )
        self._ma = m.astype(int)
        log.info(
            'MAs {} selected for digital beam.'.format(
                self._ma
            )
        )
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def beam_values(self, coords, time):
        """
            :param coords:
            :type coords: :class:`~astropy.coordinates.ICRS` or 
                :class:`~astropy.coordinates.SkyCoord`
            :param time:
            :type time: :class:`~astropy.time.Time`
        """
        if not isinstance(coords, (ICRS, SkyCoord)):
            raise TypeError(
                'coords should be ICRS or SkyCoord object'
            )
        if not isinstance(time, Time):
            raise TypeError(
                'time should be Time object'
            )

        # Build the Mini-Array 'summed' response
        abeams = {}
        for ma in self.ma:
            rot = ma_info['rot'][ma_info['ma'] == ma][0]
            if str(rot%60) not in abeams.keys():
                ana = ABeam(
                    freq=self.freq,
                    polar=self.polar,
                    azana=self.azana,
                    elana=self.elana,
                    ma=ma
                )
                anavals = ana.beam_values(
                    coords=coords,
                    time=time
                )
                abeams[str(rot%60)] = anavals.copy()
            if not 'summa' in locals():
                summa = abeams[str(rot%60)]
            else:
                summa += abeams[str(rot%60)]
         # Array factor
        phase_center = ho_coord(
                az=self.azdig,
                alt=self.eldig,
                time=time
        )
        altazcoords = toAltaz(
            skycoord=coords,
            time=time
        )  
        arrfac = self.array_factor(
            phase_center=phase_center,
            coords=altazcoords,
            antpos=ma_pos[np.isin(ma_info['ma'], self.ma)]
        )
        # Arrayfactor * summed MA response
        digigain = arrfac * summa
        log.info(
            'Digibeam computed for {} pixels.'.format(
                digigain.size
            )
        )
        return digigain
# ============================================================= #

