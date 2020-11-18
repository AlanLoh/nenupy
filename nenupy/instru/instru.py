#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********************
    Instrument functions
    ********************

    Below are defined a set of useful NenuFAR instrumental
    functions, hereafter summarized:
    
    * :func:`~nenupy.instru.instru.analog_pointing`: NenuFAR real analog pointing conversion
    * :func:`~nenupy.instru.instru.desquint_elevation`: Correct for beamsquint
    * :func:`~nenupy.instru.instru.nenufar_ant_gain`: NenuFAR antenna gain in HEALPix
    * :func:`~nenupy.instru.instru.read_cal_table`: Read the antenna delay calibration table
    * :func:`~nenupy.instru.instru.effective_area`: NenuFAR effective area
    * :func:`~nenupy.instru.instru.sky_temperature`: Compute sky temperature at a given frequency
    * :func:`~nenupy.instru.instru.inst_temperature`: NenuFAR temperature
    * :func:`~nenupy.instru.instru.sefd`: NenuFAR SEFD
    * :func:`~nenupy.instru.instru.sensitivity`: NenuFAR sensitivity
    * :func:`~nenupy.instru.instru.resolution`: NenuFAR resolution
    * :func:`~nenupy.instru.instru.confusion_noise`: NenuFAR confusion noise
    * :func:`~nenupy.instru.instru.data_rate`: NenuFAR data rate estimation
    * :func:`~nenupy.instru.instru.freq2sb`: Conversion from frequency to sub-band index
    * :func:`~nenupy.instru.instru.sb2freq`: Conversion sub-band index to sub-band start frequency

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'ma_antpos',
    'ma_info',
    'ma_pos',
    '_HiddenPrints',
    'getMAL93',
    'nenufar_loc',
    'analog_pointing',
    'desquint_elevation',
    'nenufar_ant_gain',
    'read_cal_table',
    'effective_area',
    'sky_temperature',
    'inst_temperature',
    'sefd',
    'sensitivity',
    'resolution',
    'confusion_noise',
    'data_rate',
    'freq2sb',
    'sb2freq'
]


from astropy.io.fits import getdata
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import (
    ICRS,
    AltAz,
    EarthLocation
)
import astropy.units as u
from astropy import constants as const
import numpy as np
import os, sys
from os.path import join, dirname
from scipy.io.idl import readsav
from scipy.interpolate import interp1d

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- ma_antpos ------------------------- #
# ============================================================= #
def ma_antpos(rot):
    """ MiniArray rotation in degrees
    """
    nenufar_antpos = np.array(
        [
            -5.50000000e+00, -9.52627850e+00,  0.00000000e+00,
             0.00000000e+00, -9.52627850e+00,  0.00000000e+00,
             5.50000000e+00, -9.52627850e+00,  0.00000000e+00,
            -8.25000000e+00, -4.76313877e+00,  0.00000000e+00,
            -2.75000000e+00, -4.76313877e+00,  0.00000000e+00,
             2.75000000e+00, -4.76313877e+00,  0.00000000e+00,
             8.25000000e+00, -4.76313877e+00,  0.00000000e+00,
            -1.10000000e+01,  9.53674316e-07,  0.00000000e+00,
            -5.50000000e+00,  9.53674316e-07,  0.00000000e+00,
             0.00000000e+00,  9.53674316e-07,  0.00000000e+00,
             5.50000000e+00,  9.53674316e-07,  0.00000000e+00,
             1.10000000e+01,  9.53674316e-07,  0.00000000e+00,
            -8.25000000e+00,  4.76314068e+00,  0.00000000e+00,
            -2.75000000e+00,  4.76314068e+00,  0.00000000e+00,
             2.75000000e+00,  4.76314068e+00,  0.00000000e+00,
             8.25000000e+00,  4.76314068e+00,  0.00000000e+00,
            -5.50000000e+00,  9.52628040e+00,  0.00000000e+00,
             0.00000000e+00,  9.52628040e+00,  0.00000000e+00,
             5.50000000e+00,  9.52628040e+00,  0.00000000e+00
        ]
    ).reshape(19, 3)
    rot = np.radians(rot - 90)
    rotation = np.array(
        [
            [ np.cos(rot), np.sin(rot), 0],
            [-np.sin(rot), np.cos(rot), 0],
            [ 0,           0,           1]
        ]
    )
    return np.dot(nenufar_antpos, rotation)
# ============================================================= #


# ============================================================= #
# -------------------------- ma_info -------------------------- #
# ============================================================= #
ma_info = np.array(
    [
        (0 , 0  , np.array([6.39113316e+05, 6.69766347e+06, 1.81735000e+02]), 440.5 , 29.5),
        (1 , 30 , np.array([6.39094578e+05, 6.69764471e+06, 1.81750000e+02]), 364   , 30  ),
        (2 , 300, np.array([6.39069472e+05, 6.69763443e+06, 1.81761000e+02]), 150   , 31  ),
        (3 , 200, np.array([6.39038120e+05, 6.69761975e+06, 1.81757000e+02]), 145.5 , 31  ),
        (4 , 20 , np.array([6.39020122e+05, 6.69759892e+06, 1.81762000e+02]), 464.5 , 28.5),
        (5 , 180, np.array([6.39062298e+05, 6.69765911e+06, 1.81671000e+02]), 384.5 , 30  ),
        (6 , 180, np.array([6.39039218e+05, 6.69764638e+06, 1.81718000e+02]), 276.5 , 31  ),
        (7 , 230, np.array([6.38985155e+05, 6.69762795e+06, 1.81620000e+02]), 471   , 28.5),
        (8 , 150, np.array([6.39002711e+05, 6.69764788e+06, 1.81677000e+02]), 411.5 , 29.5),
        (9 , 240, np.array([6.39006567e+05, 6.69767471e+06, 1.81737000e+02]), 428.5 , 29.5),
        (10, 290, np.array([6.39033717e+05, 6.69769736e+06, 1.81762000e+02]), 496   , 28.5),
        (11, 310, np.array([6.39040955e+05, 6.69772821e+06, 1.81813000e+02]), 663   , 27.5),
        (12, 250, np.array([6.39061482e+05, 6.69770986e+06, 1.81727000e+02]), 659.5 , 25  ),
        (13, 40 , np.array([6.39081586e+05, 6.69774800e+06, 1.81997000e+02]), 958   , 24.5),
        (14, 330, np.array([6.39099636e+05, 6.69778020e+06, 1.82152000e+02]), 1032  , 24.5),
        (15, 280, np.array([6.39098493e+05, 6.69772636e+06, 1.81912000e+02]), 793   , 26.5),
        (16, 60 , np.array([6.39128375e+05, 6.69774506e+06, 1.82005000e+02]), 1054.5, 24  ),
        (17, 110, np.array([6.39153064e+05, 6.69776096e+06, 1.82062000e+02]), 1233  , 23  ),
        (18, 10 , np.array([6.39201475e+05, 6.69776774e+06, 1.82083000e+02]), 1355  , 22  ),
        (19, 210, np.array([6.39146673e+05, 6.69779044e+06, 1.82199000e+02]), 1249.5, 22.5),
        (20, 320, np.array([6.39191912e+05, 6.69780784e+06, 1.82057000e+02]), 1352.5, 21.5),
        (21, 260, np.array([6.39158373e+05, 6.69784497e+06, 1.82201000e+02]), 1461  , 21.5),
        (22, 250, np.array([6.39007359e+05, 6.69773473e+06, 1.81967000e+02]), 662   , 27.5),
        (23, 170, np.array([6.38994637e+05, 6.69778006e+06, 1.82016000e+02]), 969.5 , 25  ),
        (24, 180, np.array([6.38974900e+05, 6.69779771e+06, 1.82057000e+02]), 1082.5, 24.5),
        (25, 50 , np.array([6.39039664e+05, 6.69779603e+06, 1.82092000e+02]), 1052  , 23.5),
        (26, 300, np.array([6.39051439e+05, 6.69782833e+06, 1.82133000e+02]), 1112  , 23.5),
        (27, 210, np.array([6.39037314e+05, 6.69786239e+06, 1.82535000e+02]), 1387  , 21  ),
        (28, 320, np.array([6.39106516e+05, 6.69788052e+06, 1.82416000e+02]), 1527.5, 20.5),
        (29, 330, np.array([6.39085345e+05, 6.69782694e+06, 1.82164000e+02]), 1226  , 22  ),
        (30, 60 , np.array([6.39124407e+05, 6.69781362e+06, 1.82277000e+02]), 1530.5, 20.5),
        (31, 20 , np.array([6.39118449e+05, 6.69784678e+06, 1.82334000e+02]), 1462  , 21  ),
        (32, 290, np.array([6.38980493e+05, 6.69766162e+06, 1.81773000e+02]), 573   , 28  ),
        (33, 240, np.array([6.38955067e+05, 6.69765390e+06, 1.81787000e+02]), 616.5 , 27.5),
        (34, 230, np.array([6.38917110e+05, 6.69764246e+06, 1.81723000e+02]), 770.5 , 26  ),
        (35, 340, np.array([6.38901511e+05, 6.69766562e+06, 1.81722000e+02]), 1024  , 24.5),
        (36, 170, np.array([6.38842551e+05, 6.69768422e+06, 1.81915000e+02]), 1182.5, 23  ),
        (37, 350, np.array([6.38881893e+05, 6.69770613e+06, 1.81971000e+02]), 1231  , 23  ),
        (38, 260, np.array([6.38828310e+05, 6.69773150e+06, 1.82165000e+02]), 1365  , 21.5),
        (39, 160, np.array([6.38798998e+05, 6.69767696e+06, 1.82180000e+02]), 1472  , 21  ),
        (40, 220, np.array([6.38828026e+05, 6.69764418e+06, 1.82034000e+02]), 1115.5, 24  ),
        (41, 120, np.array([6.38994603e+05, 6.69769938e+06, 1.81819000e+02]), 661.5 , 26.5),
        (42, 140, np.array([6.38973994e+05, 6.69773127e+06, 1.81940000e+02]), 802.5 , 25.5),
        (43, 130, np.array([6.38963888e+05, 6.69775682e+06, 1.82315000e+02]), 932   , 24.5),
        (44, 110, np.array([6.38907143e+05, 6.69775707e+06, 1.82406000e+02]), 1187  , 23  ),
        (45, 150, np.array([6.38934310e+05, 6.69776276e+06, 1.82306000e+02]), 1030  , 23.5),
        (46, 300, np.array([6.38947300e+05, 6.69779542e+06, 1.82257000e+02]), 1096  , 23.5),
        (47, 190, np.array([6.38957218e+05, 6.69768346e+06, 1.81905000e+02]), 721.5 , 26.5),
        (48, 100, np.array([6.38932724e+05, 6.69769106e+06, 1.81967000e+02]), 898   , 26  ),
        (49, 340, np.array([6.38924357e+05, 6.69772908e+06, 1.82312000e+02]), 1029  , 23.5),
        (50, 160, np.array([6.38865831e+05, 6.69778298e+06, 1.82415000e+02]), 1339.5, 23  ),
        (51, 240, np.array([6.38881847e+05, 6.69776009e+06, 1.82376000e+02]), 1138.5, 23.5),
        (52, 270, np.array([6.39169242e+05, 6.69769247e+06, 1.82180000e+02]), 1014,   23.5),
        (53, 340, np.array([6.39199806e+05, 6.69768286e+06, 1.82226000e+02]), 1014,   23.5),
        (54, 310, np.array([6.39223589e+05, 6.69770809e+06, 1.82158000e+02]), 1106,   23.5),
        (55, 90,  np.array([6.39216821e+05, 6.69765166e+06, 1.82178000e+02]), 1106,   23.5)
    ],
    np.dtype(
        [
            ('ma', int),
            ('rot', float),
            ('pos', np.ndarray),
            ('delay', float),
            ('att', float),
        ]
    )
)
# ============================================================= #


# ============================================================= #
# -------------------------- ma_pos --------------------------- #
# ============================================================= #
rot = np.radians(-90)
rotation = np.array(
    [
        [ np.cos(rot), np.sin(rot), 0],
        [-np.sin(rot), np.cos(rot), 0],
        [ 0,           0,           1]
    ]
)
ma_pos = np.dot(
    np.array([aa.tolist() for aa in ma_info['pos']]),
    rotation
)
# ============================================================= #


# ============================================================= #
# ----------------------- HiddenPrints ------------------------ #
# ============================================================= #
class _HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
# ============================================================= #


# ============================================================= #
# ------------------------ nenufar_loc ------------------------ #
# ============================================================= #
nenufar_loc = EarthLocation(
    lat=47.376511 * u.deg,
    lon=2.1924002 * u.deg,
    height=135.834 * u.m
)
# ============================================================= #


# ============================================================= #
# -------------------------- getMAL93 ------------------------- #
# ============================================================= #
def getMAL93(m):
    """
    """
    ma_pos = np.array([a.tolist() for a in ma_info['pos']])
    available_mas = np.arange(ma_pos.shape[0])
    antpos = ma_pos[np.isin(available_mas, m)]
    return antpos
# ============================================================= #


# ============================================================= #
# ---------------------- analog_pointing ---------------------- #
# ============================================================= #
def analog_pointing(azimuth, elevation):
    """ NenuFAR Mini-Array pointing is performed using analogical
        delays between antennas. Therefore, the pointing directions
        follow a discrete distribution. This function allows for
        finding the actual pointing order given an aziumuth and
        an elevation as requests.

        .. image:: ./_images/analog_pointing.png
            :width: 800

        :param azimuth:
            Requested azimuth (in degrees if `float`).
        :type azimuth: `float` or :class:`astropy.units.Quantity`
        :param elevation:
            Requested elevation (in degrees if `float`).
        :type elevation: `float` or :class:`astropy.units.Quantity`

        :returns: (Azimuth, Elevation)
        :rtype: :class:`astropy.units.Quantity`
        
        :Example:
            >>> from nenupysim.instru import analog_pointing
            >>> import astropy.units as u
            >>> analog_pointing(180*u.deg, 45*u.deg)
            (<Quantity 180. deg>, <Quantity 45.17045975 deg>)
    """
    if not isinstance(azimuth, Quantity):
        azimuth *= u.deg
    if not isinstance(elevation, Quantity):
        elevation *= u.deg
    azimuth = azimuth.to(u.deg)
    elevation = elevation.to(u.deg)
    file = join(
        dirname(__file__),
        'NenuFAR_thph.fits'
    )
    thph = getdata(file) # azimuth, zenith angle
    if np.isscalar(azimuth) and np.isscalar(elevation):
        phi_idx = int(azimuth.value/0.05 - 0.5)
        theta_idx = int((90. - elevation.value)/0.05 - 0.5)
    else:
        phi_idx = (azimuth.value/0.05 - 0.5).astype(int)
        theta_idx = ((90. - elevation.value)/0.05 - 0.5).astype(int)
    t, p = thph[:, theta_idx, phi_idx]
    azimuth = p * u.deg
    elevation = (90. - t) * u.deg
    return azimuth, elevation
# ============================================================= #


# ============================================================= #
# -------------------- desquint_elevation --------------------- #
# ============================================================= #
def desquint_elevation(elevation, opt_freq=30):
    """ Radio phased array are affected by beam squint. Combination
        of antenna response (maximal at zenith) and array factor
        of antenna distribution can shift maximal sensitivity
        towards greater elevations.
        This function allows for correcting this effect by shifting
        pointing elevation a little bit lower.
        The correction is limited to elevation greater than 20 
        deg, otherwise, analog pointing can shift drastically if
        a low elevation is required
        (see :func:`~nenupy.instru.instru.analog_pointing`).

        .. image:: ./_images/desquint.png
            :width: 800

        :param elevation:
            Requested elevation (in degrees if `float`).
        :type elevation: `float` or :class:`astropy.units.Quantity`
        :param opt_freq:
            Beam squint optimization frequency (in MHz if `float`)
        :type opt_freq: `float` or :class:`astropy.units.Quantity`

        :returns: Beamsquint-corrected elevation to point
        :rtype: :class:`~astropy.units.Quantity`

        :Example:
            >>> from nenupy.instru import desquint_elevation
            >>> desquint_elevation(
                    elevation=45,
                    opt_freq=80
                )
            44.359063°
    """
    if not isinstance(elevation, Quantity):
        elevation *= u.deg
    if not isinstance(opt_freq, Quantity):
        opt_freq *= u.MHz
    elevation = elevation.to(u.deg)
    opt_freq = opt_freq.to(u.MHz)
    squint = readsav(
        join(
            dirname(__file__),
            'squint_table.sav'
        )
    )
    freq_idx = np.argmin(
        np.abs(squint['freq'] - opt_freq.value)
    )
    elevation = interp1d(
        squint['elev_desiree'][freq_idx, :],
        squint['elev_a_pointer']
    )(elevation.value)
    # Squint is limited at 20 deg elevation, otherwise the
    # pointing can vary drasticaly as the available pointing
    # positions become sparse at low elevation.
    if np.isscalar(elevation):
        if elevation < 20.:
            elevation = 20
    else:
        elevation[elevation < 20] = 20
    return elevation * u.deg
# ============================================================= #


# ============================================================= #
# --------------------- nenufar_ant_gain ---------------------- #
# ============================================================= #
def nenufar_ant_gain(freq, polar='NW', nside=64, time=None, normalize=True):
    """ Get NenuFAR elementary antenna gain with respect to the
        frequency ``freq``, the polarization ``polar`` and
        convert it to HEALPix representation with a given
        ``nside`` that defines pixel number. The map is initially
        in horizontal coordinates, conversion to equatorial
        coordinates requires to set the ``time`` at which the 
        computation should occur.

        :param freq:
            Frequency of the returned antenna gain. Its value
            must be comprised between 10 and 80 MHz, because
            available antenna models are constrained to these
            frequencies. If the frequency value exceeds 80 MHz,
            an extrapolation is made using polynomial fits.
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param polar:
            Antenna polarization to take into account (either
            ``'NW'`` or ``'NE'``).
        :type polar: `str`
        :param nside:
            HEALPix nside parameter, must be a power of 2, less
            than 2**30 (see :func:`~healpy.pixelfunc.nside2resol`
            for corresponding angular pixel resolution).
        :type nside: `int`
        :param time:
            Time at which the computation occurs, in order to
            have the right antenna gain pattern on the sky above
            NenuFAR. If ``None`` the map is not rotated to 
            equatorial coordinates.
        :type time: `str` or :class:`~astropy.time.Time`
        :param normalize:
            Returns the normalized gain or not.
        :type normalize: `bool`

        :returns:
            Sky map in HEALPix representation of normalized
            antenna gain.
        :rtype: :class:`~numpy.ndarray`

        :Example:
            Get the antenna gain and plot it (using
            :class:`~nenupy.astro.hpxsky.HpxSky`):

            >>> from nenupy.instru import nenufar_ant_gain
            >>> from nenupy.astro import HpxSky
            >>> ant_sky = HpxSky(resolution=0.5)
            >>> ant_sky.skymap = nenufar_ant_gain(
                    freq=60,
                    polar='NW',
                    nside=ant_sky.nside,
                    time='2020-04-01 12:00:00'
                )
            >>> ant_sky.plot()

            .. image:: ./_images/antgain.png
                :width: 800

        .. seealso::
            :class:`~nenupy.beam.hpxbeam.HpxABeam`
    """
    from healpy import (
        read_map,
        ud_grade,
        nside2npix,
        pix2ang,
        Rotator
    )

    # Parameter checks
    if not isinstance(freq, Quantity):
        freq *= u.MHz
    freq = freq.to(u.MHz).value
    if polar.lower() not in ['nw', 'ne']:
        raise ValueError(
            'Polar should be either NW or NE'
        )
    polar = polar.upper()
    # Correspondance between polar/freq and field index in FITS
    gain_freqs = np.arange(10, 90, 10, dtype=int)
    count = 0
    cols = {}
    for p in ['NE', 'NW']:#['NW', 'NE']:
        for f in gain_freqs:
            cols['{}_{}'.format(p, f)] = count
            count += 1
    antgain_file = join(
        dirname(__file__),
        'NenuFAR_Ant_Hpx.fits',
    )
    if freq < 10:
        raise ValueError(
            'No antenna model < 10 MHz.'
        )
    elif freq > 80:
        log.warning(
            'NenuFAR antenna response is extrapolated > 80 MHz.'
        )
        # Will fit a polynomial along the high end of frequencies
        freq_to_fit = np.arange(40, 90, 10, dtype=int)
        gains = np.zeros((freq_to_fit.size, nside2npix(64)))
        # Construct the gain map (freqs, npix)
        for i, f in enumerate(freq_to_fit):
            gains[i, :] = read_map(
                filename=antgain_file,
                hdu=1,
                field=cols['{}_{}'.format(polar, f)],
                verbose=False,
                memmap=True,
                dtype=float
            )
        # Get the polynomial coefficients
        coeffs = np.polyfit(freq_to_fit, gains, 3)
        def poly(x, coeffs):
            """ Retrieve the polynomial from coefficients
            """
            na = np.newaxis
            order = coeffs.shape[0]
            poly = np.zeros((x.size, coeffs.shape[1]))
            for deg in range(order):
                poly += (x**deg)[:, na] * coeffs[order-deg-1, :][na, :]
            return poly
        gain = poly(np.array([freq]), coeffs).ravel()
    else:
        # Get Low and High ant gain bounding freq
        f_low = gain_freqs[gain_freqs <= freq].max()
        f_high = gain_freqs[gain_freqs >= freq].min()
        gain_low = read_map(
            filename=antgain_file,
            hdu=1,
            field=cols['{}_{}'.format(polar, f_low)],
            verbose=False,
            memmap=True,
            dtype=float
        )
        gain_high = read_map(
            filename=antgain_file,
            hdu=1,
            field=cols['{}_{}'.format(polar, f_high)],
            verbose=False,
            memmap=True,
            dtype=float
        )
        # Make interpolation
        if f_low != f_high:
            gain = gain_low * (f_high - freq)/10. +\
                gain_high * (freq - f_low)/10.
        else:
            gain = gain_low
    # Rotate the map to equatorial coordinates
    if time is not None:
        if not isinstance(time, Time):
            time = Time(time)
        altaz_origin = AltAz(
            0*u.deg,
            0*u.deg,
            location=nenufar_loc,
            obstime=time
        )
        radec_origin = altaz_origin.transform_to(ICRS)
        rot = Rotator(
            deg=True,
            rot=[radec_origin.ra.deg, radec_origin.dec.deg], 
            coord=['C', 'C'],
            inv=True
        )
        with _HiddenPrints():
            gain = rot.rotate_map_alms(gain)
    # Convert HEALPix map to required nside
    gain = ud_grade(gain, nside_out=nside)
    return gain / gain.max() if normalize else gain
# ============================================================= #


# ============================================================= #
# ---------------------- read_cal_table ----------------------- #
# ============================================================= #
def read_cal_table(calfile=None):
    """ Reads NenuFAR antenna delays calibration file.

        :param calfile: 
            Name of the calibration file to read. If ``None`` or
            ``'default'`` the standard calibration file is read.
        :type calfile: `str`

        :returns: 
            Antenna delays shaped as 
            (frequency, mini-arrays, polarizations).
        :rtype: :class:`~numpy.ndarray`
    """
    if (calfile is None) or (calfile.lower() == 'default'):
        calfile = join(
            dirname(__file__),
            'cal_pz_2_multi_2019-02-23.dat',
        )
    with open(calfile, 'rb') as f:
        log.info(
            'Loading calibration table {}'.format(
                calfile
            )
        )
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.startswith(b'HeaderStop'):
                break
    hd_size = sum([len(s) for s in header])
    dtype = np.dtype(
        [
            ('data', 'float64', (512, 96, 2, 2))
        ]
    )
    tmp = np.memmap(
        filename=calfile,
        dtype='int8',
        mode='r',
        offset=hd_size
    )
    decoded = tmp.view(dtype)[0]['data']
    data = decoded[..., 0] + 1.j*decoded[..., 1]
    return data
# ============================================================= #


# ============================================================= #
# ---------------------- effective_area ----------------------- #
# ============================================================= #
def effective_area(freq=50, antennas=None, miniarrays=None):
    """ Computes the NenuFAR array effective area.

        :param freq: 
            Frequency at which computing the effective area.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param antennas:
            Mini-Array antenna indices to take into account.
            Default is ``None`` (all 19 antennas).
        :type antennas: `int`, `list` or :class:`~numpy.ndarray`
        :param miniarrays:
            Mini-Array indices to take into account.
            Default is ``None`` (all available MAs).
        :type miniarrays: `int`, `list` or :class:`~numpy.ndarray`

        :returns: Effective area in squared meters
        :rtype: :class:`~astropy.units.Quantity`

        :Example:
            * Effective area of a single antenna from ``MA 0``:

            >>> from nenupy.instru import effective_area
            >>> effective_area(
                    freq=50,
                    antennas=10,
                    miniarrays=0
                )
            11.982422 m2

            * Effective area of ``MA 0``:
            
            >>> from nenupy.instru import effective_area
            >>> effective_area(
                    freq=50,
                    antennas=np.arange(19),
                    miniarrays=0
                )
            227.60482 m2

            * Effective area of NenuFAR with 56 MAs:
            
            >>> from nenupy.instru import effective_area
            >>> effective_area(
                    freq=50,
                    antennas=np.arange(19),
                    miniarrays=np.arange(56)
                )
            12745.87 m2

        .. seealso::
            NenuFAR Mini-Arrays `characteristics
            <https://nenufar.obs-nancay.fr/en/astronomer/#mini-arrays>`_.

    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    if antennas is None:
        antennas = np.arange(19)
    else:
        if np.isscalar(antennas):
            antennas = np.array([antennas])
        elif isinstance(antennas, list):
            antennas = np.array(antennas)
        if max(antennas) > 18:
            raise ValueError(
                'Only 19 antennas in NenuFAR MA'
            )
    if miniarrays is None:
        miniarrays = np.arange(ma_info['ma'].size)
    else:
        if np.isscalar(miniarrays):
            miniarrays = np.array([miniarrays])
        elif isinstance(miniarrays, list):
            miniarrays = np.array(miniarrays)
        if max(miniarrays) > ma_info.size:
            raise ValueError(
                'Only {} Mini-Arrays'.format(ma_info.size)
            )

    # Antenna Effective Area
    k = 3
    wavelength = const.c.to(u.m/u.s) / freq.to(u.Hz)
    ant_ea = (wavelength**2 / k).to(u.m**2)

    # Mini-Array Effective Area
    n = 1000 # grid resolution
    grid = np.zeros(
        (n, n),
        dtype=np.int32
    )
    ant_ea_radius = np.sqrt(ant_ea/np.pi).to(u.m).value
    antpos = ma_antpos(rot=0)[antennas]
    x_grid = np.linspace(
        antpos[:, 0].min() - ant_ea_radius,
        antpos[:, 0].max() + ant_ea_radius,
        n
    )
    dx = x_grid[1] - x_grid[0]
    y_grid = np.linspace(
        antpos[:, 1].min() - ant_ea_radius,
        antpos[:, 1].max() + ant_ea_radius,
        n
    )
    dy = y_grid[1] - y_grid[0]
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    for xi, yi, zi in antpos:
        dist = np.sqrt((xx_grid - xi)**2. + (yy_grid - yi)**2.)
        grid[dist <= ant_ea_radius] += 1
    grid[grid != 0] = 1
    ma_ea = (grid * dx * dy).sum() * (u.m**2)

    # NenuFAR Effective Area
    return ma_ea * miniarrays.size
# ============================================================= #


# ============================================================= #
# ---------------------- sky_temperature ---------------------- #
# ============================================================= #
def sky_temperature(freq=50):
    r""" Sky temperature at a given frequency ``freq`` (strongly
        dominated by Galactic emission).

        .. math::
            T_{\rm sky} = T_0 \lambda^{2.55}

        with :math:`T_0 = 60 \pm 20\,\rm{K}` for Galactic
        latitudes between 10 and 90 degrees.

        :param freq:
            Frequency at which computing the esky temperature.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`

        :returns: Sky temperature in Kelvins
        :rtype: :class:`~astropy.units.Quantity`

        .. seealso::
            `LOFAR website <http://old.astron.nl/radio-observatory/astronomers/lofar-imaging-capabilities-sensitivity/sensitivity-lofar-array/sensiti>`_, 
            Haslam et al. (1982) and Mozdzen et al. (2017, 2019)
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    wavelength = (const.c/freq).to(u.m).value
    t0 = 60. * u.K
    tsky = t0 * wavelength**2.55
    return tsky
# ============================================================= #


# ============================================================= #
# --------------------- inst_temperature ---------------------- #
# ============================================================= #
def inst_temperature(freq=50):
    """ Instrument temperature at a given frequency ``freq``.
        This depends on the Low Noise Amplifier characteristics.

        :param freq:
            Frequency at which computing the esky temperature.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`

        :returns: Instrument temperature in Kelvins
        :rtype: :class:`~astropy.units.Quantity`

        .. seealso::
            :func:`~nenupy.instru.instru.sky_temperature`
    """
    if isinstance(freq, u.Quantity):
        freq = freq.to(u.MHz).value
    lna_sky = np.array([
        5.0965,2.3284,1.0268,0.4399,0.2113,0.1190,0.0822,0.0686,
        0.0656,0.0683,0.0728,0.0770,0.0795,0.0799,0.0783,0.0751,
        0.0710,0.0667,0.0629,0.0610,0.0614,0.0630,0.0651,0.0672,
        0.0694,0.0714,0.0728,0.0739,0.0751,0.0769,0.0797,0.0837,
        0.0889,0.0952,0.1027,0.1114,0.1212,0.1318,0.1434,0.1562,
        0.1700,0.1841,0.1971,0.2072,0.2135,0.2168,0.2175,0.2159,
        0.2121,0.2070,0.2022,0.1985,0.1974,0.2001,0.2063,0.2148,
        0.2246,0.2348,0.2462,0.2600,0.2783,0.3040,0.3390,0.3846,
        0.4425,0.5167,0.6183,0.7689,1.0086,1.4042,2.0732
    ])
    freqs = (np.arange(71) + 15) # MHz
    tsky = sky_temperature(freq=freq)
    tinst = tsky * lna_sky[ np.abs(freqs - freq).argmin() ]
    return tinst
# ============================================================= #


# ============================================================= #
# --------------------------- sefd ---------------------------- #
# ============================================================= #
def sefd(freq=50, antennas=None, miniarrays=None):
    r""" Computes the System Equivalent Flux Density (SEFD or
        system sensitivity).
        
        .. math::
            S_{\rm sys} = \frac{2 \eta k_{\rm B}}{ A_{\rm eff}} T_{\rm sys}
        
        with :math:`T_{\rm sys} = T_{\rm sky} + T_{\rm inst}`,
        the efficiency :math:`\eta = 1` and :math:`k_{\rm B}` the
        Boltzmann constant.

        :param freq: 
            Frequency at which computing the SEFD.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param antennas:
            Mini-Array antenna indices to take into account.
            Default is ``None`` (all 19 antennas).
        :type antennas: `int`, `list` or :class:`~numpy.ndarray`
        :param miniarrays:
            Mini-Array indices to take into account.
            Default is ``None`` (all available MAs).
        :type miniarrays: `int`, `list` or :class:`~numpy.ndarray`

        :returns: SEFD in Janskys
        :rtype: :class:`~astropy.units.Quantity`
    
        .. seealso::
            `LOFAR website <http://old.astron.nl/radio-observatory/astronomers/lofar-imaging-capabilities-sensitivity/sensitivity-lofar-array/sensiti>`_, 
            :func:`~nenupy.instru.instru.sky_temperature` for :math:`T_{\rm sky}`,
            :func:`~nenupy.instru.instru.inst_temperature` for :math:`T_{\rm inst}`,
            :func:`~nenupy.instru.instru.effective_area` for :math:`A_{\rm eff}`.
    """
    efficiency = 1.
    aeff = effective_area(
        freq=freq,
        antennas=antennas,
        miniarrays=miniarrays)
    tsky = sky_temperature(freq=freq)
    tinst = inst_temperature(freq=freq)
    tsys = tsky + tinst
    sefd = 2 * efficiency * const.k_B * tsys / aeff
    return sefd.to(u.Jy)
# ============================================================= #


# ============================================================= #
# ------------------------ sensitivity ------------------------ #
# ============================================================= #
def sensitivity(mode='imaging', freq=50, antennas=None, miniarrays=None, dt=1, df=1):
    r""" Returns the sensitivity of NenuFAR.

        For the imaging mode:

        .. math::
            \Delta S_{\rm im} = \frac{S_{\rm sys}}{
                \sqrt{N(N-1) 2 \delta \nu \delta t}
            }

        For the beamforming mode:

        .. math::
            \Delta S_{\rm bf} = \frac{S_{\rm sys}}{
                \sqrt{\delta \nu \delta t}
            }

        where :math:`S_{\rm sys}` is the System Equivalent Flux
        Density, :math:`N` is the number of Mini-Arrays involved,
        :math:`\delta t` is the integration time and
        :math:`\delta \nu` is the bandwidth.

        :param mode:
            Observation mode (either ``'imaging'`` or
            ``'beamforming'``)
        :type mode: `str`
        :param freq: 
            Frequency at which computing the sensitivity.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param antennas:
            Mini-Array antenna indices to take into account.
            Default is ``None`` (all 19 antennas).
        :type antennas: `int`, `list` or :class:`~numpy.ndarray`
        :param miniarrays:
            Mini-Array indices to take into account.
            Default is ``None`` (all available MAs).
        :type miniarrays: `int`, `list` or :class:`~numpy.ndarray`
        :param dt:
            Integration time (in sec if `float`)
        :type dt: `float` or class:`~astropy.units.Quantity`
        :param df:
            Bandwidth (in MHz if `float`)
        :type df: `float` or :class:`~astropy.units.Quantity`

        :returns: Sensitivity in Janskys
        :rtype: :class:`~astropy.units.Quantity`

        .. seealso::
            :func:`~nenupy.instru.instru.sefd` for :math:`S_{\rm sys}`.
    """
    if not isinstance(dt, u.Quantity):
        dt *= u.s
    if not isinstance(df, u.Quantity):
        df *= u.MHz
    ssys = sefd(
        freq=freq,
        antennas=antennas,
        miniarrays=miniarrays
    )
    if miniarrays is None:
        miniarrays = np.arange(ma_info['ma'].size)
    else:
        if np.isscalar(miniarrays):
            miniarrays = np.array([miniarrays])
        elif isinstance(miniarrays, list):
            miniarrays = np.array(miniarrays)
        if max(miniarrays) > ma_info.size:
            raise ValueError(
                'Only {} Mini-Arrays'.format(ma_info.size)
            )
    nant = miniarrays.size
    if mode.lower() == 'imaging':
        sensitivity = ssys / np.sqrt(nant*(nant-1) * 2 * dt * df)
    elif mode.lower() == 'beamforming':
        sensitivity = ssys / np.sqrt(dt * df)
    else:
        raise ValueError(
            'Observation mode not understood'
        )
    return sensitivity.to(u.Jy)
# ============================================================= #


# ============================================================= #
# ------------------------ resolution ------------------------- #
# ============================================================= #
def resolution(freq=50, miniarrays=None):
    """ Returns the resolution for a given NenuFAR configuration.

        :param freq: 
            Frequency at which computing the resolution.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param miniarrays:
            Mini-Array indices to take into account.
            Default is ``None`` (all available MAs).
        :type miniarrays: `int`, `list` or :class:`~numpy.ndarray`

        :returns: Resolution in degrees
        :rtype: :class:`~astropy.units.Quantity`

        :Example:
            Resolution of the full NenuFAR array:

            >>> from nenupy.instru import resolution
            >>> resolution(freq=50, miniarrays=None)
            0.89189836°

            Resolution of a single Mini-Array:

            >>> from nenupy.instru import resolution
            >>> resolution(freq=50, miniarrays=0)
            13.741474°

        .. seealso::
            `NenuFAR characteristics <https://nenufar.obs-nancay.fr/en/astronomer/#mini-arrays>`_.

        .. warning::
            A 25m diameter is used for a Mini-Array maximum
            baseline.
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    wavelength = (const.c / freq).to(u.m)

    if miniarrays is None:
        miniarrays = np.arange(ma_info['ma'].size)
    
    if np.isscalar(miniarrays):
        # size = np.sqrt(
        #     np.sum(
        #         (ma_antpos(0) - np.mean(ma_antpos(0), axis=0))**2,
        #         axis=-1
        #     )
        # ).max()*2* u.m
        size = 25 * u.m
    else:
        if isinstance(miniarrays, list):
            miniarrays = np.array(miniarrays)
        if max(miniarrays) > ma_info.size:
            raise ValueError(
                'Only {} Mini-Arrays'.format(ma_info.size)
            )
        positions = ma_pos[miniarrays]
        size = np.sqrt(
            np.sum(
                (positions - np.mean(positions, axis=0))**2,
                axis=-1
            )
        ).max()*2 * u.m / 1.2
    psf = wavelength / size
    return (psf.value * u.rad).to(u.deg)
# ============================================================= #


# ============================================================= #
# ---------------------- confusion_noise ---------------------- #
# ============================================================= #
def confusion_noise(freq=50, miniarrays=None):
    r""" Confusion rms noise :math:`\sigma_{\rm c}` (parameter
        used for specifying the width of the confusion
        distribution) computed as:

        .. math::
            \left( \frac{\sigma_{\rm c}}{\rm{mJy}\, \rm{beam}^{-1}} \right) \simeq
            0.2 \left( \frac{\nu}{\rm GHz} \right)^{-0.7} 
            \left( \frac{\theta}{\rm arcmin} \right)^{2}
        
        where :math:`\nu` is the frequency and :math:`\theta` is
        the radiotelescope FWHM.
        
        Individual sources fainter than about 
        :math:`5\sigma_{\rm c}` cannot be detected reliably.

        :param freq:
            Frequency at which computing the confusion noise.
            In MHz if no unit is provided. Default is ``50 MHz``.
        :type freq: `float` or :class:`~astropy.units.Quantity`
        :param miniarrays:
            Mini-Array indices to take into account.
            Default is ``None`` (all available MAs).
        :type miniarrays: `int`, `list` or :class:`~numpy.ndarray`

        :returns: Confusion rms noise in mJy/beam
        :rtype: :class:`~astropy.units.Quantity`

        :Example:
            >>> from nenupy.instru import confusion_noise
            >>> confusion_noise(
                    freq=50,
                    miniarrays=None
                )
            4663.202 mJy

        .. see also::
            `NRAO lecture <https://www.cv.nrao.edu/course/astr534/Radiometers.html>`_ (eq. 3E6),
            `Takeuchi and Ishii, 2004 <https://ui.adsabs.harvard.edu/abs/2004ApJ...604...40T/abstract>`_.
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    resol = resolution(
        freq=freq,
        miniarrays=miniarrays
    )
    norm_freq = freq.to(u.GHz).value
    norm_res = resol.to(u.arcmin).value
    conf = 0.2 * norm_freq**(-0.7) * norm_res**2

    return conf * u.mJy # mJy/beam
# ============================================================= #


# ============================================================= #
# ------------------------- data_rate ------------------------- #
# ============================================================= #
def data_rate(mode='imaging', mas=96, dt=1, nchan=64, bandwidth=75, nb=1):
    r""" Estimates the NenuFAR data rate product. To get the total
        observation size, a simple multiplication with a
        :class:`~astropy.units.Quantity` time instance,
        corresponding to total exposure time, is needed. To convert
        bytes in binary base (i.e. 1 kB = 1024 B), use `astropy`
        unit conversions (with prefixes ``Ki``, ``Mi``, ``Gi`` or
        ``Ti`` for KiloBytes MegaBytes, GigaBytes and TeraBytes,
        see `astropy unit prefixes <https://docs.astropy.org/en/stable/units/standard_units.html#prefixes>`_).

        Data rates (in bytes/s) are computed as follows:

        * Imaging mode (*NICKEL* correlator): :math:`r_{\rm im} = n_{\rm correlations} d_{\rm complex\, 64\, bits} n_{\rm channels} n_{\rm baselines} n_{\rm subbands} / \delta t`;
        * Beamforming mode (*UnDySPuTeD* backend): :math:`r_{\rm bf} = n_{\rm correlations} d_{\rm float\, 32\, bits} n_{\rm channels} n_{\rm subbands} / \delta t`;
        * Waveform mode (*LaNewBa* backend): :math:`r_{\rm wf} = 195312.5 \times n_{\rm raw} n_{\rm b} n_{\rm subbands}`;
        * Transient Buffer mode (*LaNewBa* backend): :math:`r_{\rm tbb} = 195312.5 \times 1024 \times n_{\rm raw} n_{\rm b} n_{\rm mas}`;
        
        where :math:`n_{\rm correlations} = 4` (XX, XY, YX, YY),
        :math:`n_{\rm baselines} = n_{\rm mas}*(n_{\rm mas}-1)/2 + n_{\rm mas}`,
        :math:`n_{\rm subbands} = \Delta \nu / 195.3125\, \rm{kHz}`,
        :math:`n_{\rm raw} = 4` (Re(X), Im(X), Re(Y), Im(Y)),
        :math:`d_{\rm complex\, 64\, bits}` and :math:`d_{\rm float\, 32\, bits}`
        are data sizes in bytes.
        
        :param mode:
            Observation mode (either ``'imaging'`` or
            ``'beamforming'`` or ``'waveform'`` or ``'tbb'``).
        :type mode: `str`
        :param mas: Number of Mini-Arrays to take into account
            :math:`n_{\rm mas}`. Default is ``96``.
        :type mas: `int`
        :param dt: Observation time step :math:`\delta t` (in
            seconds if no unit is provided). Default is ``1 sec``.
        :type dt: `float` or :class:`~astropy.units.Quantity`
        :param nchan: Number of channels per subband :math:`n_{\rm channels}`.
            Each subband is 195.3125 kHz. Default is ``64``.
        :type nchan: `int`
        :param bandwidth: Observation bandwidth :math:`\Delta \nu`
            (in MHz if no unit is provided). Default is ``75 MHz``.
        :type bandwidth: `float` or :class:`~astropy.units.Quantity`
        :param nb: Number of bytes of raw data samples :math:`n_{\rm b}`
            (``1 = 8 bits``, ``2 = 16 bits``).
        :type nb: `int`

        :returns: Data rate in bytes/s.
        :rtype: :class:`~astropy.units.Quantity`

        :example:
            Imaging data rate and total size for a 1h exposure, converted in TB:

            >>> from nenupy.instru import data_rate
            >>> import astropy.units as u
            >>> rate = data_rate(
                    mode='imaging',
                    mas=96,
                    dt=1*u.s,
                    nchan=64,
                    bandwidth=75*u.MHz
                )
            >>> print(rate)
            3.6616274×10^9 byte/s
            >>> exposure = 3600*u.s
            >>> size = rate * exposure
            >>> print(size)
            1.3181859×10^13 byte
            >>> print(size.to(u.Tibyte))
            11.988831 Tibyte

        .. note::
            IDL original version v1, PZ, 2019-03-19
            
            PYTHON transcript v1, JG, 2020-05-09
            
            Pythonized for `nenupy`, AL, 2020-05-11

        .. versionadded:: 1.1.0
    """
    # Input checks
    available_modes = ['imaging', 'beamforming', 'waveform', 'tbb']
    if not mode in available_modes:
        raise ValueError(
            'mode should be one of {}'.format(available_modes)
        )
    if not isinstance(mas, int):
        raise TypeError(
            'mas should be an integer'
        )
    if not isinstance(dt, u.Quantity):
        dt *= u.s
    if not isinstance(nchan, int):
        raise TypeError(
            'nchan should be an integer'
        )
    if nchan > 64:
        raise ValueError(
            'NenuFAR maximal number of channels per subband is 64'
        )
    if not isinstance(bandwidth, u.Quantity):
        bandwidth *= u.MHz
    if bandwidth > 150*u.MHz:
        raise ValueError(
            'NenuFAR maximal bandwidth is 150 MHz.'
        )

    # NenuFAR backend properties
    sb_width = 195.3125*u.kHz # subband bandwidth
    n_sb_max = 768 # maximal number of subbands
    n_corr = 4 # XX, XY, YX, YY 

    # Number of sub-bands involved
    n_sb = int(np.round(bandwidth.to(u.kHz)/sb_width))

    if mode == 'imaging':
        # Complex in bytes
        nenucomplex = np.complex64().itemsize * u.byte
        n_baselines = mas*(mas - 1)/2 + mas
        rate_sb = n_corr*nenucomplex*nchan*n_baselines/dt
        rate = rate_sb * n_sb
    elif mode == 'beamforming':
        # Floats as bytes in NenuFAR calculators
        nenufloat = np.float32().itemsize * u.byte
        rate_sb = n_corr*nenufloat*nchan/dt
        rate = rate_sb * n_sb
    elif mode == 'waveform':
        rate_sb = 4*nb*sb_width.to(u.Hz).value
        rate = rate_sb * n_sb * u.byte / u.s
    elif mode == 'tbb':
        rate = 4*nb*sb_width.to(u.Hz).value*1024*mas  * u.byte / u.s

    return rate
# ============================================================= #


# ============================================================= #
# ------------------------- freq2sb --------------------------- #
# ============================================================= #
def freq2sb(freq):
    r""" Conversion between the frequency :math:`\nu` and the
        NenuFAR sub-band index :math:`n_{\rm SB}`.
        Each NenuFAR sub-band has a bandwidth of
        :math:`\Delta \nu = 195.3125\, \rm{kHz}`:

        .. math::
            n_{\rm SB} = \frac{512 \times \nu}{\Delta \nu}

        :param freq:
            Frequency to convert in sub-band index (assumed in
            MHz if no unit is provided).
        :type freq:
            `float`, :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`

        :returns:
            Sub-band index, same dimension as ``freq``.
        :rtype: `int` or :class:`~numpy.ndarray`

        :example:
            >>> from nenupy.instru import freq2sb
            >>> freq2sb(freq=50.5)
            258
            >>> freq2sb(freq=[50.5, 51])
            array([258, 261])

        .. versionadded:: 1.1.0
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    if (freq.min() < 0 * u.MHz) or (freq.max() > 100 * u.MHz):
        raise ValueError(
            'freq should be between 0 and 100 MHz.'
        )
    freq = freq.to(u.MHz)
    sb_width = 100. * u.MHz
    sb_idx = np.floor((freq * 512) / sb_width)
    return sb_idx.astype(int).value
# ============================================================= #


# ============================================================= #
# ------------------------- freq2sb --------------------------- #
# ============================================================= #
def sb2freq(sb):
    r""" Conversion between NenuFAR sub-band index :math:`n_{\rm SB}`
        to sub-band starting frequency :math:`\nu_{\rm start}`:

        .. math::
            \nu_{\rm start} = \frac{n_{\rm SB} \times \Delta \nu}{512}

        Each NenuFAR sub-band has a bandwidth of
        :math:`\Delta \nu = 195.3125\, \rm{kHz}`, therefore, the
        sub-band :math:`n_{\rm SB}` goes from :math:`\nu_{\rm start}`
        to :math:`\nu_{\rm stop} = \nu_{\rm start} + \Delta \nu`.

        :param sb:
            Sub-band index (from 0 to 511).
        :type sb: `int` or :class:`~numpy.ndarray` of `int`

        :returns:
            Sub-band start frequency :math:`\nu_{\rm start}` in MHz.
        :rtype: :class:`~astropy.units.Quantity`

        :example:
            >>> from nenupy.instru import sb2freq
            >>> sb2freq(sb=1)
            [0.1953125] MHz
            >>> sb2freq(sb=[1, 2, 3, 4])
            [0.1953125, 0.390625, 0.5859375, 0.78125] MHz

        .. versionadded:: 1.1.0
    """
    if np.isscalar(sb):
        sb = np.array([sb])
    else:
        sb = np.array(sb)
    if sb.dtype.name not in ['int32', 'int64']:
        raise TypeError(
            'sb should be integers.'
        )
    if (sb.min() < 0) or (sb.max() > 511):
        raise ValueError(
            'sb should be between 0 and 511.'
        )
    sb_width = 100. * u.MHz
    freq_start = (sb * sb_width) / 512
    return freq_start
# ============================================================= #

