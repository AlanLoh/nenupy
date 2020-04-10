#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********************
    Instrument functions
    ********************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HiddenPrints',
    'nenufar_loc',
    'analog_pointing',
    'desquint_elevation',
    'nenufar_ant_gain',
    'read_cal_table'
]


from astropy.io.fits import getdata
from astropy.units import Quantity
from astropy.coordinates import (
    ICRS,
    AltAz,
    EarthLocation
)
import astropy.units as u
from healpy import (
    read_map,
    ud_grade,
    nside2npix,
    pix2ang,
    Rotator
)
import numpy as np
import os, sys
from os.path import join, dirname
from scipy.io.idl import readsav
from scipy.interpolate import interp1d

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ----------------------- HiddenPrints ------------------------ #
# ============================================================= #
class HiddenPrints:
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
    lon=2.192400 * u.deg,
    height=150 * u.m
)
# nenufar_loc = EarthLocation( # old
#     lat=47.375944 * u.deg,
#     lon=2.193361 * u.deg,
#     height=136.195 * u.m
# )
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

        :param azimuth:
            Requested azimuth (in degrees if `float`).
        :type azimuth: `float` or :class:`astropy.units.Quantity`
        :param elevation:
            Requested elevation (in degrees if `float`).
        :type elevation: `float` or :class:`astropy.units.Quantity`

        :returns: (Azimuth, Elevation)
        :rtype: :class:`astropy.units.Quantity`
        
        >>> from nenupysim.instru import analog_pointing
        >>> import astropy.units as u
        >>> analog_pointing(180*u.deg, 45*u.deg)
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
    phi_idx = int(azimuth.value/0.05 - 0.5)  
    theta_idx = int((90. - elevation.value)/0.05 - 0.5)
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

        :param elevation:
            Requested elevation (in degrees if `float`).
        :type elevation: `float` or :class:`astropy.units.Quantity`
        :param opt_freq:
            Beam squint optimization frequency (in MHs if `float`)
        :type opt_freq: `float` or :class:`astropy.units.Quantity`

        :returns: Elevation to point
        :rtype: :class:`astropy.units.Quantity`
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
    if elevation < 20.:
        elevation = 20
    return elevation * u.deg
# ============================================================= #


# ============================================================= #
# --------------------- nenufar_ant_gain ---------------------- #
# ============================================================= #
def nenufar_ant_gain(freq, polar='NW', nside=64, time=None):
    """
    """
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
    # Get Low and High ant gain bounding freq
    f_low = gain_freqs[gain_freqs <= freq].max()
    f_high = gain_freqs[gain_freqs >= freq].min()
    antgain_file = join(
        dirname(__file__),
        'NenuFAR_Ant_Hpx.fits',
    )
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
        with HiddenPrints():
            gain = rot.rotate_map_alms(gain)
    # Convert HEALPix map to required nside
    gain = ud_grade(gain, nside_out=nside)
    return gain / gain.max()
# ============================================================= #


# ============================================================= #
# ---------------------- read_cal_table ----------------------- #
# ============================================================= #
def read_cal_table(calfile=None):
    """
        data(sb, ma, pol)
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


