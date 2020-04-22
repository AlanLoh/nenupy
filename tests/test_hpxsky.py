#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
import pytest

from nenupy.astro import HpxSky


# ============================================================= #
# ------------------ test_hpxsky_resolution ------------------- #
# ============================================================= #
def test_hpxsky_resolution():
    # Float input
    sky = HpxSky(
        resolution=20
    )
    assert sky.nside == 4
    assert isinstance(sky.resolution, Angle)
    assert sky.resolution.deg == 20.
    # Astropy Quantity input
    sky = HpxSky(
        resolution=20*u.deg
    )
    assert sky.nside == 4
    assert isinstance(sky.resolution, Angle)
    assert sky.resolution.deg == 20.
# ============================================================= #


# ============================================================= #
# --------------------- test_hpxsky_time ---------------------- #
# ============================================================= #
def test_hpxsky_time():
    sky = HpxSky(
        resolution=20
    )
    # String input
    sky.time = '2020-04-01 12:00:00'
    assert isinstance(sky.time, Time)
    assert sky.time.jd == 2458941.0
    # Astropy Time input
    sky.time = Time('2020-04-01 12:00:00')
    assert isinstance(sky.time, Time)
    assert sky.time.jd == 2458941.0
# ============================================================= #


# ============================================================= #
# -------------------- test_hpxsky_skymap --------------------- #
# ============================================================= #
def test_hpxsky_skymap():
    sky = HpxSky(
        resolution=20
    )
    assert sky.skymap.size == 192
    assert isinstance(sky.skymap, np.ma.core.MaskedArray)
# ============================================================= #


# ============================================================= #
# -------------------- test_hpxsky_visible -------------------- #
# ============================================================= #
def test_hpxsky_visible():
    sky = HpxSky(
        resolution=50
    )
    assert sky.skymap.size == 12
    assert isinstance(sky._is_visible, np.ndarray)
    assert sky.skymap.size == sky._is_visible.size
    allvisible = np.ones(sky.skymap.size, dtype=bool)
    assert all(sky._is_visible == allvisible)
    sky.time = '2020-04-01 12:00:00'
    halfvisible = allvisible.copy()
    halfvisible[6:] = False
    assert all(sky._is_visible == halfvisible)
    sky.visible_sky = True
    assert sky.eq_coords.size == 6
    assert sky.ho_coords.size == 6
    assert isinstance(sky.skymap[-1], np.ma.core.MaskedConstant)
    sky.visible_sky = False
    assert sky.eq_coords.size == 12
    assert sky.ho_coords.size == 12
    assert sky.skymap[-1] == 0.0
# ============================================================= #


# ============================================================= #
# -------------------- test_hpxsky_eqcoord -------------------- #
# ============================================================= #
def test_hpxsky_eqcoord():
    sky = HpxSky(
        resolution=50
    )
    assert isinstance(sky.eq_coords, SkyCoord)
    
    dec_12 = 41.8103
    hpxcoords = SkyCoord(
        ra=np.array([
            45., 135., 225., 315.,   0.,  90.,
            180., 270.,  45., 135., 225., 315.
        ])*u.deg,
        dec=np.array([
            dec_12, dec_12, dec_12, dec_12, 0., 0.,
            0., 0., -dec_12, -dec_12, -dec_12, -dec_12
        ])*u.deg,
        frame='icrs'
    )
    compra = hpxcoords.ra.deg
    compdec = hpxcoords.dec.deg
    tol = 1e-4
    assert sky.eq_coords.ra.deg == pytest.approx(compra, tol)
    assert sky.eq_coords.dec.deg == pytest.approx(compdec, tol)
    sky.time = '2020-04-01 12:00:00'
    assert sky.eq_coords.ra.deg == pytest.approx(compra[:6], tol)
    assert sky.eq_coords.dec.deg == pytest.approx(compdec[:6], tol)
# ============================================================= #


# ============================================================= #
# -------------------- test_hpxsky_hocoord -------------------- #
# ============================================================= #
def test_hpxsky_hocoord():
    sky = HpxSky(
        resolution=50
    )
    sky.time = '2020-04-01 12:00:00'
    assert isinstance(sky.ho_coords, SkyCoord)
    compaz = np.array([
        91.16,  39.95, 336.17, 284.00, 196.46, 99.07,
        16.45, 279.07, 156.18, 104.01, 271.15, 219.96
    ])
    compalt = np.array([
        66.19,  12.46,   3.69,  49.77, 41.54, 8.25,
        -41.55,  -8.25, -3.70, -49.78, -66.19, -12.46
    ])
    tol = 1e-2
    sky.visible_sky = True
    assert sky.ho_coords.az.deg == pytest.approx(compaz[:6], tol)
    assert sky.ho_coords.alt.deg == pytest.approx(compalt[:6], tol)
    sky.visible_sky = False
    assert sky.ho_coords.az.deg == pytest.approx(compaz, tol)
    assert sky.ho_coords.alt.deg == pytest.approx(compalt, tol)
# ============================================================= #

