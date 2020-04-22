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
    # Astropy Angle input
    sky = HpxSky(
        resolution=Angle(20, unit='deg')
    )
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
    assert isinstance(sky.time, Time) # now
    sky.time = None
    assert isinstance(sky.time, Time) # now
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


# ============================================================= #
# ---------------------- test_hpxsky_lmn ---------------------- #
# ============================================================= #
def test_hpxsky_lmn():
    sky = HpxSky(
        resolution=50
    )
    center = SkyCoord(
        ra=0.*u.deg,
        dec=0.*u.deg
    )
    l, m, n = sky.lmn(phase_center=center)
    assert isinstance(l, np.ndarray)
    assert isinstance(m, np.ndarray)
    assert isinstance(n, np.ndarray)
    assert l.size == 12
    assert m.size == 12
    assert n.size == 12
    tol = 1e-2
    lcomp = np.array([
        0.53, 0.53, -0.53, -0.53, 0.00, 1.00,
        0.00, -1.00, 0.53, 0.53, -0.53, -0.53,
    ])
    mcomp = np.array([
        0.67, 0.67, 0.67, 0.67, 0.00, 0.00,
        0.00, 0.00, -0.67, -0.67, -0.67, -0.67
    ])
    ncomp = np.array([
        0.53, 0.53, 0.53, 0.53, 1.00, 0.00,
        1.00, 0.00, 0.53, 0.53, 0.53, 0.53,
    ])
    assert l == pytest.approx(lcomp, tol)
    assert m == pytest.approx(mcomp, tol)
    assert n == pytest.approx(ncomp, tol)
# ============================================================= #


# ============================================================= #
# --------------------- test_hpxsky_radec --------------------- #
# ============================================================= #
def test_hpxsky_radec():
    sky = HpxSky(
        resolution=50
    )
    sky.skymap[:] = np.arange(sky.skymap.size)
    # Float input
    vals = sky.radec_value(ra=0, n=5)
    # Astropy Quantity input
    vals = sky.radec_value(ra=0*u.deg, n=5)
    assert all(vals == np.array([4., 4., 0., 0., 0.]))
    assert sky.radec_value(ra=180, dec=-45) == 10.
    vals = sky.radec_value(dec=45, n=5)
    vals = sky.radec_value(dec=45*u.deg, n=5)
    assert all(vals == np.array([0., 1., 2., 3., 0.]))
# ============================================================= #


# ============================================================= #
# --------------------- test_hpxsky_azel ---------------------- #
# ============================================================= #
def test_hpxsky_azel():
    sky = HpxSky(
        resolution=50
    )
    sky.time = '2020-04-01 12:00:00'
    sky.visible_sky = False
    sky.skymap[:] = np.arange(sky.skymap.size)
    # Float input
    vals = sky.azel_value(az=0, n=5)
    # Astropy Quantity input
    vals = sky.azel_value(az=0*u.deg, n=5)
    assert all(vals == np.array([2., 2., 2., 0., 0.]))
    assert sky.azel_value(az=0, el=-90) == 10.
    vals = sky.azel_value(el=0, n=5)
    vals = sky.azel_value(el=0*u.deg, n=5)
    assert all(vals == np.array([2., 5., 8., 7., 2.]))
# ============================================================= #


# ============================================================= #
# --------------------- test_hpxsky_plot ---------------------- #
# ============================================================= #
def test_hpxsky_plot():
    sky = HpxSky(
        resolution=50
    )
    sky.time = '2020-04-01 12:00:00'
    sky.visible_sky = False
    sky.skymap[:] = np.arange(sky.skymap.size)
    fig, ax = sky.plot(
        figname='return',
        db=False,
        figsize=(10, 10),
        center=SkyCoord(ra=0*u.deg, dec=0*u.deg),
        size=50,
        cblabel='test',
        title='title',
        cmap='viridis',
        vmin=0,
        vmax=9,
        tickscol='white',
        grid=True,
        indices=(np.arange(12), 20, 'black'),
        scatter=([0, 10], [0, 10], 10, 'tab:red'),
        curve=([0, 10], [0, 10], (':'), 'tab:red'),
    )
    fig, ax = sky.plot(
        figname='return',
        db=True,
        grid=True,
    )
    sky.visible_sky = True
    fig, ax = sky.plot(
        figname='return',
        db=True,
        size=50*u.deg,
        grid=False,
        vmin=None,
        vmax=None
    )
# ============================================================= #

