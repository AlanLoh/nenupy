#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
from unittest.mock import patch
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np

from nenupy.astro.sky import Sky, HpxSky


# ============================================================= #
# ----------------------- test_sky_init ----------------------- #
# ============================================================= #
def test_sky_init():
    sky = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 6)*u.MHz,
        polarization="NE",
        value=0
    )
    assert sky.shape == (2, 6, 1, 3)
    assert sky.time.size == 2
    assert sky.frequency.size == 6
    assert sky.polarization.size == 1
    assert sky.coordinates.size == 3
    assert sky.visible_mask.shape == (2, 6, 1, 3)
    assert sky.visible_mask[0, 0, 0, 1]
    assert sky.horizontal_coordinates.shape == (2, 3)
    assert str(sky) == "<class 'nenupy.astro.sky.Sky'> instance\nvalue: (2, 6, 1, 3)\n\t* time: (2,)\n\t* frequency: (6,)\n\t* polarization: (1,)\n\t* coordinates: (3,)\n"
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- test_sky_operations -------------------- #
# ============================================================= #
def test_sky_operations():
    sky1 = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 2)*u.MHz,
        polarization="NE",
        value=6
    )
    sky2 = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 2)*u.MHz,
        polarization="NE",
        value=2
    )
    result = sky1/sky2
    assert np.unique(result.value)[0] == 3.

    sky1 = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 2)*u.MHz,
        polarization="NE",
        value=6
    )
    sky2 = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 2)*u.MHz,
        polarization="NE",
        value=2
    )
    result = sky1*sky2
    assert np.unique(result.value)[0] == 12.

    sky1 = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 2)*u.MHz,
        polarization="NE",
        value=6
    )
    sky2 = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 3)*u.MHz,
        polarization="NE",
        value=2
    )
    with pytest.raises(ValueError):
        result = sky1/sky2
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- test_sky_lmn ------------------------ #
# ============================================================= #
def test_sky_lmn():
    sky = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 6)*u.MHz,
        polarization="NE",
        value=0
    )
    l, m, n = sky.compute_lmn(phase_center=SkyCoord(200, 20, unit="deg"))
    assert l.shape ==(1, 3)
    assert l[0, 1] == 0.
    assert m[0, 1] == 0.
    assert n[0, 1] == 1.
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- test_sky_get ------------------------ #
# ============================================================= #
@patch("matplotlib.pyplot.show")
def test_sky_get(mock_show):
    sky = Sky(
        coordinates=SkyCoord([100, 200, 300], [10, 20, 30], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 6)*u.MHz,
        polarization="NE",
        value=10
    )
    sky_slice = sky[0, 0, 0]
    assert sky_slice.value.shape == (3,)
    assert sky_slice.visible_sky.shape == (3,)
    sky_slice.plot(decibel=True, altaz_overlay=True)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_hpxsky_init ---------------------- #
# ============================================================= #
def test_hpxsky_init():
    sky = HpxSky(
        resolution=2*u.deg,
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 4)*u.MHz,
        polarization="NW",
        value=5
    )
    assert sky.shape == (2, 4, 1, 12288)
    assert sky.nside == 32
    assert sky.resolution.to(u.deg).value == pytest.approx(1.83, 1e-2)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------ test_hpxsky_shaped_like ------------------ #
# ============================================================= #
def test_hpxsky_shaped_like():
    sky1 = HpxSky(
        resolution=2*u.deg,
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 4)*u.MHz,
        polarization="NW",
        value=5
    )
    sky2 = HpxSky.shaped_like(sky1)
    assert sky2.shape == sky1.shape
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- test_hpxsky_get ---------------------- #
# ============================================================= #
@patch("matplotlib.pyplot.show")
def test_hpxsky_get(mock_show):
    sky = HpxSky(
        resolution=2*u.deg,
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        frequency=np.linspace(30, 50, 4)*u.MHz,
        polarization="NW",
        value=5
    )
    sky_slice = sky[0, 0, 0]
    assert sky_slice.value.shape == (12288,)
    assert sky_slice.visible_sky.shape == (12288,)
    sky_slice.plot(decibel=True, altaz_overlay=True)
    sky_slice.plot(decibel=True, altaz_overlay=True)
# ============================================================= #
# ============================================================= #

