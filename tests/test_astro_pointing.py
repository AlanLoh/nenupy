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
from os.path import join, dirname
from astropy.coordinates import SkyCoord, Longitude
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np

from nenupy.io.bst import BST
from nenupy.astro.pointing import Pointing
from nenupy.astro.target import FixedTarget, SolarSystemTarget


# ============================================================= #
# ----------------------- test_pointing ----------------------- #
# ============================================================= #
def test_pointing():
    pointing = Pointing(
        coordinates=SkyCoord(300, 40, unit="deg"),
        time=Time("2022-01-01T12:00:00")
    )

    assert isinstance(pointing.duration, TimeDelta)
    assert pointing.duration.sec == 1.0

    altaz = pointing.horizontal_coordinates 
    assert altaz.shape == (1, 1)
    assert altaz[0, 0].az.deg == pytest.approx(114.884, 1e-3)
    assert altaz[0, 0].alt.deg == pytest.approx(75.821, 1e-3)

    ha = pointing.hour_angle()
    assert isinstance(ha, Longitude)
    assert ha.shape == (1,)
    assert ha[0].deg == pytest.approx(343.315, 1e-3)

    lst = pointing.local_sidereal_time()
    assert isinstance(lst, Longitude)
    assert lst.shape == (1,)
    assert lst[0].deg == pytest.approx(283.315, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- test_pointing_plot --------------------- #
# ============================================================= #
@patch('matplotlib.pyplot.show')
def test_pointing_plot(mock_show):
    pointing = Pointing(
        coordinates=SkyCoord(300, 40, unit="deg"),
        time=Time("2022-01-01T12:00:00")
    )
    pointing.plot(display_duration=True)
    pointing.plot(display_duration=False)
# ============================================================= #
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------ test_pointing_from_file ------------------ #
# ============================================================= #
def test_pointing_from_file():
    altaza_full = join(dirname(__file__), "test_data/full.altazA")
    altaza_no_corr = join(dirname(__file__), "test_data/no_corr.altazA")
    altaza_no_beamsquint = join(dirname(__file__), "test_data/no_beamsquint.altazA")
    altazb_full = join(dirname(__file__), "test_data/full.altazB")

    with pytest.raises(ValueError):
        pointing = Pointing.from_file(
            file_name=altaza_full,
            beam_index=1,
            include_corrections=True
        )

    pointing = Pointing.from_file(
        file_name=altaza_full,
        beam_index=0,
        include_corrections=True
    )
    assert pointing.coordinates.size == 80
    assert pointing.coordinates[0].ra.deg == pytest.approx(180.346, 1e-3)
    pointing = Pointing.from_file(
        file_name=altaza_full,
        beam_index=0,
        include_corrections=False
    )
    assert pointing.coordinates[0].ra.deg == pytest.approx(176.985, 1e-3)

    pointing = Pointing.from_file(
        file_name=altaza_no_corr,
        beam_index=0,
        include_corrections=True
    )
    assert pointing.coordinates.size == 80
    assert pointing.coordinates[0].ra.deg == pytest.approx(180.518, 1e-3)
    pointing = Pointing.from_file(
        file_name=altaza_no_corr,
        beam_index=0,
        include_corrections=False
    )
    assert pointing.coordinates[0].ra.deg == pytest.approx(176.985, 1e-3)

    pointing = Pointing.from_file(
        file_name=altaza_no_beamsquint,
        beam_index=0,
        include_corrections=True
    )
    assert pointing.coordinates.size == 80
    assert pointing.coordinates[0].ra.deg == pytest.approx(177.772, 1e-3)
    pointing = Pointing.from_file(
        file_name=altaza_no_beamsquint,
        beam_index=0,
        include_corrections=False
    )
    assert pointing.coordinates[0].ra.deg == pytest.approx(176.985, 1e-3)

    with pytest.raises(ValueError):
        pointing = Pointing.from_file(
            file_name=altazb_full,
            beam_index=4,
            include_corrections=True
        )

    pointing = Pointing.from_file(
        file_name=altazb_full,
        beam_index=2
    )
    assert pointing.coordinates.size == 163
    assert pointing.coordinates[0].ra.deg == pytest.approx(144.776, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_pointing_get --------------------- #
# ============================================================= #
def test_pointing_get():
    pointing = Pointing(
        coordinates=SkyCoord(300, 40, unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        duration=TimeDelta([3600, 1800], format="sec")
    )
    times = Time([
        "2022-01-01T12:30:00", # in first pointing
        "2022-01-01T13:30:00", # in between pointings...
        "2022-01-01T14:10:00", # in second pointing
        "2022-01-01T18:00:00"  # after everything
    ])
    sub_pointing = pointing[times]
    assert sub_pointing.coordinates.shape == (4,)
    assert sub_pointing.coordinates[0].ra.deg == pytest.approx(300.0, 1e-1)
    assert sub_pointing.coordinates[1].ra.deg == pytest.approx(305.709, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------- test_pointing_target_tracking --------------- #
# ============================================================= #
def test_pointing_target_tracking():
    pointing = Pointing.target_tracking(
        target=FixedTarget.from_name("Cas A"),
        time=Time("2022-01-01T12:00:00")
    )
    assert pointing.coordinates.size == 1

    pointing = Pointing.target_tracking(
        target=FixedTarget.from_name("Cas A"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"])
    )
    assert pointing.coordinates.size == 2
    assert pointing.horizontal_coordinates[1].az.deg == pytest.approx(48.45, 1e-2)

    pointing = Pointing.target_tracking(
        target=SolarSystemTarget.from_name("Sun"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"])
    )
    assert np.unique(pointing.coordinates.ra.deg).size == 2
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------- test_pointing_target_transit ---------------- #
# ============================================================= #
def test_pointing_target_transit():
    pointing = Pointing.target_transit(
        target=FixedTarget.from_name("Cyg A"),
        t_min=Time("2022-01-01T12:00:00"),
        duration=TimeDelta(3600, format="sec"),
        dt=TimeDelta(3600, format="sec"),
        azimuth=180*u.deg
    )
    assert pointing.custom_ho_coordinates[0, 0].az.deg == pytest.approx(179.954, 1e-3)
    assert pointing.custom_ho_coordinates[0, 0].alt.deg == pytest.approx(83.416, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------- test_pointing_zenith_tracking --------------- #
# ============================================================= #
def test_pointing_zenith_tracking():
    pointing = Pointing.zenith_tracking(
        time=Time([
            "2022-01-01T12:30:00",
            "2022-01-01T13:30:00",
            "2022-01-01T14:10:00",
            "2022-01-01T18:00:00"
        ]),
        duration=TimeDelta(1800, format="sec")
    )
    sub_pointing = pointing[Time("2022-01-01T12:45:00")]
    sub_pointing.custom_ho_coordinates[0].az.deg == 0.
    sub_pointing.custom_ho_coordinates[0].alt.deg == 0.
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------ test_pointing_from_bst ------------------- #
# ============================================================= #
def test_pointing_from_bst():
    bst = BST(join(dirname(__file__), "test_data/BST.fits"))

    pointing = Pointing.from_bst(
        bst=bst,
        beam=0,
        analog=True
    )
    assert pointing.custom_ho_coordinates[0].alt.deg == pytest.approx(35.579, 1e-3)

    pointing = Pointing.from_bst(
        bst=bst,
        beam=0,
        analog=False
    )
    assert pointing.custom_ho_coordinates[0].alt.deg == pytest.approx(34.229, 1e-3)

    pointing = Pointing.from_bst(
        bst=bst,
        beam=16,
        analog=False
    )
    assert pointing.custom_ho_coordinates[0].alt.deg == pytest.approx(36.929, 1e-3)
# ============================================================= #
# ============================================================= #

