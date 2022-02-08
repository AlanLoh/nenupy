#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u

from nenupy.astro.target import FixedTarget, SolarSystemTarget


# ============================================================= #
# ---------------- test_fixedtarget_properties ---------------- #
# ============================================================= #
def test_fixedtarget_properties():
    src = FixedTarget(
        coordinates=SkyCoord(300, 40, unit="deg"),
        time=Time("2022-01-01T12:00:00")
    )

    ha = src.hour_angle(fast_compute=True)
    assert ha[0].deg == pytest.approx(343.315, 1e-3)
    ha = src.hour_angle(fast_compute=False)
    assert ha[0].deg == pytest.approx(343.311, 1e-3)

    lst = src.local_sidereal_time(fast_compute=True)
    assert lst[0].deg == pytest.approx(283.315, 1e-3)
    lst = src.local_sidereal_time(fast_compute=False)
    assert lst[0].deg == pytest.approx(283.311, 1e-3)

    assert src.culmination_azimuth.to(u.deg).value == 180.0

    assert not src.is_circumpolar

    src = FixedTarget(
        coordinates=SkyCoord(100, 80, unit="deg"),
        time=Time("2022-01-01T12:00:00")
    )

    assert src.is_circumpolar
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------- test_solarsystemtarget_properties ------------- #
# ============================================================= #
def test_solarsystemtarget_properties():
    src = SolarSystemTarget.from_name(
        name="Sun",
        time=Time("2022-01-01T12:00:00")
    )

    ha = src.hour_angle(fast_compute=True)
    assert ha[0].deg == pytest.approx(1.640, 1e-3)
    ha = src.hour_angle(fast_compute=False)
    assert ha[0].deg == pytest.approx(1.636, 1e-3)

    lst = src.local_sidereal_time(fast_compute=True)
    assert lst[0].deg == pytest.approx(283.315, 1e-3)
    lst = src.local_sidereal_time(fast_compute=False)
    assert lst[0].deg == pytest.approx(283.311, 1e-3)

    assert src.culmination_azimuth.to(u.deg).value == 180.0

    assert not src.is_circumpolar
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------- test_fixedtarget_meridian_transit ------------- #
# ============================================================= #
def test_fixedtarget_meridian_transit():
    src = FixedTarget.from_name(
        name="Cyg A",
        time=Time("2022-01-01T12:00:00")
    )
    transit_time = src.meridian_transit(
        t_min=Time("2022-01-01T12:00:00"),
        duration=TimeDelta(86400, format='sec'),
        precision=TimeDelta(5, format='sec'),
        fast_compute=True
    )
    assert transit_time[0].jd == pytest.approx(2459581.0464, 1e-4)

    transit_times = src.meridian_transit(
        t_min=Time("2022-01-01T12:00:00"),
        duration=TimeDelta(86400*2, format='sec'),
        precision=TimeDelta(5, format='sec'),
        fast_compute=False
    )
    assert transit_times.size == 2
    assert transit_times[1].jd == pytest.approx(2459582.0437, 1e-4)

    transit_time = src.next_meridian_transit(time=Time("2022-01-01T12:00:00"))
    assert transit_time.isscalar
    assert transit_time.jd == pytest.approx(2459581.0464, 1e-4)

    transit_time = src.previous_meridian_transit(time=Time("2022-01-01T12:00:00"))
    assert transit_time.isscalar
    assert transit_time.jd == pytest.approx(2459580.0491, 1e-4)

    az_transit = src.azimuth_transit(
        azimuth=200*u.deg,
        t_min=Time("2022-01-01T12:00:00")
    )
    assert az_transit.size == 1
    assert az_transit.jd == pytest.approx(2459581.0551, 1e-4)

    az_transit = src.azimuth_transit(
        azimuth=330*u.deg,
        t_min=Time("2022-01-01T12:00:00")
    )
    assert az_transit.size == 0

    src_circum = FixedTarget.from_name(
        name="Cas A",
        time=Time("2022-01-01T12:00:00")
    )
    az_transit = src_circum.azimuth_transit(
        azimuth=350*u.deg,
        t_min=Time("2022-01-01T12:00:00")
    )
    assert az_transit.size == 2
    assert az_transit[0].jd == pytest.approx(2459581.1987, 1e-4)
    assert az_transit[1].jd == pytest.approx(2459581.6345, 1e-4)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------- test_solarsystemtarget_meridian_transit ---------- #
# ============================================================= #
def test_solarsystemtarget_meridian_transit():
    src = SolarSystemTarget.from_name(
        name="Sun",
        time=Time("2022-01-01T12:00:00")
    )
    transit_time = src.meridian_transit(
        t_min=Time("2022-01-01T12:00:00"),
        duration=TimeDelta(86400, format='sec'),
        precision=TimeDelta(5, format='sec'),
        fast_compute=True
    )
    assert transit_time[0].jd == pytest.approx(2459581.9967, 1e-4)

    transit_times = src.meridian_transit(
        t_min=Time("2022-01-01T12:00:00"),
        duration=TimeDelta(86400*2, format='sec'),
        precision=TimeDelta(5, format='sec'),
        fast_compute=False
    )
    assert transit_times.size == 2
    assert transit_times[1].jd == pytest.approx(2459582.9970, 1e-4)

    transit_time = src.next_meridian_transit(time=Time("2022-01-01T12:00:00"))
    assert transit_time.isscalar
    assert transit_time.jd == pytest.approx(2459581.9967, 1e-4)

    transit_time = src.previous_meridian_transit(time=Time("2022-01-01T12:00:00"))
    assert transit_time.isscalar
    assert transit_time.jd == pytest.approx(2459580.9964, 1e-4)

    az_transit = src.azimuth_transit(
        azimuth=200*u.deg,
        t_min=Time("2022-01-01T12:00:00")
    )
    assert az_transit.size == 1
    assert az_transit.jd == pytest.approx(2459581.0541, 1e-4)

    az_transit = src.azimuth_transit(
        azimuth=330*u.deg,
        t_min=Time("2022-01-01T12:00:00")
    )
    assert az_transit.size == 0
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------- test_fixedtarget_rise ------------------- #
# ============================================================= #
def test_fixedtarget_rise():
    src = FixedTarget.from_name(
        name="Cyg A",
        time=Time("2022-01-01T12:00:00")
    )
    rise_times = src.rise_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=0*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert rise_times.size == 2
    assert rise_times[0].jd == pytest.approx(2459581.601, 1e-3)
    assert rise_times[1].jd == pytest.approx(2459582.599, 1e-3)

    rise_time = src.next_rise_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert rise_time.isscalar
    assert rise_time.jd == pytest.approx(2459581.692, 1e-3)

    rise_time = src.previous_rise_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert rise_time.isscalar
    assert rise_time.jd == pytest.approx(2459580.695, 1e-3)

    src_circum = FixedTarget.from_name(
        name="Cas A",
        time=Time("2022-01-01T12:00:00")
    )
    rise_times = src_circum.rise_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=0*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert rise_times.size == 0
    rise_times = src_circum.rise_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=40*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert rise_times.size == 2
    assert rise_times[0].jd == pytest.approx(2459581.941, 1e-3)
    assert rise_times[1].jd == pytest.approx(2459582.939, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------- test_solarsystemtarget_rise ---------------- #
# ============================================================= #
def test_solarsystemtarget_rise():
    src = SolarSystemTarget.from_name(
        name="Sun",
        time=Time("2022-01-01T12:00:00")
    )
    rise_times = src.rise_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=0*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert rise_times.size == 2
    assert rise_times[0].jd == pytest.approx(2459581.823, 1e-3)
    assert rise_times[1].jd == pytest.approx(2459582.823, 1e-3)

    rise_time = src.next_rise_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert rise_time.isscalar
    assert rise_time.jd == pytest.approx(2459581.879, 1e-3)

    rise_time = src.previous_rise_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert rise_time.isscalar
    assert rise_time.jd == pytest.approx(2459580.879, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------- test_fixedtarget_set -------------------- #
# ============================================================= #
def test_fixedtarget_set():
    src = FixedTarget.from_name(
        name="Cyg A",
        time=Time("2022-01-01T12:00:00")
    )
    set_times = src.set_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=0*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert set_times.size == 2
    assert set_times[0].jd == pytest.approx(2459581.489, 1e-3)
    assert set_times[1].jd == pytest.approx(2459582.486, 1e-3)

    set_time = src.next_set_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert set_time.isscalar
    assert set_time.jd == pytest.approx(2459581.398, 1e-3)

    set_time = src.previous_set_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert set_time.isscalar
    assert set_time.jd == pytest.approx(2459580.400, 1e-3)

    src_circum = FixedTarget.from_name(
        name="Cas A",
        time=Time("2022-01-01T12:00:00")
    )
    set_times = src_circum.set_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=0*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert set_times.size == 0
    set_times = src_circum.set_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=40*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert set_times.size == 2
    assert set_times[0].jd == pytest.approx(2459581.431, 1e-3)
    assert set_times[1].jd == pytest.approx(2459582.429, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------- test_solarsystemtarget_set ----------------- #
# ============================================================= #
def test_solarsystemtarget_set():
    src = SolarSystemTarget.from_name(
        name="Sun",
        time=Time("2022-01-01T12:00:00")
    )
    set_times = src.set_time(
        t_min=Time("2022-01-01T12:00:00"),
        elevation=0*u.deg,
        duration=TimeDelta(86400*2, format="sec")
    )
    assert set_times.size == 2
    assert set_times[0].jd == pytest.approx(2459581.170, 1e-3)
    assert set_times[1].jd == pytest.approx(2459582.171, 1e-3)

    set_time = src.next_set_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert set_time.isscalar
    assert set_time.jd == pytest.approx(2459581.114, 1e-3)

    set_time = src.previous_set_time(
        time=Time("2022-01-01T12:00:00"),
        elevation=10*u.deg
    )
    assert set_time.isscalar
    assert set_time.jd == pytest.approx(2459580.113, 1e-3)
# ============================================================= #
# ============================================================= #

