#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
from astropy.coordinates import SkyCoord, Longitude, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy import nenufar_position
from nenupy.astro import (
    solar_system_source,
    local_sidereal_time,
    hour_angle,
    radec_to_altaz,
    altaz_to_radec,
    sky_temperature,
    dispersion_delay,
    wavelength,
    l93_to_etrs,
    geo_to_etrs,
    etrs_to_enu
)


# ============================================================= #
# ----------------- test_solar_system_source ------------------ #
# ============================================================= #
def test_solar_system_source():
    sun = solar_system_source(
        name="Sun",
        time=Time("2022-01-01T12:00:00")
    )
    assert isinstance(sun, SkyCoord)
    assert sun.ra.deg == pytest.approx(281.675, 1e-3)
    assert sun.dec.deg == pytest.approx(-23.005, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------- test_local_sidereal_time ------------------ #
# ============================================================= #
def test_local_sidereal_time():
    lst_fast = local_sidereal_time(
        time=Time("2022-01-01T12:00:00"),
        fast_compute=True
    )
    assert isinstance(lst_fast, Longitude)
    assert lst_fast.deg == pytest.approx(283.315, 1e-3)

    lst_slow = local_sidereal_time(
        time=Time("2022-01-01T12:00:00"),
        fast_compute=False
    )
    assert isinstance(lst_slow, Longitude)
    assert lst_slow.deg == pytest.approx(283.311, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- test_hour_angle ---------------------- #
# ============================================================= #
def test_hour_angle():
    ha_fast = hour_angle(
        radec=SkyCoord(300, 45, unit="deg"),
        time=Time("2022-01-01T12:00:00"),
        fast_compute=True
    )
    assert isinstance(ha_fast, Longitude)
    assert ha_fast.deg == pytest.approx(343.315, 1e-3)

    ha_slow = hour_angle(
        radec=SkyCoord(300, 45, unit="deg"),
        time=Time("2022-01-01T12:00:00"),
        fast_compute=False
    )
    assert isinstance(ha_slow, Longitude)
    assert ha_slow.deg == pytest.approx(343.311, 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- test_radec_to_altaz -------------------- #
# ============================================================= #
def test_radec_to_altaz():
    altaz_fast = radec_to_altaz(
        radec=SkyCoord(300, 45, unit="deg"),
        time=Time("2022-01-01T12:00:00"),
        fast_compute=True
    )
    assert isinstance(altaz_fast, SkyCoord)
    assert altaz_fast.az.deg == pytest.approx(95.036, 1e-3)
    assert altaz_fast.alt.deg == pytest.approx(78.132, 1e-3)

    altaz_slow = radec_to_altaz(
        radec=SkyCoord(300, 45, unit="deg"),
        time=Time("2022-01-01T12:00:00"),
        fast_compute=False
    )
    assert isinstance(altaz_fast, SkyCoord)
    assert altaz_slow.az.deg == pytest.approx(95.046, 1e-3)
    assert altaz_slow.alt.deg == pytest.approx(78.136, 1e-3)

    altaz_array_1 = radec_to_altaz(
        radec=SkyCoord([300, 200], [45, 45], unit="deg"),
        time=Time("2022-01-01T12:00:00"),
        fast_compute=True
    )
    assert altaz_array_1.size == 2

    altaz_array_2 = radec_to_altaz(
        radec=SkyCoord([300, 200], [45, 45], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00", "2022-01-01T16:00:00"]),
        fast_compute=True
    )
    assert altaz_array_2.shape == (3, 2)

    altaz_array_3 = radec_to_altaz(
        radec=SkyCoord([300, 200, 100], [45, 45, 45], unit="deg"),
        time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
        fast_compute=True
    )
    assert altaz_array_3.shape == (2, 3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- test_altaz_to_radec -------------------- #
# ============================================================= #
def test_altaz_to_radec():
    radec_fast = altaz_to_radec(
        altaz=SkyCoord(
            300, 45, unit="deg",
            frame=AltAz(
                obstime=Time("2022-01-01T12:00:00"),
                location=nenufar_position
            )
        ),
        fast_compute=True
    )
    assert isinstance(radec_fast, SkyCoord)
    assert radec_fast.ra.deg == pytest.approx(212.967, 1e-3)
    assert radec_fast.dec.deg == pytest.approx(49.440, 1e-3)

    radec_slow = altaz_to_radec(
        altaz=SkyCoord(
            300, 45, unit="deg",
            frame=AltAz(
                obstime=Time("2022-01-01T12:00:00"),
                location=nenufar_position
            )
        ),
        fast_compute=False
    )
    assert isinstance(radec_slow, SkyCoord)
    assert radec_slow.ra.deg == pytest.approx(212.764, 1e-3)
    assert radec_slow.dec.deg == pytest.approx(49.546, 1e-3)

    radec_array = altaz_to_radec(
        altaz=SkyCoord(
            [100, 300], [45, 45], unit="deg",
            frame=AltAz(
                obstime=Time("2022-01-01T12:00:00"),
                location=nenufar_position
            )
        ),
        fast_compute=True
    )
    assert radec_array.shape == (2,)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------- test_sky_temperature -------------------- #
# ============================================================= #
def test_sky_temperature():
    temp = sky_temperature(frequency=50*u.MHz)
    assert isinstance(temp, u.Quantity)
    assert temp.to(u.K).value == pytest.approx(5776.5765, 1e-4)

    temp_array = sky_temperature(frequency=[30, 50]*u.MHz)
    assert temp_array.size == 2
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------- test_dispersion_delay ------------------- #
# ============================================================= #
def test_dispersion_delay():
    dt = dispersion_delay(
        frequency=50*u.MHz,
        dispersion_measure=12.4*u.pc/(u.cm**3)
    )
    assert isinstance(dt, u.Quantity)
    assert dt.to(u.s).value == pytest.approx(20.5344, 1e-4)

    dt_array = dispersion_delay(
        frequency=[30, 50]*u.MHz,
        dispersion_measure=12.4*u.pc/(u.cm**3)
    )
    assert dt_array.size == 2

    freqs = [30, 40, 50]*u.MHz
    dms = [12.4, 14]*u.pc/(u.cm**3)
    dt_array_2 = dispersion_delay(
        frequency=freqs[:, None],
        dispersion_measure=dms[None, :]
    )
    assert dt_array_2.shape == (3, 2)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- test_wavelength ---------------------- #
# ============================================================= #
def test_wavelength():
    wvl = wavelength(30*u.MHz)
    assert isinstance(wvl, u.Quantity)
    assert wvl.to(u.m).value == pytest.approx(9.993, 1e-3)

    wvl_array = wavelength([10, 20, 30]*u.MHz)
    assert wvl_array.shape == (3,)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_l93_to_etrs ---------------------- #
# ============================================================= #
def test_l93_to_etrs():
    l93 = np.array([
        [6.39113316e+05, 6.69766347e+06, 1.81735000e+02],
        [6.39094578e+05, 6.69764471e+06, 1.81750000e+02]
    ])
    etrs = l93_to_etrs(positions=l93)
    assert etrs.shape == (2, 3)
    assert etrs[1, 2] == pytest.approx(4670332.180, 1e-3)

    with pytest.raises(ValueError):
        l93_to_etrs(
            np.array([6.39113316e+05, 6.69766347e+06, 1.81735000e+02])
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_geo_to_etrs ---------------------- #
# ============================================================= #
def test_geo_to_etrs():
    nenufar_etrs = geo_to_etrs(location=nenufar_position)
    assert nenufar_etrs.shape == (1, 3)
    assert nenufar_etrs[0, 1] == pytest.approx(165533.668, 1e-3)

    positions = EarthLocation(
        lat=[30, 40] * u.deg,
        lon=[0, 10] * u.deg,
        height=[100, 200] * u.m
    )
    arrays_etrs = geo_to_etrs(location=positions)
    assert arrays_etrs.shape == (2, 3)
    assert arrays_etrs[1, 2] == pytest.approx(4078114.130, 1e-3)

    # l93_to_etrs and geo_to_etrs should give the same results (MA 00)
    etrs_from_l93 = l93_to_etrs(
        positions=np.array([
            6.39113316e+05, 6.69766347e+06, 1.81735000e+02
        ]).reshape(1, 3)
    )
    etrs_from_geo = geo_to_etrs(
        EarthLocation(
            lat=47.37650985*u.deg,
            lon=2.19307873*u.deg,
            height=181.7350*u.m
        )
    )
    assert etrs_from_l93[0, 0] == pytest.approx(etrs_from_geo[0, 0], 1e-3)
    assert etrs_from_l93[0, 1] == pytest.approx(etrs_from_geo[0, 1], 1e-3)
    assert etrs_from_l93[0, 2] == pytest.approx(etrs_from_geo[0, 2], 1e-3)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_etrs_to_enu ---------------------- #
# ============================================================= #
def test_etrs_to_enu():
    etrs_positions = np.array([
        [4323934.57369062,  165585.71569665, 4670345.01314493],
        [4323949.24009871,  165567.70236494, 4670332.18016874]
    ])
    enu = etrs_to_enu(
        positions=etrs_positions,
        location=nenufar_position
    )
    assert enu.shape == (2, 3)
    assert enu[1, 1] == pytest.approx(-19.09, 1e-2)
# ============================================================= #
# ============================================================= #

