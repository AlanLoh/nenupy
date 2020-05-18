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
from astropy.coordinates import (
    EarthLocation,
    Angle,
    AltAz,
    ICRS,
    SkyCoord
)
from astropy.time import Time, TimeDelta
import pytest

from nenupy.astro import (
    nenufar_loc,
    lst,
    lha,
    wavelength,
    ho_coord,
    eq_coord,
    to_radec,
    to_altaz,
    ho_zenith,
    eq_zenith,
    radio_sources,
    meridian_transit,
    dispersion_delay
)


# ============================================================= #
# ---------------------- test_nenufarloc ---------------------- #
# ============================================================= #
def test_nenufarloc():
    loc = nenufar_loc()
    assert isinstance(loc, EarthLocation)
    assert loc.lon.deg == pytest.approx(2.19, 1e-2)
    assert loc.lat.deg == pytest.approx(47.38, 1e-2)
    assert loc.height.to(u.m).value == pytest.approx(150.00, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------------- test_lst -------------------------- #
# ============================================================= #
def test_lst():
    with pytest.raises(TypeError):
        lst_time = lst(
            time='2020-04-01 12:00:00'
        )
    lst_time = lst(
        time=Time('2020-04-01 12:00:00')
    )
    assert isinstance(lst_time, Angle)
    assert lst_time.deg == pytest.approx(12.50, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------------- test_lha -------------------------- #
# ============================================================= #
def test_lha():
    with pytest.raises(TypeError):
        hour_angle = lha(
            time='2020-04-01 12:00:00',
            ra=0
        )
    # Float input
    hour_angle = lha(
        time=Time('2020-04-01 12:00:00'),
        ra=0*u.deg
    )
    # Angle input
    hour_angle = lha(
        time=Time('2020-04-01 12:00:00'),
        ra=Angle(0, unit='deg')
    )
    # Quantity input
    hour_angle = lha(
        time=Time('2020-04-01 12:00:00'),
        ra=0*u.deg
    )
    assert isinstance(hour_angle, Angle)
    assert hour_angle.deg == pytest.approx(12.50, 1e-2)
    # Negative hourangle
    hour_angle = lha(
        time=Time('2020-04-01 12:00:00'),
        ra=300*u.deg
    )
    # > 360 hourangle
    hour_angle = lha(
        time=Time('2020-04-01 12:00:00'),
        ra=-360*u.deg
    )
    # Non scalar object
    dts = TimeDelta(np.arange(2), format='sec')
    hour_angle = lha(
        time=Time('2020-04-01 12:00:00') + dts,
        ra=100*u.deg
    )
# ============================================================= #


# ============================================================= #
# ---------------------- test_wavelength ---------------------- #
# ============================================================= #
def test_wavelength():
    # Float input
    wl = wavelength(30)
    assert isinstance(wl, u.Quantity)
    assert wl.unit == 'm'
    assert wl.to(u.m).value == pytest.approx(10., 1e-2)
    # Nnumpy ndarray input
    freqs = np.array([10, 20, 30, 40])
    wavel = np.array([30, 15 ,  10,  7.5])
    wl = wavelength(freqs)
    assert isinstance(wl, u.Quantity)
    assert wl.unit == 'm'
    assert wl.to(u.m).value == pytest.approx(wavel, 1e-2)
    # Astropy Quantity input
    freqs = freqs * 1e6 * u.Hz
    wl = wavelength(freqs)
    assert isinstance(wl, u.Quantity)
    assert wl.unit == 'm'
    assert wl.to(u.m).value == pytest.approx(wavel, 1e-2)
# ============================================================= #


# ============================================================= #
# ----------------------- test_hocoord ------------------------ #
# ============================================================= #
def test_hocoord():
    # Float and string inputs
    altaz = ho_coord(
        alt=90,
        az=180,
        time='2020-04-01 12:00:00'
    )
    # Quantity and Time inputs
    altaz = ho_coord(
        alt=90*u.deg,
        az=180*u.deg,
        time=Time('2020-04-01 12:00:00')
    )
    assert isinstance(altaz, AltAz)
    assert altaz.az.deg == 180.0
    assert altaz.alt.deg == 90.0
# ============================================================= #


# ============================================================= #
# ----------------------- test_eqcoord ------------------------ #
# ============================================================= #
def test_eqcoord():
    # Float inputs
    radec = eq_coord(
        ra=180,
        dec=45
    )
    # Quantity inputs
    radec = eq_coord(
        ra=180*u.deg,
        dec=45*u.deg
    )
    assert isinstance(radec, ICRS)
    assert radec.ra.deg == 180.0
    assert radec.dec.deg == 45.0
# ============================================================= #


# ============================================================= #
# ----------------------- test_toradec ------------------------ #
# ============================================================= #
def test_toradec():
    with pytest.raises(TypeError):
        radec = to_radec(1.)
    altaz = ho_coord(
        alt=90,
        az=180,
        time='2020-04-01 12:00:00'
    )
    radec = to_radec(altaz)
    assert isinstance(radec, ICRS)
    assert radec.ra.deg == pytest.approx(12.22, 1e-2)
    assert radec.dec.deg == pytest.approx(47.27, 1e-2)
# ============================================================= #


# ============================================================= #
# ----------------------- test_toaltaz ------------------------ #
# ============================================================= #
def test_toaltaz():
    with pytest.raises(TypeError):
        altaz = to_altaz(1., '2020-04-01 12:00:00')
    radec = eq_coord(
        ra=180,
        dec=45
    )
    altaz = to_altaz(radec, Time('2020-04-01 12:00:00'))
    assert isinstance(altaz, AltAz)
    assert altaz.az.deg == pytest.approx(8.64, 1e-2)
    assert altaz.alt.deg == pytest.approx(2.89, 1e-2)
# ============================================================= #


# ============================================================= #
# ----------------------- test_hozenith ----------------------- #
# ============================================================= #
def test_hozenith():
    # String input
    zen = ho_zenith(
        time='2020-04-01 12:00:00'
    )
    assert isinstance(zen, AltAz)
    assert zen.az.deg == 0.
    assert zen.alt.deg == 90.
    # Time input
    zen = ho_zenith(
        time=Time('2020-04-01 12:00:00')
    )
    assert isinstance(zen, AltAz)
    assert zen.az.deg == 0.
    assert zen.alt.deg == 90.
    # Non scalar time input
    dts = TimeDelta(np.arange(2), format='sec')
    zen = ho_zenith(
        time=Time('2020-04-01 12:00:00') + dts
    )
    assert isinstance(zen, AltAz)
    assert all(zen.az.deg == np.array([0., 0.]))
    assert all(zen.alt.deg == np.array([90., 90.]))
# ============================================================= #


# ============================================================= #
# ----------------------- test_eqzenith ----------------------- #
# ============================================================= #
def test_eqzenith():
    tol = 1e-2
    # String input
    zen = eq_zenith(
        time='2020-04-01 12:00:00'
    )
    assert isinstance(zen, ICRS)
    assert zen.ra.deg == pytest.approx(12.22, tol)
    assert zen.dec.deg == pytest.approx(47.27, tol)
    # Time input
    zen = eq_zenith(
        time=Time('2020-04-01 12:00:00')
    )
    assert isinstance(zen, ICRS)
    assert zen.ra.deg == pytest.approx(12.22, tol)
    assert zen.dec.deg == pytest.approx(47.27, tol)
    # Non scalar time input
    dts = TimeDelta(np.arange(2), format='sec')
    zen = eq_zenith(
        time=Time('2020-04-01 12:00:00') + dts
    )
    assert isinstance(zen, ICRS)
    ras = np.array([12.2226, 12.2268])
    assert zen.ra.deg == pytest.approx(ras, 1e-4)
    decs = np.array([47.269802, 47.269804])
    assert zen.dec.deg == pytest.approx(decs, 1e-6)
# ============================================================= #


# ============================================================= #
# --------------------- test_radiosources --------------------- #
# ============================================================= #
def test_radiosources():
    with pytest.raises(ValueError):
        dts = TimeDelta(np.arange(2), format='sec')
        srcs = radio_sources(
            time=Time('2020-04-01 12:00:00') + dts
        )
    # String input
    srcs = radio_sources(
        time='2020-04-01 12:00:00'
    )
    # Time input
    srcs = radio_sources(
        time=Time('2020-04-01 12:00:00')
    )
    src_list = [
        'vira', 'cyga', 'casa', 'hera', 'hyda',
        'taua', 'sun', 'moon', 'jupiter'
    ]
    assert all(np.isin(list(srcs.keys()), src_list))
    assert isinstance(srcs['vira'], AltAz)
# ============================================================= #


# ============================================================= #
# -------------------- test_meridiantransit ------------------- #
# ============================================================= #
def test_meridiantransit():
    with pytest.raises(TypeError):
        transit = meridian_transit(
            source='wrong format',
            from_time=Time('2020-04-01 12:00:00'),
            npoints=10
        )
    with pytest.raises(TypeError):
        transit = meridian_transit(
            source=SkyCoord(ra=299.86*u.deg, dec=40.73*u.deg),
            from_time='wrong format',
            npoints=10
        )
    transit = meridian_transit(
        source=SkyCoord(ra=299.86*u.deg, dec=40.73*u.deg),
        from_time=Time('2020-04-01 12:00:00'),
        npoints=100
    )
    assert isinstance(transit, Time)
    assert transit.isot == '2020-04-02T07:07:03.968'
# ============================================================= #


# ============================================================= #
# -------------------- test_dispersiondelay ------------------- #
# ============================================================= #
def test_dispersiondelay():
    delay = dispersion_delay(
        freq=50,
        dm=12.4
    )
    assert isinstance(delay, u.Quantity)
    assert delay.value == pytest.approx(20.53, 1e-2)
    delay = dispersion_delay(
        freq=50*u.MHz,
        dm=12.4*u.pc/(u.cm**3)
    )
    assert isinstance(delay, u.Quantity)
    assert delay.value == pytest.approx(20.53, 1e-2)
# ============================================================= #

