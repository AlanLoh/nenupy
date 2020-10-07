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
    Longitude,
    AltAz,
    ICRS,
    SkyCoord
)
from astropy.time import Time, TimeDelta
import pytest

from nenupy.astro import (
    lst,
    lha,
    toFK5,
    wavelength,
    ho_coord,
    eq_coord,
    to_radec,
    toAltaz,
    ho_zenith,
    eq_zenith,
    radio_sources,
    meridianTransit,
    dispersion_delay
)


# ============================================================= #
# ------------------------- test_lst -------------------------- #
# ============================================================= #
def test_lst():
    with pytest.raises(TypeError):
        lst_time = lst(
            time='2020-04-01 12:00:00',
            kind=1
        )
    with pytest.raises(TypeError):
        lst_time = lst(
            time='2020-24-01 12:00:00',
            kind='fast'
        )
    lst_time = lst(
        time=Time('2020-04-01 12:00:00'),
        kind='fast'
    )
    assert isinstance(lst_time, Longitude)
    assert lst_time.deg == pytest.approx(12.50, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------------- test_tofk5 ------------------------ #
# ============================================================= #
def test_tofk5():
    casa_icrs = SkyCoord(
        350.850000,
        +58.815000,
        unit='deg',
        frame='icrs'
    )
    time = Time('2020-04-01 12:00:00')
    with pytest.raises(TypeError):
        srcfk5 = toFK5(
            skycoord=1,
            time=time
        )
    with pytest.raises(TypeError):
        srcfk5 = toFK5(
            skycoord=casa_icrs,
            time=1
        )
    srcfk5 = toFK5(
        skycoord=casa_icrs,
        time=time
    )
    assert isinstance(srcfk5, SkyCoord)
    assert isinstance(srcfk5.equinox, Time)
    assert srcfk5.equinox.jd == 2458941.0
    assert srcfk5.ra.deg == pytest.approx(351.0801, 1e-4)
    assert srcfk5.dec.deg == pytest.approx(58.9263, 1e-4)
# ============================================================= #


# ============================================================= #
# ------------------------- test_lha -------------------------- #
# ============================================================= #
def test_lha():
    time = Time('2020-04-01 12:00:00')
    lstTime = lst(
        time=time,
        kind='apparent'
    )
    src1 = SkyCoord(
        0.0,
        +45.0,
        unit='deg',
        frame='icrs'
    )
    src2 = SkyCoord(
        300.0,
        +45.0,
        unit='deg',
        frame='icrs'
    )
    src1fk5 = toFK5(src1, time)
    src2fk5 = toFK5(src2, time)
    with pytest.raises(TypeError):
        hour_angle = lha(
            lst=1,
            skycoord=src1
        )
    with pytest.raises(TypeError):
        hour_angle = lha(
            lst=lstTime,
            skycoord=1
        )

    hour_angle = lha(
        lst=lstTime,
        skycoord=src1fk5
    )
    assert isinstance(hour_angle, Angle)
    assert hour_angle.deg == pytest.approx(12.2358, 1e-4)
    # Negative hourangle
    hour_angle = lha(
        lst=lstTime,
        skycoord=src2fk5
    )
    assert hour_angle.deg == pytest.approx(72.3337, 1e-4)
    # Non scalar object
    dts = TimeDelta(np.arange(0, 180, 60), format='sec')
    lstTime = lst(
        time=time + dts,
        kind='apparent'
    )
    src1fk5 = toFK5(src1, time + dts)
    hour_angle = lha(
        lst=lstTime,
        skycoord=src1fk5
    )
    ha = np.array([12.2358, 12.4865, 12.7372])
    assert hour_angle.deg == pytest.approx(ha, 1e-4)
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
    source = SkyCoord(180, 45, unit='deg')
    time = Time('2020-04-01 12:00:00')
    with pytest.raises(TypeError):
        altaz = toAltaz(
            skycoord=1,
            time=Time.now()
        )
    with pytest.raises(TypeError):
        altaz = toAltaz(
            skycoord=source,
            time=1
        )
    with pytest.raises(TypeError):
        altaz = toAltaz(
            skycoord=source,
            time=time,
            kind=source
        )
    altaz = toAltaz(
        skycoord=source,
        time=time,
        kind='fast'
    )
    assert isinstance(altaz, SkyCoord)
    assert altaz.az.deg == pytest.approx(8.651, 1e-3)
    assert altaz.alt.deg == pytest.approx(2.8895, 1e-4)
    altaz = toAltaz(
        skycoord=source,
        time=time,
        kind='normal'
    )
    assert isinstance(altaz, SkyCoord)
    assert altaz.az.deg == pytest.approx(8.645, 1e-3)
    assert altaz.alt.deg == pytest.approx(2.8898, 1e-4)
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
    src = SkyCoord.from_name('3C 274')
    time = Time('2020-09-10')
    with pytest.raises(TypeError):
        meridianTransit(
            source=1,
            fromTime=time,
            duration=TimeDelta(1)
        )
    with pytest.raises(TypeError):
        meridianTransit(
            source=src,
            fromTime=1,
            duration=TimeDelta(1)
        )
    with pytest.raises(TypeError):
        meridianTransit(
            source=src,
            fromTime=time,
            duration=1
        )
    with pytest.raises(TypeError):
        meridianTransit(
            source=src,
            fromTime=time,
            duration=TimeDelta(1),
            kind=1
        )
    transits = meridianTransit(
        source=src,
        fromTime=time,
        duration=TimeDelta(1),
        kind='apparent'
    )
    assert isinstance(transits, Time)
    assert transits.size == 1
    assert transits[0].isot == '2020-09-10T13:03:00.500'
    # Double transit
    casa = SkyCoord.from_name('Cas A')
    time = Time('2020-09-09')
    transits = meridianTransit(
        source=casa,
        fromTime=time,
        duration=TimeDelta(1),
        kind='apparent'
    )
    assert transits.size == 2
    assert transits[0].isot == '2020-09-09T00:01:34.500'
    assert transits[1].isot == '2020-09-09T23:57:38.500'
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

