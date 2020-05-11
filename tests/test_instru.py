#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import astropy.units as u
import numpy as np
import pytest

from nenupy.instru import (
    analog_pointing,
    desquint_elevation,
    nenufar_ant_gain,
    read_cal_table,
    effective_area,
    sky_temperature,
    inst_temperature,
    sefd,
    sensitivity,
    resolution,
    confusion_noise,
    data_rate
)


# ============================================================= #
# -------------------- test_instru_analog --------------------- #
# ============================================================= #
# Cannot CI test this function because the NenuFAR_thph.fits file
# is not hosted on github
@pytest.mark.skip
def test_instru_analog():
    tol = 1e-2
    # Float inputs
    az, el = analog_pointing(
        azimuth=180,
        elevation=45
    )
    assert isinstance(az, u.Quantity)
    assert isinstance(el, u.Quantity)
    assert az.to(u.deg).value == pytest.approx(180.00, tol)
    assert el.to(u.deg).value == pytest.approx(45.17, tol)
    # Quantity inputs
    az, el = analog_pointing(
        azimuth=180*u.deg,
        elevation=45*u.deg
    )
    assert isinstance(az, u.Quantity)
    assert isinstance(el, u.Quantity)
    assert az.to(u.deg).value == pytest.approx(180.00, tol)
    assert el.to(u.deg).value == pytest.approx(45.17, tol)
    # Non scalar inputs
    az, el = analog_pointing(
        azimuth=np.array([0, 180]),
        elevation=np.array([20, 45])
    )
    assert isinstance(az, u.Quantity)
    assert isinstance(el, u.Quantity)
    azs = np.array([0., 180.])
    els = np.array([19.09, 45.17])
    assert all(az.to(u.deg).value == azs)
    assert el.to(u.deg).value == pytest.approx(els, tol)
# ============================================================= #


# ============================================================= #
# ------------------- test_instru_desquint -------------------- #
# ============================================================= #
def test_instru_desquint():
    tol = 1e-2
    # Float inputs
    el = desquint_elevation(
        elevation=20,
        opt_freq=80
    )
    assert isinstance(el, u.Quantity)
    assert el.to(u.deg).value == pytest.approx(20.00, tol)
    # Quantity inputs
    el = desquint_elevation(
        elevation=45*u.deg,
        opt_freq=80*u.MHz
    )
    assert isinstance(el, u.Quantity)
    assert el.to(u.deg).value == pytest.approx(44.36, tol)
    # Non scalar inputs
    el = desquint_elevation(
        elevation=np.array([30, 45])*u.deg,
        opt_freq=80*u.MHz
    )
    assert isinstance(el, u.Quantity)
    els = np.array([28.36, 44.36])
    assert el.to(u.deg).value == pytest.approx(els, tol)
# ============================================================= #


# ============================================================= #
# -------------------- test_instru_antgain -------------------- #
# ============================================================= #
def test_instru_antgain():
    with pytest.raises(ValueError):
        antgain = nenufar_ant_gain(
            freq=5, # too low frequency
            polar='NW',
            nside=4,
            time=None
        )
    with pytest.raises(ValueError):
        antgain = nenufar_ant_gain(
            freq=50,
            polar='wrongpolar',
            nside=4,
            time=None
        )
    antgain = nenufar_ant_gain(
        freq=50,
        polar='NW',
        nside=4,
        time='2020-04-01 12:00:00'
    )
    assert isinstance(antgain, np.ndarray)
    assert antgain.size == 192
    assert antgain[10] == pytest.approx(0.68, 1e-2)
    antgain = nenufar_ant_gain(
        freq=55*u.MHz,
        polar='NE',
        nside=4,
        time='2020-04-01 12:00:00'
    )
    assert antgain[10] == pytest.approx(0.56, 1e-2)
    # Above 80 MHz
    antgain = nenufar_ant_gain(
        freq=90,
        polar='NW',
        nside=4,
        time='2020-04-01 12:00:00'
    )
    assert antgain[10] == pytest.approx(0.68, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------- test_instru_caltable -------------------- #
# ============================================================= #
def test_instru_caltable():
    cal = read_cal_table(
        calfile='default'
    )
    assert isinstance(cal, np.ndarray)
    assert cal.shape == (512, 96, 2)
    assert cal[0, 0, 0] == pytest.approx(0.99 + 0.14j, 1e-2)
# ============================================================= #


# ============================================================= #
# -------------------- test_instru_effarea -------------------- #
# ============================================================= #
def test_instru_effarea():
    with pytest.raises(ValueError):
        effa = effective_area(
            freq=50,
            antennas=np.arange(25), # wrong antennas
            miniarrays=None
        )
    with pytest.raises(ValueError):
        effa = effective_area(
            freq=50,
            antennas=None,
            miniarrays=np.arange(200) # wrong mas
        )
    effa = effective_area(
        freq=50,
        antennas=None,
        miniarrays=None
    )
    assert isinstance(effa, u.Quantity)
    # Possible configurations tests
    effa = effective_area(
        freq=50*u.MHz,
        antennas=10,
        miniarrays=0
    )
    assert effa.to(u.m**2).value == pytest.approx(11.98, 1e-2)
    effa = effective_area(
        freq=50*u.MHz,
        antennas=[9, 10],
        miniarrays=0
    )
    assert effa.to(u.m**2).value == pytest.approx(23.96, 1e-2)
    effa = effective_area(
        freq=50,
        antennas=[9, 10],
        miniarrays=[0, 1, 2]
    )
    assert effa.to(u.m**2).value == pytest.approx(71.89, 1e-2)
# ============================================================= #


# ============================================================= #
# -------------------- test_instru_skytemp -------------------- #
# ============================================================= #
def test_instru_skytemp():
    temp = sky_temperature(freq=50)
    assert isinstance(temp, u.Quantity)
    assert temp.to(u.K).value == pytest.approx(5776.58, 1e-2)
    temp = sky_temperature(freq=66*u.MHz)
    assert isinstance(temp, u.Quantity)
    assert temp.to(u.K).value == pytest.approx(2845.82, 1e-2)
# ============================================================= #


# ============================================================= #
# -------------------- test_instru_instemp -------------------- #
# ============================================================= #
def test_instru_instemp():
    temp = inst_temperature(freq=50)
    assert isinstance(temp, u.Quantity)
    assert temp.to(u.K).value == pytest.approx(643.51, 1e-2)
    temp = inst_temperature(freq=66*u.MHz)
    assert isinstance(temp, u.Quantity)
    assert temp.to(u.K).value == pytest.approx(564.89, 1e-2)
# ============================================================= #


# ============================================================= #
# --------------------- test_instru_sefd ---------------------- #
# ============================================================= #
def test_instru_sefd():
    se = sefd(freq=50, antennas=10, miniarrays=[0, 1])
    assert isinstance(se, u.Quantity)
    assert se.to(u.kJy).value == pytest.approx(739.74, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------ test_instru_sensitivity ------------------ #
# ============================================================= #
def test_instru_sensitivity():
    with pytest.raises(ValueError):
        sens = sensitivity(
            mode='imaging',
            freq=50,
            antennas=None,
            miniarrays=np.arange(200),
            dt=10,
            df=3
        )
    with pytest.raises(ValueError):
        sens = sensitivity(
            mode='unknown mode',
            freq=50,
            antennas=None,
            miniarrays=None,
            dt=10,
            df=3
        )
    with pytest.raises(ValueError):
        sens = sensitivity(
            mode='beamforming',
            freq=50,
            antennas=None,
            miniarrays=np.arange(200),
            dt=10,
            df=3
        )
    sens = sensitivity(
        mode='imaging',
        freq=50,
        antennas=None,
        miniarrays=None,
        dt=1,
        df=3
    )
    assert isinstance(sens, u.Quantity)
    sens = sensitivity(
        mode='imaging',
        freq=50,
        antennas=10,
        miniarrays=[0, 1],
        dt=10,
        df=3
    )
    assert sens.to(u.Jy).value == pytest.approx(67.53, 1e-2)
    sens = sensitivity(
        mode='beamforming',
        freq=50*u.MHz,
        antennas=[9, 10],
        miniarrays=0,
        dt=60*u.s,
        df=10*u.MHz
    )
    assert sens.to(u.Jy).value == pytest.approx(30.20, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------ test_instru_resolution ------------------- #
# ============================================================= #
def test_instru_resolution():
    with pytest.raises(ValueError):
        res = resolution(
            freq=50,
            miniarrays=np.arange(200),
        )
    res = resolution(
        freq=50,
        miniarrays=None
    )
    assert isinstance(res, u.Quantity)
    res = resolution(
        freq=50,
        miniarrays=[0, 1]
    )
    assert isinstance(res, u.Quantity)
    assert res.to(u.deg).value == pytest.approx(15.55, 1e-2)
    res = resolution(
        freq=50*u.MHz,
        miniarrays=0
    )
    assert isinstance(res, u.Quantity)
    assert res.to(u.deg).value == pytest.approx(13.74, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------- test_instru_confusion ------------------- #
# ============================================================= #
def test_instru_confusion():
    conf = confusion_noise(
        freq=50,
        miniarrays=[0, 1]
    )
    assert isinstance(conf, u.Quantity)
    assert conf.to(u.Jy).value == pytest.approx(1417.02, 1e-2)
    conf = confusion_noise(
        freq=80*u.MHz,
        miniarrays=np.arange(30)
    )
    assert isinstance(conf, u.Quantity)
    assert conf.to(u.Jy).value == pytest.approx(3.09, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------- test_instru_datarate -------------------- #
# ============================================================= #
def test_instru_datarate():
    # errors
    with pytest.raises(ValueError):
        rate = data_rate(
            mode='unkown_mode',
        )
    with pytest.raises(TypeError):
        rate = data_rate(
            mode='imaging',
            mas='wrong type'
        )
    with pytest.raises(TypeError):
        rate = data_rate(
            mode='imaging',
            nchan='wrong type'
        )
    with pytest.raises(ValueError):
        rate = data_rate(
            mode='imaging',
            nchan=100
        )
    with pytest.raises(ValueError):
        rate = data_rate(
            mode='imaging',
            bandwidth=1000
        )
    # imaging
    rate = data_rate(
        mode='imaging',
        mas=96,
        dt=1,
        nchan=64,
        bandwidth=75
    )
    assert isinstance(rate, u.Quantity)
    assert rate.value == 3661627392.0
    assert (rate*3600*u.s).to(u.Tibyte).value == pytest.approx(11.99, 1e-2)
    # beamformed
    rate = data_rate(
        mode='beamforming',
        dt=1*u.s,
        nchan=64,
        bandwidth=75*u.MHz
    )
    assert isinstance(rate, u.Quantity)
    assert rate.value == 393216.0
    assert (rate*3600*u.s).to(u.Gibyte).value == pytest.approx(1.32, 1e-2)
    # waveform
# ============================================================= #

