#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS, AltAz
from unittest.mock import patch

from nenupy.beam import Beam, ABeam, DBeam

# ============================================================= #
# --------------------- test_beam_inputs ---------------------- #
# ============================================================= #
def test_beam_inputs():
    # Standard inputs
    beam = Beam(
        freq=50,
        polar='NW'
    )
    assert isinstance(beam.freq, u.Quantity)
    assert beam.freq.to(u.kHz).value == 50000.0
    assert beam.polar == 'NW'
    beam = Beam(
        freq=60*u.MHz,
        polar='NE'
    )
    assert beam.freq.to(u.kHz).value == 60000.0
    # Errors
    with pytest.raises(ValueError):
        beam.freq = [10, 20]
    with pytest.raises(ValueError):
        beam.polar = ['NW', 'NE']
    with pytest.raises(ValueError):
        beam.polar = 'wrong pol'
    with pytest.raises(AttributeError):
        beam.polar = 2
# ============================================================= #


# ============================================================= #
# --------------------- test_beam_antgain --------------------- #
# ============================================================= #
def test_beam_antgain():
    beam = Beam(
        freq=50,
        polar='NW'
    )
    # Errors
    with pytest.raises(TypeError):
        gain = beam.ant_gain(
            coords=2,
            time=Time('2020-04-01 12:00:00')
        )
    with pytest.raises(TypeError):
        gain = beam.ant_gain(
            coords=ICRS(0*u.deg, 90*u.deg),
            time='2020-04-01 12:00:00'
        )
    # Check returns
    gain = beam.ant_gain(
        coords=ICRS(0*u.deg, 90*u.deg),
        time=Time('2020-04-01 12:00:00')
    )
    assert isinstance(gain, np.float64)
    assert gain == pytest.approx(0.62, 1e-2)
    gain = beam.ant_gain(
        coords=SkyCoord(
            ra=np.array([10, 20, 30])*u.deg,
            dec=np.array([50, 60, 80])*u.deg
        ),
        time=Time('2020-04-01 12:00:00')
    )
    assert isinstance(gain, np.ndarray)
    expected = np.array([0.998, 0.945, 0.738])
    assert gain == pytest.approx(expected, 1e-3)
# ============================================================= #


# ============================================================= #
# ------------------- test_beam_arrayfactor ------------------- #
# ============================================================= #
def test_beam_arrayfactor():
    positions = np.array([
            [0., 0., 0.],
            [5., 5., 0.],
            [-5., -5., 0.],
            [-5., 5., 0.],
            [5., -5., 0.],
        ])
    beam = Beam(
        freq=50,
        polar='NW'
    )
    # Errors
    with pytest.raises(TypeError):
        af = beam.array_factor(
            phase_center='wrong_type',
            coords=AltAz(
                az=180*u.deg,
                alt=90*u.deg,
                obstime=Time('2020-04-01 12:00:00')
            ),
            antpos=positions
        )
    with pytest.raises(TypeError):
        af = beam.array_factor(
            phase_center=AltAz(
                az=180*u.deg,
                alt=90*u.deg,
                obstime=Time('2020-04-01 12:00:00')
            ),
            coords='wrong_type',
            antpos=positions
        )
    with pytest.raises(TypeError):
        af = beam.array_factor(
            phase_center=AltAz(
                az=180*u.deg,
                alt=90*u.deg,
                obstime=Time('2020-04-01 12:00:00')
            ),
            coords=AltAz(
                az=180*u.deg,
                alt=90*u.deg,
                obstime=Time('2020-04-01 12:00:00')
            ),
            antpos='wrong_type'
        )
    with pytest.raises(IndexError):
        af = beam.array_factor(
            phase_center=AltAz(
                az=180*u.deg,
                alt=90*u.deg,
                obstime=Time('2020-04-01 12:00:00')
            ),
            coords=AltAz(
                az=180*u.deg,
                alt=90*u.deg,
                obstime=Time('2020-04-01 12:00:00')
            ),
            antpos=positions.T
        )
    # Check returns
    az = np.linspace(0, 360, 10)
    alt = np.linspace(0, 90, 10)
    az, alt = np.meshgrid(az, alt)
    af = beam.array_factor(
        phase_center=AltAz(
            az=180*u.deg,
            alt=90*u.deg,
            obstime=Time('2020-04-01 12:00:00')
        ),
        coords=AltAz(
            az=az.ravel()*u.deg,
            alt=alt.ravel()*u.deg,
            obstime=Time('2020-04-01 12:00:00')
        ),
        antpos=positions
    )
    assert isinstance(af, np.ndarray)
    assert af.size == 100
    assert af[20] == pytest.approx(3.38, 1e-2)
# ============================================================= #


# ============================================================= #
# --------------------- test_abeam_inputs --------------------- #
# ============================================================= #
def test_abeam_inputs():
    ana = ABeam(
        freq=50,
        polar='NW', 
        azana=180,
        elana=90,
        ma=0
    )
    assert isinstance(ana.azana, u.Quantity)
    assert ana.azana.value == 180.0
    assert isinstance(ana.elana, u.Quantity)
    assert ana.elana.value == 90.0
    assert isinstance(ana.polar, str)
    assert ana.polar == 'NW'
    assert isinstance(ana.ma, int)
    assert ana.ma == 0
    # squint_freq
    with pytest.raises(ValueError):
        ana.squint_freq = [30, 40]
    # ma
    with pytest.raises(TypeError):
        ana.ma = 'wrong'
    with pytest.raises(ValueError):
        ana.ma = 200
    # beamsquint
    with pytest.raises(TypeError):
        ana.beamsquint = 'wrong'
# ============================================================= #


# ============================================================= #
# --------------------- test_abeam_values --------------------- #
# ============================================================= #
def test_abeam_values():
    with patch('nenupy.beam.beam.analog_pointing') as mock_anapoint:
        mock_anapoint.return_value = (180*u.deg, 45*u.deg)
        ana = ABeam(
            freq=50,
            polar='NW', 
            azana=180,
            elana=45,
            ma=0
        )
        vals = ana.beam_values(
            coords=SkyCoord([0, 10]*u.deg, [90, 80]*u.deg),
            time=Time('2020-04-01 12:00:00')
        )
        assert isinstance(vals, np.ndarray)
        expected = np.array([108.93, 272.41])
        assert vals == pytest.approx(expected, 1e-2)
        ana.beamsquint = False
        vals = ana.beam_values(
            coords=SkyCoord([0, 10]*u.deg, [90, 80]*u.deg),
            time=Time('2020-04-01 12:00:00')
        )
        expected = np.array([108.93, 272.41])
        assert vals == pytest.approx(expected, 1e-2)
# ============================================================= #


# ============================================================= #
# --------------------- test_dbeam_inputs --------------------- #
# ============================================================= #
def test_dbeam_inputs():
    digi = DBeam(
        freq=50,
        polar='NW',
        azdig=180,
        eldig=90,
        ma=np.arange(2),
        azana=None,
        elana=None,
        squint_freq=30,
        beamsquint=True
    )
    assert isinstance(digi.azdig, u.Quantity)
    assert digi.azdig.value == 180.0
    assert isinstance(digi.eldig, u.Quantity)
    assert digi.eldig.value == 90.0
    assert isinstance(digi.ma, np.ndarray)
    assert all(digi.ma == np.array([0, 1], dtype=int))
    digi.ma = [1, 2, 3]
    assert all(digi.ma == np.array([1, 2, 3], dtype=int))
    with pytest.raises(ValueError):
        digi.ma = 0
    with pytest.raises(ValueError):
        digi.ma = np.array([0, 200])
# ============================================================= #


# ============================================================= #
# --------------------- test_abeam_values --------------------- #
# ============================================================= #
def test_dbeam_values():
    with patch('nenupy.beam.beam.analog_pointing') as mock_anapoint:
        mock_anapoint.return_value = (180*u.deg, 45*u.deg)
        digi = DBeam(
            freq=50,
            polar='NW',
            azdig=180,
            eldig=90,
            ma=np.arange(10),
            azana=None,
            elana=None,
            squint_freq=30,
            beamsquint=True
        )
        vals = digi.beam_values(
            coords=SkyCoord([0, 10]*u.deg, [90, 80]*u.deg),
            time=Time('2020-04-01 12:00:00')
        )
        assert isinstance(vals, np.ndarray)
        expected = np.array([15153.35, 6942.92])
        assert vals == pytest.approx(expected, 1e-2)
# ============================================================= #

