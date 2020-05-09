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
from unittest.mock import patch

from nenupy.beam.hpxbeam import HpxBeam
from nenupy.beam import HpxABeam, HpxDBeam


@pytest.fixture(scope='module')
def lowresbeam():
    lowb = HpxBeam(
        resolution=20*u.deg
    )
    lowb.time = '2020-04-01 12:00:00'
    return lowb

# ============================================================= #
# --------------------- test_ana_pointing --------------------- #
# ============================================================= #
def test_ana_pointing(lowresbeam):
    with pytest.raises(ValueError):
        lowresbeam.azana = None
    with pytest.raises(ValueError):
        lowresbeam.elana = None
    # Float input
    lowresbeam.azana = 180
    lowresbeam.elana = 45
    # Quanity input
    lowresbeam.azana = 180*u.deg
    lowresbeam.elana = 45*u.deg
    assert isinstance(lowresbeam.azana, u.Quantity)
    assert isinstance(lowresbeam.elana, u.Quantity)
    assert lowresbeam.azana.to(u.deg).value == 180.
    assert lowresbeam.elana.to(u.deg).value == 45.
# ============================================================= #


# ============================================================= #
# --------------------- test_dig_pointing --------------------- #
# ============================================================= #
def test_dig_pointing(lowresbeam):
    lowresbeam.azana = 180
    lowresbeam.elana = 45
    # No selection
    lowresbeam.azdig = None
    lowresbeam.eldig = None
    assert lowresbeam.azdig.to(u.deg).value == 180.
    assert lowresbeam.eldig.to(u.deg).value == 45. 
    # Float input
    lowresbeam.azdig = 180
    lowresbeam.eldig = 45
    # Quanity input
    lowresbeam.azdig = 180*u.deg
    lowresbeam.eldig = 45*u.deg
    assert isinstance(lowresbeam.azdig, u.Quantity)
    assert isinstance(lowresbeam.eldig, u.Quantity)
    assert lowresbeam.azdig.to(u.deg).value == 180.
    assert lowresbeam.eldig.to(u.deg).value == 45.
# ============================================================= #


# ============================================================= #
# --------------------- test_arrayfactor ---------------------- #
# ============================================================= #
def test_arrayfactor(lowresbeam):
    positions = np.array([
        [0., 0., 0.],
        [5., 5., 0.],
        [-5., -5., 0.],
        [-5., 5., 0.],
        [5., -5., 0.],
    ])
    # Float input
    af = lowresbeam.array_factor(
        az=0,
        el=90,
        antpos=positions,
        freq=50
    )
    assert isinstance(af, np.ndarray)
    assert af.size == 96
    assert af[48] == pytest.approx(0.397, 1e-3)
    # Multi-process and Quanity inputs
    lowresbeam.ncpus = 2
    af = lowresbeam.array_factor(
        az=0*u.deg,
        el=90*u.deg,
        antpos=positions,
        freq=50*u.MHz
    )
    assert af.size == 96
    assert af[48] == pytest.approx(0.397, 1e-3)
# ============================================================= #


# ============================================================= #
# -------------------- test_radialprofile --------------------- #
# ============================================================= #
def test_radialprofile(lowresbeam):
    # It should be run after test_arrayfactor()
    sep, profile = lowresbeam.radial_profile(
        da=20*u.deg
    )
    assert isinstance(sep, u.Quantity)
    assert isinstance(profile, np.ndarray)
    sep_expected = np.array([15.5, 35.5, 55.5, 75.5, 95.5])
    assert sep.to(u.deg).value == pytest.approx(sep_expected, 1e-1)
    assert all(profile == 0.) # empty skymap
    try:
        del lowresbeam.phase_center
    except AttributeError:
        pass
    with pytest.raises(Exception):
        sep, profile = lowresbeam.radial_profile()
# ============================================================= #


# ============================================================= #
# ---------------------- test_hpxanabeam ---------------------- #
# ============================================================= #
def test_hpxanabeam():
    with patch('nenupy.beam.hpxbeam.analog_pointing') as mock_anapoint:
        mock_anapoint.return_value = (180*u.deg, 90*u.deg)
        ana = HpxABeam(resolution=5)
        ana.beam(
            azana=0,
            elana=45,
            time=Time('2020-04-01 12:00:00')
        )
        assert ana.skymap.size == 3072
        assert ana.skymap[1000] == pytest.approx(6.288, 1e-3)
# ============================================================= #

