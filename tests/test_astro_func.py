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
    eq_zenith
)


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


