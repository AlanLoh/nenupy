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
import astropy.units as u
import numpy as np

from nenupy.instru import (
    freq2sb,
    sb2freq,
    miniarrays_rotated_like,
    read_cal_table,
    generate_nenufar_subarrays
)


# ============================================================= #
# ----------------------- test_freq2sb ------------------------ #
# ============================================================= #
def test_freq2sb():
    sbs = freq2sb(frequency=50*u.MHz)
    assert sbs == 256
    sbs = freq2sb(frequency=[20.1, 20.3]*u.MHz)
    assert np.all(sbs == np.array([103, 104]))
    with pytest.raises(TypeError):
        freq2sb(frequency=50)
# ============================================================= #
# ============================================================= #

# ============================================================= #
# ----------------------- test_sb2freq ----------------------- #
# ============================================================= #
def test_sb2freq():
    freqs = sb2freq(subband=256)
    assert np.testing.assert_allclose(
            freqs.to(u.MHz).value,
            np.array([49.902344]),
            atol=1e-2
        ) is None # other an AssertionError is raised
    freqs = sb2freq([34, 500])
    assert np.testing.assert_allclose(
            freqs.to(u.MHz).value,
            np.array([6.5429688, 97.558594]),
            atol=1e-2
        ) is None # other an AssertionError is raised
    
    with pytest.raises(TypeError):
        sb2freq(subband="a")
    with pytest.raises(ValueError):
        sb2freq(subband=[34, 543])
# ============================================================= #
# ============================================================= #

# ============================================================= #
# --------------- test_miniarrays_rotated_like ---------------- #
# ============================================================= #
def test_miniarrays_rotated_like():
    mas = miniarrays_rotated_like([0])
    assert mas.size == 32
    assert miniarrays_rotated_like([10])[0] == 11
# ============================================================= #
# ============================================================= #

# ============================================================= #
# -------------------- test_read_cal_table -------------------- #
# ============================================================= #
def test_read_cal_table():
    cal = read_cal_table(calibration_file=None)
    assert cal.shape == (512, 96, 2)
# ============================================================= #
# ============================================================= #

# ============================================================= #
# -------------- test_generate_nenufar_subarrays -------------- #
# ============================================================= #
def test_generate_nenufar_subarrays():
    sub_arrays = generate_nenufar_subarrays(
        n_subarrays=2,
        include_remote_mas=True
    )
    assert sum(map(len, sub_arrays)) == 102
    sub_arrays = generate_nenufar_subarrays(
        n_subarrays=3,
        include_remote_mas=False
    )
    assert sum(map(len, sub_arrays)) == 96
# ============================================================= #
# ============================================================= #

