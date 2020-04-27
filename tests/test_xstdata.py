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
from os.path import join, dirname
import pytest

from nenupy.crosslet import XST_Data
from nenupy.beamlet import SData


# ============================================================= #
# ----------------------- test_xstread ------------------------ #
# ============================================================= #
def test_xstread():
    xst = XST_Data(
        xstfile=join(dirname(__file__), 'test_data/XST.fits')
    )
    time_size = xst.times.size
    sb_size = xst.sb_idx.size
    assert sb_size == 16
    assert xst.vis.shape[:-1] == (time_size, sb_size)
# ============================================================= #


# ============================================================= #
# --------------------- test_xstbeamform ---------------------- #
# ============================================================= #
def test_xstbeamform():
    xst = XST_Data(
        xstfile=join(dirname(__file__), 'test_data/XST.fits')
    )
    bf = xst.beamform(
        az=180*u.deg,
        el=90*u.deg,
        pol='NW',
        ma=None
    )
    assert isinstance(bf, SData)
    assert bf.time[0].isot == '2020-02-19T18:00:03.000'
    assert bf.freq.mean().value == pytest.approx(74.67, 1e-2)
    assert bf.db[8] == pytest.approx(80.73, 1e-2)
    assert bf.amp[15] == pytest.approx(71325439, 1e0)
    bf = xst.beamform(
        az=180,
        el=90,
        pol='NE',
        ma=[0, 1],
        calibration='none'
    )
    assert isinstance(bf, SData)
    assert bf.db[8] == pytest.approx(60.82, 1e-2)
    assert bf.amp[15] == pytest.approx(748046, 1e0)
# ============================================================= #

