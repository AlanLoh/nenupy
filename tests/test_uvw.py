#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord
import pytest

from nenupy.crosslet import UVW


# ============================================================= #
# ---------------------- test_nenufarloc ---------------------- #
# ============================================================= #
def test_uvwcompute():
    dts = TimeDelta(np.arange(2), format='sec')
    timestamps = Time('2020-04-01 12:00:00') + dts
    uvw = UVW(
        times=timestamps,
        freqs=np.linspace(50, 60, 5)*u.MHz,
        mas=np.arange(5)
    )
    # No phase_center --> zenith
    uvw.compute(
        phase_center=None
    )
    assert isinstance(uvw.uvw, np.ndarray)
    assert uvw.uvw.shape == (2, 15, 3)
    assert uvw.uvw[0, 7, 0] == pytest.approx(55.39, 1e-2)
    # Bad phase_center
    with pytest.raises(TypeError):
        uvw.compute(
            phase_center='wrong type'
        )
    # SkyCoord scalar input
    ncp = SkyCoord(
        ra=0*u.deg,
        dec=90*u.deg
    )
    uvw.compute(
        phase_center=ncp
    )
    assert isinstance(uvw.uvw, np.ndarray)
    assert uvw.uvw.shape == (2, 15, 3)
    assert uvw.uvw[0, 7, 0] == pytest.approx(49.64, 1e-2)
    # SkyCoord non-scalar input
    ncp = SkyCoord(
        ra=np.array([0, 0])*u.deg,
        dec=np.array([90, 90])*u.deg
    )
    uvw.compute(
        phase_center=ncp
    )
    assert isinstance(uvw.uvw, np.ndarray)
    assert uvw.uvw.shape == (2, 15, 3)
    assert uvw.uvw[0, 7, 0] == pytest.approx(49.641, 1e-3)
    assert uvw.uvw[0, 7, 0] == pytest.approx(49.638, 1e-3)
    # SkyCoord non-scalar wrong size input
    with pytest.raises(ValueError):
        ncp = SkyCoord(
            ra=np.array([0, 0, 0])*u.deg,
            dec=np.array([90, 90, 90])*u.deg
        )
        uvw.compute(
            phase_center=ncp
        )
# ============================================================= #