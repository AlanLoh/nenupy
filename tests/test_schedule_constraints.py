#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = "Alan Loh"
__copyright__ = "Copyright 2025, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"

from nenupy.schedule import (
    ESTarget,
    Constraint,
    TargetConstraint,
    ScheduleConstraint,
    ElevationCnst,
    MeridianTransitCnst,
    AzimuthCnst,
    LocalSiderealTimeCnst,
    LocalTimeCnst,
    TimeRangeCnst,
    NightTimeCnst,
    PeriodicCnst,
    Constraints
)
import pytest
import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np

# ============================================================= #
# ----------------- test_elevation_cnst_error ----------------- #
def test_elevation_cnst_error():
    with pytest.raises(ValueError):
        _ = ElevationCnst(elevationMin=120)
    
    with pytest.raises(ValueError):
        _ = ElevationCnst(elevationMin=-2.)
    
    with pytest.raises(Exception):
        _ = ElevationCnst(elevationMin=10 * u.m)

# ============================================================= #
# ----------------- test_elevation_scale_cnst ----------------- #
def test_elevation_scale_cnst():

    times = Time("2025-06-01") + np.arange(24) * TimeDelta(3600, format="sec")
    target = ESTarget.fromName("Cyg A")
    target.computePosition(times)

    cnst = ElevationCnst(elevationMin=0., scale_elevation=True)
    score = cnst(target)

    expected = np.array([
        0.74501608, 0.86909344, 0.97726726, 1.        , 0.91705472,
       0.79550933, 0.67080604, 0.54859903, 0.43151765, 0.32190874,
       0.2223457 , 0.13576637, 0.06542279, 0.01461947,        np.nan,
              np.nan, 0.00237296, 0.04586779, 0.10995296, 0.1914074 ,
       0.28692013, 0.39343169, 0.50825091])
    
    assert np.testing.assert_allclose(
        score,
        expected,
        atol=1e-3
    ) is None

    assert np.testing.assert_almost_equal(cnst.get_score([0, 1, 2]), 1., 3) is None
    assert np.testing.assert_almost_equal(cnst.get_score([13, 14]), 0.5, 3) is None


# ============================================================= #
# ----------------- test_elevation_scale_cnst ----------------- #
def test_elevation_cnst():

    times = Time("2025-06-01") + np.arange(24) * TimeDelta(3600, format="sec")
    target = ESTarget.fromName("Cyg A")
    target.computePosition(times)

    cnst = ElevationCnst(elevationMin=0., scale_elevation=False)
    score = cnst(target)

    expected = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1., np.nan, np.nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    
    assert np.testing.assert_allclose(
        score,
        expected,
        atol=1e-3
    ) is None

    assert np.testing.assert_almost_equal(cnst.get_score([0, 1, 2]), 1., 3) is None
    assert np.testing.assert_almost_equal(cnst.get_score([13, 14]), 0.5, 3) is None


