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
from astropy.time import Time, TimeDelta
import pytest
from unittest.mock import patch 

from nenupy.beamlet import SData


# ============================================================= #
# ------------------- test_sdata_def_error -------------------- #
# ============================================================= #
def test_sdata_def_error():
    dts = TimeDelta(np.arange(2), format='sec')
    with pytest.raises(AssertionError):
        # Time shape error
        s = SData(
            data=np.ones((3, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
    with pytest.raises(AssertionError):
        # Freq shape error
        s = SData(
            data=np.ones((2, 4, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
    with pytest.raises(AssertionError):
        # Polar shape error
        s = SData(
            data=np.ones((2, 3, 2)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
    with pytest.raises(TypeError):
        # Time type error
        s = SData(
            data=np.ones((2, 3, 1)),
            time=np.arange(2),
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
    with pytest.raises(TypeError):
        # Freq type error
        s = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3),
            polar=['NE']
        )
# ============================================================= #


# ============================================================= #
# ---------------------- test_sdata_attr ---------------------- #
# ============================================================= #
def test_sdata_attr():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar='NE'
    )
    assert s1.amp.shape == (2, 3)
    assert (s1.amp == 1.).all()
    assert s1.db.shape == (2, 3)
    assert (s1.db == 0.).all()
    assert s1.mjd.size == 2
    testmjd = np.array([0.50000, 0.50001])
    assert testmjd == pytest.approx(s1.mjd-58940, 1e-5)
    assert s1.jd.size == 2
    assert s1.datetime.size == 2
# ============================================================= #


# ============================================================= #
# ---------------------- test_sdata_add ----------------------- #
# ============================================================= #
def test_sdata_add():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s2 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s = s1 + s2
    assert isinstance(s, SData)
    assert (s.data == 2.).all()
    s = s1 + 2
    assert (s.data == 3.).all()
    s = s1 + np.ones((2, 3, 1)) * 3
    assert (s.data == 4.).all()
    # Test operation errors
    with pytest.raises(ValueError):
        # Data not same shape
        s3 = SData(
            data=np.ones((2, 4, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(4) * u.MHz,
            polar=['NE']
        )
        s = s1 + s3
    with pytest.raises(ValueError):
        # Polar not same
        s3 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NW']
        )
        s = s1 + s3
    with pytest.raises(ValueError):
        # Time not same
        s3 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 13:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
        s = s1 + s3
    with pytest.raises(ValueError):
        # Freq not same
        s3 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=(np.arange(3)+1) * u.MHz,
            polar=['NE']
        )
        s = s1 + s3
    with pytest.raises(Exception):
        s = s1 + 'wrong'
    with pytest.raises(ValueError):
        s = s1 + np.ones((1, 3, 1)) * 3
# ============================================================= #


# ============================================================= #
# ---------------------- test_sdata_sub ----------------------- #
# ============================================================= #
def test_sdata_sub():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s2 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s = s1 - s2
    assert isinstance(s, SData)
    assert (s.data == 0.).all()
    s = s1 - 2
    (s.data == -1.).all()
# ============================================================= #


# ============================================================= #
# ---------------------- test_sdata_mul ----------------------- #
# ============================================================= #
def test_sdata_mul():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s2 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s = s1 * s2
    assert isinstance(s, SData)
    assert (s.data == 1.).all()
    s = s1 * 2
    (s.data == 2.).all()
# ============================================================= #


# ============================================================= #
# ---------------------- test_sdata_div ----------------------- #
# ============================================================= #
def test_sdata_div():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s2 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s = s1 / s2
    assert isinstance(s, SData)
    assert (s.data == 1.).all()
    s = s1 / 2
    (s.data == 0.5).all()
# ============================================================= #


# ============================================================= #
# -------------------- test_sdata_concat_t -------------------- #
# ============================================================= #
def test_sdata_concat_t():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    with pytest.raises(TypeError):
        s = s1 | 'other'
    with pytest.raises(ValueError):
        s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 13:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NW']
        )
        # Different polars
        s = s1 | s2
    with pytest.raises(ValueError):
        s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 13:00:00') + dts,
            freq=(1 + np.arange(3)) * u.MHz,
            polar=['NE']
        )
        # Different frequencies
        s = s1 | s2
    s2 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 13:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    s = s1 | s2
    assert s.data.shape == (4, 3, 1)
# ============================================================= #


# ============================================================= #
# -------------------- test_sdata_concat_f -------------------- #
# ============================================================= #
def test_sdata_concat_f():
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    with pytest.raises(TypeError):
        s = s1 & 'other'
    with pytest.raises(ValueError):
        s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=(1 + np.arange(3)) * u.MHz,
            polar=['NW']
        )
        # Different polars
        s = s1 & s2
    with pytest.raises(ValueError):
        s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 13:00:00') + dts,
            freq=(1 + np.arange(3)) * u.MHz,
            polar=['NE']
        )
        # Different times
        s = s1 & s2
    s2 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=(1 + np.arange(3)) * u.MHz,
        polar=['NE']
    )
    s = s1 & s2
    assert s.data.shape == (2, 6, 1)
# ============================================================= #


# ============================================================= #
# ---------------------- test_sdata_plot ---------------------- #
# ============================================================= #
@patch('matplotlib.pyplot.show')
def test_sdata_plot(mock_show):
    dts = TimeDelta(np.arange(2), format='sec')
    s1 = SData(
        data=np.ones((2, 3, 1)),
        time=Time('2020-04-01 12:00:00') + dts,
        freq=np.arange(3) * u.MHz,
        polar=['NE']
    )
    # Default plot
    s1.plot()
    # Custom plot
    s1.plot(
        db=False,
        cmap='Blues',
        vmin=-0.1,
        vmax=0.1,
        title='test title',
        cblabel='test cblabel',
        figsize=(6, 6),
    )
# ============================================================= #

