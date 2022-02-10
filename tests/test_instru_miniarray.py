#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
import astropy.units as u

from nenupy.instru import MiniArray
from nenupy.instru.nenufar import MiniArrayBadIndexFormat, MiniArrayUnknownIndex


# ============================================================= #
# ---------------- test_miniarray_init_errors ----------------- #
# ============================================================= #
class TestMiniArrayInitErrors:

    # ========================================================= #
    # ------------ test_miniarray_init_bad_format ------------- #
    def test_miniarray_init_bad_format(self):
        with pytest.raises(MiniArrayBadIndexFormat):
            ma = MiniArray(index=['32'])


    # ========================================================= #
    # ----------- test_miniarray_init_unkwown_index ----------- #
    def test_miniarray_init_unkwown_index(self):
        with pytest.raises(MiniArrayUnknownIndex):
            ma = MiniArray(index=97)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- TestMiniArray ----------------------- #
# ============================================================= #
@pytest.fixture(scope="class")
def get_ma_instance():
    return MiniArray(index=0)

@pytest.fixture(autouse=True, scope="class")
def _use_miniarray(request, get_ma_instance):
    request.cls.ma = get_ma_instance

@pytest.mark.usefixtures("_use_miniarray")
class TestMiniArray:

    # ========================================================= #
    # --------------------- test_rotation --------------------- #
    def test_rotation(self):
        assert self.ma.rotation.to(u.deg) == 0


    def test_position(self):
        assert self.ma.position.size == 1
        assert self.ma.position.lon.deg == pytest.approx(2.193, 1e-3)


    def test_antennas(self):
        assert self.ma.antenna_names.size == 19
        assert self.ma.size == 19
        assert self.ma.antenna_names[9] == "Ant10"
        assert self.ma.antenna_positions.shape == (19, 3)
        assert self.ma.antenna_gains.size == 19


    def test_baselines(self):
        bsl = self.ma.baselines
        assert bsl.distance.shape == (19, 19)
        assert bsl.flatten.shape == ((19*18)/2+19, 3)


    def test_system_temperature(self):
        sys_temp = self.ma.system_temperature(frequency=50*u.MHz)
        assert sys_temp.to(u.K).value == pytest.approx(6264.19, 1e-2)
        sys_temp = self.ma.system_temperature(frequency=[50, 60]*u.MHz)
        assert sys_temp.size == 2
# ============================================================= #
# ============================================================= #

