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
from os.path import join, dirname
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time, TimeDelta
import numpy as np

from nenupy.io.bst import BST


# ============================================================= #
# -------------------------- TestBST -------------------------- #
# ============================================================= #

@pytest.fixture(scope="class")
def get_bst_instance():
    return BST(join(dirname(__file__), "test_data/BST.fits"))

@pytest.fixture(autouse=True, scope="class")
def _use_bst(request, get_bst_instance):
    request.cls.bst = get_bst_instance

@pytest.mark.usefixtures("_use_bst")
class TestBST:

    # ========================================================= #
    # ---------------------- test_beams ----------------------- #
    def test_beams(self):
        assert self.bst.analog_beams.size == 1
        assert self.bst.analog_beams[0] == 0
        assert self.bst.digital_beams.size == 17
        assert np.unique(self.bst.digital_beams).size == 17


    # ========================================================= #
    # ------------------- test_mini_arrays -------------------- #
    def test_mini_arrays(self):
        assert self.bst.mini_arrays.size == 55


    # ========================================================= #
    # ------------------- test_frequencies -------------------- #
    def test_frequencies(self):
        freqs = self.bst.frequencies
        assert freqs.size == 45
        assert freqs[0].to(u.MHz).value == pytest.approx(45.508, 1e-3)
        assert freqs[-1].to(u.MHz).value == pytest.approx(54.102, 1e-3)


    # ========================================================= #
    # ----------------- test_analog_pointing ------------------ #
    def test_analog_pointing(self):
        pointing = self.bst.analog_pointing
        assert pointing[0].size == 2
        assert pointing[0][0].jd == pytest.approx(2459603.959, 1e-3)
        assert pointing[1][0].to(u.deg).value == pytest.approx(231.652, 1e-3)
        assert pointing[1][1].to(u.deg).value == pytest.approx(0.0, 1e-1)
        assert pointing[2][0].to(u.deg).value == pytest.approx(35.579, 1e-3)
        assert pointing[2][1].to(u.deg).value == pytest.approx(90.0, 1e-1)


    # ========================================================= #
    # ---------------- test_digital_pointing ------------------ #
    def test_digital_pointing(self):
        self.bst.beam = 1
        pointing = self.bst.digital_pointing
        assert pointing[0][0].jd == pytest.approx(2459603.959, 1e-3)
        assert pointing[1][0].to(u.deg).value == pytest.approx(231.652, 1e-3)
        assert pointing[2][0].to(u.deg).value == pytest.approx(34.398, 1e-3)

        self.bst.beam = 16
        pointing = self.bst.digital_pointing
        assert pointing[0][0].jd == pytest.approx(2459603.959, 1e-3)
        assert pointing[1][0].to(u.deg).value == pytest.approx(231.652, 1e-3)
        assert pointing[2][0].to(u.deg).value == pytest.approx(36.929, 1e-3)


    # ========================================================= #
    # ----------------------- test_get ------------------------ #
    def test_get(self):
        # Default with wrong polar
        data = self.bst.get(
            frequency_selection=None,
            time_selection=None,
            polarization="wrong value",
            beam=0
        )
        assert data.value.shape == (779, 45)
        assert data.frequency.shape == (45,)
        assert data.time.shape == (779,)


    # ========================================================= #
    # -------------------- test_lightcurve -------------------- #
    @patch("matplotlib.pyplot.show")
    def test_lightcurve(self, mock_show):
        data = self.bst.get(
            frequency_selection="==50.195312MHz",
            time_selection=None,
            polarization="NE",
            beam=0
        )
        assert data.value.shape == (779,)
        data.plot(
            digital_pointing=True,
            analog_pointing=True
        )
        fitted_data, transit_time, chi2, parameters = data.fit_transit()
        assert transit_time.jd == pytest.approx(2459603.964, 1e-3)
        rebin_10s = data.rebin(dt=10*u.s)
        assert rebin_10s.value.shape == (77,)
        rebin_100ms = data.rebin(dt=0.1*u.s)
        assert rebin_100ms.value.shape == (779,)


    # ========================================================= #
    # --------------------- test_spectrum --------------------- #
    @patch("matplotlib.pyplot.show")
    def test_spectrum(self, mock_show):
        data = self.bst.get(
            frequency_selection=None,
            time_selection="==2022-01-24T11:05:10.000",
            polarization="NE",
            beam=0
        )
        assert data.value.shape == (45,)
        data.plot()
        rebin_1MHz = data.rebin(df=1*u.MHz)
        assert rebin_1MHz.value.shape == (9,)
        rebin_100kHz = data.rebin(df=0.1*u.MHz)
        assert rebin_100kHz.value.shape == (45,)


    # ========================================================= #
    # ----------------- test_dynamic_spectrum ----------------- #
    @patch("matplotlib.pyplot.show")
    def test_dynamic_spectrum(self, mock_show):
        data = self.bst.get(
            frequency_selection=">=52MHz",
            time_selection='>=2022-01-24T11:08:10.000 & <= 2022-01-24T11:14:08.000',
            polarization="NW",
            beam=8
        )
        assert data.value.shape == (359, 11)
        vals = np.zeros((30, 100), dtype=bool)
        vals[5:20, 20:70] = True
        data.plot(
            digital_pointing=True,
            analog_pointing=True,
            hatched_overlay=(
                Time("2022-01-24T11:01:00")+np.arange(100)*TimeDelta(2, format='sec'),
                np.linspace(47, 52, 30)*u.MHz,
                vals
            )
        )
        data_rebinned = data.rebin(df=1*u.MHz, dt=2*u.s)
        assert data_rebinned.value.shape == (179, 2)
# ============================================================= #
# ============================================================= #

