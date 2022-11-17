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

from nenupy.io.xst import XST
from nenupy.astro.pointing import Pointing


# ============================================================= #
# -------------------------- TestXST -------------------------- #
# ============================================================= #

@pytest.fixture(scope="class")
def get_xst_instance():
    return XST(join(dirname(__file__), "test_data/XST.fits"))

@pytest.fixture(autouse=True, scope="class")
def _use_xst(request, get_xst_instance):
    request.cls.xst = get_xst_instance

@pytest.mark.usefixtures("_use_xst")
class TestXST:

    # ========================================================= #
    # ------------------- test_frequencies -------------------- #
    def test_frequencies(self):
        assert self.xst.frequencies.size == 16
        assert self.xst.frequencies[0][0].to(u.MHz).value == pytest.approx(68.55, 1e-2)


    # ========================================================= #
    # -------------------- test_miniarrays -------------------- #
    def test_miniarrays(self):
        assert self.xst.mini_arrays.size == 55


    # ========================================================= #
    # ----------------------- test_get ------------------------ #
    def test_get(self):
        with pytest.raises(IndexError):
            data = self.xst.get(
                polarization="XY",
                miniarray_selection=np.array([1, 44])
            )
        data = self.xst.get(
            polarization="XX",
            miniarray_selection=np.array([17, 25, 44])
        )
        assert data.value.shape == (1, 16, 6)
        assert np.imag(data.value[0, 0, 0]) == 0.0
        assert np.imag(data.value[0, 0, 2]) == 0.0
        assert np.imag(data.value[0, 0, 5]) == 0.0
        data = self.xst.get(
            polarization="XY",
            miniarray_selection=np.array([17, 44])
        )
        assert data.value.shape == (1, 16, 3)
        assert np.imag(data.value[0, 0, 0]) == 15057.
        assert np.imag(data.value[0, 0, 2]) == -3462.


    # ========================================================= #
    # -------------------- test_get_stokes -------------------- #
    def test_get_stokes(self):
        with pytest.raises(KeyError):
            data = self.xst.get_stokes(
                stokes="RR"
            )
        data = self.xst.get_stokes(
            stokes="I",
            miniarray_selection=np.array([17, 44])
        )
        assert data.value.shape == (1, 16, 3)
        assert np.real(data.value[0, 0, 0]) == 2895175.


    # ========================================================= #
    # ------------------- test_get_beamform ------------------- #
    def test_get_beamform(self):

        pointing = Pointing.zenith_tracking(
            time=self.xst.time[0],
            duration=TimeDelta(3600, format="sec")
        )

        data_cal = self.xst.get_beamform(
            pointing=pointing,
            frequency_selection=">=73.828125MHz",
            time_selection="<2020-02-20T00:00:00",
            mini_arrays=np.array([17, 44]),
            polarization="NW",
            calibration="default"
        )
        assert data_cal.value.shape == (12,)
        assert data_cal.value[0] == pytest.approx(4376021.03, 1e-2)

        data_no_cal = self.xst.get_beamform(
            pointing=pointing,
            frequency_selection=">=73.828125MHz",
            time_selection="<2020-02-20T00:00:00",
            mini_arrays=np.array([17, 44]),
            polarization="NW",
            calibration="none"
        )
        assert data_no_cal.value.shape == (12,)
        assert data_no_cal.value[0] == pytest.approx(4388607.82, 1e-2)


    # ========================================================= #
    # --------- test_xst_slice_plot_correlaton_matrix --------- #
    @patch("matplotlib.pyplot.show")
    def test_xst_slice_plot_correlaton_matrix(self, mock_show):
        data = self.xst.get(
            polarization="XX",
            miniarray_selection=np.array([17, 25, 44])
        )
        data.plot_correlaton_matrix(mask_autocorrelations=True)


    # ========================================================= #
    # ----------------- test_xst_slice_image ------------------ #
    def test_xst_slice_image(self):
        data = self.xst.get(
            polarization="XX",
            miniarray_selection=np.array([17, 25, 44])
        )
        im = data.make_image(
            resolution=1*u.deg,
            fov_radius=5*u.deg,
            phase_center=SkyCoord.from_name("Cyg A"),
            stokes="I"
        )
        assert im.shape == (1, 1, 1, 49152)
        im_zenith = data.make_image(
            resolution=1*u.deg,
            fov_radius=5*u.deg,
            phase_center=None,
            stokes="I"
        )
        assert im_zenith.value[0, 0, 0][~np.isnan(im_zenith.value[0, 0, 0])][0] == pytest.approx(-1645.46, 1e-2)


    # ========================================================= #
    # --------------- test_xst_slice_nearfield ---------------- #
    def test_xst_slice_nearfield(self):
        data = self.xst.get(
            polarization="XX",
            miniarray_selection=np.array([17, 25, 44])
        )
        nf, nf_sources = data.make_nearfield(
            radius=400*u.m,
            npix=32,
            sources=["Cas A", "Cyg A"]
        )
        assert nf.shape == (32, 32)
        assert nf[16, 16] == pytest.approx(16293.4, 1e-1)
        assert nf_sources["Cas A"].shape == (32, 32)
        assert nf_sources["Cyg A"].shape == (32, 32)
# ============================================================= #
# ============================================================= #

