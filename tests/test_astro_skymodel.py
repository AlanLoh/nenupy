#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
import sys
from unittest.mock import MagicMock, patch
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
from os.path import join, dirname

sys.modules["pygsm"] = MagicMock()

from nenupy.astro.skymodel import HpxGSM


@pytest.fixture(scope="module")
def create_mock_gsm():
    with patch("nenupy.astro.skymodel.GlobalSkyModel") as mockgsm:
        mockgsm.return_value.generate.return_value = np.expand_dims(
            np.load(
                join(dirname(__file__), "test_data/gsm_55mhz.npy")
            ),
            axis=0
        )
        gsm  = HpxGSM(
            resolution=2*u.deg,
            time=Time(["2022-01-01T12:00:00", "2022-01-01T14:00:00"]),
            frequency=55*u.MHz
        )
        return gsm


@pytest.fixture(autouse=True, scope="class")
def _init_gsm(request, create_mock_gsm):
    request.cls._gsm = create_mock_gsm


# ============================================================= #
# ------------------------ TestHpxGSM ------------------------- #
# ============================================================= #
@pytest.mark.usefixtures("_init_gsm")
class TestHpxGSM:


    # ========================================================= #
    # -------------------- test_gsm_shape --------------------- #
    def test_gsm_shape(self):
        assert self._gsm.value.shape == (2, 1, 1, 12288)


    # ========================================================= #
    # -------------------- test_gsm_value --------------------- #
    def test_gsm_value(self):
        assert self._gsm.value[0, 0, 0, 100].compute() == pytest.approx(3251.89, 1e-2)


    # ========================================================= #
    # --------------------- test_gsm_plot --------------------- #
    @patch("matplotlib.pyplot.show")
    def test_gsm_plot(self, mock_show):
        source_names = ["Cyg A", "Cas A"]
        ras = []
        decs = []
        for name in source_names:
            src = SkyCoord.from_name(name)
            ras.append(src.ra)
            decs.append(src.dec)
        sources = SkyCoord(ras, decs)

        self._gsm[0, 0, 0].plot(
            decibel=True,
            altaz_overlay=True,
            contour=(self._gsm.value[0, 0, 0].compute(), None, "copper"),
            only_visible=False,
            scatter=(sources, 10, "white"),
            text=(sources, source_names, "white")
        )
# ============================================================= #
# ============================================================= #

