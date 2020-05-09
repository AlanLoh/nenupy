# #! /usr/bin/python3
# # -*- coding: utf-8 -*-


# __author__ = 'Alan Loh'
# __copyright__ = 'Copyright 2020, nenupy'
# __credits__ = ['Alan Loh']
# __maintainer__ = 'Alan'
# __email__ = 'alan.loh@obspm.fr'
# __status__ = 'Production'


import numpy as np
import astropy.units as u
import pytest
import sys
from os.path import join, dirname
from unittest.mock import MagicMock, patch

sys.modules['pygsm'] = MagicMock()

from nenupy.skymodel import HpxGSM, HpxLOFAR


# ============================================================= #
# ------------------------- test_gsm -------------------------- #
# ============================================================= #
# Need pygsm to test, impossible remotley on CI
# @pytest.mark.skip
def test_gsm():
    with patch('nenupy.skymodel.hpxgsm.GlobalSkyModel') as mockgsm:
        mockgsm.return_value.generate.return_value = np.load(
            join(dirname(__file__), 'test_data/gsm_55mhz.npy')
        )
        # Fail because too high resolution
        with pytest.raises(ValueError):
            gsm = HpxGSM(
                freq=55*u.MHz,
                resolution=10*u.deg
            )
        gsm = HpxGSM(
            freq=55*u.MHz,
            resolution=1.8*u.deg
        )
        assert isinstance(gsm.skymap, np.ma.core.MaskedArray)
        assert gsm.skymap.size == 12288
        assert gsm.skymap[6000] == pytest.approx(1793.298, 1e-3)
# ============================================================= #


# ============================================================= #
# ------------------------ test_lofar ------------------------- #
# ============================================================= #
def test_lofar():
    lofar = HpxLOFAR(
        freq=55,
        resolution=10*u.deg
    )
    assert isinstance(lofar.skymap, np.ma.core.MaskedArray)
    assert lofar.skymap.size == 768
    assert lofar.skymap[350] == pytest.approx(4.718, 1e-3)
# ============================================================= #

