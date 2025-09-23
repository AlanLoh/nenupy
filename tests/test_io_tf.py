#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = "Alan Loh"
__copyright__ = "Copyright 2024, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"


import pytest
from nenupy.io.tf_utils import get_bandpass
import numpy as np

# ============================================================= #
# -------------------------- test_get_bandpass -------------------------- #
def test_get_bandpass():
    bp = get_bandpass(16)

    expected = np.array([
        1.96591532, 0.99159353, 1.00350477, 1.06808059, 0.99481222,
        1.05979879, 1.00530128, 1.05058502, 1.01044723, 1.05058502,
        1.00530128, 1.05979879, 0.99481222, 1.06808059, 1.00350477,
        0.99159353])
    
    assert np.testing.assert_allclose(
        bp,
        expected,
        atol=1e-3
    ) is None

    with pytest.raises(ValueError):
        bp = get_bandpass(n_channels=14)

    with pytest.raises(ValueError):
        bp =get_bandpass(n_channels=21)