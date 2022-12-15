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
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import numpy as np

from nenupy.instru import NenuFAR, NenuFAR_Configuration, Polarization
from nenupy.instru.interferometer import ObservingMode, AntennaNameError, AntennaIndexError, DuplicateAntennaError
from nenupy.astro.sky import Sky
from nenupy.astro.pointing import Pointing


# ============================================================= #
# ---------------- test_miniarray_init_errors ----------------- #
# ============================================================= #
class TestNenuFARInitErrors:

    # ========================================================= #
    # ------------- test_nenufar_init_bad_format -------------- #
    def test_nenufar_init_bad_format(self):
        with pytest.raises(AntennaNameError):
            nenu = NenuFAR(
                miniarray_antennas="a",
            )
    
    # ========================================================= #
    # -------------- test_nenufar_init_bad_index -------------- #
    def test_nenufar_init_bad_format(self):
        with pytest.raises(AntennaIndexError):
            nenu = NenuFAR(
                miniarray_antennas=np.arange(21),
            )

    # ========================================================= #
    # ---------- test_nenufar_init_duplicate_antenna ---------- #
    def test_nenufar_init_duplicate_antenna(self):
        with pytest.raises(DuplicateAntennaError):
            nenu = NenuFAR(
                miniarray_antennas=[1, 1]
            )


    # ========================================================= #
    # -------------- test_nenufar_init_non_bool --------------- #
    def test_nenufar_init_non_bool(self):
        with pytest.raises(TypeError):
            nenu = NenuFAR(
                include_remote_mas="Test"
            )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ TestNenuFAR ------------------------ #
# ============================================================= #
@pytest.fixture(scope="class")
def get_nenufar_instance():
    return NenuFAR()

@pytest.fixture(autouse=True, scope="class")
def _use_nenufar(request, get_nenufar_instance):
    request.cls.nenu = get_nenufar_instance

@pytest.mark.usefixtures("_use_nenufar")
class TestMiniArray:

    # ========================================================= #
    # -------------------- test_rotations --------------------- #
    def test_rotations(self):
        rotations = self.nenu.miniarray_rotations
        assert rotations.size == 96


    # ========================================================= #
    # --------------------- test_position --------------------- #
    def test_position(self):
        assert self.nenu.position.size == 1
        assert self.nenu.position.lon.deg == pytest.approx(2.192, 1e-3)


    # ========================================================= #
    # --------------------- test_antennas --------------------- #
    def test_antennas(self):
        assert self.nenu.antenna_names.size == 96
        assert self.nenu.size == 96
        assert self.nenu.antenna_names[55] == "MA055"
        assert self.nenu.antenna_positions.shape == (96, 3)
        assert self.nenu.antenna_gains.size == 96


    # ========================================================= #
    # -------------------- test_baselines --------------------- #
    def test_baselines(self):
        bsl = self.nenu.baselines
        assert bsl.distance.shape == (96, 96)
        assert bsl.flatten.shape == ((96*95)/2 + 96, 3)
        assert bsl[0, 1].distance[0, 1].to(u.m).value == pytest.approx(26.82, 1e-2)


    # ========================================================= #
    # ----------------------- test_beam ----------------------- #
    def test_beam(self):
        beam = self.nenu.beam(
            sky=Sky(
                coordinates=SkyCoord([100, 200, 300], [10, 50, 90], unit="deg"),
                time=Time("2022-01-01T12:00:00"),
                frequency=50*u.MHz,
                polarization=Polarization.NW
            ),
            pointing=Pointing.zenith_tracking(
                time=Time("2022-01-01T11:00:00"),
                duration=TimeDelta(7200, format="sec")
            ),
            configuration=NenuFAR_Configuration(
                beamsquint_correction=True,
                beamsquint_frequency=50*u.MHz
            )
        )
        assert beam.value.shape == (1, 1, 1, 3)
        beam_values = beam[0, 0, 0].value.compute()
        assert np.ma.is_masked(beam_values[0])
        # assert beam_values[1] == pytest.approx(198722, abs=1e0) # before normalization
        # assert beam_values[2] == pytest.approx(105078.590, 1e-3)
        assert beam_values[1] == pytest.approx(8.24e-6, abs=1e-8)
        assert beam_values[2] == pytest.approx(4.36e-6, abs=1e-8)


    # ========================================================= #
    # ------------------ test_effective_area ------------------ #
    @pytest.mark.parametrize(
        'frequency, elevation, expected',
        [
            (50*u.MHz, 60*u.deg, 18929.273),
            ([50, 60]*u.MHz, 80*u.deg, np.array([21525.575, 14926.085])),
            (50*u.MHz, [10, 90]*u.deg, np.array([3795.5396, 21857.641])),
            ([50, 60]*u.MHz, [10, 90]*u.deg, np.array([3795.5396, 15156.344]))
        ]
    )
    def test_effective_area(self, frequency, elevation, expected):
        assert np.testing.assert_allclose(
            self.nenu.effective_area(frequency=frequency, elevation=elevation).to(u.m**2).value,
            expected,
            atol=1e-1
        ) is None # other an AssertionError is raised


    # ========================================================= #
    # ----------------------- test_plot ----------------------- #
    @patch("matplotlib.pyplot.show")
    def test_plot(self, mock_show):
        self.nenu.plot()


    # ========================================================= #
    # ----------------------- test_sefd ----------------------- #
    def test_sefd(self):
        sefd = self.nenu.sefd(
            frequency=50*u.MHz,
            elevation=50*u.deg
        )
        assert sefd.to(u.Jy).value == pytest.approx(1033.0, 1e-1)


    # ========================================================= #
    # ------------------- test_sensitivity -------------------- #
    @pytest.mark.parametrize(
        'mode, dt, df, expected',
        [
            (ObservingMode.BEAMFORMING, 1.*u.s, 195.3125*u.kHz, 1.2661786),
            (ObservingMode.IMAGING, 1.*u.s, 195.3125*u.kHz, 0.013258597),
            (ObservingMode.BEAMFORMING, [1., 10, 20]*u.s, 195.3125*u.kHz, np.array([1.2661786, 0.40040084, 0.28312615])),
        ]
    )
    def test_sensitivity(self, mode, dt, df, expected):
        assert np.testing.assert_allclose(
            self.nenu.sensitivity(
                frequency=50*u.MHz,
                mode=mode,
                dt=dt,
                df=df,
                elevation=90*u.deg,
                efficiency=1.,
                decoherence=1.
            ).to(u.Jy).value,
            expected,
            atol=1e-2
        ) is None # other an AssertionError is raised


    # ========================================================= #
    # ------------------ angular_resolution ------------------- #
    @pytest.mark.parametrize(
        'freq, expected',
        [
            (50*u.MHz, 0.63354018),
            ([50, 60]*u.MHz, np.array([0.63354018, 0.52795015]))
        ]
    )
    def angular_resolution(self, freq, expected):
        assert np.testing.assert_allclose(
            self.nenu.angular_resolution(frequency=freq).to(u.deg).value,
            expected,
            atol=1e-2
        ) is None


    # ========================================================= #
    # -------------------- confusion_noise -------------------- #
    @pytest.mark.parametrize(
        'freq, lofar, expected',
        [
            (50*u.MHz, True, 5.8579076),
            ([50, 60]*u.MHz, True, np.array([5.8579076, 3.8938259])),
            (50*u.MHz, False, 2.352891),
            ([50, 60]*u.MHz, False, np.array([2.352891, 1.4381774]))
        ]
    )
    def confusion_noise(self, freq, lofar, expected):
        assert np.testing.assert_allclose(
            self.nenu.confusion_noise(
                frequency=freq,
                lofar=lofar
            ).to(u.deg).value,
            expected,
            atol=1e-2
        ) is None
# ============================================================= #
# ============================================================= #

