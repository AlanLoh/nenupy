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
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time, TimeDelta
import numpy as np

from nenupy import nenufar_position
from nenupy.instru import MiniArray, NenuFAR_Configuration, Polarization
from nenupy.instru.nenufar import MiniArrayBadIndexFormat, MiniArrayUnknownIndex
from nenupy.astro.sky import HpxSky, Sky
from nenupy.astro.target import FixedTarget
from nenupy.astro.pointing import Pointing


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


    # ========================================================= #
    # --------------------- test_position --------------------- #
    def test_position(self):
        assert self.ma.position.size == 1
        assert self.ma.position.lon.deg == pytest.approx(2.193, 1e-3)


    # ========================================================= #
    # --------------------- test_antennas --------------------- #
    def test_antennas(self):
        assert self.ma.antenna_names.size == 19
        assert self.ma.size == 19
        assert self.ma.antenna_names[9] == "Ant10"
        assert self.ma.antenna_positions.shape == (19, 3)
        assert self.ma.antenna_gains.size == 19


    # ========================================================= #
    # -------------------- test_baselines --------------------- #
    def test_baselines(self):
        bsl = self.ma.baselines
        assert bsl.distance.shape == (19, 19)
        assert bsl.flatten.shape == ((19*18)/2+19, 3)


    # ========================================================= #
    # ----------------------- test_get ------------------------ #
    def test_get(self):
        sub_ma = self.ma[1]
        assert sub_ma.size == 1
        sub_ma = self.ma["Ant01", "Ant02"]
        assert sub_ma.size == 2
        sub_ma = self.ma[np.arange(10)]
        assert sub_ma.size == 10

    # ========================================================= #
    # ----------------------- test_get ------------------------ #
    def test_add(self):
        sub_ma = MiniArray()["Ant01", "Ant02", "Ant03"]
        new_array = self.ma + sub_ma
        assert new_array.size == 19


    # ========================================================= #
    # ----------------------- test_get ------------------------ #
    def test_sub(self):
        sub_ma = MiniArray()["Ant01", "Ant02", "Ant03"]
        new_array = self.ma - sub_ma
        assert new_array.size == 16


    # ========================================================= #
    # ------------------ test_effective_area ------------------ #
    @pytest.mark.parametrize(
        'frequency, elevation, expected',
        [
            (50*u.MHz, 60*u.deg, 197.17992),
            ([50, 60]*u.MHz, 80*u.deg, np.array([224.22474, 155.48005])),
            (50*u.MHz, [10, 90]*u.deg, np.array([39.536871, 227.68377])),
            ([50, 60]*u.MHz, [10, 90]*u.deg, np.array([39.536871, 157.87858]))
        ]
    )
    def test_effective_area(self, frequency, elevation, expected):
        assert np.testing.assert_allclose(
            self.ma.effective_area(frequency=frequency, elevation=elevation).to(u.m**2).value,
            expected,
            atol=1e-1
        ) is None # other an AssertionError is raised


    # ========================================================= #
    # ----------------------- test_beam ----------------------- #
    def test_beam(self):
        beam = self.ma.beam(
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
        # assert beam_values[1] == pytest.approx(23.2, abs=1e-1) # before normalization
        # assert beam_values[2] == pytest.approx(300.6, abs=1e-1)
        assert beam_values[1] == pytest.approx(8.53e-4, abs=1e-6)
        assert beam_values[2] == pytest.approx(1.10e-2, abs=1e-4)


    # ========================================================= #
    # -------------- test_instrument_temperature -------------- #
    @pytest.mark.parametrize(
        'frequency, filter, expected',
        [
            (50*u.MHz, 0, 487.61479),
            ([50, 60]*u.MHz, 3, np.array([643.51063, 786.71536]))
        ]
    )
    def test_instrument_temperature(self, frequency, filter, expected):
        assert np.testing.assert_allclose(
            self.ma.instrument_temperature(frequency=frequency, lna_filter=filter).to(u.K).value,
            expected,
            atol=1e-1
        ) is None # other an AssertionError is raised


    # ========================================================= #
    # ------------- test_attenuation_from_zenith -------------- #
    def test_attenuation_from_zenith(self):
        time = Time("2022-01-01T12:00:00")
        zenith_radec = SkyCoord(
            0, 90, unit="deg",
            frame=AltAz(
                location=nenufar_position,
                obstime=time
            )
        ).transform_to("icrs")
        coords = SkyCoord([300, zenith_radec.ra.deg], [40, zenith_radec.dec.deg], unit="deg")
        attenuation = self.ma.attenuation_from_zenith(
            coordinates=coords,
            time=time,
            frequency=50*u.MHz,
            polarization=Polarization.NW
        )
        assert attenuation.shape == (1, 1, 1, 2)
        assert attenuation[0, 0, 0, 0] == pytest.approx(0.026, 1e-1)
        assert attenuation[0, 0, 0, 1] == pytest.approx(1.000, 1e-3)


    # ========================================================= #
    # -------------- test_beamsquint_correction --------------- #
    def test_beamsquint_correction(self):
        source = SkyCoord.from_name("Vir A")
        altaz_frame = AltAz(
            location=nenufar_position,
            obstime=Time("2022-01-01T12:00:00")
        )
        new_coords = self.ma.beamsquint_correction(
            coords=source.transform_to(altaz_frame),
            frequency=50*u.MHz
        )
        assert new_coords.az.deg == pytest.approx(282.219, 1e-3)
        assert new_coords.alt.deg == pytest.approx(0.389, 1e-3)


    # ========================================================= #
    # ----------------- test_analog_pointing ------------------ #
    def test_analog_pointing(self):
        analog_pointing = self.ma.analog_pointing(
            pointing=Pointing.target_tracking(
                target=FixedTarget.from_name("Vir A"),
                time=Time(["2022-01-01T11:00:00", "2022-01-01T14:00:00"]),
                duration=TimeDelta(7200, format="sec")
            ),
            configuration=NenuFAR_Configuration(
                beamsquint_correction=False,
                beamsquint_frequency=20*u.MHz
            )
        )
        assert analog_pointing._custom_ho_coordinates.shape == (2,)
        assert analog_pointing._custom_ho_coordinates[0].alt.deg == pytest.approx(21.55, 1e-2)
        assert analog_pointing._custom_ho_coordinates[1].alt.deg == pytest.approx(11.98, 1e-2)

        analog_pointing = self.ma.analog_pointing(
            pointing=Pointing.target_tracking(
                target=FixedTarget.from_name("Vir A"),
                time=Time(["2022-01-01T11:00:00", "2022-01-01T14:00:00"]),
                duration=TimeDelta(7200, format="sec")
            ),
            configuration=NenuFAR_Configuration(
                beamsquint_correction=True,
                beamsquint_frequency=20*u.MHz
            )
        )
        assert analog_pointing._custom_ho_coordinates[0].alt.deg == pytest.approx(12.26, 1e-2)
        assert analog_pointing._custom_ho_coordinates[1].alt.deg == pytest.approx(11.98, 1e-2)


    # ========================================================= #
    # ---------------- test_system_temperature ---------------- #
    def test_system_temperature(self):
        sys_temp = self.ma.system_temperature(frequency=50*u.MHz)
        assert sys_temp.to(u.K).value == pytest.approx(6264.19, 1e-2)
        sys_temp = self.ma.system_temperature(frequency=[50, 60]*u.MHz)
        assert sys_temp.size == 2
    
        def casa_spectrum(frequency):
            """ Cas A spectrum """
            a0 = 3.3584
            a1 = -0.7518
            a2 = -0.0347
            a3 = -0.0705
            log_nu = np.log10(frequency.to(u.GHz).value)
            return 0.7 * np.power(10, (a0 + a1*log_nu + a2*log_nu**2 + a3*log_nu**3) )*u.Jy
        sys_temp = self.ma.system_temperature(frequency=50*u.MHz, source_spectrum={"Cas A": casa_spectrum})
        assert sys_temp.to(u.K).value == pytest.approx(7828.6, 1e-1)
# ============================================================= #
# ============================================================= #

