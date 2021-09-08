#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


from os.path import join, dirname
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np
import pytest
from unittest.mock import patch 

from nenupy.observation import Parset
from nenupy.observation import (
    ObsConfig,
    BSTConfig,
    NICKELConfig,
    TFConfig,
    PulsarFoldConfig,
    PulsarSingleConfig,
    PulsarWaveConfig,
    RAWConfig
) 


# ============================================================= #
# ---------------------- test_bstconfig ----------------------- #
# ============================================================= #
def test_bstconfig():
    bst = BSTConfig()
    assert bst.nSubBands == 768
    assert bst.nPolars == 2
    assert bst.durationSec == 0

    bst = BSTConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=384,
        nPolars=2
    )
    vol = bst.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Kibyte).value == 5400
# ============================================================= #


# ============================================================= #
# ------------------- test_bstconfig_parset ------------------- #
# ============================================================= #
def test_bstconfig_parset():
    bst = BSTConfig.fromParset(
        join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
    )
    assert bst.nSubBands == 768
    assert bst.nPolars == 2
    assert bst.durationSec == 7200

    bst = BSTConfig.fromParset(
        Parset(
            join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
        )
    )
    assert bst.volume.value == pytest.approx(42.19, 1e-2)
# ============================================================= #


# ============================================================= #
# --------------------- test_nickelconfig --------------------- #
# ============================================================= #
def test_nickelconfig():
    nickel = NICKELConfig()
    assert nickel.nSubBands == 384
    assert nickel.nPolars == 4
    assert nickel.durationSec == 0
    assert nickel.nChannels == 64
    assert nickel.timeRes == pytest.approx(1.0, 1e-1)
    assert nickel.nMAs == 96

    nickel = NICKELConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=192,
        nPolars=4,
        nChannels=32,
        timeRes=2.,
        nMAs=56
    )
    vol = nickel.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Tibyte).value == pytest.approx(0.257, 1e-3)
# ============================================================= #


# ============================================================= #
# ----------------- test_nickelconfig_parset ------------------ #
# ============================================================= #
def test_nickelconfig_parset():
    nickel = NICKELConfig.fromParset(
        join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
    )
    assert nickel.nSubBands == 32
    assert nickel.nPolars == 4
    assert nickel.durationSec == 7200
    assert nickel.nChannels == 64
    assert nickel.nMAs == 57
    assert nickel.timeRes == 1

    nickel = NICKELConfig.fromParset(
        Parset(
            join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
        )
    )
    assert nickel.volume.value == pytest.approx(726.42, 1e-2)
# ============================================================= #


# ============================================================= #
# ----------------------- test_tfconfig ----------------------- #
# ============================================================= #
def test_tfconfig():
    tf = TFConfig()
    assert tf.nSubBands == 768
    assert tf.nPolars == 4
    assert tf.durationSec == 0
    assert tf.timeRes * 1e3 == pytest.approx(5.24, 1e-2)
    assert tf.freqRes == pytest.approx(6103.5, 1e-1)

    tf = TFConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=192,
        nPolars=4,
        timeRes=0.020,
        freqRes=3000
    )
    assert tf.freqRes == pytest.approx(3051.76, 1e-2)
    assert tf.timeRes * 1e2 == pytest.approx(2.10, 1e-2)
    vol = tf.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Gibyte).value == pytest.approx(15.72, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------- test_tfconfig_parset -------------------- #
# ============================================================= #
def test_tfconfig_parset():
    tf = TFConfig.fromParset(
        join(dirname(__file__), 'test_data/venus_tracking_tf.parset')
    )
    assert len(tf._beamConfigs) == 4
    assert tf._beamConfigs[0].nSubBands == 192
    assert tf._beamConfigs[0].nPolars == 4
    assert tf._beamConfigs[0].durationSec == 14400
    assert tf._beamConfigs[0].timeRes * 1e2 == pytest.approx(1.05, 1e-2)
    assert tf._beamConfigs[0].freqRes == pytest.approx(3051.8, 1e-1)

    assert tf.volume.value == pytest.approx(1005.8, 1e-1)

    tf = TFConfig.fromParset(
        Parset(
            join(dirname(__file__), 'test_data/zenith_tracking_tf.parset')
        )
    )
    assert len(tf._beamConfigs) == 1
    assert tf._beamConfigs[0].nSubBands == 192
    assert tf._beamConfigs[0].nPolars == 4
    assert tf._beamConfigs[0].durationSec == 3520
    assert tf._beamConfigs[0].timeRes * 1e2 == pytest.approx(2.10, 1e-2)
    assert tf._beamConfigs[0].freqRes == pytest.approx(3051.8, 1e-1)

    assert tf.volume.value == pytest.approx(30.73, 1e-2)
# ============================================================= #


# ============================================================= #
# ---------------------- test_rawconfig ----------------------- #
# ============================================================= #
def test_rawconfig():
    raw = RAWConfig()
    assert raw.nSubBands == 192
    assert raw.nPolars == 4
    assert raw.durationSec == 0
    assert raw.nBits == 8

    raw = RAWConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=192,
        nPolars=4,
        nBits=8,
    )
    vol = raw.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Gibyte).value == pytest.approx(243.1, 1e-1)
# ============================================================= #


# ============================================================= #
# ------------------- test_rawconfig_parset ------------------- #
# ============================================================= #
def test_rawconfig_parset():
    raw = RAWConfig.fromParset(
        join(dirname(__file__), 'test_data/seti_tracking_raw.parset')
    )
    assert len(raw._beamConfigs) == 5
    assert raw._beamConfigs[0].nSubBands == 144
    assert raw._beamConfigs[0].nPolars == 4
    assert raw._beamConfigs[0].durationSec == 600
    assert raw._beamConfigs[0].nBits == 8

    assert raw.volume.value == pytest.approx(282.89, 1e-2)
# ============================================================= #


# ============================================================= #
# ---------------------- test_foldconfig ---------------------- #
# ============================================================= #
def test_foldconfig():
    fold = PulsarFoldConfig()
    assert fold.nSubBands == 192
    assert fold.nPolars == 4
    assert fold.durationSec == 0
    assert fold.tFold == pytest.approx(10.74, 1e-2)
    assert fold.nBins == 2048

    fold = PulsarFoldConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=192,
        nPolars=4,
        nBins=1024,
        tFold=10.73741824
    )
    vol = fold.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Mibyte).value == pytest.approx(486.15, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------- test_foldconfig_parset ------------------ #
# ============================================================= #
def test_foldconfig_parset():
    fold = PulsarFoldConfig.fromParset(
        join(dirname(__file__), 'test_data/pulsar_tracking_fold.parset')
    )
    assert len(fold._beamConfigs) == 2
    assert fold._beamConfigs[0].nSubBands == 192
    assert fold._beamConfigs[0].nPolars == 4
    assert fold._beamConfigs[0].durationSec == 7140
    assert fold._beamConfigs[0].nBins == 2048
    assert fold._beamConfigs[0].tFold == pytest.approx(10.74, 1e-2)

    assert fold.volume.value == pytest.approx(7.73, 1e-2)
# ============================================================= #


# ============================================================= #
# --------------------- test_singleconfig --------------------- #
# ============================================================= #
def test_singleconfig():
    single = PulsarSingleConfig()
    assert single.nSubBands == 192
    assert single.nPolars == 4
    assert single.durationSec == 0
    assert single.nBits == 32
    assert single.dsTime == 128

    single = PulsarSingleConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=192,
        nPolars=4,
        nBits=32,
        dsTime=64
    )
    vol = single.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Gibyte).value == pytest.approx(15.19, 1e-2)
# ============================================================= #


# ============================================================= #
# ----------------- test_singleconfig_parset ------------------ #
# ============================================================= #
def test_singleconfig_parset():
    single = PulsarSingleConfig.fromParset(
        join(dirname(__file__), 'test_data/pulsar_tracking_single.parset')
    )
    assert len(single._beamConfigs) == 3
    assert single._beamConfigs[0].nSubBands == 192
    assert single._beamConfigs[0].nPolars == 1
    assert single._beamConfigs[0].durationSec == 4740
    assert single._beamConfigs[0].nBits == 32
    assert single._beamConfigs[0].dsTime == 128

    assert single.volume.value == pytest.approx(15.32, 1e-2)
# ============================================================= #


# ============================================================= #
# -------------------- test_waveolafconfig -------------------- #
# ============================================================= #
def test_waveolafconfig():
    wave = PulsarWaveConfig()
    assert wave.nSubBands == 192
    assert wave.durationSec == 0

    wave = PulsarWaveConfig(
        durationSec=TimeDelta(1800, format='sec'),
        nSubBands=192
    )
    vol = wave.volume
    assert isinstance(vol, u.Quantity)
    assert vol.to(u.Gibyte).value == pytest.approx(140.5, 1e-1)
# ============================================================= #


# ============================================================= #
# ---------------- test_waveaolafconfig_parset ---------------- #
# ============================================================= #
def test_waveaolafconfig_parset():
    wave = PulsarWaveConfig.fromParset(
        join(dirname(__file__), 'test_data/pulsar_tracking_waveolaf.parset')
    )
    assert len(wave._beamConfigs) == 2
    assert wave._beamConfigs[0].nSubBands == 192
    assert wave._beamConfigs[0].durationSec == 3520

    assert wave.volume.value == pytest.approx(558.79, 1e-2)
# ============================================================= #


# ============================================================= #
# ------------------- test_obsconfig_parset ------------------- #
# ============================================================= #
def test_obsconfig_parset():
    obs = ObsConfig.fromParset(
        join(dirname(__file__), 'test_data/pulsar_tracking_waveolaf.parset')
    )
    assert len(obs.nickel) == 1
    assert isinstance(obs.nickel[0], NICKELConfig)
    assert isinstance(obs.tf[0], TFConfig)
    assert isinstance(obs.raw[0], RAWConfig)
    assert isinstance(obs.pulsar_fold[0], PulsarFoldConfig)
    assert isinstance(obs.pulsar_waveolaf[0], PulsarWaveConfig)
    assert isinstance(obs.pulsar_single[0], PulsarSingleConfig)
    assert isinstance(obs.volume, dict)
    assert obs.volume['pulsar_fold'].to('Mibyte').value == pytest.approx(3867.0, 1e-1)
# ============================================================= #


# ============================================================= #
# ----------------- test_obsconfig_parsetlist ----------------- #
# ============================================================= #
def test_obsconfig_parsetlist():
    parsets = [
        join(dirname(__file__), 'test_data/pulsar_tracking_waveolaf.parset'),
        join(dirname(__file__), 'test_data/zenith_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/seti_tracking_raw.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/venus_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_single.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_fold.parset'),
        join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
    ]
    obs = ObsConfig.fromParsetList(
        parsets
    )
    assert len(obs.nickel) == 8
    assert isinstance(obs.volume, dict)
    assert obs.volume['bst'].to('Mibyte').value == pytest.approx(264.5, 1e-1)
# ============================================================= #


# ============================================================= #
# ----------------- test_obsconfig_cumulative ----------------- #
# ============================================================= #
def test_obsconfig_cumulative():
    parsets = [
        join(dirname(__file__), 'test_data/pulsar_tracking_waveolaf.parset'),
        join(dirname(__file__), 'test_data/zenith_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/seti_tracking_raw.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/venus_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_single.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_fold.parset'),
        join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
    ]
    obs = ObsConfig.fromParsetList(
        parsets
    )
    times, volumes = obs.getCumulativeVolume('nickel')
    assert isinstance(times, Time)
    assert times.size == 8
    assert isinstance(volumes, np.ndarray)
    assert volumes.size == 8
    assert volumes[7] * 10 == pytest.approx(7.10, 1e-2) 
# ============================================================= #


# ============================================================= #
# --------------- test_obsconfig_plotcumulative --------------- #
# ============================================================= #
@patch('matplotlib.pyplot.show')
def test_obsconfig_plotcumulative(mock_show):
    parsets = [
        join(dirname(__file__), 'test_data/pulsar_tracking_waveolaf.parset'),
        join(dirname(__file__), 'test_data/zenith_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/seti_tracking_raw.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/venus_tracking_tf.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_single.parset'),
        join(dirname(__file__), 'test_data/pulsar_tracking_fold.parset'),
        join(dirname(__file__), 'test_data/vira_tracking_nickel.parset')
    ]
    obs = ObsConfig.fromParsetList(
        parsets
    )
    # Default plot
    obs.plotCumulativeVolume()
    # Custom plot
    obs.plotCumulativeVolume(
        figname='',
        receivers=['nickel', 'bst'],
        unit='Mibyte',
        title='Titre de la figure',
        figsize=(10, 7),
        scale='log',
        grid=True,
        tMin='2020-01-01',
        tMax='2020-12-31'
    )
# ============================================================= #

