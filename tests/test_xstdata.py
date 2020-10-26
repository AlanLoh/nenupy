#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
import numpy as np
from os.path import join, dirname
import pytest
from unittest.mock import patch 

from nenupy.crosslet import XST_Data, NenuFarTV
from nenupy.beamlet import SData


# ============================================================= #
# ----------------------- test_xstread ------------------------ #
# ============================================================= #
def test_xstread():
    xst = XST_Data(
        xstfile=join(dirname(__file__), 'test_data/XST.fits')
    )
    time_size = xst.times.size
    sb_size = xst.sb_idx.size
    assert sb_size == 16
    assert xst.vis.shape[:-1] == (time_size, sb_size)
# ============================================================= #


# ============================================================= #
# --------------------- test_xstbeamform ---------------------- #
# ============================================================= #
def test_xstbeamform():
    xst = XST_Data(
        xstfile=join(dirname(__file__), 'test_data/XST.fits')
    )
    bf = xst.beamform(
        az=180*u.deg,
        el=90*u.deg,
        pol='NW',
        ma=None
    )
    assert isinstance(bf, SData)
    assert bf.time[0].isot == '2020-02-19T18:00:03.000'
    assert bf.freq.mean().value == pytest.approx(74.67, 1e-2)
    assert bf.db[8] == pytest.approx(80.73, 1e-2)
    assert bf.amp[15] == pytest.approx(71325439, 1e0)
    bf = xst.beamform(
        az=180,
        el=90,
        pol='NE',
        ma=[0, 1],
        calibration='none'
    )
    assert isinstance(bf, SData)
    assert bf.db[8] == pytest.approx(60.82, 1e-2)
    assert bf.amp[15] == pytest.approx(748046, 1e0)
# ============================================================= #


# ============================================================= #
# ----------------------- test_xstimage ----------------------- #
# ============================================================= #
@patch('matplotlib.pyplot.show')
def test_xstimage(mock_show):
    xst = XST_Data(
        xstfile=join(dirname(__file__), 'test_data/XST.fits')
    )
    with pytest.raises(IndexError):
        image = xst.image(
            fIndices=np.array([17])
        )
    with pytest.raises(IndexError):
        image = xst.image(
            fIndices=np.arange(17)
        )
    with pytest.raises(TypeError):
        image = xst.image(
            fIndices=(1, 2)
        )
    # Zenith, full bandwidth
    image = xst.image(
        resolution=1,
        fov=5,
        center=None,
        fIndices=None
    )
    assert isinstance(image, NenuFarTV)
    assert image.analogPointing.az.deg == 0.
    assert image.analogPointing.alt.deg == 90.
    assert image.phaseCenter.ra.deg == pytest.approx(60.998, 1e-3)
    assert image.phaseCenter.dec.deg == pytest.approx(47.321, 1e-3)
    assert image.time.isot == '2020-02-19T18:00:03.000'
    assert image.fov.value == 5.
    assert image.meanFreq.to(u.MHz).value == pytest.approx(74.67, 1e-2)
    assert image.nside == 64
    assert image.skymap.data.max() == pytest.approx(1230, 1e0)
    # Center on Tau A, first frequency
    image = xst.image(
        resolution=0.2,
        fov=5,
        center=SkyCoord.from_name('Tau A'),
        fIndices=np.array([0])
    )
    assert image.phaseCenter.ra.deg == pytest.approx(83.633, 1e-3)
    assert image.phaseCenter.dec.deg == pytest.approx(22.014, 1e-3)
    assert image.nside == 256
    assert image.skymap.data.max() == pytest.approx(33353, 1e0)
    # Plot
    image.plot(
        db=False,
        center=image.phaseCenter,
        size=5
    )
    # image.savePng()
# ============================================================= #


# ============================================================= #
# --------------------- test_xstnearfield --------------------- #
# ============================================================= #
@patch('matplotlib.pyplot.show')
def test_xstnearfield(mock_show):
    xst = XST_Data(
        xstfile=join(dirname(__file__), 'test_data/XST.fits')
    )
    with pytest.raises(IndexError):
        nf = xst.nearfield(
            fIndices=np.array([17])
        )
    with pytest.raises(IndexError):
        nf = xst.nearfield(
            fIndices=np.arange(17)
        )
    with pytest.raises(TypeError):
        nf = xst.nearfield(
            fIndices=(1, 2)
        )
    nf = xst.nearfield(
        radius=400,
        npix=32,
        sources=['Vir A', 'Tau A'])
    assert nf.meanFreq.to(u.MHz).value == pytest.approx(74.67, 1e-2)
    assert nf.nPix == 32
    assert nf.radius.value == 400.
    assert nf.antNames.size == 55
    assert nf.obsTime.isot == '2020-02-19T18:00:03.000'
    assert isinstance(nf.simuSources, dict)
    assert 'Tau A' in nf.simuSources.keys()
    assert isinstance(nf.maxPosition, EarthLocation)
    assert nf.maxPosition[0].lon.deg == pytest.approx(2.196, 1e-3)
    assert nf.maxPosition[0].lat.deg == pytest.approx(47.373, 1e-3)
    assert isinstance(nf.nfImage, np.ndarray)
    assert nf.nfImage.shape == (32, 32)
    assert nf.nfImage[16, 16] == pytest.approx(183852, 1e0)
    # Plot
    nf.plot()
# ============================================================= #

