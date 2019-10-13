#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

__author__ = ['Alan Loh']
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = [
    'to_icrs',
    'from_bst',
    'from_beam',
    'ma_beam_size',
    'grating_lobe_sep',
    'test'
    ]


try:
    from mocpy import STMOC, TimeMOC, MOC
except ModuleNotFoundError:
    raise ModuleNotFoundError('You need to install mocpy first.')

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.units import Unit as u
from astropy import constants as const

# ------------------------------------------------------------- #
def to_icrs(az, alt, time):
    """
    """
    from astropy.coordinates import AltAz, ICRS, EarthLocation

    nenufar = EarthLocation.from_geodetic(
        lat=47.376511*u('deg'),
        lon=2.1924002*u('deg'),
        height=136*u('m'))

    # nenufar_times = Time(hd[6].data['timestamp'])
    altaz = AltAz(alt=alt, az=az, obstime=time, location=nenufar)
    icrs = altaz.transform_to(ICRS)

    return icrs.ra, icrs.dec
# ------------------------------------------------------------- #


# ------------------------------------------------------------- #
def from_bst(bst):
    """ From a NenuFAR BST observation, read the configuration
        and returns a TMOC object from digital pointing infos.
        
        Parameters
        ----------
        bst : str
            Filename of the BST observation.
    """
    from nenupy.read.BST_v2 import BST
    bst = BST(bst)
    bst.track_dbeam = 1

    starts = Time(bst._meta['pbe']['timestamp'])
    stops = starts + TimeDelta(bst.duration)

    mean_time = starts + (stops - starts)/2.
    mean_time = Time(mean_time)

    ra, dec = to_icrs(
        az=bst.all_azdig*u('deg'),
        alt=bst.all_eldig*u('deg'),
        time=mean_time
        )

    stmoc = STMOC.from_times_positions(
        times=mean_time,
        time_depth=18,
        lon=ra,
        lat=dec,
        spatial_depth=10
        )
    # nenufar_stmoc.write('./stmoc.fits', overwrite=True)
    # nenufar_tmoc.plot()

    return stmoc
# ------------------------------------------------------------- #


# ------------------------------------------------------------- #
def ma_beam_size(frequency=50):
    """ Returns the Mini Array beam size.
        The size is computed as lambda/d where lambda is the
        wavelength and d is the typical MA diameter (25 m).

        Parameters
        ----------
        frequency  float
            Frequency og observation in MHz

        Returns
        -------
        beam_size : `astropy.quantity`
            Beam size in degrees
    """
    frequency *= u('MHz')
    wavelength = const.c.to(u('m')/u('s')) / frequency.to(u('Hz'))

    ma_diameter = 25 * u('m')
    psf = wavelength / ma_diameter
    return (psf.value * u('rad')).to(u('deg'))
# ------------------------------------------------------------- #


# ------------------------------------------------------------- #
def grating_lobe_sep(frequency=50, order=0):
    """ Returns the angular separation of the grating lobe with
        respect to the pointing for a NenuFAR mini-array.
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.846.4976&rep=rep1&type=pdf
    """
    # p = [4, 3.5, 3, 2.5, 2, 1.5, 1]
    halfbeam = ma_beam_size(frequency) / 2.
    frequency *= u('MHz')
    wavelength = const.c.to(u('m')/u('s')) / frequency.to(u('Hz'))

    k = 2 * np.pi / wavelength

    theta, phi = (0, 0) # zenith
    dx = 5.5 * u('m') #* p[order]
    dy = np.sqrt(3) * d /2.

    ax = k * dx * np.sin(theta) * np.cos(phi)
    ay = k * dy * np.sin(theta) * np.sin(phi)

    # sep = np.arctan((ay + wavelength/dy) / (ax + wavelength/dx))
    sep = np.arcsin(
            np.sqrt(
                (ax +  wavelength/dx)**2. + (ay + wavelength/dy)**2.
            )
        )
    return sep.to(u('deg')) - order * halfbeam
# ------------------------------------------------------------- #


# ------------------------------------------------------------- #
def from_beam(frequency):
    """
    """
    return 
# ------------------------------------------------------------- #


def test(frequency=50, order=4):
    import healpy as hp
    import pylab as plt; import numpy as np
    from nenupy.hpx import Anabeam
    ma_beam = Anabeam(ma=0, freq=frequency, azana=0, elana=90, resol=0.5)
    ma_beam.get_anabeam()
    hp.mollview(ma_beam.anabeam, title='Mini-Array 0 beam at {} MHz'.format(frequency))
    hp.graticule()

    azs = np.linspace(0, 360, 50)
    lat = 90-grating_lobe_sep(frequency, order).value
    # hp.projplot(azs, np.ones(azs.size)*lat, lonlat=True)
    hp.projscatter(azs, np.ones(azs.size)*lat, lonlat=True, s=10, color='red')

    hp.projscatter(azs, 90-np.ones(azs.size)*ma_beam_size(frequency).value/2, lonlat=True, s=10, color='green')


    plt.show()

