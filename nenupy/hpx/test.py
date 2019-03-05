#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
"""

from astropy.time import Time, TimeDelta
from nenupy.hpx import Anabeam, Digibeam, Skymodel
import pylab as plt
import numpy as np

from nenupy import BST

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['Skymodel']

def test(t0='2018-10-18T20:56:10', t1='2018-10-18T22:06:10', dt=60, az=359.9944, el=78.4559, freq=60, polar='NW', resol=0.1):
# def test(t0='2019-02-28 07:19:40', t1='2019-02-28 11:19:40', dt=600, az=180.0247, el=83.4075, freq=60, polar='NW', resol=0.2):
    t0 = Time(t0)
    t1 = Time(t1)
    dt = TimeDelta(dt, format='sec')

    tstart = Time.now()
    db = Digibeam(azana=az,
                  azdig=az,
                  elana=el,
                  eldig=el,
                  miniarrays=None,#[40, 55],
                  freq=freq,
                  polar=polar,
                  resol=resol)
    beam = db.get_digibeam()
    print('Beam computed {} sec'.format( (Time.now()-tstart).sec ))


    import healpy as hp
    # hp.orthview( np.log10(beam), rot=[0, 90, 90])
    hp.cartview( np.log10(beam))
    hp.graticule()
    plt.show()
    plt.close('all')
    # stop

    tarray = []
    aarray = []

    current = t0
    while current < t1:
        sm = Skymodel(nside=db._nside,
                      freq=freq)

        tstart = Time.now()
        sky = sm.get_skymodel(time=current)
        print('Time {} sec'.format( (Time.now()-tstart).sec ))

        # hp.cartview( np.log10(sky))
        # hp.graticule()
        # plt.show()
        # plt.close('all')

        integ = np.sum(beam * sky)
        
        tarray.append(current.mjd)
        aarray.append(integ)
        
        current += dt

    return Time(np.array(tarray), format='mjd'), np.array(aarray)

x, y = test()
plt.plot((x-x[0]).sec, y)

try:
    # b = BST('/Users/aloh/Desktop/NenuFAR_Data/20190228_071900_BST.fits')
    b = BST('/Users/aloh/Desktop/NenuFAR_Data/20181018_195600_BST.fits')
    b.select(freq=60)
    
    scale = np.median(b.data['amp']) / np.median(y)

    plt.plot( (b.data['time']-x[0]).sec, b.data['amp']/scale )
except:
    pass

plt.show()