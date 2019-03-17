#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
"""

from nenupy.hpx import Anabeam, Digibeam, Skymodel, Tracking, Transit
import pylab as plt
import numpy as np

from nenupy import BST, SST


# # TEST TRACKING
# b = BST('/Users/aloh/Desktop/NenuFAR_Data/20181030_200800_BST.fits')
# b.select(freq=60)

# t=Tracking()
# t.from_bst(obs='/Users/aloh/Desktop/NenuFAR_Data/20181030_200800_BST.fits', dt=240, resol=0.2, freq=60)

# x = (t.simulation['time'] - t.simulation['time'][0]).sec / 60
# y = t.simulation['amp']

# scale = np.median(b.data['amp']) / np.median(y)
# plt.plot(x, 10 * np.log10(y))

# plt.plot( (b.data['time']-t.simulation['time'][0]).sec/60, 10 * np.log10(b.data['amp']/scale) )

# plt.xlabel('Time (min since {})'.format(t.simulation['time'][0].iso))
# plt.ylabel('dB')
# plt.show()
# plt.close('all')



# # TEST TRANSIT CYGA
# b = BST('/Users/aloh/Desktop/NenuFAR_Data/20181018_195600_BST.fits')
# b.select(freq=60)
# t=Transit()
# t.from_bst(obs='/Users/aloh/Desktop/NenuFAR_Data/20181018_195600_BST.fits', dt=60, resol=0.2, freq=60)
# x = (t.simulation['time'] - t.simulation['time'][0]).sec / 60
# y = t.simulation['amp']
# scale = np.median(b.data['amp']) / np.median(y)
# plt.plot(x, 10 * np.log10(y))
# plt.plot( (b.data['time']-t.simulation['time'][0]).sec/60, 10 * np.log10(b.data['amp']/scale) )
# plt.xlabel('Time (min since {})'.format(t.simulation['time'][0].iso))
# plt.ylabel('dB')
# plt.show()
# plt.close('all')


# TEST TRANSIT CYGA 2 MINIARRAYS
t = Transit()
t.from_bst(obs='/Users/aloh/Desktop/NenuFAR_Data/20190228_071900_BST.fits', dt=600, freq=60, resol=0.2)
b = BST('/Users/aloh/Desktop/NenuFAR_Data/20190228_071900_BST.fits')
b.select(freq=60)
x = (t.simulation['time'] - t.simulation['time'][0]).sec / 60
y = t.simulation['amp']
scale = np.median(b.data['amp']) / np.median(y)
plt.plot(x, 10 * np.log10(y*scale))
plt.plot( (b.data['time']-t.simulation['time'][0]).sec/60, 10 * np.log10(b.data['amp']) )
plt.ylim((69.5, 74))
plt.xlabel('Time (min since {})'.format(t.simulation['time'][0].iso))
plt.ylabel('dB')
plt.show()
plt.close('all')


# # TEST TRANSIT SST ANT UNIQUE ZENITH
# from astropy.io import fits
# from astropy.time import Time
# FREQ=60
# files = ['/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_140000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_150000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_160000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_170000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_180000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_190000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_200000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_210000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_220000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161020_230000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_000000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_010000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_020000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_030000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_040000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_050000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_060000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_070000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_080000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_090000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_100000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_110000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_120000_MR0_NW_SST.fits',
#     '/Users/aloh/Documents/Work/NenuFAR/Tests/20161021_130000_MR0_NW_SST.fits']
# freqs = np.arange(0.09765625, 99.90234375+0.1953125, 0.1953125)
# f_ind = (np.abs(freqs - FREQ)).argmin()
# for ff in sorted(files):
#     data = fits.getdata(ff, ext=1, memmap=True)
#     if 'mr0data' in locals():
#         mr0data = np.hstack( (mr0data, data['data'][:, f_ind]) )
#         mr0time = np.hstack( (mr0time, data['jd']) )
#     else :
#         mr0data = data['data'][:, f_ind]
#         mr0time = data['jd']
# mr0time = Time(mr0time, format='jd')
# t=Transit()
# t.predict(src=None, resol=1, time=mr0time[0], freq=FREQ, duration=86400, miniarrays=[1], azana=180, elana=90, azdig=180, eldig=90, dt=3600, polar='NW')
# x = (t.simulation['time'] - t.simulation['time'][0]).sec / 3600
# y = t.simulation['amp']
# scale = np.median(mr0data) / np.median(y)
# plt.plot( (mr0time-t.simulation['time'][0]).sec/3600, 10 * np.log10(mr0data) )
# plt.plot(x, 10 * np.log10(y*scale))
# plt.xlabel('Time (min since {})'.format(t.simulation['time'][0].iso))
# plt.ylabel('dB')
# plt.show()
# plt.close('all')


# # TEST TRANSIT SST ZENITH
# import glob
# FREQ=60
# for mr in range(40):
#     MR=mr
#     sst = SST(sorted(glob.glob('/Users/aloh/Documents/Work/NenuFAR/Nenupy3_Tests/Zenith_SST/*SST.fits'))[0:23])
#     sst.select(freq=FREQ, ma=mr)
#     t=Transit()
#     t.predict(src=None, resol=0.4, time=sst.data['time'][0], freq=FREQ, duration=86400, miniarrays=[MR], azana=180, elana=90, azdig=180, eldig=90, dt=1800, polar='NW')
#     x = (t.simulation['time'] - t.simulation['time'][0]).sec / 3600
#     y = t.simulation['amp']
#     scale = np.median(sst.data['amp']) / np.median(y)
#     plt.plot( (sst.data['time']-t.simulation['time'][0]).sec/3600, 10 * np.log10(sst.data['amp']) )
#     plt.plot(x, 10 * np.log10(y*scale))
#     plt.xlabel('Time (hour since {})'.format(t.simulation['time'][0].iso))
#     plt.ylabel('dB')
#     plt.savefig('/Users/aloh/Documents/Work/NenuFAR/Nenupy3_Tests/Zenith_SST/mr{}_zenith_60mhz.png'.format(MR))
#     # plt.show()
#     plt.close('all')


