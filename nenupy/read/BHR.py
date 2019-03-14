#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read NenuFAR BHR data
        Originally by C. Viou
        Adapted to Nenupy by A. Loh
"""

__author__ = ['Alan Loh', 'Cedric Viou']
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh', 'Cedric Viou']
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = ['BHR']


import os
import numpy as np

from astropy import units as u
from astropy.time import Time

# ============================================================= #
# ---------------------------- BHR ---------------------------- #
# ============================================================= #

class BHR(object):
    """
    """
    def __init__(self, obsfile):
        self.obsfile = obsfile


    # ========================================================= #
    # --------------------- Getter/Setter --------------------- #
    @property
    def obsfile(self):
        """ BHR observaiton file, should end with '.spectra'
        """
        return self._obsfile
    @obsfile.setter
    def obsfile(self, o):
        assert o.endswith('.spectra'), 'Wrong file type'
        o = os.path.abspath(o)
        assert os.path.isfile(o), 'File {} not found'.format(o)

        head_dtype = np.dtype([('idx', 'uint64'),
                          ('TIMESTAMP', 'uint64'),
                          ('BLOCKSEQNUMBER', 'uint64'),
                          ('fftlen', 'int32'),
                          ('nfft2int', 'int32'),
                          ('fftovlp', 'int32'),
                          ('apodisation', 'int32'),
                          ('nffte', 'int32'),
                          ('nbchan', 'int32')])
        header = np.fromfile(o, count=1, dtype=head_dtype)[0]
        # self.lane = header['idx']
        self.nchan = header['nbchan']
        self.nffte = header['nffte']
        self.fftlen = header['fftlen']
        self.freq_res = 0.1953125 / self.fftlen * u.MHz
        self.time_res = self.fftlen * header['nfft2int'] * 5.12 * u.us

        chan_dtype = np.dtype([('lane', 'int32'),
                               ('beam', 'int32'),
                               ('channel', 'int32'),
                               ('fft0', 'float32', (header['nffte'], header['fftlen'], 2)),
                               ('fft1', 'float32', (header['nffte'], header['fftlen'], 2)),
                               ])

        block_dtype = np.dtype([('idx', 'uint64'),
                             ('TIMESTAMP', 'uint64'),
                             ('BLOCKSEQNUMBER', 'uint64'),
                             ('fftlen', 'int32'),
                             ('nfft2int', 'int32'),
                             ('fftovlp', 'int32'),
                             ('apodisation', 'int32'),
                             ('nffte', 'int32'),
                             ('nbchan', 'int32'),
                             ('data', chan_dtype, (header['nbchan'])),
                             ])

        self.alldata = np.fromfile(o, count=-1, dtype=block_dtype)



    # ========================================================= #
    # ------------------------ Methods ------------------------ #
    def select(self):
        """
        """
        max_bsn = 200e6 / 1024
        self.times = self.alldata['TIMESTAMP'] + self.alldata['BLOCKSEQNUMBER']/max_bsn
        self.times = Time(self.times, format='unix')

        # data_selected1 = b.alldata['data']['fft0'][0, :, :, 1, 0] # block number, nbchan, nffte, fftlen, polar
        # data_selected2 = b.alldata['data']['fft0'][1, :, :, 1, 0]
        # return np.hstack((data_selected1, data_selected2))

        # data_selected = b.alldata['data']['fft0'][0:3, :, :, :, 0]
        # nblocks, nchan, nt, fftl = data_selected.shape
        # data_selected = data_selected.transpose((0, 2, 1, 3)).copy()
        # data = data_selected.reshape((nblocks * self.nffte,
        #                      nchan * self.fftlen))

        data_selected = b.alldata['data']['fft0'][0:80, :, :, 0, 0]
        nblocks, nchan, nt = data_selected.shape
        data_selected = data_selected.transpose((0, 2, 1))
        data = data_selected.reshape((nblocks * self.nffte, nchan))
        return data

# ============================================================= #
# ============================================================= #

import pylab as plt
b = BHR(obsfile='/Users/aloh/Documents/Work/NenuFAR/Nenupy3_Tests/BHR/B1919+21_B1_TRACKING_BHR_20190212_123033_0.spectra')
# print(b.time_res, b.freq_res)
data = b.select()
plt.imshow(np.log10(data.T), origin='lower', aspect='auto')
plt.show()

















"""

Description of .spectra files as produced by 'tf' code with undysputed           (Feb 11, 2019)
/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
for each block
 ULONG    first effective sample index as incremented by the 'tf' code
            used for this block (starting with 0!)
 ULONG    TIMESTAMP of first sample (see NenuFAR/LOFAR beamlet description)
 ULONG    BLOCKSEQNUMBER of first sample (see NenuFAR/LOFAR beamlet description)
 LONGINT  fftlen : FFT length -> freq resolution= 0.1953125/fftlen MHz
 LONGINT  nfft2int : Nber of integrated FFTs -> time resolution= fftlen*nfft2int*5.12 us
 LONGINT  ffovlp : FFToverlap 0/1                    (not used yet)
 LONGINT  apodisation : (0: none, 54=hamming, ...)   (not used yet)
 LONGINT  nffte : nber of FFTs to be read per beamlet within this block
             (so we have 'nffte' spectra/time_sample per block)
 LONGINT  nbchan : nbr of subsequent beamlets/channels (nb of NenuFAR/LOFAR 195kHz beamlets)
             (so we have 'fftlen*nbchan' frequencies for each spectra)

 for each beamlet
  LONGINT  lane index (0, 1, 2 or 3)
  LONGINT  beam index (numerical beam : 0, 1, ...)
  LONGINT  beamlet/channel index (0...768, 0 is 0.000MHz)
  FLOAT    pol0 data : 'nffte' time_samples each made of 'fftlen' pairs [XrXr+XiXi : YrYr+YiYi]
  FLOAT    pol1 data : 'nffte' time_samples each made of 'fftlen' pairs [XrYr+XiYi : XrYi-XiYr]
/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/


Calculs de Stokes (booklog du 6 janvier 2009)

LINEAR FEEDS
  les polars complexes sont (X,Y) ou (Xr,Xi,Yr,Yi) pour (E,W) ou (H,V)

ASP files
  dans les fichiers .asp on trouve
       XX = XrXr + XiXi
       YY = YrYr + YiYi
       Re(X*Y) = XrYr + XiYi
       Im(X*Y) = XrYi - XiYr

ASC files
  in .asc files without "-S" option of asp_prof2Dn (added on Nov 16, 2012)
       XX = XrXr + XiXi
       YY = YrYr + YiYi
       Re(X*Y) = XrYr + XiYi
       Im(X*Y) = XrYi - XiYr

ASC files
  dans les fichiers .asc avec option "-S" de asp_prof, asp_prof2Dn
      I = (XX+YY)/2
      Q = (XX-YY)/2
      U = Re(X*Y)
      V = Im(X*Y)           voir p121 Handbook of Pulsar Astronomy

dans les plots/figures avec l'option 'pol' de pltasc
      IntTot = I
      Circ = V
      Lin = sqrt(Q*Q+U*U)
      PA = 1/2 ArcTg(U/Q)   voir p187 Handbook of Pulsar Astronomy

Added on Nov 23, 2014

CIRCULAR FEEDS
  les polars complexes sont (R,L) ou (Rr,Ri,Lr,Li), alors
     RR = RrRr + RiRi
     LL = LrLr + LiLi
     Re(R*.L) = RrLr + RiLi
     Im(R*.L) = RrLi - RiLr
  et
     I = (RR+LL)/2
     Q = Re(R*.L)
     U = Im(R*.L)
     V = (RR-LL)/2
  si (X,Y) permettent de calculer XX, YY, Re(X*Y) et Im(X*Y),
 alors on peut retrouver RR et LL avec :
   RR = (XX+YY)/2 + Im(X*Y)
   LL = (XX+YY)/2 - Im(X*Y)


"""


