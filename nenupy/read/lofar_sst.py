#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
    Documentation:
    http://lofar.ie/wp-content/uploads/2018/03/station_data_cookbook_v1.2.pdf

"""



from os.path import isfile, abspath, basename
import numpy as np
from astropy.time import Time

class LOFAR_SST(object):
    """
    """


    def __init__(self, sst, bitmode=8):
        self.start = None
        self.rcu = None

        self.sst = sst

        self.dtype = np.dtype([
            ('data', 'float64')
            ])


    @property
    def sst(self):
        return self._sst
    @sst.setter
    def sst(self, s):
        s = abspath(s)
        if not isfile(s):
            raise FileNotFoundError(
                'Unable to find {}.'.format(s)
                )
        if not s.endswith('.dat'):
            raise TypeError(
                '{} is not a .dat file.'.format(s)
                )
        if not 'sst' in s.lower():
            raise ValueError(
                '{} is not a SST file.'.format(s)
                )
        self._sst = s
        self._parse_filename()
        return

    def load_data(self):
        """
        """
        with open(self.sst, 'rb') as rf:
            data = np.memmap(rf, dtype=self.dtype, mode='r')
        return data['data'].reshape((int(data['data'].size/512), 512))


    def _parse_filename(self):
        """ e.g. 20111205_105031_sst_rcu000.dat
        """
        import re
        fname = basename(self._sst)
        match = re.search(
            '(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_sst_rcu(\d{3})',
            fname
            )
        if not match:
            raise ValueError(
                'Not able to parse {}'.format(fname)
                )
        ye, mo, da, ho, mi, se, rc = match.groups()
        date = '-'.join([ye, mo, da])
        hour = ':'.join([ho, mi, se])
        self.start = Time('T'.join([date, hour]), format='isot')
        self.rcu = rc
        return



