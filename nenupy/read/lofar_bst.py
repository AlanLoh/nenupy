#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
    Documentation:
    http://lofar.ie/wp-content/uploads/2018/03/station_data_cookbook_v1.2.pdf

"""

from os.path import isfile, abspath, basename
import numpy as np
from astropy.time import Time

class LOFAR_BST(object):
    """
    """


    def __init__(self, bst, bitmode=8):
        self.start = None
        self.polar = None
        self.ring = None

        self.bst = bst
        self.bitmode = bitmode

        self.dtype = np.dtype([
            ('data', 'float64')
            ])


    @property
    def bst(self):
        return self._bst
    @bst.setter
    def bst(self, b):
        b = abspath(b)
        if not isfile(b):
            raise FileNotFoundError(
                'Unable to find {}.'.format(b)
                )
        if not b.endswith('.dat'):
            raise TypeError(
                '{} is not a .dat file.'.format(b)
                )
        if not 'bst' in b.lower():
            raise ValueError(
                '{} is not a BST file.'.format(b)
                )
        self._bst = b
        self._parse_filename()
        return


    @property
    def bitmode(self):
        return self._bitmode
    @bitmode.setter
    def bitmode(self, b):
        if not isinstance(b, int):
            raise TypeError(
                '`bitmode` should be integer.'
                )
        bit2beamlets = {
            '8': 488,
            '16': 244
            }
        if not str(b) in bit2beamlets.keys():
            raise ValueError(
                '`bitmode` only accept values 8 or 16.'
                )
        self.nbeamlets = bit2beamlets[str(b)]
        self._bitmode = b
        return    


    def load_data(self):
        """
        """
        with open(self.bst, 'rb') as rf:
            data = np.memmap(rf, dtype=self.dtype, mode='r')
        size = data['data'].size
        return data['data'].reshape((size//self.nbeamlets, self.nbeamlets))


    def _parse_filename(self):
        """ e.g. 20111205_105947_bst_00Y.dat
        """
        import re
        fname = basename(self._bst)
        match = re.search(
            '(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_bst_0(\d{1})(\S{1})',
            fname
            )
        if not match:
            raise ValueError(
                'Not able to parse {}'.format(fname)
                )
        ye, mo, da, ho, mi, se, ri, po = match.groups()
        date = '-'.join([ye, mo, da])
        hour = ':'.join([ho, mi, se])
        self.start = Time('T'.join([date, hour]), format='isot')
        self.polar = po
        self.ring = ri
        return


