#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST file
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "StatisticsData"
]


from abc import ABC
import operator
import re
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy.instru.instrument_tools import sb2freq

import logging
log = logging.getLogger(__name__)


ops = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
}


# ============================================================= #
# ---------------------- StatisticsData ----------------------- #
# ============================================================= #
class StatisticsData(ABC):
    """ """

    def __init__(self, file_name: str):
        self.file_name = file_name
        #self.instrument = None
        #self.pointing = None

        self._meta_data = {}
        self._lazy_load()
    

    def _lazy_load(self):
        """ """
        with fits.open(self.file_name,
            mode='readonly',
            ignore_missing_end=True,
            memmap=True
        ) as f:
            # Metadata loading
            # self.meta['hea'] = f[0].header
            self._meta_data['ins'] = f[1].data
            self._meta_data['obs'] = f[2].data
            self._meta_data['ana'] = f[3].data
            self._meta_data['bea'] = f[4].data
            self._meta_data['pan'] = f[5].data
            self._meta_data['pbe'] = f[6].data

            # # Data loading 
            self.time = Time(f[7].data['JD'], format='jd')
            self.data = f[7].data['data']
            try:
                # For XST data, the frequencies are in the data extension
                self.frequencies = sb2freq(
                    np.unique(f[7].data['xstsubband']).astype("int")
                ) + 195.3125*u.kHz/2 # mid frequency
            except KeyError:
                pass

        return


    @staticmethod
    def _parse_frequency_condition(conditions: str):
        """ """
        condition_list = conditions.replace(" ", "").split("&")

        cond = []
        for condition in condition_list:
            op = re.search('((>=)|(<=)|(==)|(<)|(>))', condition).group(0)
            val = re.search(f'(?<={op})(.*)', condition).group(0)
            val = u.Quantity(val)
            op = ops[op]
            cond.append( lambda x, op=op, val=val: op(x, val) )

        if len(cond) == 2:
            return lambda x, cond1=cond[0], cond2=cond[1]: operator.and_(cond1(x), cond2(x))
        elif len(cond) == 1:
            return cond[0]
        else:
            raise Exception


    @staticmethod
    def _parse_condition(conditions, converter):
        """ """
        condition_list = conditions.replace(" ", "").split("&")

        cond = []
        for condition in condition_list:
            op = re.search('((>=)|(<=)|(==)|(<)|(>))', condition).group(0)
            val = re.search(f'(?<={op})(.*)', condition).group(0)
            val = converter(val)
            op = ops[op]
            cond.append( lambda x, op=op, val=val: op(converter(x), val) )

        if len(cond) == 2:
            return lambda x, cond1=cond[0], cond2=cond[1]: operator.and_(cond1(x), cond2(x))
        elif len(cond) == 1:
            return cond[0]
        else:
            raise Exception


    def _parse_time_condition(self, conditions):
        """ """
        return self._parse_condition(conditions, lambda t: Time(t).jd)


    def _parse_frequency_condition(self, conditions):
        """ """
        return self._parse_condition(conditions, lambda f: u.Quantity(f))
# ============================================================= #
# ============================================================= #

