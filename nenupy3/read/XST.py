#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to read XST NenuFAR data
        by A. Loh
"""

import os
import sys
import numpy as np


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['XST']


class XST():
    def __init__(self, obsfile):
        self.obsfile = obsfile

    # ================================================================= #
    # =========================== Methods ============================= #
    def convertMS(self):
        """ Convert the XST data into a Measurement Set
        """
        return

    
