#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
Class to compute a BST NenuFAR beam
        by A. Loh
"""

import os
import sys
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.time import Time

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['BSTBeam']

class BSTbeam():
    def __init_(self):
        return