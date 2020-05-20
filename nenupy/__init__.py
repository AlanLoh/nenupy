#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '1.1.0'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'


import logging
import sys


logging.basicConfig(
    # filename='nenupy.log',
    # filemode='w',
    stream=sys.stdout,
    level=logging.WARNING,
    format='%(asctime)s -- %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

