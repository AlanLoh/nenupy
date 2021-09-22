#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '1.2.26'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'


import logging
import sys
import json
from os.path import join, dirname
from astropy.coordinates import EarthLocation
import astropy.units as u


logging.basicConfig(
    # filename='nenupy.log',
    # filemode='w',
    stream=sys.stdout,
    level=logging.WARNING,
    format='%(asctime)s -- %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


with open(join(dirname(__file__), "telescopes.json")) as array_file:
    arrays = json.load(array_file)
    nenufar_position = EarthLocation(
        lat=arrays["nenufar"]["lat"] * u.deg,
        lon=arrays["nenufar"]["lon"] * u.deg,
        height=arrays["nenufar"]["height"] * u.m
    )
