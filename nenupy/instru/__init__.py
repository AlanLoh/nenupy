#! /usr/bin/python3
# -*- coding: utf-8 -*-


import json
import logging
log = logging.getLogger(__name__)
from os.path import join, dirname
from scipy.io import readsav


with open(join(dirname(__file__), "nenufar_array.json")) as nenufar:
    nenufar_miniarrays = json.load(nenufar)
    log.debug("NenuFAR Mini-Array positions loaded.")

with open(join(dirname(__file__), "miniarray_antennas.json")) as miniarray:
    miniarray_antennas = json.load(miniarray)
    log.debug("Mini-Array antenna positions loaded.")

with open(join(dirname(__file__), "low_noise_amplifier.json")) as lna:
    lna_gain = json.load(lna)
    log.debug("Low Noise Amplifier gains loaded.")

squint_table = readsav(join(dirname(__file__), "squint_table.sav"))
log.debug("Beam squint table loaded.")


from nenupy.instru.instrument_tools import *
from nenupy.instru.nenufar import Polarization, MiniArray, NenuFAR, NenuFAR_Configuration

