#! /usr/bin/python3
# -*- coding: utf-8 -*-

import json
from os.path import join, dirname

with open(join(dirname(__file__), 'parset_user_options.json')) as parset_options:
    PARSET_OPTIONS = json.load(parset_options)
    for field in PARSET_OPTIONS:
        for key in PARSET_OPTIONS[field]:
            PARSET_OPTIONS[field][key]["modified"] = False

# from .tapdatabase import ObsDatabase
from .sqldatabase import ParsetDataBase
from .parset import Parset, ParsetUser
# from .pointing_obs import *
from .obs_config import *
