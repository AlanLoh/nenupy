#! /usr/bin/python3
# -*- coding: utf-8 -*-

import json
from os.path import join, dirname

with open(join(dirname(__file__), 'parset_user_options.json')) as parset_options:
    PARSET_OPTIONS = json.load(parset_options)

# from .tapdatabase import ObsDatabase
from .sqldatabase import ParsetDataBase
from .parset import Parset, ParsetUser
from .pointing import *
from .obs_config import *
