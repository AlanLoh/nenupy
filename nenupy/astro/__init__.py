#! /usr/bin/python3
# -*- coding: utf-8 -*-

from .astro_tools import *
import json
from os.path import join, dirname

with open(join(dirname(__file__), "common_sources.json")) as sources:
    common_sources = json.load(sources)

