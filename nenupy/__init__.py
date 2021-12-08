#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '2.0.13'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'


import logging
import sys
import json
import os
from os.path import join, dirname
import functools
import inspect

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time


# ============================================================= #
# ------------------- Logging configuration ------------------- #
# ============================================================= #
logging.basicConfig(
    # filename='nenupy.log',
    # filemode='w',
    stream=sys.stdout,
    level=logging.WARNING,
    #format='%(asctime)s -- %(levelname)s: %(message)s',
    #format='\033[1m%(asctime)s\033[0m | %(levelname)s: \033[34m%(message)s\033[0m',
    format='%(asctime)s | %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- HiddenPrints ------------------------ #
# ============================================================= #
class HiddenPrints:
    """ Hides unwanted prints nd warning.
        Usage:
            >>> with HiddenPrints():
            >>>    function_with_prints()
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- nenufar_position ---------------------- #
# ============================================================= #
with open(join(dirname(__file__), "telescopes.json")) as array_file:
    arrays = json.load(array_file)
    nenufar_position = EarthLocation(
        lat=arrays["nenufar"]["lat"] * u.deg,
        lon=arrays["nenufar"]["lon"] * u.deg,
        height=arrays["nenufar"]["height"] * u.m
    )
    log.debug("NenuFAR position loaded.")
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- logdebug -------------------------- #
# ============================================================= #
def getDefaultArgs(func):
    """ Gets the default arguments of a function. """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def logdebug(func):
    """ Decorates a function to add a log of the call. """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        funcArgs = inspect.signature(func).bind(*args, **kwargs).arguments
        kwds = getDefaultArgs(func)
        kwds.update(funcArgs)
        signature = ", ".join(
            "{}={!r}".format(*item) for item in kwds.items()
        )
        log.debug(
            f"Calling {func.__qualname__}({signature})."
        )
        t0 = Time.now()
        result = func(*args, **kwargs)
        t1 = Time.now()
        log.info(
            f"{func.__qualname__} completed in {(t1 - t0).sec:.3f} sec."
        )
        return result
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- LogMethodMetaClass --------------------- #
# ============================================================= #
class LogMethodMetaClass(type):
    """ Metaclass with all the methods decorated with logdebug. """


    def __new__(cls, name, bases, local):
        for attr in local:
            if callable(local[attr]) and not attr.startswith("_"):
                local[attr] = logdebug(local[attr])
        return type.__new__(cls, name, bases, local)

    """
    or add this to the class
        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            local = cls.__dict__#.items()
            
            for attr in cls.__dict__:
                if callable(getattr(cls, attr)) and not attr.startswith("_"):
                    setattr(cls, attr, logdebug(getattr(cls, attr)))
            return instance
    """
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ DummyCtMgr ------------------------- #
# ============================================================= #
class DummyCtMgr(object):
    """ Class to be used as default context manager, does nothing. """


    def __enter__(self):
        pass


    def __exit__(self, ext, exv, trb):
        pass
# ============================================================= #
# ============================================================= #
