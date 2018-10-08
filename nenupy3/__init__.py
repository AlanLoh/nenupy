#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

from .read import SST, BST, XST
from .read import __all__ as allread
# import read
# from . import beam
# from . import skymodel
# from . import astro

__all__ = [] 
__all__.extend(allread)
# __all__.extend(['beam', 'skymodel', 'astro'])