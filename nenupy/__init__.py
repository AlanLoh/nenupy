#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

from .read import SST, BST, XST
from .read import __all__ as allread

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2018, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.3.11'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'

__all__ = [] 
__all__.extend(allread)
# __all__.extend(['beam', 'skymodel', 'astro'])