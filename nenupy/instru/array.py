#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ['Alan Loh']
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['core_ma',
           'remote_ma']


def core_ma():
    """ Iterator over the core Mini-Arrays
    """
    ma_index = 0
    while ma_index < 96:
        yield ma_index
        ma_index += 1