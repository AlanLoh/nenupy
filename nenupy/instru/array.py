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


import numpy as np


class MiniArrays(object):
    """ NenuFAR Mini-Arrays class
    """
    def __init__(self, ma_indices):
        self.indices = ma_indices

    def itcore():
        """ Iterator over the core Mini-Arrays
        """
        ma_index = 0
        while ma_index < 96:
            yield ma_index
            ma_index += 1


class UVW(object):
    """ Class to handle UVW distributions
    """

    def __init__(self, antennas):
        self.antennas = antennas
        self.autocor = False

    @property
    def antennas(self):
        return self._antennas
    @antennas.setter
    def antenas(self, ant):
        if not isinstance(ant, np.ndarray):
            ant = np.array(ant)
        assert ant.shape[1] == 3,\
            'ant axis 1 should be 3D (x, y, z)'
        self._antennas = ant
        self.n_ant = ant.shape[0]
        return

    def _baselines(self):
        """
        """
        baselines = []
        auto = int(not self.autocor)
        for i in range(0, self.n_ant-auto):
            for j in range(i+int(not auto), self.n_ant):
                ant1 = self.antenas[i]
                ant2 = self.antenas[j]
                baselines.append( (ant1, ant2) )
        return np.array(baselines)
    