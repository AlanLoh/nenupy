#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *************
    Parset reader
    *************
"""


__author__ = 'Alan Loh, Baptiste Cecconi'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_ParsetProperty',
    'Parset'
]


from os.path import abspath, isfile
from collections.abc import MutableMapping
from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy.observation import ParsetDataBase

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ============================================================= #
# ---------------------- _ParsetProperty ---------------------- #
# ============================================================= #
class _ParsetProperty(MutableMapping):
    """ Class which mimics a dictionnary object, adapted to
        store parset metadata per category. It understands the
        different data types from raw strings it can encounter.
    """

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)

    def __getitem__(self, key):
        return self.mapping[key]

    def __delitem__(self, key):
        del self.mapping[key]

    def __setitem__(self, key, value):
        """
        """
        value = value.replace('\n', '')
        value = value.replace('"', '')

        if value.startswith('[') and value.endswith(']'):
            # This is a list
            val = value[1:-1].split(',')
            value = []
            # Parse according to syntax
            for i in range(len(val)):
                if '..' in val[i]:
                    # This is a subband syntax
                    subBandSart, subBanStop = val[i].split('..')
                    value.extend(
                        list(
                            range(
                                int(subBandSart),
                                int(subBanStop) + 1
                            )
                        )
                    )
                elif ':' in val[i]:
                    # Might be a time object
                    try:
                        item = Time(val[i], precision=0)
                    except ValueError:
                        item = val[i]
                    value.append(item)
                elif val[i].isdigit():
                    # Integers (there are not list of floats)
                    value.append(int(val[i]))
                else:
                    # A simple string
                    value.append(val[i])

        elif value.lower() in ['on', 'enable', 'true']:
            # This is a 'True' boolean
            value = True

        elif value.lower() in ['off', 'disable', 'false']:
            # This is a 'False' boolean
            value = False
        
        elif 'angle' in key.lower():
            # This is a float angle in degrees
            value = float(value) * u.deg
        
        elif value.isdigit():
            value = int(value)
        
        elif ':' in value:
            # Might be a time object
            try:
                value = Time(value, precision=0)
            except ValueError:
                pass

        else:
            pass
        
        # if key in self:
        #     del self[self[key]]

        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f'{type(self).__name__}({self.mapping})'
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------------- Parset -------------------------- #
# ============================================================= #
class Parset(object):
    """
    """

    def __init__(self, parset):
        self.observation = _ParsetProperty()
        self.output = _ParsetProperty()
        self.anabeams = {} # dict of _ParsetProperty
        self.digibeams = {} # dict of _ParsetProperty
        self.parset = parset


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def parset(self):
        """
        """
        return self._parset
    @parset.setter
    def parset(self, p):
        if not isinstance(p, str):
            raise TypeError(
                'parset must be a string.'
            )
        if not p.endswith('.parset'):
            raise ValueError(
                'parset file must end with .parset'
            )
        p = abspath(p)
        if not isfile(p):
            raise FileNotFoundError(
                f'Unable to find {p}'
            )
        self._parset = p
        self._decodeParset()


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def createDatabase(self):
        """
        """
        return


    def addToDatabase(self, dataBaseName):
        """
        """
        parsetDB = ParsetDataBase(dataBaseName)
        parsetDB.parset = self.parset
        parsetDB.addTable(
            {**self.observation, **self.output}, # dict merging
            desc='observation'
        )
        for anaIdx in self.anabeams.keys():
            parsetDB.addTable(
                self.anabeams[anaIdx],
                desc='anabeam'
            )
        for digiIdx in self.digibeams.keys():
            parsetDB.addTable(
                self.digibeams[digiIdx],
                desc='digibeam'
            )

        log.info(
            f'Parset {self.parset} added to database {dataBaseName}'
        )
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _decodeParset(self):
        """
        """
        log.info(
            f'Decoding parset {self._parset}...'
        )

        with open(self.parset, 'r') as file_object:
            line = file_object.readline()
            
            while line:
                try:
                    dicoName, content = line.split('.', 1)
                except ValueError:
                    # This is a blank line
                    pass
                
                key, value = content.split('=', 1)
                
                if line.startswith('Observation'):
                    self.observation[key] = value
                
                elif line.startswith('Output'):
                    self.output[key] = value
                
                elif line.startswith('AnaBeam'):
                    anaIdx = int(dicoName[-2])
                    if anaIdx not in self.anabeams.keys():
                        self.anabeams[anaIdx] = _ParsetProperty()
                        self.anabeams[anaIdx]['anaIdx'] = str(anaIdx)
                    self.anabeams[anaIdx][key] = value
                
                elif line.startswith('Beam'):
                    digiIdx = int(dicoName[-2])
                    if digiIdx not in self.digibeams.keys():
                        self.digibeams[digiIdx] = _ParsetProperty()
                    self.digibeams[digiIdx][key] = value
                
                line = file_object.readline()

            log.info(
                f'Parset {self._parset} decoded.'
            )
        return
# ============================================================= #

