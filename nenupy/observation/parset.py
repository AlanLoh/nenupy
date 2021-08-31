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
    'Parset',
    'ParseUser'
]


from os.path import abspath, isfile
from collections.abc import MutableMapping
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np

from nenupy.observation import PARSET_OPTIONS

import logging
log = logging.getLogger(__name__)


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
        from nenupy.observation import ParsetDataBase

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


# ============================================================= #
# ------------------------ ParsetUser ------------------------- #
# ============================================================= #
class _ParsetBlock:
    """
    """

    def __init__(self, field):
        self.field = field
        self.configuration = PARSET_OPTIONS[self.field].copy()
    

    def __setitem__(self, key, value):
        """
        """
        self._modify_properties(**{key: value})


    def _modify_properties(self, **kwargs):
        """
        """
        for key, value in kwargs.items():
            
            # If the key exists, it will be udpated
            if key in self.configuration:
                
                # If the value is an astropy.Time instance, the format is 'YYYY-MM-DDThh:mm:ssZ'
                if isinstance(value, Time):
                    value.precision = 0
                    value = value.isot + "Z"
                
                # Durations/exposures are expressed in seconds
                elif isinstance(value, TimeDelta):
                    value = str(int(np.round(value.sec))) + "s"
                
                # The boolean values needs to be translated to strings
                elif isinstance(value, bool):
                    value = "true" if value else "false"

                # Updates the key value
                self.configuration[key] = value
            
            # If the key doesn't exist a warning message is raised
            else:
                log.warning(
                    f"Key '{key}' is invalid. Available keys are: {self.configuration.keys()}."
                )

    def _write_block_list(self, index=None) -> str:
        """
        """
        if index is not None:
            counter = f"[{index}]"
        else:
            counter = ""
        return  "\n".join(
            "{}{}.{}={}".format(self.field, counter, *i)
            for i in self.configuration.items()
        )


class _NumericalBeamParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Beam")
        self._index = 0
        self._modify_properties(**kwargs)


    def __str__(self):
        return self._write_block_list(index=self._index)


class _AnalogBeamParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Anabeam")
        self._index = 0
        self._modify_properties(**kwargs)
        self.numerical_beams = []
        self._add_numerical_beam()


    def __str__(self):
        return self._write_block_list(index=self._index)


    def _add_numerical_beam(self, **kwargs):
        """
        """
        self.numerical_beams.append(
            _NumericalBeamParsetBlock(
                noBeam=self._index
            )
        )


class _OutputParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Output")
        self._modify_properties(**kwargs)


    def __str__(self):
        return self._write_block_list()


class _ObservationParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Observation")
        self._modify_properties(**kwargs)
        self.analog_beams = []
        self._add_analog_beam()


    def __str__(self):
        return self._write_block_list()


    def _add_analog_beam(self, **kwargs):
        """
        """
        self.analog_beams.append(
            _AnalogBeamParsetBlock()
        )


class ParsetUser:
    """
    """

    def __init__(self):
        self.observation = _ObservationParsetBlock()
        self.output = _OutputParsetBlock()


    def __str__(self):
        # Updates the number of analog and numerical beams
        nb_analog_beams = len(self.observation.analog_beams)
        nb_numerical_beams = sum(len(anabeam.numerical_beams) for anabeam in self.observation.analog_beams)
        self.observation.configuration['nrAnabeams'] = nb_analog_beams
        self.observation.configuration['nrBeams'] = nb_numerical_beams

        # Prepares the different text blocks
        observation_text = str(self.observation)
        output_text = str(self.output)
        return "\n\n".join(
            [observation_text,
            self.analog_beams_str,
            self.numerical_beams_str,
            output_text]
        )


    @property
    def analog_beams_str(self):
        """
        """
        return "\n\n".join(
            str(anabeam)
            for anabeam in self.observation.analog_beams
        )


    @property
    def numerical_beams_str(self):
        """
        """
        return "\n\n".join(
            str(numbeam)
            for anabeam in self.observation.analog_beams
            for numbeam in anabeam.numerical_beams
        )


    def add_analog_beam(self, **kwargs):
        """
        """
        self.observation._add_analog_beam(**kwargs)
        self._updates_anabeams_indices()


    def remove_analog_beam(self, index):
        """
        """
        del self.observation.analog_beams[index]
        self._updates_anabeams_indices()


    def add_numerical_beam(self, anabeam_index=0, **kwargs):
        """
        """
        # Adds a numerical beam to the analog beam 'anabeam_index'
        try:
            self.observation.analog_beams[anabeam_index]._add_numerical_beam(**kwargs)
        except IndexError:
            log.error(
                f"Requested analog beam index {anabeam_index} is out of range. Only {len(self.observation.analog_beams)} analog beams are set."
            )
            raise
        self._updates_numbeams_indices()
        

    def remove_numerical_beam(self, index):
        """
        """
        counter = 0
        for anabeam in self.observation.analog_beams:
            for i, _ in enumerate(anabeam.numerical_beams):
                if counter==index:
                    del anabeam.numerical_beams[i]
                    break
                counter += 1
            else:
                continue
            break
        self._updates_numbeams_indices()


    def write(self, file_name):
        """ Writes the current instance of :class:`~nenupy.observation.parset.ParsetUser`
            to a file called ``file_name``. 
        """
        return


    def _updates_numbeams_indices(self):
        """ Updates the indices of numerical beams. """
        numbeams_counter = 0
        for anabeam in self.observation.analog_beams:
            for numbeam in anabeam.numerical_beams:
                numbeam._index = numbeams_counter
                numbeams_counter += 1


    def _updates_anabeams_indices(self):
        """ Updates the indices of analog beams. """
        anabeams_counter = 0
        for anabeam in self.observation.analog_beams:
            anabeam._index = anabeams_counter
            anabeams_counter += 1
        self._updates_numbeams_indices()
# ============================================================= #
# ============================================================= #
