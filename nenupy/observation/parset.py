#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *************
    Parset reader
    *************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_ParsetProperty',
    'Parset',
    'ParsetUser'
]


from os.path import abspath, isfile, join, basename, dirname
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Tuple, Callable
import re
import json
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz, ICRS, Angle
import astropy.units as u
# from ipywidgets.widgets.widget_output import Output
import numpy as np
from typing import Union

from nenupy import nenufar_position
from nenupy.instru import sb2freq
from nenupy.astro.target import SolarSystemTarget
from nenupy.observation import PARSET_OPTIONS
# from nenupy.observation.sqldatabase import DuplicateParsetEntry, UserNameNotFound

import logging
log = logging.getLogger(__name__)


SB_WIDTH = 195.3125*u.kHz


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
                        item = Time(val[i].strip(), precision=0)
                    except ValueError:
                        item = val[i]
                    value.append(item)
                elif val[i].isdigit():
                    # Integers (there are not list of floats)
                    value.append(int(val[i]))
                else:
                    # A simple string
                    value.append(val[i])

        elif value.lower() in ['true']: #["on", "enable", "true"]
            # This is a 'True' boolean
            value = True

        elif value.lower() in ['false']: #["off", "disable", "false"]
            # This is a 'False' boolean
            value = False
        
        elif 'angle' in key.lower():
            # This is a float angle in degrees
            value = float(value) * u.deg
        
        elif "duration" in key.lower():
            value = f"{value}s"
        
        elif value.isdigit():
            value = int(value)
        
        elif ':' in value:
            # Might be a time object
            try:
                value = Time(value.strip(), precision=0)
            except ValueError:
                pass
        
        elif isinstance(value, str):
            value = value.lower()

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
# ------------------------ _JsonEntry ------------------------- #
# ============================================================= #
def _parse_parameters(parameters: str, pulsar: bool = False) -> Tuple[str, dict]:
    """ Parse values from the digital beam 'parameters'
        entry.
        E.g. 'TF: DF=3.05 DT=10.0 HAMM'
    """
    parameters = parameters.lower()
    mode = parameters.split(':')[0]
    if pulsar:
        configs = {
            param.split('=')[0]: param.split('=')[1]\
            for param in parameters.split('--')\
            if '=' in param
        }
        configs.update({
            param.rstrip(): True\
            for param in parameters.split('--')\
            if '=' not in param
        })
    else:
        configs = {
            param.split('=')[0]: param.split('=')[1]\
            for param in parameters.split()\
            if '=' in param
        }
        configs.update({
            param.rstrip(): True\
            for param in parameters.split('--')\
            if '=' not in param
        })
    return mode, configs

def _array_to_dict_array(array: list, unit: str = "") -> list:
    """ """
    if unit != "":
        return [
            {"value": val, "unit": unit} for val in array
        ]
    else:
        return [
            {"value": val} for val in array
        ]

def _get_pointing_center_dict(property: _ParsetProperty) -> dict:
    """ Returns a RA, Dec whatever the pointing type is. """

    def _constrain_angle(
            angle: u.Quantity,
            valmin: u.Quantity = 0.*u.deg,
            valmax: u.Quantity = 90*u.deg
        ):
        """ Constrain an angle between two values. """
        if angle < valmin:
            angle = valmin 
        elif angle > valmax:
            angle = valmax
        else:
            pass
        return angle

    # Sort out the beam start and stop times
    duration = TimeDelta(property['duration'] , format='sec')
    start_time = property['startTime']
    stop_time = (property['startTime'] + duration)

    if "azelFile" in property:
        # In case of pointing described by an azelfile
        # it will be treated as a zenith pointing (wrong but best compromise for the database)
        property["directionType"] = "azelgeo_azelfile"

    # Deal with coordinates and pointing types
    direction_type = property['directionType'].lower()
    if direction_type == "j2000":
        ra = property['angle1'].to(u.deg)
        dec = property['angle2'].to(u.deg)
        if ("decal_az" in property) or ("decal_el" in property):
            altaz = SkyCoord(ra, dec).transform_to(
                AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            )
            radec = SkyCoord(
                _constrain_angle(
                    altaz.az + float(property.get("decal_az", 0.0))*u.deg,
                    valmin=0.*u.deg,
                    valmax=360.*u.deg
                ),
                _constrain_angle(
                    altaz.alt + float(property.get("decal_el", 0.0))*u.deg,
                    valmin=0.*u.deg,
                    valmax=90.*u.deg
                ),
                frame=AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
            ra = radec.ra
            dec = radec.dec
        # Nothing else to do
        decal_ra = float(property.get("decal_ra", 0.0))*u.deg
        decal_dec = float(property.get("decal_dec", 0.0))*u.deg
        right_ascension = _constrain_angle(
            (ra + decal_ra).value,
            valmin=0.,
            valmax=360.
        )
        declination = _constrain_angle(
            (dec + decal_dec).value,
            valmin=-90.,
            valmax=90.
        )

    elif direction_type == "azelgeo":
        # This is a transit observation, compute the mean RA/Dec
        # Convert AltAz to RA/Dec
        radec = SkyCoord(
            _constrain_angle(
                property['angle1'] + float(property.get("decal_az", 0.0))*u.deg,
                valmin=0.*u.deg,
                valmax=360.*u.deg
            ),
            _constrain_angle(
                property['angle2'] + float(property.get("decal_el", 0.0))*u.deg,
                valmin=0.*u.deg,
                valmax=90.*u.deg
            ),
            frame=AltAz(
                obstime=start_time + duration/2.,
                location=nenufar_position
            )
        ).transform_to(ICRS)
        right_ascension = _constrain_angle(
            radec.ra.deg + float(property.get("decal_ra", 0.0)),
            valmin=0.,
            valmax=360.
        )
        declination = _constrain_angle(
            radec.dec.deg + float(property.get("decal_dec", 0.0)),
            valmin=-90.,
            valmax=90.
        )
    
    elif direction_type == "azelgeo_azelfile":
        # This observation was made using an azelfile
        radec = SkyCoord(
            0.*u.deg,
            90*u.deg,
            frame=AltAz(
                obstime=start_time + duration/2.,
                location=nenufar_position
            )
        ).transform_to(ICRS)
        right_ascension = radec.ra.deg
        declination = radec.dec.deg
    
    elif direction_type == "natif":
        # This is a test observation, unable to parse the RA/Dec
        right_ascension = None
        declination = None

    else:
        # Dealing with a Solar System source
        solar_system_target = SolarSystemTarget.from_name(
            name=direction_type,
            time=start_time + duration/2.
        )
        radec = solar_system_target.coordinates
        if ("decal_az" in property) or ("decal_el" in property):
            altaz = solar_system_target.horizontal_coordinates[0]
            radec = SkyCoord(
                _constrain_angle(
                    altaz.az + float(property.get("decal_az", 0.0))*u.deg,
                    valmin=0.*u.deg,
                    valmax=360.*u.deg
                ),
                _constrain_angle(
                    altaz.alt + float(property.get("decal_el", 0.0))*u.deg,
                    valmin=0.*u.deg,
                    valmax=90.*u.deg
                ),
                frame=AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
        decal_ra = float(property.get("decal_ra", 0.0))*u.deg
        decal_dec = float(property.get("decal_dec", 0.0))*u.deg
        right_ascension = _constrain_angle(
            radec.ra.deg + decal_ra.value,
            valmin=0.,
            valmax=360.
        )
        declination = _constrain_angle(
            radec.dec.deg + decal_dec.value,
            valmin=-90.,
            valmax=90.
        )

    return {
        "ra": {
            "value": right_ascension,
            "unit": "deg"
        },
        "dec": {
            "value": declination,
            "unit": "deg"
        },
        "obs_direction_type": property["directionType"].lower()
    }

def _get_time_dict(property: _ParsetProperty) -> dict:
    """ """
    # Sort out the beam start and stop times
    duration = TimeDelta(property['duration'] , format='sec')
    start_time = property['startTime']
    stop_time = (property['startTime'] + duration)
    return {
        "startstop":
            {
                "gte": start_time.isot,
                "lte": stop_time.isot
            },
        "duration": {
            "value": np.round(duration.sec, 3),
            "unit": "s"
        }
    }

def _get_frequency_dict(property: _ParsetProperty, field: str = "subbandList") -> dict:
        """ """
        subband_list = property[field]
        # Find consecutive subbands groups:
        subband_list_groups = np.split(
            subband_list,
            np.where(np.diff(subband_list) != 1)[0] + 1
        )
        return [
            {
                "value": {
                    "gte": sb2freq(group.min())[0].to(u.MHz).value,
                    "lt": (sb2freq(group.max()) + SB_WIDTH)[0].to(u.MHz).value,
                },
                "unit": "MHz"
            } for group in subband_list_groups
        ]

def _default_setting(digibeam: _ParsetProperty, output: _ParsetProperty, version: tuple) -> dict:
    return {
        "name": "LaNewBa",
        "dt": {
            "value": 1,
            "unit": "s"
        },
        "df": {
            "value": SB_WIDTH.to(u.kHz).value,
            "unit": "kHz"
        },
        "frequency": _get_frequency_dict(digibeam, field="subbandList")
    }

def _pulsar_setting(digibeam: _ParsetProperty, output: _ParsetProperty, version: tuple) -> dict:
    # Parse the parameters
    try:
        mode, config = _parse_parameters(digibeam["parameters"], pulsar=True)
    except KeyError:
        log.warning(
            f"No 'parameters' for numerical beam {digibeam['noBeam']})."
        )
        return {}
    
    # Fill out the receiver configuration depending on the observing mode
    if mode == "fold":
        return {
            "name": "undysputed",
            "mode": "pulsar_fold",
            "source_name": config["src"].rstrip(),
            "n_polars": 1 if config.get("onlyi", False) else 4,
            "frequency": _get_frequency_dict(digibeam, field="subbandList")
        }
    elif mode == "single":
        return {
            "name": "undysputed",
            "mode": "pulsar_single",
            "source_name": config["src"].rstrip(),
            "downsampling": int(config["dstime"]),
            "n_polars": 1 if config.get("onlyi", False) else 4,
            "frequency": _get_frequency_dict(digibeam, field="subbandList")
        }
    elif mode == "waveolaf":
        return {
            "name": "undysputed",
            "mode": "pulsar_waveolaf",
            "source_name": config["src"].rstrip(),
            "frequency": _get_frequency_dict(digibeam, field="subbandList")
        }
    elif mode == "wave":
        return {
            "name": "undysputed",
            "mode": "pulsar_wave",
            "source_name": config["src"].rstrip(),
            "frequency": _get_frequency_dict(digibeam, field="subbandList")
        }
    else:
        log.warning(f"Pulsar mode '{mode}' not recognized.")
        return {}

def _waveform_setting(digibeam: _ParsetProperty, output: _ParsetProperty, version: tuple) -> dict:
    return {
        "name": "undysputed",
        "mode": "waveform",
        "frequency": _get_frequency_dict(digibeam, field="subbandList")
    }

def _dynamicspectrum_setting(digibeam: _ParsetProperty, output: _ParsetProperty, version: tuple) -> dict:
    # Parse the parameters
    try:
        _, config = _parse_parameters(digibeam["parameters"], pulsar=False)
    except KeyError:
        log.warning(
            f"No 'parameters' for numerical beam {digibeam['noBeam']}). Setting to default values."
        )
        # Set default value for the configuration
        config = {
            "dt": 5.00,
            "df": 6.1
        }
    try:
        if config.get("tf: rawrt", False):
            # This shouldnt be the case though...
            return _waveform_setting(digibeam, output, version)
        return {
            "name": "undysputed",
            "mode": "tf",
            "dt": {
                "value": float(config["dt"]),
                "unit": "ms"
            },
            "df": {
                "value": float(config["df"]),
                "unit": "kHz"
            },
            "frequency": _get_frequency_dict(digibeam, field="subbandList")
        }
    except KeyError:
        log.warning(
            f"Wrong '{digibeam['toDo']}' configuration: {config}."
        )
        return {}

def _nickel_setting(phasecenter: _ParsetProperty, output: _ParsetProperty, version: tuple) -> dict:
    nickel_config = {
        "name": "nickel",
            "channelization": {
                "value": output["nri_channelization"],
                "unit": None
            },
            "dumptime": {
                "value": output["nri_dumpTime"],
                "unit": "s"
            }
    }
    if version >= (1, 0):
        # Parse the parameters
        try:
            mode, config = _parse_parameters(phasecenter["parameters"], pulsar=False)
            log.warning("NICKEL parameters not taken into account. Needs to be implemented!")
        except KeyError:
            log.warning(
                f"No 'parameters' for phase center {phasecenter['noBeam']})."
            )
            #return {}
        nickel_config["frequency"] = _get_frequency_dict(phasecenter, "subbandList")
        return nickel_config
    else:
        nickel_config["frequency"] = _get_frequency_dict(output, "nri_subbandList")
        return nickel_config

def _tbd_setting(digibeam: _ParsetProperty, output: _ParsetProperty, version: tuple) -> dict:
    if ("nickel" in output.get("nri_receivers", [])) and (version < (1, 0)):
        return _nickel_setting(digibeam, output, version)
    else:
        return _default_setting(digibeam, output, version)

BEAM_SETTINGS = {
    "none": _default_setting,
    "pulsar": _pulsar_setting,
    "waveform": _waveform_setting,
    "dynamicspectrum": _dynamicspectrum_setting,
    "tbd": _tbd_setting,
    "nickel": _nickel_setting
}

class _JsonEntry:


    def __init__(self, output: _ParsetProperty):
        self.obs_metadata = {}
        self.output = output
        self.fovs = []
        self.pointings = []


    @property
    def fov_indices(self) -> np.ndarray:
        return np.array([fov["idx"] for fov in self.fovs])


    @property
    def data(self) -> dict:
        """ """
        # Fill the Field of Views with their associated pointings
        fovs = self.fovs.copy()
        for fov_idx, pointing in self.pointings:
            fovs[fov_idx]["pointings"].append(pointing)

        # Build the dictionnary of field of views        
        fov_dict = {
            "field_of_views": fovs
        }

        # Return a dictionnary that can be transformed to JSON
        return {**self.obs_metadata, **fov_dict}


    def add_observation_metadata(self, observation: _ParsetProperty, parset_file: str, parset_user: str = "") -> None:
        """ """
        self.obs_metadata["@timestamp"] = observation["startTime"].isot

        # Fill out observation tab
        self.obs_metadata["file_name"] = {
            "name": basename(parset_file),
            "path": dirname(parset_file)
        }
        self.obs_metadata["time"] = {
            "startstop": 
               {
                  "gte": observation["startTime"].isot, 
                  "lt": observation["stopTime"].isot
               },
            "duration": {
                "value": np.round((observation["stopTime"] - observation["startTime"]).sec, 3),
                "unit": "s"
            }
        }
        topic = observation.get('topic', 'LT00 DEBUG')
        # Try to capture code and topic name using regular expression
        pattern = r'^(?P<code>(ES([0]?[0-9]|1[0-7])|LT(?!08)([0]?[0-9]|1[0-3])|RP\d+[A-Z]|SP(16|17))) (?P<name>\w+)$'
        topic_decoded = re.match(pattern=pattern, string=topic)
        if topic_decoded is None:
            log.warning(f'{topic} not properly decoded!!')
            self.obs_metadata["topic"] = {
                'code': 'LT00',
                'name': 'DEBUG'
            }
        else:
            topic_decoded_dict = topic_decoded.groupdict()
            self.obs_metadata["topic"] = {
                'code': topic_decoded_dict['code'],
                'name': topic_decoded_dict['name']
            }
        key_mapping = {
            "title": "title",
            "contactName": "contact_name",
            "name": "name"
        }
        for key, value in observation.items():
            if key in key_mapping:
                self.obs_metadata[key_mapping[key]] = value
        self.obs_metadata["parset_user"] = parset_user


    def add_field_of_view(self, index: int, anabeam: _ParsetProperty) -> None:
        """ """
        fov = {}
        fov["idx"] = index
        fov["pointings"] = []
        fov["name"] = anabeam["target"]
        fov["center"] = _get_pointing_center_dict(anabeam)
        fov["time"] = _get_time_dict(anabeam)
        fov["beamsquint"] = {
            "correction": anabeam.get("beamSquint", False),
            "frequency": {
                "value": anabeam.get("optFrq", None),
                "unit": "MHz"
            }
        }
        fov["mini_arrays"] = _array_to_dict_array(anabeam["maList"])
        fov["antennas"] = _array_to_dict_array(anabeam["antList"])
        fov["filter"] = [{"name": int(fil), "start": tim.isot} for fil, tim in zip(anabeam["filter"], anabeam["filterTime"])]
        self.fovs.append(fov)


    def add_pointing(self, index: int, beam: _ParsetProperty, parset_version: tuple, pointing_setting_func: Callable = None) -> None:
        """ """
        pointing = {}

        # Mandatory keys
        pointing["idx"] = index
        pointing["name"] = beam["target"]
        pointing["center"] = _get_pointing_center_dict(beam)
        pointing["time"] = _get_time_dict(beam)

        # Select the correct function to store the receiver configuration
        if pointing_setting_func is None:
            # Automatically choose the function
            pointing_setting_func = BEAM_SETTINGS[beam.get("toDo", "none").lower()]
        pointing["receiver"] = pointing_setting_func(beam, self.output, parset_version)

        # Assign the FoV index to each pointing
        fov_idx = np.where(self.fov_indices == beam["noBeam"])[0][0]

        self.pointings.append((fov_idx, pointing))


    def add_nickel_pointing(self, anabeam: _ParsetProperty, index: int) -> None:
        """ Only for parset < 1.0 """
        nickel_pointing = {}

        # Mandatory keys
        nickel_pointing["idx"] = index
        nickel_pointing["name"] = anabeam["target"]
        nickel_pointing["center"] = _get_pointing_center_dict(anabeam)
        nickel_pointing["time"] = _get_time_dict(anabeam)
        
        nickel_pointing.update(
            _nickel_setting(anabeam, self.output, version=(0, 0))
        )

        # Add the pointing to the list, with its associated fov index
        self.pointings.append((0, nickel_pointing))

    def add_xst_pointings(self) -> None:
        """ """
        # Get the last pointing index, before adding more
        last_pointing_idx = len(self.pointings) - 1

        for i, fov in enumerate(self.fovs):
            start = Time(fov["time"]["startstop"]["gte"])
            duration = TimeDelta(fov["time"]["duration"]["value"], format="sec")
            zenith = SkyCoord(
                0, 90,
                unit="deg",
                frame=AltAz(
                    obstime=start + duration/2,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
            
            # Prepare the pointing configuration
            xst_pointing = {
                "idx": last_pointing_idx + 1 + i,
                "center": {
                    "ra": {
                        "value": zenith.ra.deg,
                        "unit": "deg"
                    },
                    "dec": {
                        "value": zenith.dec.deg,
                        "unit": "deg"
                    },
                    "obs_direction_type": "zenith_xst"
                },
                "name": "",
                "time": fov["time"],
                "receiver": {
                    "name": "LaNewBa",
                    "frequency": _get_frequency_dict(self.output, field="xst_sbList")
                }
            }

            # Add the pointing to the list, with its associated fov index
            self.pointings.append((i, xst_pointing))


    def remove_unused_miniarrays(self) -> None:
        """ """
        for fov in self.fovs:
            # Check if remote MA are there, loop out if not
            mas_in_fov = np.array([ma_in_fov["value"] for ma_in_fov in fov["mini_arrays"]])
            if not np.any(mas_in_fov > 96):
                continue

            # Find out the receivers used
            for pointing in fov["pointings"]:
                if "nickel" == pointing["receiver"]["name"]:
                    # Check if one of the associated pointings implies NICKEL
                    continue

            # Remove the remote Mini-Arrays
            remote_mas_in_fov_mask = mas_in_fov > 96
            fov["mini_arrays"] = np.array(fov["mini_arrays"])[~remote_mas_in_fov_mask].tolist()
            log.info(
                f"Remote Mini-Arrays have been removed for 'field_of_view' #{fov['idx']} because no associated 'pointing' is using the NICKEL receiver."
            )


    def save_file(self, file_name: str) -> None:
        """ Writes the JSON file. """
        with open(file_name, 'w', encoding='utf-8') as wf:
            json.dump(self.data, wf, ensure_ascii=False, indent=4)
        log.info(f"'{file_name}' written.")
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
        self.phase_centers = {}
        self.parset_user = ""
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


    @property
    def version(self) -> tuple:
        """ """
        version_str = self.observation.get("parsetVersion", "0")
        if version_str == "":
            version_tuple = (1, 0) # default version
        else:
            version_tuple = tuple(map(lambda x: int(x), version_str.split(".")))
        return version_tuple


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def to_json(self, path_name: str = None):
        """ """

        parset_version = self.version

        json_entry = _JsonEntry(output=self.output)

        json_entry.add_observation_metadata(
            observation=self.observation,
            parset_file=self.parset,
            parset_user=self.parset_user
        )

        # Parse and store every field of view = analog configurations
        for ana_idx, anabeam in self.anabeams.items():
            json_entry.add_field_of_view(ana_idx, anabeam)

        # Parse and store every pointing = digital beam configurations
        digi_idx = 0
        for digi_idx, digibeam in self.digibeams.items():
            json_entry.add_pointing(digi_idx, digibeam, parset_version)

        # Parse and store every imaging pointing = phase center configurations
        if parset_version >= (1, 0):
            # These were introduced with parset version 1.0
            for center_idx, phase_center in self.phase_centers.items():
                pc_index = center_idx + digi_idx + 1
                json_entry.add_pointing(pc_index, phase_center, parset_version)

        # Add extra pointings in some specific cases
        # If XST are used
        if self.output.get("xst_userfile", False):
            # Add a pointing per anabeam if XST data have been taken
            json_entry.add_xst_pointings()
        # If NICKEL is used, old parset versions
        if parset_version < (1, 0):
            if "nickel" in self.output.get("nri_receivers", []):
                if "TBD" not in [beam["toDo"] for beam in self.digibeams.values()]:
                    if len(self.anabeams) > 1:
                        log.warning("Found more than 1 FoV. A NICKEL pointing is added for the first one ONLY.")

                    # Add a NICKEL pointing corresponding to the analog beam
                    index = len(json_entry.pointings)
                    json_entry.add_nickel_pointing(anabeam=self.anabeams[0], index=index)

        # Remove un-necessary Mini-Arrays indices
        json_entry.remove_unused_miniarrays()

        # Save or not the data to a file
        if path_name is not None:
            json_file_name = basename(self.parset).replace(".parset", ".json")
            json_file = join(path_name, json_file_name)
            json_entry.save_file(file_name=json_file)
        else:
            return json_entry.data


    def to_json_old(self, path_name=None):
        """ """
        
        data = {}
        data["@timestamp"] = self.observation["startTime"].isot

        # Fill out observation tab
        data["file_name"] = {
            "name": basename(self.parset),
            "path": dirname(self.parset)
        }
        data["time"] = {
            "startstop": 
               {
                  "gte": self.observation["startTime"].isot, 
                  "lt": self.observation["stopTime"].isot
               },
            "duration": {
                "value": (self.observation["stopTime"] - self.observation["startTime"]).sec,
                "unit": "s"
            }
        }
        topic = self.observation.get("topic", "ES00 DEBUG")
        data["topic"] = {
            "code": topic[:4] if topic.startswith('ES') else "ES00",
            "name": topic[5:] if topic.startswith('ES') else topic
        }
        key_mapping = {
            "title": "title",
            "contactName": "contact_name",
            "name": "name"
            # "contactEmail": "contact_email",
            # "topic": "topic"
        }
        for key, value in self.observation.items():
            if key in key_mapping:
                data[key_mapping[key]] = value

        # to_dos = [digibeam["toDo"] for digibeam in self.digibeams.values()]
        receivers_used = self.output["hd_receivers"] + self.output.get("nri_receivers", [])
        # to_dos = np.unique(to_dos)
        # data["receivers"] = {"name": receiver_name for receiver_name in to_dos if receiver_name.lower() != "tbd"}
        data["receivers"] = [{"name": receiver_name} for receiver_name in receivers_used]

        # Fill out outputs
        # data["output"] = {}
        # for key, value in self.output.items():
        #     data_level, data_property = key.split("_")
        #     if data_level not in data["output"]:
        #         data["output"][data_level] = {}
        #     data["output"][data_level][data_property] = value

        # Fill out field of views (= anabeams)
        data["field_of_views"] = []
        for ana_idx, anabeam in self.anabeams.items():
            fov = {}
            
            fov["idx"] = ana_idx
            fov["pointings"] = []
            fov["name"] = anabeam["target"]
            fov["center"] = self._get_pointing_center_dict(anabeam)
            fov["time"] = self._get_time_dict(anabeam)
            fov["beamsquint"] = {
                "correction": anabeam.get("beamSquint", False),
                "frequency": {
                    "value": anabeam.get("optFrq", None),
                    "unit": "MHz"
                }
            }
            # fov["mini_arrays"] = anabeam["maList"]
            fov["mini_arrays"] = self._array_to_dict_array(anabeam["maList"])
            # fov["antennas"] = anabeam["antList"]
            fov["antennas"] = self._array_to_dict_array(anabeam["antList"])
            fov["filter"] = [{"name": int(fil), "start": tim.isot} for fil, tim in zip(anabeam["filter"], anabeam["filterTime"])]

            data["field_of_views"].append(fov)

        fov_indices = np.array([fov["idx"] for fov in data["field_of_views"]])

        for digi_idx, digibeam in self.digibeams.items():
            pointing = {}
            pointing['idx'] = digi_idx
            pointing["name"] = digibeam["target"]
            pointing["center"] = self._get_pointing_center_dict(digibeam)
            pointing["time"] = self._get_time_dict(digibeam)

            if "toDo" not in digibeam:
                pointing["receiver"] = {
                    "name": "LaNewBa",
                    "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                }
            elif digibeam["toDo"].lower() == "pulsar":
                try:
                    mode, config = self._parse_parameters(digibeam["parameters"], pulsar=True)
                except KeyError:
                    log.warning(
                        f"Parset '{self.parset}' doesn't have any 'parameters' for numerical beam {digibeam['noBeam']})."
                    )
                    continue

                if mode == "fold":
                    pointing["receiver"] = {
                        "name": "undysputed",
                        "mode": "pulsar_fold",
                        "source_name": config["src"],
                        "n_polars": 1 if config.get("onlyi", False) else 4,
                        "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                    }
                elif mode == "single":
                    pointing["receiver"] = {
                        "name": "undysputed",
                        "mode": "pulsar_single",
                        "source_name": config["src"],
                        "downsampling": int(config["dstime"]),
                        "n_polars": 1 if config.get("onlyi", False) else 4,
                        "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                    }
                elif mode == "waveolaf":
                    pointing["receiver"] = {
                        "name": "undysputed",
                        "mode": "pulsar_waveolaf",
                        "source_name": config["src"],
                        "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                    }
                elif mode == "wave":
                    pointing["receiver"] = {
                        "name": "undysputed",
                        "mode": "pulsar_wave",
                        "source_name": config["src"],
                        "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                    }
                else:
                    pointing["receiver_configuration"] = {}
            elif digibeam["toDo"].lower() == "waveform":
                pointing["receiver"] = {
                    "name": "undysputed",
                    "mode": "waveform",
                    #"source_name": config["src"],
                    "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                }
            elif digibeam["toDo"].lower() == "dynamicspectrum":
                try:
                    _, config = self._parse_parameters(digibeam["parameters"], pulsar=False)
                except KeyError:
                    log.warning(
                        f"Parset '{self.parset}' doesn't have any 'parameters' for numerical beam {digibeam['noBeam']})."
                    )
                    continue

                try:
                    pointing["receiver"] = {
                        "name": "undysputed",
                        "mode": "tf",
                        "dt": {
                            "value": float(config["dt"]),
                            "unit": "ms"
                        },
                        "df": {
                            "value": float(config["df"]),
                            "unit": "kHz"
                        },
                        "frequency": self._get_frequency_dict(digibeam, field="subbandList")
                    }
                except KeyError:
                    log.warning(
                        f"Parset '{self.parset}' has a wrong '{digibeam['toDo']}' configuration."
                    )
                    continue
            elif (digibeam["toDo"].lower() == "tbd") and ("nickel" in self.output.get("nri_receivers", [])):
            # elif digibeam["toDo"].lower() == "imaging": # to be implemented?
                pointing["receiver"] = {
                    "name": "nickel",
                    "channelization": {
                        "value": self.output["nri_channelization"],
                        "unit": None
                    },
                    "dumptime": {
                        "value": self.output["nri_dumpTime"],
                        "unit": "s"
                    },
                    "frequency": self._get_frequency_dict(self.output, "nri_subbandList")
                }           

            # Select the correct fov
            idx = np.where(fov_indices == digibeam["noBeam"])[0][0]
            associated_fov = data["field_of_views"][idx]
            associated_fov["pointings"].append(pointing)
        
        # Add a pointing per anabeam if XST data have been taken
        if self.output.get("xst_userfile", False):
            for i, fov in enumerate(data["field_of_views"]):
                start = Time(fov["time"]["startstop"]["gte"])
                duration = TimeDelta(fov["time"]["duration"]["value"], format="sec")
                zenith = SkyCoord(
                    0, 90,
                    unit="deg",
                    frame=AltAz(
                        obstime=start + duration/2,
                        location=nenufar_position
                    )
                ).transform_to(ICRS)
                try:
                    last_dig_idx = digi_idx
                except:
                    last_dig_idx = -1
                fov["pointings"].append(
                    {
                        "idx": last_dig_idx + 1 + i,
                        "center": {
                            "ra": {
                                "value": zenith.ra.deg,
                                "unit": "deg"
                            },
                            "dec": {
                                "value": zenith.dec.deg,
                                "unit": "deg"
                            },
                            "obs_direction_type": "zenith_xst"
                        },
                        "name": "",
                        "time": fov["time"],
                        "receiver": {
                            "name": "LaNewBa",
                            "frequency": self._get_frequency_dict(self.output, field="xst_sbList")
                        }
                    }
                )
        
        # Add a pointing per anabeam if NiCKEL data have been taken in // with undysputed
        # to_dos = [digibeam["toDo"] for digibeam in self.digibeams.values()]
        # if ("nickel" in self.output.get("nri_receivers", [])) & ("TBD" not in to_dos):
        #     for i, fov in enumerate(data["field_of_views"]):

        # Remove the remote Mini-Arrays if they are not used
        for i, fov in enumerate(data["field_of_views"]):

            # Check if remote MA are there
            mas_in_fov = np.array([ma_in_fov["value"] for ma_in_fov in fov["mini_arrays"]])
            if not np.any(mas_in_fov > 96):
                continue

            # Find out the receivers used
            receivers_in_fov = []
            for pointing in fov["pointings"]:
                receivers_in_fov.append(pointing["receiver"]["name"])

            # Check if one of the associated pointings implies NICKEL
            if "nickel" in receivers_in_fov:
                continue

            # Remove the remote Mini-Arrays
            remote_mas_in_fov_mask = mas_in_fov > 96
            fov["mini_arrays"] = np.array(fov["mini_arrays"])[~remote_mas_in_fov_mask].tolist()
            log.info(
                f"Remote Mini-Arrays have been removed for 'field_of_view' #{fov['idx']} because no associated 'pointing' is using the NICKEL receiver."
            )

        data['parset_user'] = self.parset_user

        if path_name is not None:
            # Write the JSON file
            json_file_name = basename(self.parset).replace(".parset", ".json")
            json_file = join(path_name, json_file_name)
            with open(json_file, 'w', encoding='utf-8') as wf:
                json.dump(data, wf, ensure_ascii=False, indent=4)
                log.info(f"'{json_file}' written.")
        else:
            return data


    # def add_to_database(self, data_base):#dataBaseName):
    #     """
    #         data_base: ParsetDataBase
    #     """
    #     parsetDB = data_base
    #     try:
    #         parsetDB.parset = self.parset
    #     except DuplicateParsetEntry:
    #         return

    #     try:
    #         parsetDB.add_row(
    #             {**self.observation, **self.output}, # dict merging
    #             desc='observation'
    #         )
    #     except UserNameNotFound:
    #         return
    #     for anaIdx in self.anabeams.keys():
    #         parsetDB.add_row(
    #             self.anabeams[anaIdx],
    #             desc='anabeam'
    #         )
    #     for digiIdx in self.digibeams.keys():
    #         parsetDB.add_row(
    #             self.digibeams[digiIdx],
    #             desc='digibeam'
    #         )

    #     log.info(
    #         f'Parset {self.parset} added to database {data_base.name}'
    #     )

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _decodeParset(self):
        """
        """
        
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
                    anaIdx = int(re.search(r'\[(\d*)\]', dicoName).group(1))
                    if anaIdx not in self.anabeams.keys():
                        self.anabeams[anaIdx] = _ParsetProperty()
                        self.anabeams[anaIdx]['anaIdx'] = str(anaIdx)
                    self.anabeams[anaIdx][key] = value
                
                elif line.startswith('Beam'):
                    digiIdx = int(re.search(r'\[(\d*)\]', dicoName).group(1))
                    if digiIdx not in self.digibeams.keys():
                        self.digibeams[digiIdx] = _ParsetProperty()
                        self.digibeams[digiIdx]['digiIdx'] = str(digiIdx)
                    self.digibeams[digiIdx][key] = value
                
                elif line.startswith('PhaseCenter'):
                    pcIdx = int(re.search(r'\[(\d*)\]', dicoName).group(1))
                    if pcIdx not in self.phase_centers.keys():
                        self.phase_centers[pcIdx] = _ParsetProperty()
                        self.phase_centers[pcIdx]['pcIdx'] = str(pcIdx)
                    self.phase_centers[pcIdx][key] = value

                line = file_object.readline()
            
            log.info(
                f"Parset '{self._parset}' loaded."
            )
        
        try:
            with open(self.parset + '_user', 'r') as file_object:
                line = file_object.readline()
                while line:
                    self.parset_user = self.parset_user + line
                    line = file_object.readline()
        except Exception as e:
            pass

        return


    @staticmethod
    def _parse_parameters(parameters, pulsar=False):
        """ Parse values from the digital beam 'parameters'
            entry.
            E.g. 'TF: DF=3.05 DT=10.0 HAMM'
        """
        parameters = parameters.lower()
        mode = parameters.split(':')[0]
        if pulsar:
            configs = {
                param.split('=')[0]: param.split('=')[1]\
                for param in parameters.split('--')\
                if '=' in param
            }
            configs.update({
                param.rstrip(): True\
                for param in parameters.split('--')\
                if '=' not in param
            })
        else:
            configs = {
                param.split('=')[0]: param.split('=')[1]\
                for param in parameters.split()\
                if '=' in param
            }
            configs.update({
                param.rstrip(): True\
                for param in parameters.split('--')\
                if '=' not in param
            })
        return mode, configs


    @staticmethod
    def _array_to_dict_array(array: list, unit: str = "") -> list:
        """ """
        if unit != "":
            return [
                {"value": val, "unit": unit} for val in array
            ]
        else:
            return [
                {"value": val} for val in array
            ]

    @staticmethod
    def _get_time_dict(property) -> dict:
        """ """
        # Sort out the beam start and stop times
        duration = TimeDelta(property['duration'] , format='sec')
        start_time = property['startTime']
        stop_time = (property['startTime'] + duration)
        # return {
        #     "start": start_time.isot,
        #     "stop": stop_time.isot,
        #     "duration": {
        #         "value": duration.sec,
        #         "unit": "s"
        #     }
        # }
        return {
            "startstop":
               {
                  "gte": start_time.isot,
                  "lte": stop_time.isot
               },
            "duration": {
                "value": np.round(duration.sec, 3),
                "unit": "s"
            }
        }


    @staticmethod
    def _get_frequency_dict(property, field="subbandList") -> dict:
        """ """
        subband_list = property[field]
        # Find consecutive subbands groups:
        subband_list_groups = np.split(
            subband_list,
            np.where(np.diff(subband_list) != 1)[0] + 1
        )
        # return {
        #     "value": [
        #         {
        #             "gte": sb2freq(group.min())[0].to(u.MHz).value,
        #             "lt": (sb2freq(group.max()) + SB_WIDTH)[0].to(u.MHz).value,
        #         } for group in subband_list_groups
        #     ],
        #     "unit": "MHz"
        # }
        return [
            {
                "value": {
                    "gte": sb2freq(group.min())[0].to(u.MHz).value,
                    "lt": (sb2freq(group.max()) + SB_WIDTH)[0].to(u.MHz).value,
                },
                "unit": "MHz"
            } for group in subband_list_groups
        ]


    @staticmethod
    def _get_miniarray_dict(mini_arrays: np.ndarray) -> dict:
        """ """
        # Find consecutive Mini-Arrays groups
        ma_groups = np.split(
            mini_arrays,
            np.where(np.diff(mini_arrays) != 1)[0] + 1
        )
        return {
            "value": [
                {
                    "gte": group[0],
                    "lte": group[-1] 
                } for group in ma_groups
            ],
            "unit": ""
        }


    @staticmethod
    def _get_pointing_center_dict(property) -> dict:
        """ Returns a RA, Dec whatever the pointing type is. """

        def _constrain_angle(
                angle: u.Quantity,
                valmin: u.Quantity = 0.*u.deg,
                valmax: u.Quantity = 90*u.deg
            ):
            """ Constrain an angle between two values. """
            if angle < valmin:
                angle = valmin 
            elif angle > valmax:
                angle = valmax
            else:
                pass
            return angle

        # Sort out the beam start and stop times
        duration = TimeDelta(property['duration'] , format='sec')
        start_time = property['startTime']
        stop_time = (property['startTime'] + duration)

        if "azelFile" in property:
            # In case of pointing described by an azelfile
            # it will be treated as a zenith pointing (wrong but best compromise for the database)
            property["directionType"] = "azelgeo_azelfile"

        # Deal with coordinates and pointing types
        direction_type = property['directionType'].lower()
        if direction_type == "j2000":
            ra = property['angle1'].to(u.deg)
            dec = property['angle2'].to(u.deg)
            if ("decal_az" in property) or ("decal_el" in property):
                altaz = SkyCoord(ra, dec).transform_to(
                    AltAz(
                        obstime=start_time + duration/2.,
                        location=nenufar_position
                    )
                )
                radec = SkyCoord(
                    _constrain_angle(
                        altaz.az + float(property.get("decal_az", 0.0))*u.deg,
                        valmin=0.*u.deg,
                        valmax=360.*u.deg
                    ),
                    _constrain_angle(
                        altaz.alt + float(property.get("decal_el", 0.0))*u.deg,
                        valmin=0.*u.deg,
                        valmax=90.*u.deg
                    ),
                    frame=AltAz(
                        obstime=start_time + duration/2.,
                        location=nenufar_position
                    )
                ).transform_to(ICRS)
                ra = radec.ra
                dec = radec.dec
            # Nothing else to do
            decal_ra = float(property.get("decal_ra", 0.0))*u.deg
            decal_dec = float(property.get("decal_dec", 0.0))*u.deg
            right_ascension = _constrain_angle(
                (ra + decal_ra).value,
                valmin=0.,
                valmax=360.
            )
            declination = _constrain_angle(
                (dec + decal_dec).value,
                valmin=-90.,
                valmax=90.
            )

        elif direction_type == "azelgeo":
            # This is a transit observation, compute the mean RA/Dec
            # Convert AltAz to RA/Dec
            radec = SkyCoord(
                _constrain_angle(
                    property['angle1'] + float(property.get("decal_az", 0.0))*u.deg,
                    valmin=0.*u.deg,
                    valmax=360.*u.deg
                ),
                _constrain_angle(
                    property['angle2'] + float(property.get("decal_el", 0.0))*u.deg,
                    valmin=0.*u.deg,
                    valmax=90.*u.deg
                ),
                frame=AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
            right_ascension = _constrain_angle(
                radec.ra.deg + float(property.get("decal_ra", 0.0)),
                valmin=0.,
                valmax=360.
            )
            declination = _constrain_angle(
                radec.dec.deg + float(property.get("decal_dec", 0.0)),
                valmin=-90.,
                valmax=90.
            )
        
        elif direction_type == "azelgeo_azelfile":
            # This observation was made using an azelfile
            radec = SkyCoord(
                0.*u.deg,
                90*u.deg,
                frame=AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
            right_ascension = radec.ra.deg
            declination = radec.dec.deg
        
        elif direction_type == "natif":
            # This is a test observation, unable to parse the RA/Dec
            right_ascension = None
            declination = None

        else:
            # Dealing with a Solar System source
            solar_system_target = SolarSystemTarget.from_name(
                name=direction_type,
                time=start_time + duration/2.
            )
            radec = solar_system_target.coordinates
            if ("decal_az" in property) or ("decal_el" in property):
                altaz = solar_system_target.horizontal_coordinates[0]
                radec = SkyCoord(
                    _constrain_angle(
                        altaz.az + float(property.get("decal_az", 0.0))*u.deg,
                        valmin=0.*u.deg,
                        valmax=360.*u.deg
                    ),
                    _constrain_angle(
                        altaz.alt + float(property.get("decal_el", 0.0))*u.deg,
                        valmin=0.*u.deg,
                        valmax=90.*u.deg
                    ),
                    frame=AltAz(
                        obstime=start_time + duration/2.,
                        location=nenufar_position
                    )
                ).transform_to(ICRS)
            decal_ra = float(property.get("decal_ra", 0.0))*u.deg
            decal_dec = float(property.get("decal_dec", 0.0))*u.deg
            right_ascension = _constrain_angle(
                radec.ra.deg + decal_ra.value,
                valmin=0.,
                valmax=360.
            )
            declination = _constrain_angle(
                radec.dec.deg + decal_dec.value,
                valmin=-90.,
                valmax=90.
            )

        return {
            "ra": {
                "value": right_ascension,
                "unit": "deg"
            },
            "dec": {
                "value": declination,
                "unit": "deg"
            },
            "obs_direction_type": property["directionType"].lower()
        }
# ============================================================= #


# ============================================================= #
# ------------------------ ParsetUser ------------------------- #
# ============================================================= #
class _ParsetBlock:
    """
    """

    def __init__(self, field):
        self.field = field
        self.configuration = deepcopy(PARSET_OPTIONS[self.field])
    

    def __setitem__(self, key, value):
        """
        """
        self._modify_properties(**{key: value})
    

    def __getitem__(self, key):
        """
        """
        return self.configuration[key]["value"]


    @property
    def fields(self):
        """ Lists the available fields. """
        return list(self.configuration.keys())


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
                # elif isinstance(value, bool):
                #     value = "true" if value else "false"

                # The lists must be squeezed
                elif isinstance(value, list):
                    value = f"[{','.join(str(element) for element in value)}]"

                # Updates the key value
                self.configuration[key]["value"] = value
                self.configuration[key]["modified"] = True
            
            # If the key doesn't exist a warning message is raised
            else:
                log.warning(
                    f"Key '{key}' is invalid. Available keys are: {self.configuration.keys()}."
                )


    def _write_block_list(self, index=None) -> str:
        """
        """
        # Prints a counter that is shown regarding the beam indices
        if index is not None:
            counter = f"[{index}]"
        else:
            counter = ""

        # Writes the parset blocks in the correct format
        return "\n".join(
            [f"{self.field}{counter}.{key}={val['value']}"
                for key, val in self.configuration.items()
                if (val['modified'] or val['required'])
            ])

# ============================================================= #

class _BeamParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, field, **kwargs):
        super().__init__(field=field)
        self.index = 0
        self._modify_properties(**kwargs)


    def __str__(self):
        return self._write_block_list(index=self.index)


    def is_above_horizon(self) -> bool:
        """ Checks that the numerical beam is pointed above the horizon. """
        # beam_start_time = Time(self["startTime"], format="isot")
        # beam_duration = self._get_duration()
        return True


    def _get_duration(self) -> TimeDelta:
        """ Reads the 'duration' field and converts it to a TimeDelta instance. """

        # Regex check to split the value and the unit
        match = re.match(
            pattern=r"(?P<value>\d+)(?P<unit>[smh])",
            string=self["duration"]
        )
        value = float(match.group("value"))

        # Prepares a dictionnary to convert unit to seconds
        to_seconds = {
            "s": 1,
            "m": 60,
            "h": 3600
        }
        conversion_factor = to_seconds[match.group("unit").lower()]

        # Converts the value to seconds
        seconds = value * conversion_factor
    
        return TimeDelta(seconds, format="sec")

# ============================================================= #

class _NumericalBeamParsetBlock(_BeamParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Beam", **kwargs)


    def __repr__(self) -> str:
        return f"<NumericalBeam(target={self['target']}, index={self.index})>"

# ============================================================= #

class _PhaseCenterParsetBlock(_BeamParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="PhaseCenter", **kwargs)


    def __repr__(self) -> str:
        return f"<PhaseCenter(target={self['target']}, index={self.index})>"

# ============================================================= #

class _AnalogBeamParsetBlock(_BeamParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Anabeam", **kwargs)
        self.numerical_beams = []
        self.phase_centers = []


    def __repr__(self) -> str:
        return f"<AnalogBeam(target={self['target']}, index={self.index})>"


    def _add_numerical_beam(self, **kwargs) -> None:
        """
        """
        self.numerical_beams.append(
            _NumericalBeamParsetBlock(
                **kwargs
            )
        )

    
    def _add_phase_center(self, **kwargs) -> None:
        """ """
        self.phase_centers.append(
            _PhaseCenterParsetBlock(
                **kwargs
            )
        )


    def _propagate_index(self) -> None:
        """
        """
        for _, numbeam in enumerate(self.numerical_beams):
            numbeam["noBeam"] = self.index
        for _, phasecenter in enumerate(self.phase_centers):
            phasecenter["noBeam"] = self.index

# ============================================================= #

class _OutputParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Output")
        self._modify_properties(**kwargs)


    def __str__(self):
        return self._write_block_list()

# ============================================================= #

class _ObservationParsetBlock(_ParsetBlock):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(field="Observation")
        self._modify_properties(**kwargs)


    def __str__(self):
        return self._write_block_list()

# ============================================================= #

class ParsetUser:
    """ Class that handles the formatting of a NenuFAR *parset_user* file.

        :Example:
            .. code-block:: python

                from nenupy.observation import ParsetUser

                # Create an instance of this class
                p = ParsetUser()

                # Add an analog beam
                p.add_analog_beam(target="Analog beam 1")

                # Add two numerical beams associated with this analog beam
                p.add_numerical_beam(0, target="One")
                p.add_numerical_beam(0, target="Two")

                # Print the analog beam parset block
                print(p.analog_beams[0])

                # Validate the parset
                p.validate()

                # Write the file
                p.write("my_parset.parset_user")


        .. seealso::

            NenuFAR `Parset_User guide <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ and the :ref:`tutorial <parset_user_doc>`.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~ParsetUser.analog_beam_fields
            ~ParsetUser.numerical_beam_fields
            ~ParsetUser.phase_center_fields
            ~ParsetUser.observation_fields
            ~ParsetUser.output_fields

        .. rubric:: Methods Summary

        .. autosummary::

            ~ParsetUser.set_observation_config
            ~ParsetUser.set_output_config
            ~ParsetUser.add_analog_beam
            ~ParsetUser.add_numerical_beam
            ~ParsetUser.modify_analog_beam
            ~ParsetUser.modify_numerical_beam
            ~ParsetUser.remove_analog_beam
            ~ParsetUser.remove_numerical_beam
            ~ParsetUser.validate
            ~ParsetUser.write

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self):
        self.observation = _ObservationParsetBlock()
        self.output = _OutputParsetBlock()
        self.analog_beams = []


    def __str__(self):
        self._update_beam_numbers()

        # Prepares the different text blocks
        observation_text = str(self.observation)
        output_text = str(self.output)
        return "\n\n".join(
            [observation_text,
            self._analog_beams_str,
            self._numerical_beams_str,
            self._phase_center_str,
            output_text]
        )


    @property
    def analog_beam_fields(self) -> list:
        """ Lists all the available *analog beam* fields. """
        return _AnalogBeamParsetBlock().fields
    

    @property
    def numerical_beam_fields(self) -> list:
        """ Lists all the available *numerical beam* fields. """
        return _NumericalBeamParsetBlock().fields


    @property
    def phase_center_fields(self) -> list:
        """ Lists all the available *phase center* fields. """
        return _PhaseCenterParsetBlock().fields


    @property
    def observation_fields(self) -> list:
        """ Lists all the available *observation* fields. """
        return _ObservationParsetBlock().fields


    @property
    def output_fields(self) -> list:
        """ Lists all the available *output* fields. """
        return _OutputParsetBlock().fields


    @property
    def _analog_beams_str(self) -> str:
        """
        """
        return "\n\n".join(
            str(anabeam)
            for anabeam in self.analog_beams
        )


    @property
    def _numerical_beams_str(self) -> str:
        """
        """
        return "\n\n".join(
            str(numbeam)
            for anabeam in self.analog_beams
            for numbeam in anabeam.numerical_beams
        )


    @property
    def _phase_center_str(self) -> str:
        """
        """
        return "\n\n".join(
            str(pc)
            for anabeam in self.analog_beams
            for pc in anabeam.phase_centers
        )


    @classmethod
    def replay_from_parset(cls, parset: Union[str, Parset], from_time: Time = Time.now(), same_lst: bool = False):

        # Initialize a Parset object if not provided
        if isinstance(parset, str):
            parset = Parset(parset)

        # Start a new ParsetUser object
        new_parset = cls()

        # Compute the time difference between the start times
        parset_start = parset.anabeams[0]["startTime"]
        if same_lst:
            new_start = "to be done"
        else:
            new_start = Time(f"{from_time.iso.split()[0]}T{parset_start.iso.split()[1]}")
        delta_time = new_start - parset_start
        if delta_time.sec < 0:
            raise ValueError(f"from_time needs to be after {parset_start.isot}")

        def new_parameters(prop: _ParsetProperty, fields: list) -> dict:
            params = {key: prop[key] for key in prop.keys() if key in fields}
            # Add coordinates information
            if "ra" in fields:
                from nenupy.astro.target import FixedTarget
                sky_coord = SkyCoord(Angle(prop["angle1"]), Angle(prop["angle2"]))
                target = FixedTarget(sky_coord)
                ra_hms = sky_coord.ra.to_string(u.hourangle, sep=":", pad=True)
                dec_dms = sky_coord.dec.to_string(u.degree, sep=":", pad=True)
                params["ra"] = f"'{ra_hms}'"
                params["dec"] = f"'{dec_dms}'"
            # Modify time instances
            try:
                params["startTime"] += delta_time
            except KeyError:
                pass
            try:
                params["filterTime"] = [filter_time + delta_time for filter_time in params["filterTime"]]
            except KeyError:
                pass
            # Add transit date
            if "transitDate" in fields:
                prev_transit = target.next_meridian_transit(params["startTime"])
                next_transit = target.next_meridian_transit(params["startTime"])
                transit = prev_transit if np.abs((prev_transit - params["startTime"]).sec) < np.abs((next_transit - params["startTime"]).sec) else next_transit
                transit.precision = 0
                params["transitDate"] = transit
            # Remove "seti" and "codalema"
            try:
                receivers = params["hd_receivers"].copy()
                receivers.remove("seti")
                receivers.remove("codalema")
                params["hd_receivers"] = receivers
            except KeyError:
                pass
            return params

        # Set up the Observation configuration part
        new_parset.set_observation_config(**new_parameters(parset.observation, new_parset.observation_fields))

        # Re-create every analog / numerical beams and phase centers
        for analog_beam in parset.anabeams.values():
            new_parset.add_analog_beam(**new_parameters(analog_beam, new_parset.analog_beam_fields))

        for digital_beam in parset.digibeams.values():
            new_parset.add_numerical_beam(
                anabeam_index=digital_beam["noBeam"],
                **new_parameters(digital_beam, new_parset.numerical_beam_fields)
            )

        for phase_center in parset.phase_centers.values():
            new_parset.add_phase_center(
                anabeam_index=phase_center["noBeam"],
                **new_parameters(phase_center, new_parset.phase_center_fields)
            )

        new_parset.set_output_config(**new_parameters(parset.output, new_parset.output_fields))

        return new_parset

    def set_observation_config(self, **kwargs) -> None:
        """ Sets the configuration of the *parset_user* observation block.
            This method ingests any valid `keyword argument <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ corresponding to an *Observation* configuration.

            :Example:
                .. code-block:: python
                    :emphasize-lines: 4

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.set_observation_config(
                        name="My observation",
                        contactName="AlanLoh",
                        contactEmail="alan.loh@obspm.fr",
                        topic="DEBUG"
                    )

        """
        for key, value in kwargs.items():
            self.observation[key] = value


    def set_output_config(self, **kwargs) -> None:
        """ Sets the configuration of the *parset_user* output block.
            This method ingests any valid `keyword argument <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ corresponding to an *Output* configuration.

            :Example:
                .. code-block:: python
                    :emphasize-lines: 4

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.set_output_config(
                        hd_bitMode=16,
                        hd_receivers="[undysputed]"
                    )

        """
        for key, value in kwargs.items():
            self.output[key] = value


    def add_analog_beam(self, **kwargs) -> None:
        """ Adds an *analog beam* to the :attr:`~nenupy.observation.parset.ParsetUser.analog_beams` attribute and updates its index based on other *analog beams*.
            This method ingests any valid `keyword argument <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ corresponding to an *Anabeam* configuration.

            :Example:
                .. code-block:: python
                    :emphasize-lines: 5

                    from nenupy.observation import ParsetUser
                    from astropy.time import Time, TimeDelta

                    p = ParsetUser()
                    p.add_analog_beam(
                        target="My fav target",
                        simbadSearch="Cygnus X-3",
                        trackingType="tracking",
                        duration=TimeDelta(3600, format="sec"), # or "3600s"
                        startTime=Time("2022-01-01 12:00:00") # or "2022-01-01T12:00:00Z"
                    )

        """
        self.analog_beams.append(
            _AnalogBeamParsetBlock(**kwargs)
        )
        self._updates_anabeams_indices()


    def modify_analog_beam(self, anabeam_index: int, **kwargs) -> None:
        """ Modifies the configuration of the *analog beam*.

            :param anabeam_index:
                Index of the *analog beam* to modify.
            :type anabeam_index:
                `int`
        
            :Example:
                .. code-block:: python
                    :emphasize-lines: 5

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="First")
                    p.modify_analog_beam(0, target="Modified_Value")

        """
        self.analog_beams[anabeam_index]._modify_properties(**kwargs)


    def remove_analog_beam(self, anabeam_index: int) -> None:
        """ Removes an *analog beam* along with its associated *numerical beams* and *phase centers*.

            :param anabeam_index:
                Index of the *analog beam* to remove.
            :type anabeam_index:
                `int`

            .. note::
                One can quickly identify the indices of the *analog beams*:

                .. code-block:: python
                    :emphasize-lines: 6

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="First")
                    p.add_analog_beam(target="Second")
                    p.analog_beams # prints: [<AnalogBeam(target=First, index=0)>, <AnalogBeam(target=Second, index=1)>] 


            :Example:
                .. code-block:: python
                    :emphasize-lines: 6

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="First")
                    p.add_analog_beam(target="Second")
                    p.remove_analog_beam(anabeam_index=0) # removes the analog beam "First"

        """
        del self.analog_beams[anabeam_index]
        self._updates_anabeams_indices()


    def add_numerical_beam(self, anabeam_index: int = 0, **kwargs) -> None:
        """ Adds a *numerical beam* to the *analog beam* '``anabeam_index``' and updates its index based on other *numerical beams*.
            This method ingests any valid `keyword argument <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ corresponding to a *Beam* configuration.

            :Example:
                .. code-block:: python
                    :emphasize-lines: 11

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(
                        target="My fav target",
                        simbadSearch="Cygnus X-3",
                        trackingType="tracking",
                        duration="3600s",
                        startTime="2022-01-01T12:00:00Z"
                    )
                    p.add_numerical_beam(
                        anabeam_index=0,
                        target="My fav target",
                        useParentPointing=True,
                        subbandList="[200..300]"
                    )

        """
        if anabeam_index >= len(self.analog_beams):
            raise IndexError(
                f"Requested analog beam index {anabeam_index} is out of range. Only {len(self.analog_beams)} analog beams are set."
            )
        anabeam = self.analog_beams[anabeam_index]
        anabeam._add_numerical_beam(**kwargs)
        anabeam._propagate_index()
        self._updates_numbeams_indices()
        

    def modify_numerical_beam(self, numbeam_index: int, **kwargs) -> None:
        """ Modifies the configuration of the *numerical beam*.

            :param numbeam_index:
                Index of the *numerical beam* to modify.
            :type numbeam_index:
                `int`
        
            :Example:
                .. code-block:: python
                    :emphasize-lines: 5

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="First")
                    p.add_numerical_beam(0, target="Initial_Value")
                    p.modify_numerical_beam(0, target="Modified_Value")

        """
        counter = 0
        for anabeam in self.analog_beams:
            for i, _ in enumerate(anabeam.numerical_beams):
                if counter==numbeam_index:
                    anabeam.numerical_beams[i]._modify_properties(**kwargs)
                    break
                counter += 1
            else:
                continue
            break


    def remove_numerical_beam(self, numbeam_index: int) -> None:
        """ Removes a *numerical beam* and updates the *numerical beam* indices.

            :param numbeam_index:
                Index of the *numerical beam* to remove.
            :type numbeam_index:
                `int`
            
            .. note::
                One can quickly identify the indices of the numerical beams:

                .. code-block:: python
                    :emphasize-lines: 9

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="Analog beam 1")
                    p.add_analog_beam(target="Analog beam 2")
                    p.add_numerical_beam(0, target="One")
                    p.add_numerical_beam(0, target="Two")
                    p.add_numerical_beam(1, target="Three")
                    [anabeam.numerical_beams for anabeam in p.analog_beams] # prints: [[<NumericalBeam(target=One, index=0)>, <NumericalBeam(target=Two, index=1)>], [<NumericalBeam(target=Three, index=2)>]]


            :Example:
                .. code-block:: python
                    :emphasize-lines: 6

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="Analog beam 1")
                    p.add_numerical_beam(0, target="One")
                    p.add_numerical_beam(0, target="Two")
                    p.remove_numerical_beam(numbeam_index=0) # removes the numerical beam "One"

        """
        counter = 0
        for anabeam in self.analog_beams:
            for i, _ in enumerate(anabeam.numerical_beams):
                if counter==numbeam_index:
                    del anabeam.numerical_beams[i]
                    break
                counter += 1
            else:
                continue
            break
        self._updates_numbeams_indices()


    def add_phase_center(self, anabeam_index: int = 0, **kwargs) -> None:
        """ Adds a *phase center* to the *analog beam* '``anabeam_index``' and updates its index based on other *phase centers*.
            This method ingests any valid `keyword argument <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ corresponding to a *PhaseCenter* configuration.

            :Example:
                .. code-block:: python
                    :emphasize-lines: 11

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(
                        target="My fav target",
                        simbadSearch="Cygnus X-3",
                        trackingType="tracking",
                        duration="3600s",
                        startTime="2022-01-01T12:00:00Z"
                    )
                    p.add_phase_center(
                        anabeam_index=0,
                        target="My fav target",
                        useParentPointing=True,
                        subbandList="[200..300]"
                    )

        """
        if anabeam_index >= len(self.analog_beams):
            raise IndexError(
                f"Requested analog beam index {anabeam_index} is out of range. Only {len(self.analog_beams)} analog beams are set."
            )
        anabeam = self.analog_beams[anabeam_index]
        anabeam._add_phase_center(**kwargs)
        anabeam._propagate_index()
        self._updates_phasecenter_indices()


    def modify_phase_center(self, phasecenter_index: int, **kwargs) -> None:
        """ Modifies the configuration of the *phase center*.

            :param phasecenter_index:
                Index of the *phase center* to modify.
            :type phasecenter_index:
                `int`
        
            :Example:
                .. code-block:: python
                    :emphasize-lines: 5

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="First")
                    p.add_phase_center(0, target="Initial_Value")
                    p.modify_phase_center(0, target="Modified_Value")

        """
        counter = 0
        for anabeam in self.analog_beams:
            for i, _ in enumerate(anabeam.phase_centers):
                if counter==phasecenter_index:
                    anabeam.phase_centers[i]._modify_properties(**kwargs)
                    break
                counter += 1
            else:
                continue
            break


    def remove_phase_center(self, phasecenter_index: int) -> None:
        """ Removes a *phase center* and updates the *phase center* indices.

            :param phasecenter_index:
                Index of the *phase center* to remove.
            :type phasecenter_index:
                `int`
            
            .. note::
                One can quickly identify the indices of the phase centers:

                .. code-block:: python
                    :emphasize-lines: 9

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="Analog beam 1")
                    p.add_analog_beam(target="Analog beam 2")
                    p.add_phase_center(0, target="One")
                    p.add_phase_center(0, target="Two")
                    p.add_phase_center(1, target="Three")
                    [anabeam.phase_centers for anabeam in p.analog_beams] # prints: [[<PhaseCenter(target=One, index=0)>, <PhaseCenter(target=Two, index=1)>], [<PhaseCenter(target=Three, index=2)>]]


            :Example:
                .. code-block:: python
                    :emphasize-lines: 6

                    from nenupy.observation import ParsetUser

                    p = ParsetUser()
                    p.add_analog_beam(target="Analog beam 1")
                    p.add_phase_center(0, target="One")
                    p.add_phase_center(0, target="Two")
                    p.remove_phase_center(phasecenter_index=0) # removes the phase center "One"

        """
        counter = 0
        for anabeam in self.analog_beams:
            for i, _ in enumerate(anabeam.phase_centers):
                if counter==phasecenter_index:
                    del anabeam.phase_centers[i]
                    break
                counter += 1
            else:
                continue
            break
        self._updates_phasecenter_indices()


    def validate(self) -> bool:
        """ Validates the syntax of each field.

            :returns:
                ``True`` if all the checked keys have a correct syntax.
            :rtype:
                `bool`

            .. warning::
                This is merely just a syntax validation made with regular expressions.
                The relevance of the parameter values are not checked at all.

        """
        is_valid = True

        # Update the beam numbers on the Observation table
        self._update_beam_numbers()

        # Check that the beams are above the horizon during the course of the observation
        for anabeam in self.analog_beams:
            if not anabeam.is_above_horizon():
                log.warning("")
            for numbeam in anabeam.numerical_beams:
                if not numbeam.is_above_horizon():
                    log.warning("")
            for pc in anabeam.phase_centers:
                if not pc.is_above_horizon():
                    log.warning("")

        def check_keys(dictionnary: dict) -> bool:
            all_keys_ok = True

            # Check each key and the corresponding regex syntax
            for key in dictionnary:
                # Get the regex syntax and if it doesn't exist, go to the next key
                try:
                    syntax_pattern = dictionnary[key]['syntax']
                except KeyError:
                    continue

                # Don't check the key if it has not been modified
                if not dictionnary[key]["modified"]:
                    continue

                # Retrieve the value that needs to be checked
                value = dictionnary[key]["value"]
                if str(value) == '':
                    log.warning(f"Empty value for key '{key}'.")

                # Perform a regex full match check, send a warning if invalid
                if re.fullmatch(pattern=syntax_pattern, string=str(value)) is None:
                    log.error(
                        f"Syntax error on '{value}' (key '{key}'). "
                        f"\n\tExpected syntax: {syntax_pattern}."
                        "\n\tPlease check https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html"
                    )
                    all_keys_ok &= False

            return all_keys_ok 

        # Check all configurations
        log.info("Checking 'observation' block...")
        is_valid &= check_keys(self.observation.configuration)

        log.info("Checking 'output' block...")
        is_valid &= check_keys(self.output.configuration)

        for anabeam in self.analog_beams:
            log.info(f"Checking 'anabeam[{anabeam.index}]' block...")
            is_valid &= check_keys(anabeam.configuration)

            for numbeam in anabeam.numerical_beams:
                log.info(f"Checking 'beam[{numbeam.index}]' block...")
                is_valid &= check_keys(numbeam.configuration)
            
            for pc in anabeam.phase_centers:
                log.info(f"Checking 'phasecenter[{pc.index}]' block...")
                is_valid &= check_keys(pc.configuration)

        return is_valid


    def write(self, file_name: str) -> None:
        """ Writes the current instance of :class:`~nenupy.observation.parset.ParsetUser`
            to a file called ``file_name``. 

            :param file_name:
                Name of the *parset_user* file to write.
                The file extension must be ``".parset_user"``.
            :type file_name:
                `str`
        """
        if not file_name.endswith(".parset_user"):
            raise ValueError(f"file_name='{file_name}' does not end with '.parset_user'.")
        with open(file_name, "w") as wfile:
            wfile.write(str(self))
        log.debug(f"Parset written in file {file_name}.")


    def _updates_numbeams_indices(self) -> None:
        """ Updates the indices of numerical beams. """
        numbeams_counter = 0
        for anabeam in self.analog_beams:
            for numbeam in anabeam.numerical_beams:
                numbeam.index = numbeams_counter
                numbeams_counter += 1


    def _updates_phasecenter_indices(self) -> None:
        """ Updates the indices of phase centers. """
        pc_counter = 0
        for anabeam in self.analog_beams:
            for pc in anabeam.phase_centers:
                pc.index = pc_counter
                pc_counter += 1


    def _updates_anabeams_indices(self) -> None:
        """ Updates the indices of analog beams. """
        anabeams_counter = 0
        for anabeam in self.analog_beams:
            anabeam.index = anabeams_counter
            anabeam._propagate_index()
            anabeams_counter += 1
        self._updates_numbeams_indices()


    def _update_beam_numbers(self) -> None:
        """ Updates the number of analog, numerical beams and phase centers. """
        nb_analog_beams = len(self.analog_beams)
        nb_numerical_beams = sum(len(anabeam.numerical_beams) for anabeam in self.analog_beams)
        nb_phase_centers = sum(len(anabeam.phase_centers) for anabeam in self.analog_beams)
        self.observation["nrAnabeams"] = str(nb_analog_beams)
        self.observation["nrBeams"] = str(nb_numerical_beams)
        self.observation["nrPhaseCenters"] = str(nb_phase_centers)
# ============================================================= #
# ============================================================= #
