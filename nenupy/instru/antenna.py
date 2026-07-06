#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
    *******
    Antenna
    *******
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2026, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "CSTModel",
    "write_healpix_antenna_file",
    "ant_pol_to_ref",
    "download_ant_ref_models"
]


from astropy.table import QTable
import astropy.units as u
import healpy as hp

import re
import os
from typing import Tuple, List
import numpy as np
import urllib.request
from scipy.interpolate import RegularGridInterpolator
import logging

from nenupy.instru import nenufar_miniarrays, miniarray_antennas

log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- CSTModel -------------------------- #
# ============================================================= #
class CSTModel:

    def __init__(self, cst_file: str, x_column: str = "Phi", y_column: str = "Theta", gain_column: str = "Abs(E)"):
        self.cst_file = cst_file
        self.frequency = self._infer_frequency_from_name(cst_file)
        self.data = self._read_cst_data(cst_file)
        self._column_names = self.data.colnames
        self.x_column = x_column
        self.y_column = y_column
        self.gain_column = gain_column
        self.shape = self._infer_2d_shape(self.data, x_coord=x_column, y_coord=y_column)
        self.x_axis, self.y_axis = self._get_coordinate_axes()

    @property
    def cst_file(self) -> str:
        return self._cst_file
    @cst_file.setter
    def cst_file(self, c: str) -> None:
        self._cst_file = c

    @property
    def x_column(self) -> str:
        return self._x_column
    @x_column.setter
    def x_column(self, col: str) -> None:
        if col not in self._column_names:
            raise ValueError(f"x column name {col} not in {self._column_names}.")
        self._x_column = col

    @property
    def y_column(self) -> str:
        return self._y_column
    @y_column.setter
    def y_column(self, col: str) -> None:
        if col not in self._column_names:
            raise ValueError(f"y column name {col} not in {self._column_names}.")
        self._y_column = col

    @property
    def gain_column(self) -> str:
        return self._gain_column
    @gain_column.setter
    def gain_column(self, col: str) -> None:
        if col not in self._column_names:
            raise ValueError(f"gain column name {col} not in {self._column_names}.")
        self._gain_column = col

    @property
    def complex_gain(self) -> np.ndarray:
        if "Abs(Left)" in self.data.columns:
            x_axis = "Left"
            y_axis = "Right"
        else:
            x_axis = "Phi"
            y_axis = "Theta"
        x_component = self.data[f"Abs({x_axis})"] * np.exp( 1j * self.data[f"Phase({x_axis})"].to_value(u.rad) )
        y_component = self.data[f"Abs({y_axis})"] * np.exp( 1j * self.data[f"Phase({y_axis})"].to_value(u.rad) )

        return x_component + y_component

    def to_healpix(self, nside: int = 64, half_sky: bool = False) -> np.ndarray:
        # Add the phi=360deg value as a duplicate of the phi=0deg so that there won't be any extrapolation
        x_axis_rad = self.x_axis.to_value(u.rad)
        y_axis_rad = self.y_axis.to_value(u.rad)

        gain = self.data[self.gain_column].reshape(self.shape)

        if self.x_column == "Phi":
            gain = np.vstack((gain, gain[0, :]))
            x_axis_rad = np.append(x_axis_rad, np.radians(360))

        # CST interpolation
        gain_interp = RegularGridInterpolator(
            (x_axis_rad, y_axis_rad),
            gain,
            bounds_error=False
        )

        # HealPIX representation
        azgrid, elgrid = hp.pix2ang(
            nside=nside,
            ipix=np.arange(hp.nside2npix(nside)),
            lonlat=True, # in degrees
            nest=False
        )
        # if half_sky:
        #     elevation_mask = elgrid >= 0
        # azgrid = np.radians(azgrid[elevation_mask][::-1]) # east is 90
        # elgrid = np.radians(elgrid[elevation_mask])
        azgrid = np.radians(azgrid[::-1]) # east is 90
        elgrid = np.radians(elgrid)

        return gain_interp((azgrid, elgrid))

    def plot(self):
        return

    @staticmethod
    def _infer_frequency_from_name(file_name: str) -> u.Quantity:
        frequency_str = re.findall(r"\(f=(.*?)\)", file_name)
        if frequency_str is None:
            return None
        return float(frequency_str[0]) * u.MHz

    @staticmethod
    def _read_cst_data(cst_file: str) -> QTable:
        """Read a CST farfield file, parse its columns and units and return
        an astropy.table.QTable

        Parameters
        ----------
        cst_file : `str`
            Output farfield file from a CST simulation.

        Returns
        -------
        :class:`~astropy.table.QTable`
            Data parsed in astropy format.
        """

        with open(cst_file, "r") as rfile:
            header_row = rfile.readline()

        # Parse column names and their physical units
        column_names = []
        units = []
        for col in header_row.split("]")[:-1]:
            name, unit = col.split("[")
            column_names.append(name.replace(" ", ""))
            units.append(u.Unit(unit.strip().replace(".", "")))

        return QTable.read(
            cst_file,
            format="ascii",
            data_start=2,
            names=tuple(column_names),
            units=tuple(units)
        )

    @staticmethod
    def _infer_2d_shape(data: QTable, x_coord: str = "Phi", y_coord: str = "Theta") -> Tuple[int, int]:
        x_size = np.unique(data[x_coord]).size
        y_size = np.unique(data[y_coord]).size
        assert len(data) / y_size == x_size, "Mismatch between data length and x/y dimensions."
        return (x_size, y_size)

    def _get_coordinate_axes(self) -> Tuple[u.Quantity, u.Quantity]:
        x = self.data[self.x_column].reshape(self.shape)
        x_axis = x[:, 0]
        y = self.data[self.y_column].reshape(self.shape)
        y_axis = y[0, :]
        if self.y_column == "Theta":
            y_axis = 90 * u.deg - y_axis
        return (x_axis, y_axis)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------- write_healpix_antenna_file ----------------- #
# ============================================================= #
def write_healpix_antenna_file(filename: str, nw_se_files: Tuple[str] = None, ne_sw_files: Tuple[str] = None, nside: int = 64) -> None:
    """_summary_

    Parameters
    ----------
    filename : str
        _description_
    nw_se_files : Tuple[str], optional
        _description_, by default None
    ne_sw_files : Tuple[str], optional
        _description_, by default None
    nside : int, optional
        _description_, by default 64

    Example
    -------
        write_healpix_antenna_file(
        filename="/Users/aloh/Desktop/NenuFAR_Ant_10_Hpx.fits",
        nw_se_files=[
            f for f in glob.glob(
                "/Users/aloh/Downloads/MR-NW-SE-Rot0-Ant1-2-5-10/*.txt"
            ) if "[10]" in f
        ]
    )
    """

    ant_gain = []
    col_name = []

    for pol in ["NW_SE", "NE_SW"]:

        if pol == "NW_SE":
            files_to_use = nw_se_files
        else:
            files_to_use = ne_sw_files

        if files_to_use is None:
            continue
        else:
            if len(files_to_use) == 0:
                raise FileNotFoundError(f"No files found for polarization {pol}.")

        # Read files:
        frequencies = []
        cst_instances = []
        for cst_file in files_to_use:
            cst = CSTModel(cst_file)
            cst_instances.append(cst)
            frequencies.append(cst.frequency.to_value(u.MHz))

        # Sort by increasing frequency
        sort_idx = np.argsort(frequencies)

        for freq, cst in zip(np.array(frequencies)[sort_idx], np.array(cst_instances)[sort_idx]):
            ant_gain.append(cst.to_healpix(nside=nside, half_sky=False))
            col_name.append(f"{pol[:2]}_{freq}")
    
    if len(ant_gain) == 0:
        return
    hp.write_map(
        filename=filename,
        m=ant_gain,
        column_names=col_name,
        overwrite=True
    )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- ant_pol_to_ref ----------------------- #
# ============================================================= #
def ant_pol_to_ref(mini_array: int = 0, antenna: str = "Ant01", polarization: str = "NW") -> List[dict]:
    """Return the corresponding reference dipole measured during the June 2026 unique antenna campaign.
    Starting from June 23rd 2026, we measured SST using Mini-Arrays 88, 91, 93, 94, 95 in single antenna mode.
    We measured the sky transits over ~24hrs, switching antennas every day.
    We took advantage of the array while these MAs were unused by beamformed observations.
    
    The reference observations are:

    | Day | Mini-Array | Antenna | Attenuation (dB) | Start | Stop |
    |:--:|:--:|:--:|:--:|:--:|:--:|
    | 1 | MA 88 | Ant 01 | 12.5 | 2026-06-24 06:30:00 | 2026-06-25 05:40:00 |
    | 1 | MA 91 | Ant 08 | 13   | 2026-06-24 06:30:00 | 2026-06-25 05:40:00 |
    | 1 | MA 93 | Ant 01 | 9.5  | 2026-06-24 06:30:00 | 2026-06-25 05:40:00 |
    | 1 | MA 94 | Ant 02 | 15.5 | 2026-06-24 06:30:00 | 2026-06-25 05:40:00 |
    | 1 | MA 95 | Ant 01 | 11.5 | 2026-06-24 06:30:00 | 2026-06-25 05:40:00 |
    | 2 | MA 88 | Ant 05 | 12.5 | 2026-06-25 06:10:00 | ... |
    | 2 | MA 91 | Ant 02 | 13   | 2026-06-25 06:10:00 | ... |
    | 2 | MA 93 | Ant 02 | 9.5  | 2026-06-25 06:10:00 | ... |
    | 2 | MA 94 | Ant 06 | 15.5 | 2026-06-25 06:10:00 | ... |
    | 2 | MA 95 | Ant 02 | 11.5 | 2026-06-25 06:10:00 | ... |


    This function returns the list of measured reference dipoles that are equivalent to the inputs provided.

    Parameters
    ----------
    mini_array : `int`, optional
        _description_, by default 0
    antenna : `str`, optional
        _description_, by default "Ant01"
    polarization : `str`, optional
        _description_, by default "NW"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    
    ref_antennas = {
        88: ["Ant01", "Ant02", "Ant03", "Ant05"], # 220°
        91: ["Ant02", "Ant08", "Ant09"], # 90°
        93: ["Ant01", "Ant02", "Ant05", "Ant10"], # 10°
        94: ["Ant02", "Ant06"], # 20°
        95: ["Ant01", "Ant02", "Ant05", "Ant10"] # 120°
    }

    polarization_vectors = {
        "NW": np.array([-1, 1]),
        "NE": np.array([1, 1])
    }
    if polarization not in polarization_vectors:
        raise ValueError("polarization should either be 'NW' or 'NE'.")

    # Compute antenna positions
    def ma_ant_pos_name(ma_id: int) -> Tuple[np.ndarray, np.ndarray]:
        antenna_names = np.array([ant for ant in miniarray_antennas.keys()])
        antPos = np.array([ant["position"] for ant in miniarray_antennas.values()])
        rotation = nenufar_miniarrays[f"MA{ma_id:03d}"]["rotation"] * u.deg
        rotation = np.radians(360 - rotation.value)
        rotMatrix = np.array(
            [
                [np.cos(rotation), -np.sin(rotation), 0],
                [-np.sin(rotation), -np.cos(rotation), 0],
                [0,           0,           1]
            ]
        )
        antenna_positions = np.dot(antPos, rotMatrix).astype(np.float32)
        return antenna_positions, antenna_names
    
    antenna_positions, antenna_names = ma_ant_pos_name(ma_id=mini_array)

    # Compute the scalar product of the desired antenna, sort the values so that they can be compared
    try: 
        ant_id = np.argwhere(antenna_names == antenna)[0][0]
    except IndexError:
        raise ValueError(f"{antenna} not recognized, please select one from {antenna_names}")
    desired_scalar = np.sort(
        np.dot(
            polarization_vectors[polarization], 
            (antenna_positions[:, 0:2] - antenna_positions[ant_id, 0:2]).T
        )
    )

    # Compute every scalar product between the dipole/polarization vectors
    # and the vectors from the given antenna towards every other antennas within a MA
    result = []
    for ma_id in ref_antennas:
        current_ma_antenna_positions, current_ma_antenna_names = ma_ant_pos_name(ma_id)
        for pol in polarization_vectors:
            for ant in ref_antennas[ma_id]:
                ant_i = np.argwhere(current_ma_antenna_names == ant)[0][0]
                reference_scalar = np.dot(
                    polarization_vectors[pol], 
                    (current_ma_antenna_positions[:, 0:2] - current_ma_antenna_positions[ant_i, 0:2]).T
                )
                reference_scalar_pos = np.sort(reference_scalar)
                reference_scalar_neg = np.sort( - reference_scalar)
                if np.all(np.isclose(desired_scalar, reference_scalar_pos)) or np.all(np.isclose(desired_scalar, reference_scalar_neg)):
                    result.append(
                        {
                            "ma": ma_id,
                            "polar": pol,
                            "antenna": ant
                        }
                    )

    return result
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------ download_ant_ref_models ------------------ #
# ============================================================= #
def download_ant_ref_models(save_path: str = "") -> None:
    """ Download NenuFAR antenna reference models from Zenodo. 

    Parameters
    ----------
    save_path : `str`, optional
        Path were the models will be saved, by default ""
    """
    log.info("Downloading antenna models...")
    ref_antennas = {
        88: ["Ant01", "Ant02", "Ant03", "Ant05"], # 220°
        91: ["Ant02", "Ant08", "Ant09"], # 90°
        93: ["Ant01", "Ant02", "Ant05", "Ant10"], # 10°
        94: ["Ant02", "Ant06"], # 20°
        95: ["Ant01", "Ant02", "Ant05", "Ant10"] # 120°
    }
    for ma_id in ref_antennas:
        for ant in ref_antennas[ma_id]:
            try:
                filename = os.path.join(
                    save_path,
                    f"nenufar_ma{ma_id}_{ant.lower()}.fits"
                )
                url = f"https://zenodo.org/records/21219004/files/{os.path.basename(filename)}?download=1"
                fname, header = urllib.request.urlretrieve(url, filename)
                log.info(f"{fname} downloaded.")
            except:
                log.error(f"Impossible to download '{os.path.basename(filename)}' from '{url}' to '{filename}'.")
                raise
# ============================================================= #
# ============================================================= #
