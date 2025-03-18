#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *************
    NenuFAR Tools
    *************

"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "freq2sb",
    "sb2freq",
    "instrument_temperature",
    "miniarrays_rotated_like",
    "read_cal_table",
    "generate_nenufar_subarrays",
    "lofar_instrument_temperature"
]


import numpy as np
import astropy.units as u
from typing import List, Tuple
from os.path import join, dirname
from scipy.interpolate import interp2d
from astropy.coordinates import Latitude, Longitude, SkyCoord

from nenupy.instru import lna_gain
from nenupy.astro.astro_tools import sky_temperature
from nenupy.instru import nenufar_miniarrays

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------- freq2sb --------------------------- #
# ============================================================= #
def freq2sb(frequency: u.Quantity):
    r""" Conversion between the frequency :math:`\nu` and the
        NenuFAR sub-band index :math:`n_{\rm SB}`.
        Each NenuFAR sub-band has a bandwidth of
        :math:`\Delta \nu = 195.3125\, \rm{kHz}`:

        .. math::
            n_{\rm SB} = \lfloor*{ \frac{\nu}{\Delta \nu} + \frac{1}{2} \rfloor

        :param frequency:
            Frequency to convert in sub-band index.
        :type frequency:
            :class:`~astropy.units.Quantity`

        :returns:
            Sub-band index, same shape as ``frequency``.
        :rtype:
            `int` or :class:`~numpy.ndarray`

        :example:
            .. code-block:: python
                
                from nenupy.instru import freq2sb
                import astropy.units as u

                freq2sb(frequency=50.5*u.MHz)
                freq2sb(frequency=[50.5, 51]*u.MHz)

    """
    if not isinstance(frequency, u.Quantity):
        raise TypeError(
            f"`frequency` - {u.Quantity} expected."
        )
    if (frequency.min() < 0 * u.MHz) or (frequency.max() > 100 * u.MHz):
        raise ValueError(
            "'frequency' should be between 0 and 100 MHz."
        )
    frequency = frequency.to(u.MHz)
    sb_width = 100. * u.MHz / 512
    sb_idx = np.floor(frequency/sb_width + 0.5)
    return sb_idx.astype(int).value
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- freq2sb --------------------------- #
# ============================================================= #
def sb2freq(subband):
    r""" Conversion between NenuFAR sub-band index :math:`n_{\rm SB}`
        to sub-band starting frequency :math:`\nu_{\rm start}`.

        .. math::
            \nu_{\rm start} = n_{\rm SB} \times \Delta \nu

        Each NenuFAR sub-band has a bandwidth of
        :math:`\Delta \nu = 195.3125\, \rm{kHz}`, therefore, the
        sub-band :math:`n_{\rm SB}` goes from :math:`\nu_{\rm start}`
        to :math:`\nu_{\rm stop} = \nu_{\rm start} + \Delta \nu`.

        :param subband:
            Sub-band index (from 0 to 511).
        :type subband:
            `int` or :class:`~numpy.ndarray` of `int`

        :returns:
            Sub-band start frequency :math:`\nu_{\rm start}` in MHz.
        :rtype:
            :class:`~astropy.units.Quantity`

        :Example:
            .. code-block:: python
            
                from nenupy.instru import sb2freq

                sb2freq(subband=1)
                sb2freq(subband=[1, 2, 3, 4])

    """
    if np.isscalar(subband):
        subband = np.array([subband])
    else:
        subband = np.array(subband)
    if subband.dtype.name not in ['int32', 'int64']:
        raise TypeError(
            "`subband` - Integer(s) expected."
        )
    if (subband.min() < 0) or (subband.max() > 511):
        raise ValueError(
            "`subband` - Values should be between 0 and 511."
        )
    sb_width = 100.*u.MHz/512 #200e6*192/1024# + 195312.5/2
    freq_start = subband*sb_width - sb_width/2
    return freq_start
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------ instrument_temperature ------------------- #
# ============================================================= #
def instrument_temperature(frequency: u.Quantity = 50*u.MHz, lna_filter: int = 0) -> u.Quantity:
    """ Instrument temperature at a given ``frequency``.
        This depends on the `Low Noise Amplifier <https://nenufar.obs-nancay.fr/en/astronomer/#antennas>`_ 
        characteristics.

        :param frequency:
            Frequency at which computing the instrument temperature.
            Default is ``50 MHz``.
        :type frequency:
            :class:`~astropy.units.Quantity`
        :param lna_filter:
            Local Noise Amplifier high-pass filter selection.
            Available values are ``0, 1, 2, 3``.
            They correspond to minimal frequencies ``10, 15, 20, 25 MHz`` respectively.
            Default is ``0``, i.e., 10 MHz filter.
        :type lna_filter:
            `int`

        :returns:
            Instrument temperature in Kelvins
        :rtype:
            :class:`~astropy.units.Quantity`
        
        .. warning::
            For the time being, only ``lna_filter`` values ``0`` and ``3`` are available.

        :Example:
            .. code-block:: python
            
                from nenupy.instru import instrument_temperature
                import astropy.units as u

                instrument_temperature(frequency=70*u.MHz)

        .. seealso::
            :func:`~nenupy.astro.astro_tools.sky_temperature`

    """
    # Available filters
    filters = {0: "no_filter", 3: "25mhz_filter"}

    # lna_gain represents T_ins/T_sky - measured
    lna = lna_gain[filters[lna_filter]]

    # Multiply lna_gain (interpolatd at the desired frequencies) by T_sky to get T_ins
    t_sky = sky_temperature(frequency=frequency)
    t_inst = t_sky * np.interp(frequency, lna["frequency"]*u.MHz, lna["gain"])

    return t_inst
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------ miniarrays_rotated_like ------------------ #
# ============================================================= #
def miniarrays_rotated_like(rotations: List[int] = [0]) -> np.ndarray:
    r""" Returns the Mini-Array indices whose rotations match the ``rotations`` argument.
        A :math:`60^{\circ}` modulo is automatically applied to all rotation parameters.

        :param rotations:
            Mini-Array rotation(s) to select.
            A ``ValueError`` is raised if the values are not integers and/or if they are not multiples of 10.
        :type rotations:
            `list`[`int`]

        :returns:
            Mini-Array indices.
        :rtype:
            :class:`~numpy.ndarray`

        :Example:
            .. code-block:: python
            
                from nenupy.instru import miniarrays_rotated_like
                
                miniarrays_rotated_like([10])

    """
    # Check that the rotation format is correct
    if not all([rot%10 == 0 for rot in rotations]):
        raise ValueError(
            f"Syntax error: miniarray_rotations={rotations}. It should be a list of integers, multiples of 10."
        )
    ma_rotations = np.array([nenufar_miniarrays[ma]["rotation"] for ma in nenufar_miniarrays])%60
    ma_indices = np.array([nenufar_miniarrays[ma]["id"] for ma in nenufar_miniarrays])

    return ma_indices[np.isin(ma_rotations, np.array(rotations)%60)]
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- read_cal_table ----------------------- #
# ============================================================= #
def read_cal_table(calibration_file: str = None) -> np.ndarray:
    """ Reads NenuFAR antenna delays calibration file.

        :param calibration_file: 
            Name of the calibration file to read. If ``None`` or
            ``'default'`` the standard calibration file is read.
        :type calibration_file: `str`

        :returns: 
            Antenna delays shaped as 
            (frequency, mini-arrays, polarizations).
        :rtype: :class:`~numpy.ndarray`
    """
    if (calibration_file is None) or (calibration_file.lower() == "default"):
        calibration_file = join(
            dirname(__file__),
            'cal_pz_2_multi_2019-02-23.dat',
        )
    with open(calibration_file, 'rb') as f:
        log.info(
            "Loading calibration table {}".format(
                calibration_file
            )
        )
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.startswith(b"HeaderStop"):
                break
    hd_size = sum([len(s) for s in header])
    dtype = np.dtype(
        [
            ('data', 'float64', (512, 96, 2, 2))
        ]
    )
    tmp = np.memmap(
        filename=calibration_file,
        dtype="int8",
        mode="r",
        offset=hd_size
    )
    decoded = tmp.view(dtype)[0]["data"]
    data = decoded[..., 0] + 1.j*decoded[..., 1]
    return data
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------- generate_nenufar_subarrays ----------------- #
# ============================================================= #
def generate_nenufar_subarrays(
        n_subarrays: int = 2,
        include_remote_mas: bool = False
    ):
    """ Generates NenuFAR sub-arrays of Mini-Arrays. The sub-arraying
        is done completely randomly.

        :param n_subarrays:
            Number of sub-arrays to generate from the NenuFAR
            Mini-Array distribution. Default is `2`.
        :type n_subarrays:
            `int`
        :param include_remote_mas:
            Include or not the remote Mini-Arrays.
        :type include_remote_mas:
            `bool`
        
        :returns:
            A list of Mini-Array names for each sub-array.
        :rtype:
            `list` of `~numpy.ndarray`

        :Example:
            .. code-block:: python

                from nenupy.instru import generate_nenufar_subarrays
                from nenupy.instru import NenuFAR

                sub_arrays = generate_nenufar_subarrays(n_subarrays=2)
                sub_array_1 = NenuFAR()[sub_arrays[0]]
                sub_array_2 = NenuFAR()[sub_arrays[1]]

    """

    if n_subarrays < 2:
        raise ValueError(
            "`n_subarrays` should be at least equal to 2."
        )

    # Number of Mini-Arrays
    n_mas = 96
    if include_remote_mas:
        n_mas = 102

    # Mini-Arrays names
    ma_names = np.array([ma_name for ma_name in nenufar_miniarrays.keys()])
    ma_names = ma_names[:n_mas]

    # Generate random values
    rng = np.random.default_rng()
    ma_values = rng.random(n_mas) # floats between [0, 1[

    # Create mask by evaluating the random value per MA
    edge_values = np.linspace(0, 1, n_subarrays + 1)
    subarray_masks = (edge_values[:-1][:, None] <= ma_values[None, :]) & (ma_values[None, :] < edge_values[1:][:, None]) 

    # Check that each Mini-Array is counted once
    if not np.all(np.sum(subarray_masks, axis=0)==1):
        raise ValueError("Something's wrong.")

    # Return a list of sub-arrays (numpy arrays of Mini-Array names)
    return [ma_names[subarray_masks[i]] for i in range(n_subarrays)]
# ============================================================= #
# ============================================================= #

# ============================================================= #
# --------------- lofar_instrument_temperature ---------------- #
# ============================================================= #
def lofar_instrument_temperature(frequency: u.Quantity) -> u.Quantity:
    """_summary_
    From Vlad Kondratiev lofar_tinst.py (polynomial fit on Wijnholds (2011))

    Parameters
    ----------
    frequency : u.Quantity
        _description_

    Returns
    -------
    u.Quantity
        _description_
    """
    lba_mask = frequency <= 90*u.MHz
    hba_mask = frequency >= 110*u.MHz
    t_inst_poly_fit_coeff_lba = np.array(
        [  
            6.2699888333e-05,
            -0.019932340239,
            2.60625093843,
            -179.560314268,
            6890.14953844,
            -140196.209123,
            1189842.07708
        ]
    )
    t_inst_poly_fit_coeff_hba = np.array(
        [
            6.64031379234e-08,
            -6.27815750717e-05,
            0.0246844426766,
            -5.16281033712,
            605.474082663,
            -37730.3913315,
            975867.990312
        ]
    ) 
    t_inst_poly_fit_lba = np.poly1d(t_inst_poly_fit_coeff_lba)
    t_inst_poly_fit_hba = np.poly1d(t_inst_poly_fit_coeff_hba)

    instrument_temperature = np.empty(frequency.shape)
    instrument_temperature[lba_mask] = t_inst_poly_fit_lba(frequency[lba_mask].to_value(u.MHz))
    instrument_temperature[hba_mask] = t_inst_poly_fit_hba(frequency[hba_mask].to_value(u.MHz))
    instrument_temperature[~(lba_mask + hba_mask)] = np.nan
    return instrument_temperature * u.K
# ============================================================= #
# ============================================================= #

# ============================================================= #
# -------------- mini_array_analog_pointing_grid -------------- #
# ============================================================= #
def mini_array_analog_pointing_grid(ma_rotation: u.Quantity = 0 * u.deg) -> Tuple[Longitude, Latitude]:
    """Compute the grod of analog pointing for a given Mini-Array rotation.

    Example
    -------
    .. code-block:: python

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=(10, 10))
        >>> ax = fig.add_subplot(projection="polar")
        >>> ax.set_rlim(90, 0)
        >>> ma_rot = 0 * u.deg
        >>> grid = SkyCoord(*mini_array_analog_pointing_grid(ma_rotation=ma_rot))
        >>> ax.scatter(grid.ra.rad, grid.dec.deg, 0.5)

    Parameters
    ----------
    ma_rotation : :class:`~astropy.units.Quantity`, optional
        Rotation of the Minni-Array, by default `0*u.deg`

    Returns
    -------
    Tuple[:class:`~astropy.coordinates.Longitude`, :class:`~astropy.coordinates.Latitude`]
        Horizontal coordinates of the pointing grid.
    """
    dx = 2 * 5.5
    # DY = dx * np.cos(np.pi / 6)
    dmin_x = 0.165
    # DMINY = dmin_x * np.cos(np.pi / 6)
    dmin_d = dmin_x / dx
    n_bits = 7
    bits = 2**(n_bits - 1)

    bits_array = np.arange(2*bits)
    xx, yy = np.meshgrid(bits_array, bits_array)

    xx_mask = xx >= 64
    yy_mask = yy >= 64

    k1 = (xx - bits + 1) * dmin_d
    k2 = (bits - 1 - yy) * dmin_d
    k1[xx_mask] = (xx[xx_mask] - bits) * dmin_d
    k2[yy_mask] = (bits - yy[yy_mask]) * dmin_d

    # theta =  0.5*np.arccos(1 - 2*(k1**2 + k2**2))
    with np.errstate(invalid="ignore"):
        theta = np.pi / 2 - (0.5 * np.arccos(1 - 2 * (k1**2 + k2**2)))
    bad_values = np.isnan(theta)
    # phi = np.arctan2(k2, k1) + np.pi
    phi = np.pi / 2 - (np.arctan2(k2, k1) + np.pi) + ma_rotation.to_value("rad")

    theta[bad_values] = - np.pi / 2
    phi[bad_values] = 0.

    return (
        Longitude(phi, unit="rad"),
        Latitude(theta, unit="rad"),
    )

def mini_array_pointing_order(coordinates: SkyCoord, ma_rotation: u.Quantity = 0 * u.deg, wrong_implementation: bool = False) -> np.ndarray:

    if coordinates.ndim > 1:
        raise Exception("coordinates can only have a single dimension.")
    elif coordinates.ndim == 0:
        coordinates = coordinates.reshape((1,))

    ma_pointing_grid_lon, ma_pointing_grid_lat = mini_array_analog_pointing_grid(ma_rotation=ma_rotation)

    # Find out the angular separation between coordinates and the ma pointing grid
    # The resulting separation shape will be (n_coordinates, 128, 128)
    def separation_vincenty(lon1, lat1, lon2, lat2):
        """https://en.wikipedia.org/wiki/Great-circle_distance"""
        lon_diff = np.abs(lon2 - lon1)
        
        cos_lon_diff = np.cos(lon_diff)
        sin_lon_diff = np.sin(lon_diff)

        cos_lat1 = np.cos(lat1)
        sin_lat1 = np.sin(lat1)
        cos_lat2 = np.cos(lat2)
        sin_lat2 = np.sin(lat2)

        part_1 = np.sqrt( (cos_lat2 * sin_lon_diff)**2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_lon_diff)**2)
        part_2 = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_lon_diff
        
        dist_rad = np.atan2(part_1, part_2)

        return dist_rad
    angular_sperarations = separation_vincenty(
        lon1=coordinates.ra.rad[:, None, None],
        lat1=coordinates.dec.rad[:, None, None],
        lon2=ma_pointing_grid_lon.rad[None, :, :],
        lat2=ma_pointing_grid_lat.rad[None, :, :]
    )
    sep_shape = angular_sperarations.shape
    order = np.array(
        np.unravel_index(
            np.argmin(
                angular_sperarations.reshape(
                    (sep_shape[0], sep_shape[1] * sep_shape[2])
                ),
                axis=1
            ),
            sep_shape[1:],
        )
    )

    # Correct for order #64 which is identical to # 63
    if wrong_implementation:
        order[order >= 64] -= 1
    else:
        order[order == 64] -= 1

    return order.T

def mini_array_pointing_coordinates(orders: np.ndarray, ma_rotation: u.Quantity = 0 * u.deg) -> SkyCoord:

    if orders.ndim != 2:
        raise Exception("orders must be 2D")
    elif orders.shape[1] != 2:
        raise Exception("orders must have its 2nd dimension of size 2")

    ma_pointing_grid_lon, ma_pointing_grid_lat = mini_array_analog_pointing_grid(ma_rotation=ma_rotation)

    return SkyCoord(
        ma_pointing_grid_lon[orders[:, 0], orders[:, 1]],
        ma_pointing_grid_lat[orders[:, 0], orders[:, 1]]
    )