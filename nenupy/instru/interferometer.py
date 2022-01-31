#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *********************
    Interferometric Array
    *********************
"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "Baseline",
    "ObservingMode",
    "Interferometer"
]


from abc import ABC, ABCMeta, abstractmethod
import copy
from enum import Enum, auto
from functools import lru_cache
from os import cpu_count
from typing import Dict, Callable
import logging
log = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import k_B
from astropy.coordinates import (
    EarthLocation,
    Angle
)
import dask.array as da
from dask.diagnostics import ProgressBar

from nenupy import LogMethodMetaClass, DummyCtMgr
from nenupy.astro.sky import Sky
from nenupy.astro.astro_tools import sky_temperature
from nenupy.astro.pointing import Pointing


# ============================================================= #
# ---------------- Interferometer class errors ---------------- #
# ============================================================= #
class CombinedMeta(LogMethodMetaClass, ABCMeta):
    """ Intermediate metaclass when inheriting from ABC. """


class AntennaNameError(Exception):
    """ Error raised when the name doesn't match existing antenna names. """

    def __init__(self,
            input_name: np.ndarray,
            available_antennas: np.ndarray
        ):
        self.input_name = input_name
        self.message = f"Valid antenna names are {available_antennas}."
        super().__init__(self.message)


    def __str__(self):
        return f"'{self.input_name}'\n{self.message}"


class AntennaIndexError(Exception):
    """ Error raised when the index doesn't match existing antenna indices. """

    def __init__(self,
            input_index: np.ndarray,
            n_available_antennas: int
        ):
        self.input_index = input_index
        self.message = f"Maximum antenna index is {n_available_antennas - 1}."
        super().__init__(self.message)


    def __str__(self):
        return f"{self.input_index}\n{self.message}"


class DuplicateAntennaError(Exception):
    """ Error raised when antennas are duplicated. """

    def __init__(self,
            input_antenna: np.ndarray,
        ):
        self.input_antenna = input_antenna
        unique, counts = np.unique(self.input_antenna, return_counts=True)
        duplicate_mask = [counts > 1]
        dup = unique[tuple(duplicate_mask)]
        cnt = counts[tuple(duplicate_mask)]

        self.message = f"antenna name/index {dup} is duplicated {cnt} times."
        super().__init__(self.message)


    def __str__(self):
        return f"{self.input_antenna}: {self.message}"
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- Baseline -------------------------- #
# ============================================================= #
class Baseline:
    """ Class to handle interferometric baseline operations. """

    def __init__(self, itrf_positions: np.ndarray):
        self.itrf_positions = itrf_positions

        if self.itrf_positions.ndim != 2:
            raise ValueError(
                "<arg 'itrf_positions'>: instance of "
                f"{np.ndarray} of dimension 2 expected. "
                f"Got {self.itrf_positions.shape} instead."
            )
        self.antenna_idx = np.arange(self.itrf_positions.shape[0])
        xyz = self.itrf_positions[..., None]
        xyz = xyz[:, :, 0][:, None]
        self.bsl = xyz.transpose(1, 0, 2) - xyz


    def __getitem__(self, n):
        pos = self.itrf_positions[n, :]
        return Baseline(
            itrf_positions=pos
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def flatten(self):
        """
        """
        return self.bsl[
            np.tril_indices(self.antenna_idx.size)
        ]


    @property
    def distance(self):
        """
        """
        return np.linalg.norm(self.bsl, axis=2) * u.m
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ObservingMode ----------------------- #
# ============================================================= #
class ObservingMode(Enum):
    """ Enumerator of the different available observing modes of NenuFAR. """

    BEAMFORMING = auto()
    IMAGING = auto()
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- Interferometer ----------------------- #
# ============================================================= #
class Interferometer(ABC, metaclass=CombinedMeta):
    """ Abstract base class for all phased-array/interferometer classes.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~nenupy.instru.interferometer.Interferometer.antenna_gains
            ~nenupy.instru.interferometer.Interferometer.antenna_names
            ~nenupy.instru.interferometer.Interferometer.antenna_positions
            ~nenupy.instru.interferometer.Interferometer.baselines
            ~nenupy.instru.interferometer.Interferometer.position
            ~nenupy.instru.interferometer.Interferometer.size

        .. rubric:: Methods Summary

        .. autosummary::

            ~nenupy.instru.interferometer.Interferometer.angular_resolution
            ~nenupy.instru.interferometer.Interferometer.array_factor
            ~nenupy.instru.interferometer.Interferometer.beam
            ~nenupy.instru.interferometer.Interferometer.confusion_noise
            ~nenupy.instru.interferometer.Interferometer.plot
            ~nenupy.instru.interferometer.Interferometer.sefd
            ~nenupy.instru.interferometer.Interferometer.sensitivity
            ~nenupy.instru.interferometer.Interferometer.system_temperature

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self,
            position: EarthLocation,
            antenna_names: np.ndarray,
            antenna_positions: np.ndarray,
            antenna_gains: np.ndarray
        ):
        self.position = position
        self.antenna_names = antenna_names
        self.antenna_positions = antenna_positions
        self.antenna_gains = antenna_gains


    def __getitem__(self, n):
        if isinstance(n, slice):
            # Convert the slice into a numpy array
            n = np.r_[n]
        elif not isinstance(n, np.ndarray):
            n = np.array(n)
            if np.isscalar(n):
                n = n.reshape((1,))

        # Checking the correct input format
        if n.ndim > 1:
            raise ValueError(
                "<class 'Interferometer'> can only be "
                f"subscriptable by 1D arrays. Got {n} instead."
            )
        if np.unique(n).size != n.size:
            raise DuplicateAntennaError(n)

        # Generating antenna mask
        if n.dtype.str.startswith('<U'):
            # Selection based on antenna names
            antenna_mask = np.isin(self.antenna_names, n)
            bad_name_index = ~np.isin(n, self.antenna_names)
            if np.any(bad_name_index):
                raise AntennaNameError(
                    n[bad_name_index],
                    self.antenna_names
                )
        else:
            # Selection based on antenna indices
            if n.max() >= self.size:
                raise AntennaIndexError(n, self.size)
            antenna_mask = np.isin(np.arange(self.size), n)

        # Constructing the new instance as a 'cutout' of its parent
        interfero = copy.deepcopy(self)
        interfero.antenna_names = self.antenna_names[antenna_mask]
        interfero.antenna_positions = self.antenna_positions[antenna_mask, :]
        interfero.antenna_gains = self.antenna_gains[antenna_mask]
        return interfero


    def __repr__(self):
        return f"{self.__class__}"


    def __str__(self):
        return f"{self.__class__.__name__}"


    def __add__(self, other):
        interferometer = copy.deepcopy(self)
        to_include = ~np.isin(other.antenna_names, self.antenna_names)
        interferometer.antenna_names = np.concatenate((self.antenna_names, other.antenna_names[to_include]))
        interferometer.antenna_positions = np.concatenate((self.antenna_positions, other.antenna_positions[to_include]))
        interferometer.antenna_gains = np.concatenate((self.antenna_gains, other.antenna_gains[to_include]))
        return interferometer
    

    def __sub__(self, other):
        interferometer = copy.deepcopy(self)
        to_keep = ~np.isin(self.antenna_names, other.antenna_names)
        interferometer.antenna_names = self.antenna_names[to_keep]
        interferometer.antenna_positions = self.antenna_positions[to_keep]
        interferometer.antenna_gains = self.antenna_gains[to_keep]
        return interferometer


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def size(self):
        """ Number of elements belonging to the array. 

            :getter: Size of the array.

            :type: `int`
        """
        return self.antenna_positions.shape[0]


    @property
    def baselines(self):
        """ Instrument baselines.

            :getter: Baselines.

            :type: :class:`~nenupy.instru.interferometer.Baseline`
        """
        try:
            antenna_position = self._toITRF()
        except AttributeError:
            log.error("No method '_toITRF()' is implemented.")
            raise
        return Baseline(itrf_positions=antenna_position)


    @property
    def antenna_names(self) -> np.ndarray:
        """ Antenna names.

            :setter: Array of antenna names.

            :getter: Array of antenna names.

            :type: :class:`~numpy.ndarray`
        """
        return self._antenna_names
    @antenna_names.setter
    def antenna_names(self, n: np.ndarray):
        self._antenna_names = n

    
    @property
    def antenna_positions(self) -> np.ndarray:
        """ Antenna positions.
            The positions should be shaped as ``(n_ant, 3)``

            :setter: Array of antenna positions.

            :getter: Array of antenna positions.

            :type: :class:`~numpy.ndarray`
        """
        return self._antenna_positions
    @antenna_positions.setter
    def antenna_positions(self, p: np.ndarray):
        self._antenna_positions = p


    @property
    def antenna_gains(self) -> np.ndarray:
        """ Antenna gains.
            This is an array of `callable` (methods or functions)
            defining the radiation pattern of each antenna.

            :setter: Array of antenna gains.

            :getter: Array of antenna gains.

            :type: :class:`~numpy.ndarray` of `callable`
        """
        return self._antenna_gains
    @antenna_gains.setter
    def antenna_gains(self, g: np.ndarray):
        self._antenna_gains = g


    @property
    def position(self) -> EarthLocation:
        """ Array's position.

            :setter: Position of the array.

            :getter: Position of the array.

            :type: :class:`~astropy.coordinates.EarthLocation`
        """
        return self._position
    @position.setter
    def position(self, p: EarthLocation):
        self._position = p


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, **kwargs) -> None:
        """ Plots the antenna distribution.

            :param figsize:
                Size of the figure. Default is ``(10, 10)``.
            :type figsize:
                `tuple`
            :param figname:
                File name of the figure to save. Default is ``''``,
                i.e. show the figure without saving it.
            :type figname:
                `str`
            :param xlim:
                X-axis limits. Default is auto-scaling.
            :type xlim:
                `tuple`
            :param ylim:
                Y-axis limits. Default is auto-scaling.
            :type ylim:
                `tuple`
        """
        fig, ax = plt.subplots(
            figsize=kwargs.get('figsize', (10, 10))
        )
        ax.scatter(
            self.antenna_positions[:, 0],
            self.antenna_positions[:, 1],
            kwargs.get('s', 30)
        )
        ax.set_xlim(kwargs.get('xlim', ax.get_xlim()))
        ax.set_ylim(kwargs.get('ylim', ax.get_ylim()))
        for i, antenna_name in enumerate(self.antenna_names):
            ax.annotate(
                antenna_name,
                (
                    self.antenna_positions[i, 0],
                    self.antenna_positions[i, 1]
                ),
                ha='center'
            )
        ax.set_aspect('equal', adjustable='datalim')

        if kwargs.get('figname', '') != '':
            plt.savefig(
                kwargs.get('figname'),
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
        else:
            plt.show()
        plt.close('all')


    def array_factor(self, sky: Sky, pointing: Pointing) -> da.Array:
        r""" Computes the array factor of the antenna distribution.

            .. math::
                \mathcal{F}(\nu, \phi, \theta) = \sum_{\rm ant} e^{ i \mathbf{k}(\nu, \phi, \theta) \cdot \mathbf{r}_{\rm ant}}

            where :math:`\mathbf{k} = \frac{2\pi}{\lambda} (\cos \phi \cos \theta, \sin \phi \cos \theta, \sin \theta )`
            is the wave vector for a wave propagation in a direction described by spherical coordinates,
            :math:`\lambda` is the wavelength, :math:`\phi` is the azimuth,
            :math:`\theta` is the elevation, :math:`\mathbf{r}_{\rm ant}`
            is the antenna position matrix.

            This method considers the ``sky`` as the desired output (in terms of
            time, frequency and sky positions). It evaluates the effective 
            pointing directions for every time step defined in ``sky`` regarding
            the ``pointing`` input.

            :param sky:
                Desired output contained in a :class:`~nenupy.astro.sky.Sky` instance.
                (:attr:`~nenupy.astro.sky.Sky.time`, :attr:`~nenupy.astro.sky.Sky.frequency` and
                :attr:`~nenupy.astro.sky.Sky.coordinates` are used as inputs for the computation).
            :type sky:
                :class:`~nenupy.astro.sky.Sky`
            :param pointing:
                Instance of :class:`~nenupy.astro.pointing.Pointing` that defines
                the targeted pointing directions over the time.
            :type pointing:
                :class:`~nenupy.astro.pointing.Pointing`

            :return:
                Array factor of the antenna distribution shaped as ``(time, frequency, 1, coordinates)``.
            :rtype:
                :class:`~dask.array.Array`                

            .. seealso::
                :class:`~nenupy.astro.sky.Sky` and :class:`~nenupy.astro.pointing.Pointing`
        """

        # Determine the effective pointing based on the time
        # informations contained in the sky instance.
        effective_pointing = pointing[sky.time]

        # Compute the geometric delays as the dot product of the
        # antenna position and the difference between sky and
        # pointing ground projections.
        geometric_delays = self._geometric_delays(sky, effective_pointing)

        # Use the sky frequency attribute to compute the wavelength
        # and prepare the coefficient of the exponential with
        # the correct dimensions.
        wavelength = sky.frequency.to(u.m, equivalencies=u.spectral())
        coeff = 2j * np.pi / wavelength.value
        coeff = coeff.reshape(
            (1, 1, wavelength.size, 1, 1)
        ) # (antenna, time, frequency, polar, coord)
    
        exponent = coeff * geometric_delays
        # coord_chunk = exponent.shape[-1]//cpu_count()
        # coord_chunk = 1 if coord_chunk == 0 else coord_chunk
        # exponent = da.rechunk(
        #     exponent,
        #     chunks=exponent.shape[:-1] + (coord_chunk,)
        # )

        complex_array_factor = np.sum(
            np.exp(exponent),
            axis=0
        )
        return (np.real(complex_array_factor * np.conjugate(complex_array_factor)))


    def beam(self, sky: Sky, pointing: Pointing) -> Sky:
        r""" Computes the phased-array response :math:`\mathcal{G}` over the ``sky`` for a given
            ``pointing``.

            .. math::
                \mathcal{G}(\nu, \phi, \theta) = \sum_{\rm ant} \mathcal{F}(\nu, \phi, \theta) \mathcal{G}_{\rm ant} (\nu, \phi, \theta)

            where :math:`\nu` is the frequency, :math:`\phi` is the azimuth,
            :math:`\theta` is the elevation,
            :math:`\mathcal{G}_{\rm ant}` is the individual array element radiation pattern and
            :math:`\mathcal{F}` is the array factor.

            This method considers the ``sky`` as the desired output (in terms of
            time, frequency, polarization and sky positions). It evaluates the effective 
            pointing directions for every time step defined in ``sky`` regarding
            the ``pointing`` input.

            :param sky:
                Desired output contained in a :class:`~nenupy.astro.sky.Sky` instance.
            :type sky:
                :class:`~nenupy.astro.sky.Sky`
            :param pointing:
                Instance of :class:`~nenupy.astro.pointing.Pointing` that defines
                the targeted pointing directions over the time.
            :type pointing:
                :class:`~nenupy.astro.pointing.Pointing`

            :return:
                The instance of :class:`~nenupy.astro.sky.Sky`
                given as input is returned, its attribute
                :attr:`~nenupy.astro.sky.Sky.value` is updated
                with the result of the beam computation (stored as
                an :class:`~dask.array.Array`) and shaped as 
                ``(time, frequency, polarization, coordinates)``.
            :rtype:
                :class:`~nenupy.astro.sky.Sky`

            .. seealso::
                :meth:`~nenupy.instru.interferometer.Interferometer.array_factor`

        """

        # Compute the array factor
        array_factor = self.array_factor(
            sky=sky,
            pointing=pointing
        )

        # Compute the total antenna gain, i.e. the sum of all
        # antenna gains for beamforming.
        antenna_gain = np.sum(
            gain(
                sky=sky,
                pointing=pointing
            ) for gain in self.antenna_gains
        )
        # antenna_gain = np.sum(
        #     np.fromiter(
        #         gain(
        #             sky=sky,
        #             pointing=pointing
        #         ) for gain in self.antenna_gains
        #     )
        # )

        # Rechunk the Dask Array before the computation
        # coord_chunk = array_factor.shape[-1]//cpu_count()
        # coord_chunk = 1 if coord_chunk == 0 else coord_chunk
        # array_factor = array_factor.rechunk(array_factor.shape[:-1] + (coord_chunk,))

        # Perform the Dask computation of array factor times antenna
        # gains. Update the sky instance values.
        sky.value = array_factor * antenna_gain

        return sky


    @lru_cache(maxsize=1)
    @abstractmethod
    def effective_area(self,
            frequency: u.Quantity = 50*u.MHz,
            elevation: u.Quantity = 90*u.deg
        ) -> u.Quantity:
        """ This method needs to be defined in child classes. """
        return


    @abstractmethod
    def instrument_temperature(self, frequency: u.Quantity = 50*u.MHz, **kwargs) -> u.Quantity:
        """ This method needs to be defined in child classes. """
        return


    def system_temperature(self,
            frequency: u.Quantity = 50*u.MHz,
            source_spectrum: Dict[str, Callable] = {},
            efficiency: float = 1.,
            elevation: u.Quantity = 90*u.deg,
            **kwargs
        ) -> u.Quantity:
        r""" Computes the System Noise Temperature :math:`T_{\rm sys}`.
            It is computed as follows:

            .. math::
                T_{\rm sys} = T_{\rm sky} + T_{\rm inst} + \sum_{\rm src} T_{\rm src}

            where :math:`T_{\rm sky}` is an approximation of the
            low-frequency sky temperature dominated by Galactic
            emission and :math:`T_{\rm inst}` is the instrumental
            noise temperature (which depends on the current
            instrument instance).
            :math:`T_{\rm src}` is the antenna temperature induced by a given source
            whose spectrum is defined in the ``source_spectrum`` argument computed as:

            .. math::
                T_{\rm src} = \frac{F_{\rm src} \eta A_{\rm eff}}{2 k_{\rm B}}
            
            where :math:`F_{\rm src}` is the source spectrum, :math:`\eta` is the
            ``efficiency`` of the effective area :math:`A_{\rm eff}}`.

            :param frequency: 
                Frequency for the System Temperature computation.
                Default is ``50 MHz``.
            :type frequency:
                :class:`~astropy.units.Quantity`
            :param elevation:
                Pointing elevation impacting the :meth:`~nenupy.instru.interferometer.Interferometer.effective_area`.
                Default is ``90 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param efficiency:
                Effective area reducing factor. Default is ``1.``, it cannot be greater than ``1.``.
            :type efficiency:
                `float`
            :param source_spectrum:
                By default the system temperature is evaluated using a mean Galactic temperature.
                However, if a bright source is targeted, the noise introduced can be under-estimated.
                Therefore, one can provide a `callable` object that takes as inputs a frequency array (of type :class:`~astropy.units.Quantity`) and
                returns the source flux density in Jansky (of type :class:`~astropy.units.Quantity`).
            :type source_spectrum:
                `dict` of `callable`

            :returns:
                System Temperature in Kelvins.
            :rtype:
                :class:`~astropy.units.Quantity`

            .. seealso::
                :func:`~nenupy.astro.astro_tools.sky_temperature`

        """
        t_gal = sky_temperature(frequency)
        t_inst = self.instrument_temperature(frequency, **kwargs)
        t_src = np.zeros(frequency.shape)*u.K
        for _, spectrum in source_spectrum.items():
            src_flux = spectrum(frequency)
            t_src += (src_flux*self.effective_area(frequency, elevation)*efficiency/(2*k_B)).to(u.K)
        return (t_gal + t_inst + t_src).to(u.K)


    def sefd(self,
            frequency: u.Quantity = 50*u.MHz,
            elevation: u.Quantity = 90*u.deg,
            efficiency: float = 1.,
            decoherence: float = 1.,
            source_spectrum: Dict[str, Callable] = {},
            **kwargs
        ) -> u.Quantity:
        r""" Computes the System Equivalent Flux Density (SEFD or
            system sensitivity).
            
            .. math::
                S_{\rm sys} = \xi \frac{2 k_{\rm B}}{ \eta A_{\rm eff}(\nu, \theta)} T_{\rm sys} (\nu)
            
            with :math:`T_{\rm sys}` the :meth:`~nenupy.instru.interferometer.Interferometer.system_temperature`,
            the efficiency :math:`\eta`, :math:`\nu` the frequency, :math:`\theta` the elevation,
            :math:`\xi` the decoherence factor, and :math:`k_{\rm B}` the
            Boltzmann constant.

            :param frequency:
                Frequency at which the SEFD will be computed.
                If an array is given as input, the output will be of same shape.
                Default if ``50 MHz``.
            :type frequency:
                :class:`~astropy.units.Quantity`
            :param elevation:
                Pointing elevation impacting the :meth:`~nenupy.instru.interferometer.Interferometer.effective_area`.
                Default is ``90 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param efficiency:
                Effective area reducing factor. Default is ``1.``, it cannot be greater than ``1.``.
            :type efficiency:
                `float`
            :param decoherence:
                Parameter that reflects other uncertainties (particularly the unperfect phasing system).
                Default is ``1.``.
            :type decoherence:
                `float`
            :param source_spectrum:
                By default the system temperature is evaluated using a mean Galactic temperature.
                However, if a bright source is targeted, the noise introduced can be under-estimated.
                Therefore, one can provide a `callable` object that takes as inputs a frequency array (of type :class:`~astropy.units.Quantity`) and
                returns the source flux density in Jansky (of type (as :class:`~astropy.units.Quantity`).
            :type source_spectrum:
                `dict` of `callable`

            :returns:
                SEFD in Janskys.
            :rtype:
                :class:`~astropy.units.Quantity`
        
            .. seealso::
                `LOFAR website <http://old.astron.nl/radio-observatory/astronomers/lofar-imaging-capabilities-sensitivity/sensitivity-lofar-array/sensiti>`_, :meth:`nenupy.instru.interferometer.Interferometer.system_temperature`

        """
        effective_area = self.effective_area(frequency, elevation)
        t_sys = self.system_temperature(frequency, source_spectrum, **kwargs)
        sefd = decoherence*2*k_B*t_sys/(efficiency*effective_area)
        return sefd.to(u.Jy)


    def sensitivity(self,
            frequency: u.Quantity = 50*u.MHz,
            mode: ObservingMode = ObservingMode.BEAMFORMING,
            dt: u.Quantity = 1.*u.s,
            df: u.Quantity = 195.3125*u.kHz,
            elevation: u.Quantity = 90*u.deg,
            efficiency: float = 1.,
            decoherence: float = 1.,
            source_spectrum: Dict[str, Callable] = {},
            **kwargs
        ) -> u.Quantity:
        r""" Computes the sensititivy of the array with respect to the observing configuration.
            The sensitivity computation depends on the observing mode of the instrument:

            * for the imaging mode:

                .. math::
                    \sigma_{\rm im} = \frac{S_{\rm sys}(\nu, \theta, \eta, \xi)}{
                        \sqrt{N(N-1) 2 \Delta \nu\, \Delta t}
                    }

            * for the beamforming mode:

                .. math::
                    \sigma_{\rm bf} = \frac{S_{\rm sys}(\nu, \theta, \eta, \xi)}{
                        \sqrt{2 \Delta \nu\, \Delta t}
                    }
            
            where :math:`\nu` is the frequency, :math:`\theta` is the elevation,
            :math:`\eta` is the effective area efficiency,
            :math:`\xi` is the decoherence factor,
            :math:`\Delta t` is the integration time,
            :math:`\Delta \nu` is the bandwidth, :math:`N` is the antenna number,
            and :math:`S_{\rm sys}` is the System Equivalent Flux Density (which also depends
            on the ``source_spectrum`` argument, see :meth:`~nenupy.instru.interferometer.Interferometer.sefd`).

            :param frequency:
                Frequency at which the sensitivity will be evaluated.
                If an array is given as input, the output will be of same shape.
                Default if ``50 MHz``.
            :type frequency:
                :class:`~astropy.units.Quantity`
            :param mode:
                Observing mode, either ``ObservingMode.BEAMFORMING`` or ``ObservingMode.IMAGING``, default is the former.
            :type mode:
                :class:`~nenupy.instru.interferometer.ObservingMode`
            :param dt:
                Integration time. Default is ``1 sec``.
            :type dt:
                :class:`~astropy.units.Quantity`
            :param df:
                Observing bandwidth. Default is ``195.3125 kHz``.
            :type df:
                :class:`~astropy.units.Quantity`
            :param elevation:
                Pointing elevation impacting the :meth:`~nenupy.instru.interferometer.Interferometer.effective_area`.
                Default is ``90 deg``.
            :type elevation:
                :class:`~astropy.units.Quantity`
            :param efficiency:
                Effective area reducing factor. Default is ``1.``, it cannot be greater than ``1.``.
            :type efficiency:
                `float`
            :param decoherence:
                Parameter that reflects other uncertainties (particularly the unperfect phasing system).
                Default is ``1.``.
            :type decoherence:
                `float`
            :param source_spectrum:
                By default the system temperature is evaluated using a mean Galactic temperature.
                However, if a bright source is targeted, the noise introduced can be under-estimated.
                Therefore, one can provide a `callable` object that takes as inputs a frequency array (of type :class:`~astropy.units.Quantity`) and
                returns the source flux density in Jansky (of type :class:`~astropy.units.Quantity`).
            :type source_spectrum:
                `dict` of `callable`

            :return:
                Array sensitivity.
            :rtype:
                :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.instru.interferometer import ObservingMode
                >>> import astropy.units as u
                >>> <instrument>.sensitivity(
                        frequency=50*u.MHz,
                        mode=ObservingMode.IMAGING,
                        dt=1*u.s,
                        df=3*u.kHz
                    )

            .. seealso::
                :meth:`~nenupy.instru.interferometer.Interferometer.sefd` and :ref:`instrument_properties_doc`.

        """
        s_sys = self.sefd(
            frequency=frequency,
            elevation=elevation,
            decoherence=decoherence,
            efficiency=efficiency,
            source_spectrum=source_spectrum,
            **kwargs
        )
        n_ant = self.antenna_names.size
        if mode is ObservingMode.IMAGING:
            sensitivity = s_sys/np.sqrt(n_ant*(n_ant - 1)*2*dt*df)
        elif mode is ObservingMode.BEAMFORMING:
            sensitivity = s_sys/np.sqrt(2*dt*df)
        else:
            raise ValueError(
                'Invalid observation mode.'
            )
        return sensitivity.to(u.Jy)


    def angular_resolution(self,
            frequency: u.Quantity = 50*u.MHz
        ) -> u.Quantity:
        r""" Computes the angular resolution of the antenna array.

            The full width at half maximum (FWHM) :math:`\theta` is approximated as follows:

            .. math::
                \theta = \frac{\lambda}{D}
            
            where :math:`\lambda` is the wavelength and :math:`D` is
            is the length of the maximum physical separation of the antennas
            in the array.

            :param frequency:
                Frequency at which the angular resolution is evaluated.
            :type frequency:
                :class:`~astropy.units.Quantity`
            
            :return:
                Angular resolution (FWHM) of the instrument.
            :rtype:
                :class:`~astropy.units.Quantity`
            
            :Example:

                >>> import astropy.units as u
                >>> <instrument>.angular_resolution(
                        frequency=50*u.MHz
                    )

        """
        if self.size == 1:
            raise Exception(
                "Angular resolution is not defined for an interferometer of 1 element."
            )
        wavelength = frequency.to(
            u.m,
            equivalencies=u.spectral()
        )
        diameter = np.max(self.baselines.distance)
        return (wavelength / diameter * u.rad).to(u.deg)


    def confusion_noise(self,
            frequency: u.Quantity = 50*u.MHz,
            lofar: bool = True
        ) -> u.Quantity:
        r""" Confusion rms noise :math:`\sigma_{\rm c}` (parameter
            used for specifying the width of the confusion
            distribution) computed as:

            .. math::
                \left( \frac{\sigma_{\rm c}}{\rm{mJy}\, \rm{beam}^{-1}} \right) \simeq
                0.2 \left( \frac{\nu}{\rm GHz} \right)^{-0.7} 
                \left( \frac{\theta}{\rm arcmin} \right)^{2}
            
            or (if ``lofar=True``):
            
            .. math::
                \left( \frac{\sigma_{\rm c}}{\mu\rm{Jy}\, \rm{beam}^{-1}} \right) \simeq
                30 \left( \frac{\nu}{74 {\rm MHz}} \right)^{-0.7} 
                \left( \frac{\theta}{\rm arcsec} \right)^{1.54}
            
            where :math:`\nu` is the frequency and :math:`\theta` is
            the radiotelescope FWHM (see :meth:`~nenupy.instru.interferometer.Interferometer.angular_resolution`).
            
            Individual sources fainter than about 
            :math:`5\sigma_{\rm c}` cannot be detected reliably.

            :param freq:
                Frequency at which computing the confusion noise.
                In MHz if no unit is provided. Default is ``50 MHz``.
            :type freq: `float` or :class:`~astropy.units.Quantity`
            :param miniarrays:
                Mini-Array indices to take into account.
                Default is ``None`` (all available MAs).
            :type miniarrays: `int`, `list` or :class:`~numpy.ndarray`
            :param lofar:
                If set to ``True`` (recommended), the confusion noise
                is estimated using Eq. 6 of `van Haarlem et al. (2013) <https://arxiv.org/pdf/1305.3550.pdf>`_.
            :type:
                `bool`

            :returns:
                Confusion rms noise in Jy/beam
            :rtype:
                :class:`~astropy.units.Quantity`

            :Example:
                >>> import astropy.units as u
                >>> <instrument>.confusion_noise(
                        frequency=50*u.MHz
                    )

            .. see also::
                `NRAO lecture <https://www.cv.nrao.edu/course/astr534/Radiometers.html>`_ (eq. 3E6),
                `Takeuchi and Ishii, 2004 <https://ui.adsabs.harvard.edu/abs/2004ApJ...604...40T/abstract>`_.
        """
        resolution = self.angular_resolution(frequency=frequency)
        if lofar: # https://arxiv.org/pdf/1305.3550.pdf
            freq_at_74mHz = (frequency.to(u.MHz)/(74*u.MHz)).value
            res_in_asec = resolution.to(u.arcsec).value
            conf = 30 * res_in_asec**1.54 * freq_at_74mHz**(-0.7) * u.uJy
        else:
            # freq_in_ghz = frequency.to(u.GHz).value
            # res_in_asec = resolution.to(u.arcsec).value
            # conf = 1.2 * (freq_in_ghz/3.02)**(-0.7) * (res_in_asec/8)**(10/3) * u.uJy # condon 2012
            freq_in_ghz = frequency.to(u.GHz).value
            res_in_amin = resolution.to(u.arcmin).value
            conf = 0.2 * freq_in_ghz**(-0.7) * res_in_amin**2 * u.mJy # Condon 2002

        return conf.to(u.Jy) # Jy/beam


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _geometric_delays(self, sky: Sky, pointing: Pointing) -> da.Array:
        """ Computes the geometric delays between the sky and the pointing direction(s).
        """
        sky_projection = sky.ground_projection
        pointing_projection = pointing.ground_projection

        # coord_chunk = sky_projection.shape[-1]//cpu_count()
        # coord_chunk = 1 if coord_chunk == 0 else coord_chunk
        # chunk_shape = sky_projection.shape[:-1] + (coord_chunk,)
        chunk_shape = (1, 1, 1, 3,) + (sky_projection.shape[-1],)

        sky_projection = da.from_array(
            sky_projection,
            chunks=chunk_shape
        )
        pointing_projection = da.from_array(
            pointing_projection,
            chunks=chunk_shape
        )

        # Put the ground antennas in the sky
        angle = np.pi/2
        rot_90 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0,           0,           1]
            ]
        )

        return np.dot(
            np.dot(self.antenna_positions, rot_90),
            sky_projection - pointing_projection
        )
# ============================================================= #
# ============================================================= #
