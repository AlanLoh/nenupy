#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *********************
    NenuFAR Array Classes
    *********************

    .. inheritance-diagram:: nenupy.instru.nenufar.MiniArray nenupy.instru.nenufar.NenuFAR
        :parts: 3

    .. autosummary::

        ~MiniArray
        ~NenuFAR

"""


__author__ = "Alan Loh"
__copyright__ = "Copyright 2021, nenupy"
__credits__ = ["Alan Loh"]
__maintainer__ = "Alan"
__email__ = "alan.loh@obspm.fr"
__status__ = "Production"
__all__ = [
    "NenuFAR_Configuration",
    "Polarization",
    "MiniArray",
    "NenuFAR"
]

from functools import lru_cache
import logging
log = logging.getLogger(__name__)

import numpy as np

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from pyproj import Transformer

from nenupy import nenufar_position
from nenupy.instru import (
    nenufar_miniarrays,
    miniarray_antennas,
    squint_table,
    instrument_temperature
)
from nenupy.instru.interferometer import Interferometer
from nenupy.astro.astro_tools import radec_to_altaz
from nenupy.astro.sky import Sky
from nenupy.astro.pointing import Pointing


# ============================================================= #
# ---------------- Polarization / Antenna Gain ---------------- #
# ============================================================= #
import healpy as hp
from scipy.interpolate import interp2d
import os
from enum import Enum


class _AntennaGain:
    """ NenuFAR antenna gain class. """


    def __init__(self, polarization='NW'):
        self.polarization = polarization

        if self.polarization == 'NW':
            fields = np.arange(8)
        elif self.polarization == 'NE':
            fields = 8 + np.arange(8)

        # Read the gain
        gain = hp.read_map(
            filename=os.path.join(os.path.dirname(__file__), './NenuFAR_Ant_Hpx.fits'),
            hdu=1,
            field=fields,
            memmap=True,
            dtype=float
        )

        # Interpolate the antenna gain on the frequency axis
        freqs = np.arange(10, 90, 10)
        self.healpix_coords = np.arange(hp.pixelfunc.nside2npix(64))
        self.interpolated_gain = interp2d(
            x=self.healpix_coords,
            y=freqs,
            z=gain,
            kind='linear'
        )


    @lru_cache(maxsize=1)
    def __getitem__(self, sky: Sky) -> np.ndarray:
        """ Return an antenna gain array shaped like (sky.time, sky.frequency, sky.coord)
        """

        horizontal_coordinates = sky.horizontal_coordinates

        log.debug(
            f"Interpolating NenuFAR antenna response ('{self.polarization}' polarization) "
            f"on the given sky (time: {sky.time.size}, freq: {sky.frequency.size}, coord: {horizontal_coordinates.size})."
        )

        # Get the frequency from the Sky instance
        freqs = sky.frequency.to(u.MHz).value

        # Find the interpolated gain at the desired frequency
        gain = self.interpolated_gain(self.healpix_coords, freqs)
        if gain.ndim == 1:
            gain = gain.reshape((1, gain.size))

        # Find the interpolated gain at the desired coordinates for each frequency
        gain = np.array([
            hp.pixelfunc.get_interp_val(
                m=gain_i,
                theta=horizontal_coordinates.az.deg,
                phi=horizontal_coordinates.alt.deg,
                nest=False,
                lonlat=True
            ) for gain_i in gain
        ])

        # Return something shaped as (time, freq, coord)
        return np.moveaxis(gain, 0, 1)


class Polarization(Enum):
    """ Enumerator of the different available polarizations of NenuFAR. """

    NW = _AntennaGain('NW')
    NE = _AntennaGain('NE')


class NenuFAR_Configuration:
    """ """

    def __init__(self,
            beamsquint_correction: bool = True,
            beamsquint_frequency: u.Quantity = 50*u.MHz
        ):
        self.beamsquint_correction = beamsquint_correction
        self.beamsquint_frequency = beamsquint_frequency
    

    @property
    def beamsquint_frequency(self):
        """ """
        return self._beamsquint_frequency
    @beamsquint_frequency.setter
    def beamsquint_frequency(self, freq):
        if not isinstance(freq, u.Quantity):
            raise TypeError(
                "'beamsquint_frequency' should be of type 'astropy.units.Quantity'."
            )
        if not freq.isscalar:
            raise ValueError(
                "'beamsquint_frequency' should be scalar."
            )
        self._beamsquint_frequency = freq
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- MiniArray ------------------------- #
# ============================================================= #
class MiniArray(Interferometer):
    """ Main class to handle a NenuFAR Mini-Array antenna distribution.

        .. versionadded:: 2.0.0

        :param index:
            Mini-Array index. 'Core' Mini-Arrays have indices ranging
            from ``0`` to ``95``. 'Remote' Mini-Arrays have indices
            ranging from ``100`` to ``105``.
        :type index:
            `int`

        :Example:
            Instantiating :class:`~nenupy.instru.nenufar.MiniArray`:

            >>> from nenupy.instru import MiniArray
            >>> ma = MiniArray(index=0)

            Sub-arraying on an existing :class:`~nenupy.instru.nenufar.MiniArray` instance:
            
            >>> sub_ma = ma["Ant01", "Ant06", "Ant11"]
            >>> sub_ma.antenna_names
            array(['Ant01', 'Ant06', 'Ant11'], dtype='<U5')

            Using `slice` object (converted in :class:`~numpy.ndarray` using `~numpy.r_`):

            >>> import numpy as np
            >>> sub_ma = ma[np.r_[2:10]]
            >>> sub_ma.size
            8

            Combining two :class:`~nenupy.instru.nenufar.MiniArray` instances:
            
            >>> ma1 = MiniArray(index=0)["Ant01", "Ant06"]
            >>> ma2 = MiniArray(index=0)["Ant08", "Ant12"]
            >>> combined_ma = ma1 + ma2
            >>> combined_ma.antenna_names
            array(['Ant01', 'Ant06', 'Ant08', 'Ant12'], dtype='<U5')

        .. seealso::
            More details on this class usage can be found in 
            :ref:`array_configuration_doc` and :ref:`instrument_properties_doc`.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~MiniArray.index
            ~MiniArray.rotation
            ~nenupy.instru.interferometer.Interferometer.position
            ~nenupy.instru.interferometer.Interferometer.antenna_names
            ~nenupy.instru.interferometer.Interferometer.antenna_positions
            ~nenupy.instru.interferometer.Interferometer.antenna_gains
            ~nenupy.instru.interferometer.Interferometer.baselines
            ~nenupy.instru.interferometer.Interferometer.size


        .. rubric:: Methods Summary

        .. autosummary::

            ~MiniArray.beam
            ~MiniArray.effective_area
            ~MiniArray.instrument_temperature
            ~MiniArray.attenuation_from_zenith
            ~MiniArray.order_to_skycoord
            ~MiniArray.skycoord_to_order
            ~MiniArray.analog_pointing
            ~nenupy.instru.interferometer.Interferometer.plot
            ~nenupy.instru.interferometer.Interferometer.array_factor
            ~nenupy.instru.interferometer.Interferometer.system_temperature
            ~nenupy.instru.interferometer.Interferometer.sefd
            ~nenupy.instru.interferometer.Interferometer.sensitivity
            ~nenupy.instru.interferometer.Interferometer.angular_resolution
            ~nenupy.instru.interferometer.Interferometer.confusion_noise

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self, index: int = 0):
        self.index = index

        ma_name = f'MA{self.index:03d}'

        position = EarthLocation(
            lat=nenufar_miniarrays[ma_name]['lat'] * u.deg,
            lon=nenufar_miniarrays[ma_name]['lon'] * u.deg,
            height=nenufar_miniarrays[ma_name]['height'] * u.m
        )
        antenna_names = np.array([ant for ant in miniarray_antennas.keys()])
        antPos = np.array([ant['position'] for ant in miniarray_antennas.values()])
        self.rotation = nenufar_miniarrays[ma_name]['rotation'] * u.deg
        #rotation = np.radians(self.rotation.value - 90)
        rotation = np.radians(self.rotation.value)
        rotMatrix = np.array(
            [
                [np.cos(rotation), -np.sin(rotation), 0],
                [-np.sin(rotation), -np.cos(rotation), 0],
                [0,           0,           1]
            ]
        )
        antenna_positions = np.dot(antPos, rotMatrix).astype(np.float32)
        antenna_gains = np.array([
            self._antenna_gain for _ in range(antenna_names.size)
        ])

        super().__init__(
            position=position,
            antenna_names=antenna_names,
            antenna_positions=antenna_positions,
            antenna_gains=antenna_gains
        )


    def __repr__(self):
        return f"{self.__class__}(index={self.index})"


    def __str__(self):
        return f"{self.__class__.__name__}(index={self.index}, antennas={self.antenna_names})"


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def index(self) -> int:
        """ Mini-Array index.
            'Core' Mini-Arrays have indices ranging
            from ``0`` to ``95``. 'Remote' Mini-Arrays have indices
            ranging from ``100`` to ``105``.

            :setter: Mini-Array index.
            
            :getter: Mini-Array index.
            
            :type: `int`
        """
        return self._index
    @index.setter
    def index(self, i: int):
        self._index = i

    
    @property
    def rotation(self) -> u.Quantity:
        """ Mini-Array rotation.
            Each NenuFAR Mini-Array has its own rotation with
            respect to the others by angles multiple of 10 deg.
        
            :setter: Mini-Array rotation.
            
            :getter: Mini-Array rotation.
            
            :type: :class:`~astropy.units.Quantity`
        
        """
        return self._rotation
    @rotation.setter
    def rotation(self, r: u.Quantity):
        self._rotation = r


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def beam(self,
            sky: Sky,
            pointing: Pointing,
            configuration: NenuFAR_Configuration = NenuFAR_Configuration()
        ) -> Sky:
        r""" Computes the Mini-Array beam over the ``sky`` for a given
            ``pointing``.

            .. math::
                \mathcal{G}_{\rm MA}(\nu, \phi, \theta) = \mathcal{F}_{\rm MA}(\nu, \phi, \theta) \mathcal{G}_{\rm ant} (\nu, \phi, \theta)

            where :math:`\nu` is the frequency, :math:`\phi` is the azimuth,
            :math:`\theta` is the elevation,
            :math:`\mathcal{G}_{\rm ant}` is the NenuFAR dipole antenna radiation pattern and
            :math:`\mathcal{F}_{\rm MA}` is the array factor.

            This method considers the ``sky`` as the desired output (in terms of
            time, frequency, polarization and sky positions). It evaluates the effective 
            pointing directions for every time step defined in ``sky`` regarding
            the ``pointing`` input.

            :param sky:
                Desired output contained in a :class:`~nenupy.astro.sky.Sky` instance.
                (:attr:`~nenupy.astro.sky.Sky.time`, :attr:`~nenupy.astro.sky.Sky.frequency`,
                :attr:`~nenupy.astro.sky.Sky.polarization` and
                :attr:`~nenupy.astro.sky.Sky.coordinates` are used as inputs for the computation).
            :type sky:
                :class:`~nenupy.astro.sky.Sky`
            :param pointing:
                Instance of :class:`~nenupy.astro.pointing.Pointing` that defines
                the targeted pointing directions over the time.
            :type pointing:
                :class:`~nenupy.astro.pointing.Pointing`
            :param configuration:
                NenuFAR configuration to consider during the beam simulation.
                The beamsquint correction and its frequency setting are defined here.
                Default is ``NenuFAR_Configuration(beamsquint_correction=True, beamsquint_frequency=50MHz)``.
            :type configuration:
                :class:`~nenupy.instru.nenufar.NenuFAR_Configuration`

            :returns:
                The instance of :class:`~nenupy.astro.sky.Sky`
                given as input is returned, its attribute
                :attr:`~nenupy.astro.sky.Sky.value` is updated
                with the result of the beam computation (stored as
                an :class:`~dask.array.Array`) and shaped as 
                ``(time, frequency, polarization, coordinates)``.
            :rtype:
                :class:`~nenupy.astro.sky.Sky`

            :Example:

                Load the required librairies:

                >>> from nenupy.instru import MiniArray, Polarization
                >>> from nenupy.astro.sky import HpxSky
                >>> from nenupy.astro.pointing import Pointing
                >>> import astropy.units as u
                >>> from astropy.time import Time, TimeDelta
                
                Define a desired :class:`~nenupy.astro.sky.Sky` output:

                >>> sky = HpxSky(
                        resolution=1.*u.deg,
                        frequency=np.array([25, 50, 75])*u.MHz,
                        polarization=np.array([Polarization.NW, Polarization.NE]),
                        time=Time("2021-10-15 20:00:00")
                    )
                
                Define the pointing of the Mini-Array:

                >>> ma_pointing = Pointing.zenith_tracking(
                        time=Time("2021-10-15 00:00:00"),
                        duration=TimeDelta(3600*24, format="sec")
                    )
                
                Select the Mini-Array (and possibly its antenna distribution) and compute its response pattern:

                >>> ma = MiniArray(1)
                >>> beam = ma.beam(
                        sky=sky,
                        pointing=ma_pointing
                    )
                
                Calling :meth:`print` on a :class:`~nenupy.astro.sky.Sky` object
                enables the display of its :attr:`~nenupy.astro.sky.Sky.value` attribute structure
                (which matches the definition of the ``sky`` instance):

                >>> print(beam)
                <class 'nenupy.astro.sky.HpxSky'> instance
                value: (1, 3, 2, 49152)
                    * time: (1,)
                    * frequency: (3,)
                    * polarization: (2,)
                    * coordinates: (49152,)
                
                To :meth:`~nenupy.astro.sky.SkySliceBase.plot` the computed Mini-Array response at 75 MHz, in NE polarization:

                >>> beam[0, 2, 1].plot(
                        decibel=True,
                        colorbar_label=''
                    )

                .. image:: ./_images/instru_images/ma1_beam.png
                    :width: 800

            .. seealso::
                :meth:`~nenupy.instru.interferometer.Interferometer.array_factor` and :ref:`beam_simulation_doc`

        """
        log.info(
            f"Computing <class 'MiniArray'> beam ({self.size} "
            f"antennas, {sky.time.size} time and "
            f"{sky.frequency.size} frequency slots)."
        )

        # Computing the Mini-Array effective area.
        # aeff = self.effective_area(sky.frequency).to(u.m**2).value

        # The beam is computed thanks to the Interferometer super method.
        # The returned value is only divided by Aeff.
        return super().beam(
            sky=sky,
            pointing=self.analog_pointing(pointing, configuration=configuration)
        )# / aeff[None, :, None, None]


    def effective_area(self,
            frequency: u.Quantity = 50*u.MHz,
            elevation: u.Quantity = 90*u.deg
        ) -> u.Quantity:
        r""" Computes the effective area of a NenuFAR Mini-Array. 
            The effective area of a Mini-Array (:math:`\mathcal{A}_{\rm eff,\ MA}`) is
            computed as the sum of dipole effective areas (:math:`\mathcal{A}_{\rm eff, ant}`),
            while taking into account overlaps.
            This is a function of ``frequency`` (:math:`\nu`) and ``elevation``
            (:math:`\theta`):

            .. math::
                \mathcal{A}_{\rm eff,\ MA} (\nu) = \sum_{\rm ant} \mathcal{A}_{\rm eff, ant} (\nu) \sin( \theta )
            
            with

            .. math::
                \mathcal{A}_{\rm eff, ant} (\nu) = \frac{\lambda^2}{3}

            the NenuFAR dipole antenna effective area.

            :param frequency:
                Frequency at which the effective area is computed.
                Default is 50 MHz.
            :type frequency:
                :class:`~astropy.units.Quantity`
            :param elevation:
                Elevation at which the effective area is computed.
                Default is 90 deg, i.e., as seen from the zenith.
            :type elevation:
                :class:`~astropy.units.Quantity`

            :returns:
                Effective area of a Mini-Array shaped as ``frequency``.
            :rtype:
                :class:`~astropy.units.Quantity`

            :Example:

                >>> from nenupy.instru import MiniArray
                >>> import astropy.units as u
                >>> ma = MiniArray()
                >>> ma.effective_area(50*u.MHz)
                227.68377 m2

                >>> ma = MiniArray()
                >>> ma.effective_area(frequency=50*u.MHz, elevation=45*u.deg)
                160.99673 m2

                >>> ma = MiniArray()["Ant01"]
                >>> ma.effective_area(50*u.MHz)
                11.979179 m2

                >>> ma = MiniArray()
                >>> ma.effective_area(u.Quantity([20, 30, 40], unit='MHz'))
                [693.44216, 532.97815, 355.85306] m2

            .. seealso::
                :ref:`effective_area_sec`

        """

        log.debug(
            f"Mini-Array effective area, using {self.size} Antennas."
        )

        # Antenna Effective Area, formula for a dipole antenna.
        k = 3
        wavelength = frequency.to(
            u.m,
            equivalencies=u.spectral()
        )
        antenna_effective_area = wavelength**2 / k
        radius_ant_eff_area = np.sqrt(antenna_effective_area/np.pi)
        max_radius = np.max(radius_ant_eff_area)

        n = 500 # grid resolution
        ant_pos = self.antenna_positions * u.m
        x_grid = np.linspace(
            ant_pos[:, 0].min() - max_radius,
            ant_pos[:, 0].max() + max_radius,
            n
        )
        dx = x_grid[1] - x_grid[0]
        y_grid = np.linspace(
            ant_pos[:, 1].min() - max_radius,
            ant_pos[:, 1].max() + max_radius,
            n
        )
        dy = y_grid[1] - y_grid[0]
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

        dist = np.linalg.norm(
            ant_pos[:, :2][..., None, None] -\
                np.array([xx_grid, yy_grid]) * u.m,
            axis=1
        )

        return np.sum(
            np.any(
                (dist <= radius_ant_eff_area) if radius_ant_eff_area.isscalar else\
                (dist[..., None] <= radius_ant_eff_area),
                axis=0
            ),
            axis=(0, 1)
        ) * dx * dy * np.sin(elevation.to(u.rad).value)


    def attenuation_from_zenith(self,
            coordinates,
            time: Time = Time.now(),
            frequency: u.Quantity = 50*u.MHz,
            polarization: Polarization = Polarization.NW
        ):
        """ Returns the attenuation factor evaluated at given ``coordinates``
            compared to the zenithal Mini-Array beam gain.

            :param coordinates:
                Sky positions equatorial coordinates.
            :type coordinates:
                :class:`~astropy.coordinates.SkyCoord`
            :param time:
                UTC time at which the attenuation is evaluated. Default is ``now``.
            :type time:
                :class:`~astropy.time.Time`
            :param frequency:
                Frequency at which the attenuation is evaluated. Default is ``50 MHz``.
            :type frquency:
                :class:`~astropy.units.Quantity`
            :param polarization:
                NenuFAR antenna polarization. Default is ``Polarization.NW``.
            :type polarization:
                :class:`~nenupy.instru.nenufar.Polarization`
    
            :returns:
                Attenuation factor shaped as ``(time, frequency, polarization, coordinates)``.
                ``NaN`` is returned for any ``coordinates`` that is below the horizon.
            :rtype:
                :class:`~numpy.ndarray`

            :Example:
                >>> from nenupy.instru.nenufar import MiniArray
                >>> from astropy.coordinates import SkyCoord
                >>> ma = MiniArray(index=0)
                >>> attenuation = ma.attenuation_from_zenith(
                        coordinates=SkyCoord.from_name("Cyg A")
                    )
                
                >>> from nenupy.instru.nenufar import MiniArray
                >>> from astropy.coordinates import SkyCoord
                >>> import astropy.units as u
                >>> ma = MiniArray(index=0)
                >>> attenuation = ma.attenuation_from_zenith(
                        coordinates=SkyCoord.from_name("Cyg A"),
                        frequency=np.linspace(20, 80, 10)*u.MHz
                    )

            .. versionadded:: 2.0.0

        """
        # Define the pointing towards the zenith
        pointing = Pointing.zenith_tracking(
            time=time.reshape((1,)),
            duration=TimeDelta(10, format="sec")
        )

        # Compute the local zenith in equatorial coordinates
        local_zenith = SkyCoord(180, 90,
            unit="deg",
            frame=AltAz(
                obstime=time,
                location=nenufar_position
            )
        ).transform_to(coordinates.frame)

        # Find the coordinates below the horizon and compute a mask
        input_coord_altaz = radec_to_altaz(
            radec=coordinates,
            time=time
        )
        invisible_mask = input_coord_altaz.alt.deg <= 0

        # Concatenate local_zenith and coordinates
        if coordinates.obstime is None:
            coordinates.obstime = local_zenith.obstime
        if  coordinates.location is None:
            coordinates.location = local_zenith.location
        if coordinates.isscalar:
            coordinates = coordinates.reshape((1,))
        coordinates = coordinates.insert(0, local_zenith)

        # Prepare a Sky instance for the beam simulation
        sky = Sky(
            coordinates=coordinates,
            frequency=frequency,
            time=time,
            polarization=polarization
        )

        # Compute the beam
        beam = self.beam(sky=sky, pointing=pointing)

        # Compute the attenuation factor relative to the zenith (first member)
        values = beam.value.compute()
        output_values = values[..., 1:]/np.expand_dims(values[..., 0], 3)
        output_values[..., invisible_mask] = np.nan

        return output_values


    @staticmethod
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

            :Example:

                >>> from nenupy.instru import MiniArray
                >>> import astropy.units as u
                >>> ma = MiniArray()
                >>> ma.instrument_temperature(frequency=70*u.MHz)
                526.11213 K

            .. seealso::
                :func:`~nenupy.astro.astro_tools.sky_temperature`

        """
        return instrument_temperature(frequency=frequency, lna_filter=lna_filter)


    def _order_to_skycoord(self, order: tuple) -> SkyCoord:
        """ """
        pointing_grid = self._generate_analog_directions()
        return pointing_grid[order]


    def _skycoord_to_order(self, coordinates: SkyCoord) -> tuple:
        """ """
        if coordinates.size != 1:
            raise ValueError(
                "Only size 1 `coordinates` are accepted."
            )
        pointing_grid = self._generate_analog_directions()
        separations = coordinates.separation(pointing_grid)
        order = np.array(
            np.unravel_index(
                np.argmin(separations, axis=None),
                separations.shape
            )
        )
        order[order >= 64] -= 1
        return tuple(order)


    def beamsquint_correction(self, coords: SkyCoord, frequency: u.Quantity = 50*u.MHz) -> SkyCoord:
        """ Corrects for the beamsquint effect.

            :Example:
                >>> from astropy.coordinates import SkyCoord, AltAz
                >>> from astropy.time import Time
                >>> import astropy.units as u
                >>> from nenupy import nenufar_position
                >>> from nenupy.instru import MiniArray
                >>> position = SkyCoord(
                        0*u.deg,
                        30*u.deg,
                        frame=AltAz(
                            obstime=Time("2021-01-01 12:00:00"),
                            location=nenufar_position
                        )
                    )
                >>> ma = MiniArray()
                >>> corrected_position = ma.beamsquint_correction(
                        coords=position,
                        frequency=50*u.MHz
                    )
                >>> corrected_position.az.deg, corrected_position.alt.deg
                (0., 22.91422672)

        """
        freq_idx = np.argmin(
            np.abs(squint_table['freq'] - frequency.to(u.MHz).value)
        )
        azimuths = coords.az
        elevations = coords.alt
        elevations = np.interp(elevations.deg, squint_table['elev_desiree'][freq_idx, :], squint_table['elev_a_pointer'])
        # Squint is limited at 20 deg elevation, otherwise the
        # pointing can vary drasticaly as the available pointing
        # positions become sparse at low elevation.
        elevations[elevations < 20] = 20

        return SkyCoord(
            azimuths,
            elevations * u.deg,
            frame=coords.frame
        )


    def analog_pointing(self, pointing: Pointing, configuration: NenuFAR_Configuration) -> Pointing:
        """ Converts the desired pointing to the effective pointing
            which depends on the available pointing positions defined
            on a grid due to analog cable delays.
        """
        # Put the horizontal coordinates in a good shape
        pointing_ho_coords = pointing.horizontal_coordinates
        if pointing_ho_coords.isscalar:
            pointing_ho_coords = pointing_ho_coords.reshape((1,))

        # Correct the pointing for beamsquint effect, that is, point at a
        # lower elevation than the one desired
        if configuration.beamsquint_correction:
            pointing_ho_coords = self.beamsquint_correction(
                coords=pointing_ho_coords,
                frequency=configuration.beamsquint_frequency
            )

        coord = SkyCoord(
            pointing_ho_coords.az,
            pointing_ho_coords.alt
        )
        orders = list(map(self._skycoord_to_order, coord))
        altaz_list = list(map(self._order_to_skycoord, orders))

        azimuths = [position.ra.deg for position in altaz_list] * u.deg
        elevations = [position.dec.deg for position in altaz_list] * u.deg

        pointing.custom_ho_coordinates = SkyCoord(
            azimuths.reshape(pointing_ho_coords.shape),
            elevations.reshape(pointing_ho_coords.shape),
            frame=pointing_ho_coords.frame
        )
        return pointing


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    #def _antenna_gain(self, sky: Sky, pointing: Pointing):
    #    return 1.
    @lru_cache(maxsize=1)
    def _antenna_gain(self, sky: Sky, pointing: Pointing):
        """
        """
        import dask.array as da
        gain = da.ones(
            (
                sky.time.size,
                sky.frequency.size,
                sky.polarization.size,
                sky.coordinates.size
            ),
            dtype=np.float64
        )
        for i, pol in enumerate(sky.polarization):
            if not isinstance(pol, Polarization):
                log.warning(
                    f"Invalid value encountered in <attr 'Sky.polarization'>: '{pol}'. "
                    f"Polarization has been set to '{Polarization.NW}' by default."
                )
                pol = Polarization.NW
            t0 = Time.now()
            gain[:, :, i, :] = pol.value[sky]
        return gain


    def _toITRF(self):
        """
        """
        return self.antenna_positions


    def _generate_analog_directions(self) -> SkyCoord:
        """ """
        from astropy.coordinates import Latitude, Longitude, SkyCoord
        DX = 2*5.5
        DY = DX*np.cos(np.pi/6)
        DMINX = 0.165
        DMINY = DMINX*np.cos(np.pi/6)
        DMIN_D = DMINX/DX
        NBITS = 7
        BITS = 2**(NBITS - 1)

        bits = np.arange(2*BITS)
        xx, yy = np.meshgrid(bits, bits)

        xx_mask = xx >= 64
        yy_mask = yy >= 64

        k1 = (xx - BITS + 1)*DMIN_D
        k2 = (BITS - 1 - yy)*DMIN_D
        k1[xx_mask] = (xx[xx_mask] - BITS)*DMIN_D
        k2[yy_mask] = (BITS - yy[yy_mask])*DMIN_D

        # theta =  0.5*np.arccos(1 - 2*(k1**2 + k2**2))
        with np.errstate(invalid='ignore'):
            theta = np.pi/2 - ( 0.5*np.arccos(1 - 2*(k1**2 + k2**2)) )
        bad_values = np.isnan(theta)
        # phi = np.arctan2(k2, k1) + np.pi
        phi = np.pi/2 - (np.arctan2(k2, k1) + np.pi) + self.rotation.to("rad").value

        theta[bad_values] = -np.pi/2
        phi[bad_values] = 0.
        return SkyCoord(
            Longitude(phi, unit="rad"),
            Latitude(theta, unit="rad"),
        ).T
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------------- NenuFAR -------------------------- #
# ============================================================= #
class NenuFAR(Interferometer):
    """ Main class to handle a NenuFAR array.

        .. versionadded:: 2.0.0

        :param miniarray_antennas:
            Mini-Arrays antennas selection. 
            Default is ``numpy.r_[:19]``, i.e., the full 19 dipole antennas.
            See :class:`~nenupy.instru.nenufar.MiniArray` for different input values.
        :type miniarray_antennas:
            `numpy.ndarray` or `slice`
        :param include_remote_mas:
            Include or not the remote Mini-Arrays.
            Default is ``False``, i.e., only the dense 'core' of 96 Mini-Arrays is considered.
        :type include_remote_mas:
            `bool`

        :Example:
            Instantiating :class:`~nenupy.instru.nenufar.NenuFAR`:

            >>> from nenupy.instru import NenuFAR
            >>> nenufar = NenuFAR()

            Sub-arraying on an existing :class:`~nenupy.instru.nenufar.NenuFAR` instance:
            
            >>> sub_nenufar = NenuFAR()["MA001", "MA002", "MA104"]
            >>> sub_nenufar.antenna_names
            array(['MA001', 'MA002'], dtype='<U5')

            If :attr:`~nenupy.instru.nenufar.NenuFAR.include_remote_mas` is ``True``,
            the remote Mini-Arrays are included in the array and selecting ``MA104``
            as above would take this remote Mini-Array into account:

            >>> sub_nenufar = NenuFAR(include_remote_mas=True)["MA001", "MA002", "MA104"]
            >>> sub_nenufar.antenna_names
            array(['MA001', 'MA002', 'MA104'], dtype='<U5')

            Combining two :class:`~nenupy.instru.nenufar.NenuFAR` instances:
            
            >>> nenufar1 = NenuFAR()["MA001", "MA006"]
            >>> nenufar2 = NenuFAR()["MA010", "MA056"]
            >>> resulting_array = nenufar1 + nenufar2
            >>> resulting_array.antenna_names
            array(['MA001', 'MA006', 'MA010', 'MA056'], dtype='<U5')

            .. note::
                The result of the addition operation, namely ``resulting_array`` in this example
                will conserve the properties of the first member, namely ``nenufar1``.
                This is particularly true for the attributes :attr:`~nenupy.instru.nenufar.NenuFAR.include_remote_mas`
                and :attr:`~nenupy.instru.nenufar.NenuFAR.miniarray_antennas`.

        .. seealso::
            More details on this class usage can be found in 
            :ref:`array_configuration_doc` and :ref:`instrument_properties_doc`.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~NenuFAR.miniarray_antennas
            ~NenuFAR.include_remote_mas
            ~nenuFAR.miniarray_rotations
            ~nenupy.instru.interferometer.Interferometer.position
            ~nenupy.instru.interferometer.Interferometer.antenna_names
            ~nenupy.instru.interferometer.Interferometer.antenna_positions
            ~nenupy.instru.interferometer.Interferometer.antenna_gains
            ~nenupy.instru.interferometer.Interferometer.baselines
            ~nenupy.instru.interferometer.Interferometer.size


        .. rubric:: Methods Summary

        .. autosummary::

            ~NenuFAR.beam
            ~NenuFAR.effective_area
            ~NenuFAR.attenuation_from_zenith
            ~NenuFAR.instrument_temperature
            ~nenupy.instru.interferometer.Interferometer.plot
            ~nenupy.instru.interferometer.Interferometer.array_factor
            ~nenupy.instru.interferometer.Interferometer.system_temperature
            ~nenupy.instru.interferometer.Interferometer.sefd
            ~nenupy.instru.interferometer.Interferometer.sensitivity
            ~nenupy.instru.interferometer.Interferometer.angular_resolution
            ~nenupy.instru.interferometer.Interferometer.confusion_noise

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self, miniarray_antennas: np.ndarray = np.r_[:19], include_remote_mas: bool = False):
        self.miniarray_antennas = miniarray_antennas
        self.include_remote_mas = include_remote_mas

        antenna_names = np.array([ma for ma in nenufar_miniarrays.keys()])
        antenna_positions = np.array(
            [ma['position'] for ma in nenufar_miniarrays.values()],
            dtype=np.float32
        )
        antenna_gains = np.array([
            MiniArray(
                index=ma['id']
            )[self.miniarray_antennas].beam for ma in nenufar_miniarrays.values()
        ])

        if not self.include_remote_mas:
            # Exclude the distant Mini-Arrays from the element list
            mask_distant = ~np.array([name.startswith('MA1') for name in antenna_names])
            antenna_names = antenna_names[mask_distant]
            antenna_positions = antenna_positions[mask_distant, :]
            antenna_gains = antenna_gains[mask_distant]

        super().__init__(
            position=nenufar_position,
            antenna_names=antenna_names,
            antenna_positions=antenna_positions,
            antenna_gains=antenna_gains
        )


    def __repr__(self):
        """
        """
        return f"{self.__class__}(nMAS={self.size})"


    def __str__(self):
        """
        """
        return f"{self.__class__.__name__}"


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def miniarray_rotations(self):
        """
        """
        return np.array([
            nenufar_miniarrays[ma]['rotation'] for ma in self.antenna_names
        ])


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def beam(self,
            sky: Sky,
            pointing: Pointing,
            analog_pointing: Pointing = None,
            configuration: NenuFAR_Configuration = NenuFAR_Configuration()
        ) -> Sky:
        r""" Computes the NenuFAR beam over the ``sky`` for a given
            ``pointing``.

            .. math::
                \mathcal{G}_{\rm NenuFAR}(\nu, \phi, \theta) = \mathcal{F}_{\rm NenuFAR} (\nu, \phi, \theta) \sum_{\rm MA} \mathcal{G}_{\rm MA}(\nu, \phi, \theta)

            where :math:`\nu` is the frequency, :math:`\phi` is the azimuth,
            :math:`\theta` is the elevation,
            :math:`\mathcal{G}_{\rm MA}` is the MiniArray response (see :meth:`~nenupy.instru.nenufar.MiniArray.beam`)
            and :math:`\mathcal{F}_{\rm NenuFAR}` is the array factor.

            This method considers the ``sky`` as the desired output (in terms of
            time, frequency, polarization and sky positions). It evaluates the effective 
            pointing directions for every time step defined in ``sky`` regarding
            the ``pointing`` input.

            :param sky:
                Desired output contained in a :class:`~nenupy.astro.sky.Sky` instance.
                (:attr:`~nenupy.astro.sky.Sky.time`, :attr:`~nenupy.astro.sky.Sky.frequency`,
                :attr:`~nenupy.astro.sky.Sky.polarization` and
                :attr:`~nenupy.astro.sky.Sky.coordinates` are used as inputs for the computation).
            :type sky:
                :class:`~nenupy.astro.sky.Sky`
            :param pointing:
                Instance of :class:`~nenupy.astro.pointing.Pointing` that defines
                the targeted **numerical** pointing directions over the time.
            :type pointing:
                :class:`~nenupy.astro.pointing.Pointing`
            :param analog_pointing:
                Instance of :class:`~nenupy.astro.pointing.Pointing` that defines
                the **analog** pointing directions over the time.
                This pointing is subject to beamsquint corrections.
            :type analog_pointing:
                :class:`~nenupy.astro.pointing.Pointing`
            :param configuration:
                NenuFAR configuration to consider during the beam simulation.
                The beamsquint correction and its frequency setting are defined here.
                Default is ``NenuFAR_Configuration(beamsquint_correction=True, beamsquint_frequency=50MHz)``.
            :type configuration:
                :class:`~nenupy.instru.nenufar.NenuFAR_Configuration`

            :returns:
                The instance of :class:`~nenupy.astro.sky.Sky`
                given as input is returned, its attribute
                :attr:`~nenupy.astro.sky.Sky.value` is updated
                with the result of the beam computation (stored as
                an :class:`~dask.array.Array`) and shaped as 
                ``(time, frequency, polarization, coordinates)``.
            :rtype:
                :class:`~nenupy.astro.sky.Sky`

            .. seealso::
                :meth:`~nenupy.instru.interferometer.Interferometer.array_factor` and :ref:`beam_simulation_doc`

        """
        log.info(
            f"Computing <class 'NenuFAR'> beam ({self.size} "
            f"Mini-Arrays, {sky.time.size} time and "
            f"{sky.frequency.size} frequency slots)."
        )

        # Sorting out the analog pointing, make it equal to the
        # numerical pointing if it is not specifically defined.
        if not analog_pointing:
            analog_pointing = pointing
            log.info(
                "Analog pointing is set according to the numerical pointing."
            )

        # Computing the Array Factor of the whole NenuFAR array.
        array_factor = self.array_factor(
            sky=sky,
            pointing=pointing
        )

        # Finding the unique Mini-Array rotations and the number
        # of MAs corresponding to each rotation.
        rots, indices, counts = np.unique(
            self.miniarray_rotations%60,
            return_counts=True,
            return_index=True
        )

        # Summing all different (due to rotation) Mini-Array beam
        # patterns, although only executing it at most 6 times
        # because there could only be 6 different rotations.
        # Even though antGain updates the same sky instance, the
        # value attr * count creates new memeory allocations.
        antenna_gain = np.sum(
            gain(
                sky=sky,
                pointing=analog_pointing,
                configuration=configuration
            ).value*count for gain, count in zip(self.antenna_gains[indices], counts)
        )

        # Updating the sky object value array where the the sky
        # is above the horizon as the product of the NenuFAR array
        # factor and the combined Mini-Array gain patterns.
        sky.value = array_factor * antenna_gain

        return sky


    def effective_area(self,
            frequency: u.Quantity = 50*u.MHz,
            elevation: u.Quantity = 90*u.deg
        ) -> u.Quantity:
        r""" Computes the effective area of NenuFAR. 
            The effective area of NenuFAR (:math:`\mathcal{A}_{\rm eff,\ NenuFAR}`) 
            is computed as :math:`n_{\rm Mini-Arrays}` times the effective area
            of one Mini-Array (:math:`\mathcal{A}_{\rm eff,\ MA}`) as a 
            function of the ``frequency`` :math:`\nu`, where
            :math:`n_{\rm Mini-Arrays}` is the number of Mini-Arrays included. 
            This method also takes into account the active antennas within
            each Mini-Array (such as defined by 
            :attr:`~nenupy.instru.nenufar.NenuFAR.miniarray_antennas`).

            .. math::
                \mathcal{A}_{\rm eff,\ NenuFAR} (\nu) = n_{\rm Mini-Arrays} \mathcal{A}_{\rm eff,\ MA} (\nu)

            :param frequency:
                Frequency at which the effective area is computed.
                Default is 50 MHz.
            :type frequency:
                :class:`~astropy.units.Quantity`
            :param elevation:
                Elevation at which the effective area is computed.
                Default is 90 deg, i.e., as seen from the zenith.
            :type elevation:
                :class:`~astropy.units.Quantity`

            :returns:
                Effective area of NenuFAR shaped as ``frequency``.
            :rtype:
                :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.instru import NenuFAR
                >>> import astropy.units as u
                >>> nenufar = NenuFAR()
                >>> nenufar.effective_area(50*u.MHz)
                18214.701 m2

                >>> from nenupy.instru import NenuFAR
                >>> import astropy.units as u
                >>> nenufar = NenuFAR()
                >>> nenufar.effective_area(u.Quantity([20, 30, 40], unit='MHz'))
                [55475.372, 42638.252, 28468.245] m2

            .. seealso::
                :meth:`~nenupy.instru.nenufar.MiniArray.effective_area` 
                for the computation of :math:`\mathcal{A}_{\rm eff,\ MA}`
                and :ref:`effective_area_sec`.

        """
        log.debug(
            f"NenuFAR effective area, using {self.size} Mini-Arrays "
            f"of {self.miniarray_antennas.size} antennas each."
        )

        # Compute the Mini-Array effective area. Select the active
        # antennas in case not all of them are used. By default the
        # MA 0 is used but it's the same for every MA.
        miniarray_effective_area = MiniArray()[self.miniarray_antennas].effective_area(
            frequency=frequency,
            elevation=elevation
        )

        # The NenuFAR array effective area is then only the Mini-Array
        # effective area times the number of MAs since there is no
        # overlay between individual MA Aeff.
        return miniarray_effective_area * self.size


    @staticmethod
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

            :Example:

                >>> from nenupy.instru import MiniArray
                >>> import astropy.units as u
                >>> ma = MiniArray()
                >>> ma.instrument_temperature(frequency=70*u.MHz)
                526.11213 K

            .. seealso::
                :func:`~nenupy.astro.astro_tools.sky_temperature`

        """
        return instrument_temperature(frequency=frequency, lna_filter=lna_filter)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _toITRF(self):
        """
        """
        t = Transformer.from_crs(
            crs_from='EPSG:2154', # RGF93
            crs_to='EPSG:4896'# ITRF2005
        )
        antPos = self.antenna_positions.copy()
        antPos[:, 0], antPos[:, 1], antPos[:, 2] = t.transform(
            xx=antPos[:, 0],
            yy=antPos[:, 1],
            zz=antPos[:, 2]
        )
        return antPos
# ============================================================= #
# ============================================================= #