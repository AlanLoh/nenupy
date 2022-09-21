#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    BST file
    ********

    .. inheritance-diagram:: nenupy.io.bst.BST
        :parts: 3

    .. autosummary::

        ~BST

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "BST"
]


from astropy.time import Time
import astropy.units as u
import numpy as np

from nenupy.io.io_tools import StatisticsData, ST_Slice

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ---------------------------- BST ---------------------------- #
# ============================================================= #
class BST(StatisticsData):
    """ Beamlet STatistics reading class.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~BST.analog_beams
            ~BST.digital_beams
            ~BST.analog_beam
            ~BST.beam
            ~BST.analog_pointing
            ~BST.digital_pointing
            ~BST.frequencies
            ~BST.mini_arrays

        .. rubric:: Methods Summary

        .. autosummary::

            ~BST.get

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self, file_name, beam=0):
        super().__init__(file_name=file_name)
        self.beam = beam


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def analog_beams(self):
        """ Lists the analog beam indices used for the recording of BST data.

            :getter: List of analog beam indices.

            :type: :class:`~numpy.ndarray`
        """
        return np.arange(self._meta_data["ana"].size)


    @property
    def digital_beams(self):
        """ Lists the digital beam indices used for the recording of BST data.

            :getter: List of digital beam indices.

            :type: :class:`~numpy.ndarray`
        """
        return np.arange(self._meta_data["bea"].size)


    @property
    def analog_beam(self):
        """ Prints the analog beam index currently corresponding to the digital
            :attr:`~nenupy.io.bst.BST.beam` index.

            :getter: Analog beam index.

            :type: `int`
        """
        log.debug(
            f"Retrieving analog beam index associated with beam #{self.beam}."
        )
        return self._meta_data["bea"]["NoAnaBeam"][self.beam]


    @property
    def analog_pointing(self):
        """ Retrieves the analog pointing associated to the current :attr:`~nenupy.io.bst.BST.analog_beam` selected.

            :getter: Analog pointing (time, azimuth, elevation).

            :type:
                `tuple`(:class:`~astropy.time.Time`, :class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`)
        """
        analog_beam = self.analog_beam
        log.debug(
            f"Retrieving analog pointing associated with analog beam #{analog_beam}."
        )
        analog_mask = self._meta_data["pan"]["noAnaBeam"] == analog_beam
        pointing = self._meta_data["pan"][analog_mask]
        return Time(pointing["timestamp"]), pointing["az"]*u.deg, pointing["el"]*u.deg


    @property
    def digital_pointing(self):
        """ Retrieves the digital pointing associated to the current :attr:`~nenupy.io.bst.BST.beam` selected.

            :getter: Digital pointing (time, azimuth, elevation).

            :type:
                `tuple`(:class:`~astropy.time.Time`, :class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`)
        """
        log.debug(
            f"Retrieving digital pointing associated with beam #{self.beam}."
        )
        digital_mask = self._meta_data["pbe"]["noBeam"] == self.beam
        pointing = self._meta_data["pbe"][digital_mask]
        return Time(pointing["timestamp"]), pointing["az"]*u.deg, pointing["el"]*u.deg


    @property
    def frequencies(self) -> u.Quantity:
        """ Retrieves the sub-band middle frequency of all the sub-bands recorded for the selected :attr:`~nenupy.io.bst.BST.beam`.

            :getter: Sub-band mid frequencies.

            :type: :class:`~astropy.units.Quantity`
        """
        log.debug(
            f"Retrieving frequencies associated with beam #{self.beam}."
        )
        beamlets = self._meta_data["bea"]["nbBeamlet"][self.beam]
        subband_half_width = 195.3125*u.kHz
        freqs = self._meta_data["bea"]["freqList"][self.beam][:beamlets]*u.MHz
        return freqs - subband_half_width/2


    @property
    def mini_arrays(self) -> np.ndarray:
        """ Retrieves the list of Mini-Arrays used to record BST data for the selected :attr:`~nenupy.io.bst.BST.analog_beam`.

            :getter: Mini-Arrays list.

            :type: :class:`~numpy.ndarray`
        """
        analog_beam = self.analog_beam
        log.debug(
            f"Retrieving Mini-Arrays associated with analog beam #{analog_beam}."
        )
        analog_config = self._meta_data["ana"][analog_beam]
        nb_mini_arrays = analog_config["nbMRUsed"]
        return analog_config["MRList"][:nb_mini_arrays]


    @property
    def beam(self) -> int:
        """ Digital beam index.

            :setter: Beam index.
            
            :getter: Beam index.
            
            :type: `int`
        """
        return self._beam
    @beam.setter
    def beam(self, b: int):
        if b not in self.digital_beams:
            log.error(
                f"Selected beam #{b} should be one of {self.digital_beams}."
            )
            raise IndexError()
        self._beam = b


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self,
            frequency_selection: str = None,
            time_selection: str = None,
            polarization: str = "NW",
            beam: int = 0
        ) -> ST_Slice:
        """ Sub-selects BST data.
            ``frequency_selection`` and ``time_selection``
            arguments accept `str` values formatted as, e.g.,
            ``'>={value}'`` or ``'>={value_1} & <{value_2}'`` or ``'=={value}'``.

            :param frequency_selection:
                Frequency selection. The expected ``'{value}'`` format is frequency units, e.g. ``'>=50MHz'`` or ``'< 1 GHz'``.
                Default is ``None`` (i.e., no selection upon frequency).
            :type frequency_selection:
                `str`
            :param time_selection:
                Time selection. The expected ``'{value}'`` format is ISOT, e.g. ``'>=2022-01-01T12:00:00'``.
                Default is ``None`` (i.e., no selection upon time).
            :type time_selection:
                `str`
            :param polarization:
                Polarization selection, must be either ``'NW'`` or ``'NE'``.
                Default is ``'NW'``.
            :type polarization:
                `str`
            :param beam:
                Digital beam index selection.
                Default is ``0``.
            :type beam:
                `int`
            
            :returns:
                BST data subset.
            :rtype:
                :class:`~nenupy.io.io_tools.ST_Slice`
            
            :Example:
                .. code-block:: python

                    from nenupy.io.bst import BST

                    bst = BST("/path/to/BST.fits")
                    data = bst.get(
                        frequency_selection="<=52MHz",
                        time_selection='>=2022-01-24T11:08:10 & <2022-01-24T11:14:08',
                        polarization="NW",
                        beam=8
                    )

        """
        self.beam = beam

        # Frequency selection
        frequencies = self.frequencies
        if frequency_selection is None:
            frequency_selection = f">={frequencies.min()} & <= {frequencies.max()}"
        frequency_mask = self._parse_frequency_condition(frequency_selection)(frequencies)
        n_beamlets = self._meta_data["bea"]["nbBeamlet"][self.beam]
        beamlets = self._meta_data["bea"]['BeamletList'][self.beam][:n_beamlets]
        freq_idx = beamlets[frequency_mask]

        # Time selection
        if time_selection is None:
            time_selection = f">={self.time[0].isot} & <= {self.time[-1].isot}"
        time_mask = self._parse_time_condition(time_selection)(self.time)

        # Polarization selection
        polars = self._meta_data['ins']['spol'][0]
        if polarization not in polars:
            log.warning(
                f"`polarization` - unknown '{polarization}', setting default value ('NW')."
            )
            polarization = "NW"
        polar_idx = np.where(polars == polarization)[0]

        log.info(
            "BST selection applied\n"
            f"\t- time ({np.sum(time_mask)},): '{time_selection}'\n"
            f"\t- frequency ({np.sum(frequency_mask)},): '{frequency_selection}'\n"
            f"\t- polarization (1,): '{polarization}'\n"
            f"\t- beam (1,): {self.beam}"
        )

        return ST_Slice(
            time=self.time[time_mask],
            frequency=frequencies[frequency_mask],
            value=np.squeeze(self.data[
                np.ix_(
                    time_mask,
                    polar_idx,
                    freq_idx
                )
            ]),
            analog_pointing_times=self.analog_pointing[0],
            digital_pointing_times=self.digital_pointing[0]
        )
# ============================================================= #
# ============================================================= #

