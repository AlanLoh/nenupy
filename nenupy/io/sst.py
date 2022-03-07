#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    SST file
    ********

    .. inheritance-diagram:: nenupy.io.sst.SST
        :parts: 3

    .. autosummary::

        ~SST

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "SST"
]


import numpy as np
import astropy.units as u

from nenupy.io.io_tools import StatisticsData, ST_Slice

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ---------------------------- BST ---------------------------- #
# ============================================================= #
class SST(StatisticsData):
    """ Spectral STatistics reading class.

        .. rubric:: Attributes Summary

        .. autosummary::

            ~SST.frequencies
            ~SST.mini_arrays

        .. rubric:: Methods Summary

        .. autosummary::

            ~SST.get

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(self, file_name):
        super().__init__(file_name=file_name)
        
    
    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def frequencies(self) -> u.Quantity:
        """ Retrieves the sub-band middle frequency of all the
            sub-bands recorded.

            :getter: Sub-band mid frequencies.

            :type: :class:`~astropy.units.Quantity`
        """
        return self._meta_data["ins"]["frq"][0, :]*u.MHz


    @property
    def mini_arrays(self) -> np.ndarray:
        """ Retrieves the list of Mini-Arrays whose data have been
            recorded.

            :getter: Mini-Arrays list.

            :type: :class:`~numpy.ndarray`
        """
        return self._meta_data["ins"]["noMROn"][0, :]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self,
            mini_array: int = 0,
            frequency_selection: str = None,
            time_selection: str = None,
            polarization: str = "NW"
        ) -> ST_Slice:
        """ Sub-selects SST data.
            ``frequency_selection`` and ``time_selection``
            arguments accept `str` values formatted as, e.g.,
            ``'>={value}'`` or ``'>={value_1} & <{value_2}'`` or ``'=={value}'``.

            :param mini_array:
                Mini-Array index.
                Default is ``0``.
            :type mini_array:
                `int`
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
            
            :returns:
                SST data subset.
            :rtype:
                :class:`~nenupy.io.io_tools.ST_Slice`
            
            :Example:
                .. code-block:: python

                    from nenupy.io.sst import SST

                    sst = SST("/path/to/SST.fits")
                    data = sst.get(
                        mini_array=10,
                        frequency_selection="<=52MHz",
                        time_selection='>=2022-01-24T11:08:10 & <2022-01-24T11:14:08',
                        polarization="NW"
                    )

        """

        # Frequency selection
        frequencies = self.frequencies
        if frequency_selection is None:
            frequency_selection = f">={frequencies.min()} & <= {frequencies.max()}"
        frequency_mask = self._parse_frequency_condition(frequency_selection)(frequencies)

        # Time selection
        if time_selection is None:
            time_selection = f">={self.time[0].isot} & <= {self.time[-1].isot}"
        time_mask = self._parse_time_condition(time_selection)(self.time)

        # Mini-Array selection
        ma_mask = self.mini_arrays == mini_array
        if np.all(~ma_mask):
            raise ValueError(
                f"Unable to locate Mini-Array {mini_array} in current data."
            )
        elif np.sum(ma_mask) != 1:
            raise IndexError(
                f"Selection upon multiple Mini-Arrays is not permitted."
            )

        # Polarization selection
        polars = self._meta_data["ins"]["spol"][0]
        if polarization not in polars:
            log.warning(
                f"`polarization` - unknown '{polarization}', setting default value ('NW')."
            )
            polarization = "NW"
        polar_idx = np.where(polars == polarization)[0]

        log.info(
            "SST selection applied\n"
            f"\t- mini-array (1,): '{mini_array}'\n"
            f"\t- time ({np.sum(time_mask)},): '{time_selection}'\n"
            f"\t- frequency ({np.sum(frequency_mask)},): '{frequency_selection}'\n"
            f"\t- polarization (1,): '{polarization}'"
        )

        return ST_Slice(
            time=self.time[time_mask],
            frequency=frequencies[frequency_mask],
            value=np.squeeze(self.data[
                np.ix_(
                    time_mask,
                    ma_mask,
                    polar_idx,
                    frequency_mask
                )
            ])
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #
# ============================================================= #

