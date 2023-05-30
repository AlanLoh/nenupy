#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    .. _schedule_obsblocks:

    **************
    Booking Blocks
    **************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Block',
    'ObsBlock',
    'ReservedBlock'
]


import numpy as np
import functools
from copy import deepcopy
from astropy.time import Time, TimeDelta

from nenupy.schedule.targets import _Target
from nenupy.schedule.constraints import Constraints

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ============================================================= #
KPS = {
    'es00': {
        'name': 'Comissioning',
        'color': '#7DCEA0'
    },
    'es01': {
        'name': 'Cosmic Dawn',
        'color': '#825A2C'
    },
    'es02': {
        'name': 'Exoplanets and stars',
        'color': '#87794E'
    },
    'es03': {
        'name': 'Pulsars',
        'color': '#3F729C'
    },
    'es04': {
        'name': 'Transients',
        'color': '#8800CC'
    },
    'es05': {
        'name': 'Fast Radio Bursts',
        'color': '#00ABA9'
    },
    'es06': {
        'name': 'Planetary Lightning',
        'color': '#D80073'
    },
    'es07': {
        'name': 'Jupiter',
        'color': '#B8860B'
    },
    'es08': {
        'name': 'Galaxy clusters and AGNs',
        'color': '#0050EF'
    },
    'es09': {
        'name': 'Cluster filament and cosmic magnetism',
        'color': '#CC66FF'
    },
    'es10': {
        'name': 'Recombination Lines',
        'color': '#FA6800'
    },
    'es11': {
        'name': 'Sun',
        'color': '#008A00'
    },
    'es12': {
        'name': 'Radio Gamma',
        'color': '#A20025'
    },
    'es13': {
        'name': 'SETI',
        'color': '#F4A460'
    },
    'es14': {
        'name': 'Cas A',
        'color': '#517A7A'
    },
    'es15': {
        'name': 'Large Survey',
        'color': '#6699FF'
    },
    'es16': {
        'name': 'LOFAR-NenuFAR',
        'color': '#708D23'
    },
    'es17': {
        'name': 'Radio-Amateurs',
        'color': '#424242'
    },
    'lt00': {
        'name': 'Comissioning',
        'color': '#7DCEA0'
    },
    'lt01': {
        'name': 'Cosmic Dawn',
        'color': '#825A2C'
    },
    'lt02': {
        'name': 'Exoplanets and stars',
        'color': '#87794E'
    },
    'lt03': {
        'name': 'Pulsars',
        'color': '#3F729C'
    },
    'lt04': {
        'name': 'Transients',
        'color': '#8800CC'
    },
    'lt05': {
        'name': 'Fast Radio Bursts',
        'color': '#00ABA9'
    },
    'lt06': {
        'name': 'Planetary Lightning',
        'color': '#D80073'
    },
    'lt07': {
        'name': 'Jupiter joint studies',
        'color': '#B8860B'
    },
    'lt09': {
        'name': 'Galaxies',
        'color': '#CC66FF'
    },
    'lt10': {
        'name': 'Recombination Lines',
        'color': '#FA6800'
    },
    'lt11': {
        'name': 'Sun',
        'color': '#008A00'
    },
    'lt12': {
        'name': 'Radio Gamma',
        'color': '#A20025'
    },
    'lt13': {
        'name': 'SETI',
        'color': '#F4A460'
    },
    'rp1a': {
        'name' : 'Faraday tomography of 3C196 field',
        'color': '#000000' # TO BE DEFINED
    },
    'rp1b': {
        'name': 'NenuFAR Low-Frequency Sky Survey',
        'color': '#6699FF'
    },
    'rp1c': {
        'name': 'Free-free absorption in Cassiopeia A',
        'color': '#517A7A'
    },
    'sp16': {
        'name': 'Student training',
        'color': '#708D23'
    },
    'sp17': {
        'name': 'Radio-Amateurs',
        'color': '#424242'
    }
}


STATUS = {
    'good': '#8CC152',
    'medium': '#ECB23C',
    'bad': '#E9573F'
}
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------------- Block --------------------------- #
# ============================================================= #
class Block(object):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, *blocks):
        self._blocks = blocks
        self._assign_indices()


    def __len__(self):
        """
        """
        return len(self._blocks)


    def __mul__(self, n):
        """ Duplicates the :class:`~nenupy.schedule.obsblock.Block`
            ``n`` times.
        """
        blocks = []
        for block in self._blocks:
            for i in range(n):
                copied_block = deepcopy(block)
                try:
                    # Do not get a copy of the target attribute
                    # if it exists. Therefore any costly computation
                    # on target coordinates will not happen n times.
                    copied_block.target = block.target
                    copied_block.constraints = block.constraints
                except AttributeError:
                    # Attribute target is not found for ReservedBlock
                    pass
                blocks.append(copied_block)
        return Block(*blocks)


    def __add__(self, other):
        """
        """
        if not isinstance(other, Block):
            raise TypeError(
                'Addition may be made from two `Blocks` '
                'instances. Instead one instance is of type '
                f'`{type(other)}`.'
            )
        blocks = self._blocks + other._blocks
        return Block(*blocks)


    def __radd__(self, other):
        if other == 0:
            return self
        else:
            print(other)
            return self.__add__(other)


    def __getitem__(self, n):
        """
        """
        return self._blocks[n]


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def size(self):
        """
        """
        return len(self)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self, **kwargs):
        """
            Example:
                bb = aa.get(program='es00')
        """
        if len(kwargs) != 1:
            raise ValueError(
                'Only one key=value is allowed.'
            )
        (attr, value), = kwargs.items()
        blocks = (
            blk for blk in self if getattr(blk, attr)==value
        )
        return Block(*blocks)


    def reset(self):
        """ 
        """
        for block in self._blocks:
            block.constraints = None


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _assign_indices(self):
        """
        """
        for i, block in enumerate(self._blocks):
            block.blockIdx = i
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- ObsBlock -------------------------- #
# ============================================================= #
class ObsBlock(Block):
    """ Class to handle observation blocks.

        :param name:
            The name of the observation, for further reference.
        :type name:
            `str`
        :param program:
            The NenuFAR scientific program to which this observation belongs.
        :type program:
            `str`
        :param target:
            The celestial target.
        :type target:
            :class:`~nenupy.schedule.targets.ESTarget` or :class:`~nenupy.schedule.targets.SSTarget`
        :param constraints:
            The observing constraints to apply.
        :type constraints:
            :class:`~nenupy.schedule.constraints.Constraints`
        :param duration:
            The requested duration of the observation.
        :type duration:
            :class:`~astropy.time.TimeDelta`
        :param processing_delay:
            Time delay needed after the observation for the processing to take place.
            Only :class:`~nenupy.schedule.obsblocks.ObsBlock`s with this parameter set are compared with each other while computing the scheduling.
            This parameter particularly suits the imaging data.
        :type processing_delay:
            :class:`~astropy.time.TimeDelta`

        .. seealso::
            :ref:`observation_request_sec`
        
        .. rubric:: Attributes Summary
        
        .. autosummary::

            ~ObsBlock.program
            ~ObsBlock.target
            ~ObsBlock.constraints

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(
        self, name, program, target,
        constraints=None,
        duration=TimeDelta(3600, format='sec'),
        processing_delay: TimeDelta = None
    ):
        self.name = name
        self.program = program
        self.target = target
        self.constraints = constraints
        self.duration = duration
        self.processing_delay = processing_delay

        self.blockIdx = 0
        
        super().__init__(self)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def program(self):
        """
        """
        return self._program
    @program.setter
    def program(self, pg):
        pg = pg.lower()
        self._isKP(pg)
        self._program = pg


    @property
    def target(self):
        """
        """
        return self._target
    @target.setter
    def target(self, src):
        if src is None:
            pass
        elif not isinstance(src, _Target):
            raise TypeError(
                f'`target` should be of type {_Target}.'
            )
        elif src._lst is not None:
            # Clear the target from previous computations
            src.reset()
        self._target = src


    @property
    def constraints(self):
        """
        """
        return self._constraints
    @constraints.setter
    def constraints(self, ct):
        if ct is None:
            ct = Constraints()
        self._constraints = ct



    @property
    def kpColor(self):
        """
        """
        return KPS[self.program]['color']


    @property
    def title(self):
        """
        """
        _char_limit = 23
        block_id = f'ID: {self.blockIdx}'
        kp_infos = ' - '.join([
            self.program.upper(),
            KPS[self.program]['name']
        ])
        kp_infos = kp_infos[:_char_limit]
        obs_name = self.name[:_char_limit]
        return f'{block_id}\n{kp_infos}\n{obs_name}'


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    # @classmethod
    # def fromJson(self, jsonFile):
    #     """
    #     """

    # def evaluateScore(self, time):
    #     """
    #     """
    #     # Evaluate the target positions over time and compute the
    #     # score (product of each contraint score)
    #     # If it has already been evaluated do not do twice
    #     if self.target._lst is None:
    #         self.target.computePosition(time)

    #         # Compute the product of the score for each constraint
    #         # self.constraints.computeWeight(
    #         #     target=self.target
    #         # )
    #         self.constraints.evaluate(
    #             target=self.target,
    #             time=time,
    #             nslots=self.nSlots
    #         )

    #         log.debug(
    #             f"<ObsBlock> named '{self.name}': Constraint "
    #             "score evaluated."
    #         )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _isKP(kp):
        """
        """
        if kp not in KPS.keys():
            raise KeyError(
                '`program` is not a valid NenuFAR Key Science '
                f'Program, i.e. one of {KPS.keys()}.'
            )
        return True
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- ObsBlock -------------------------- #
# ============================================================= #
class ObsBlock2(Block):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(
        self, name, program, target,
        constraints=None,
        duration=TimeDelta(3600, format='sec')
    ):
        self.name = name
        self.program = program
        self.target = target
        self.constraints = constraints
        self.duration = duration

        self.blockIdx = 0
        self.isBooked = False
        
        # These atrributes are filled once the ObsBlock has been
        # evaluated over a time range
        self._idx = None
        self.time_min = None
        self.time_max = None
        self.nSlots = 0
        self.startIdx = None
        
        super().__init__(self)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def program(self):
        """
        """
        return self._program
    @program.setter
    def program(self, pg):
        pg = pg.lower()
        self._isKP(pg)
        self._program = pg


    @property
    def target(self):
        """
        """
        return self._target
    @target.setter
    def target(self, src):
        if not isinstance(src, _Target):
            raise TypeError(
                f'`target` should be of type {_Target}.'
            )
        self._target = src


    @property
    def constraints(self):
        """
        """
        return self._constraints
    @constraints.setter
    def constraints(self, ct):
        # if ct is None:
        #     ct = ElevationConstraint(0.)
        # elif not isinstance(ct, Constraint):
        #     raise TypeError(
        #         "Argument `constraints` should be provided "
        #         f"with a '{Constraint.__class__}'' instance."
        #     )
        # else:
        #     hasEl = [isinstance(ci, ElevationConstraint) for ci in ct]
        #     if not any(hasEl):
        #         ct += ElevationConstraint(0.)
        if ct is None:
            ct = Constraints()
        self._constraints = ct


    @property
    def startIdx(self):
        """
        """
        return self._startIdx
    @startIdx.setter
    def startIdx(self, st):
        if st is not None:
            idxRange = np.arange(self.nSlots)
            if np.isscalar(st):
                self._idx = idxRange + st
            else:
                # Useful for genetic algorithm
                self._idx = st[:, None] + idxRange[None, :]
        self._startIdx = st


    @property
    def kpColor(self):
        """
        """
        return KPS[self.program]['color']


    @property
    def statusColor(self):
        """
        """
        if 0 <= self.score < 0.5:
            return STATUS['bad']
        elif 0.5 <= self.score < 0.8:
            return STATUS['medium']
        elif 0.8 <= self.score <=1.:
            return STATUS['good']
        else:
            log.warning('Strange...')
            return STATUS['bad']


    @property
    def title(self):
        """
        """
        _charLimit = 23
        blkId = f'ID: {self.blockIdx}'
        kpInfos = ' - '.join(
            [
                self.program.upper(),
                KPS[self.program]['name']
            ]
        )
        kpInfos = kpInfos[:_charLimit]
        obsName = self.name[:_charLimit]
        return f'{blkId}\n{kpInfos}\n{obsName}'


    @property
    def score(self):
        """
        """
        if self._idx is None:
            return 0.
        else:
            scores = []
            for constraint in self.constraints:
                scores.append(constraint.getScore(self._idx))
            return np.mean(scores, axis=0)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    # @classmethod
    # def fromJson(self, jsonFile):
    #     """
    #     """



    def evaluateScore(self, time, **kwargs):
        """
        """
        # Evaluate the target positions over time and compute the
        # score (product of each contraint score)
        # If it has already been evaluated do not do twice
        if self.target._lst is None:
            self.target.computePosition(time)

            # Compute the product of the score for each constraint
            # self.constraints.computeWeight(
            #     target=self.target
            # )
            self.constraints.evaluate(
                target=self.target,
                time=time,
                method=kwargs.get('method', 'prod')
            )

            log.debug(
                f"<ObsBlock> #{self.blockIdx} named '{self.name}': "
                "Constraint score evaluated."
            )


    def plot(self, **kwargs):
        """
            kwargs
                figsize
                nPoints
                figName
        """
        import matplotlib.pyplot as plt

        # Check if the obsblock has been booked in the schedule
        if self.time_min is None:
            log.warning(
                f"<ObsBlock> #{self.blockIdx} named '{self.name}' "
                "is not booked."
            )
            return
        
        # Compute the target position
        nPoints = kwargs.get('nPoints', 50)
        dt = (self.time_max - self.time_min)/nPoints
        times = self.time_min + np.arange(nPoints + 1)*dt
        # Create a copy of the target object to keep self.target intact
        target = deepcopy(self.target)
        target.computePosition(times)

        # Plot the figure
        fig, ax1 = plt.subplots(
            figsize=kwargs.get('figsize', (10, 5))
        )
        
        color1 = 'tab:blue'
        ax1.plot(
            times.datetime,
            target.elevation,
            color=color1,
            label='Elevation'
        )
        ax1.axvline(
            self.time_min.datetime,
            color='black',
            linestyle='-.'
        )
        ax1.axvline(
            self.time_max.datetime,
            color='black',
            linestyle='-.'
        )
        ax1.set_title(f'{self.title}')
        ax1.set_xlabel('Time (UTC)')
        ax1.set_ylabel('Elevation (deg)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Azimuth (deg)', color=color2)
        ax2.plot(
            times.datetime,
            target.azimuth,
            color=color2,
            label='Azimuth'
        )
        ax2.tick_params(axis='y', labelcolor=color2)
        
        fig.tight_layout()

        # Save or show the figure
        figName = kwargs.get('figName', '')
        if figName != '':
            plt.savefig(
                figName,
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            log.info(f"Figure '{figName}' saved.")
        else:
            plt.show()
        plt.close('all')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _isKP(kp):
        """
        """
        if kp not in KPS.keys():
            raise KeyError(
                '`program` is not a valid NenuFAR Key Science '
                f'Program, i.e. one of {KPS.keys()}.'
            )
        return True


    def _display(self, ax):
        """
        """
        import matplotlib.dates as mdates

        if self.time_min is None:
            return

        # Show the block rectangle
        ax.axvspan(
            self.time_min.datetime,
            self.time_max.datetime,
            facecolor=self.kpColor,
            edgecolor='black',
            alpha=0.6
        )

        # Indicate the status
        ax.axvspan(
            self.time_min.datetime,
            self.time_max.datetime,
            ymin=0.9,
            facecolor=self.statusColor,
            edgecolor='black',
        )
        ax.axvspan(
            self.time_min.datetime,
            self.time_max.datetime,
            ymax=0.1,
            facecolor=self.statusColor,
            edgecolor='black',
        )

        # Show the observation block title
        xMin, xMax = ax.get_xlim()
        textPos = (self.time_min + (self.time_max - self.time_min)/2)
        textPosMDate = mdates.date2num(textPos.datetime)
        if (xMin <= textPosMDate) & (textPosMDate < xMax):
            ax.text(
                x=textPos.datetime,
                y=0.5,
                s=self.title,
                horizontalalignment='center',
                verticalalignment='center',
                color='black',
                fontweight='bold',
                rotation=90,
                fontsize=8
            )

            ax.text(
                x=textPos.datetime,
                y=0.05,
                s=f'{self.score:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                color='black',
                fontsize=8
            )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ReservedBlock ----------------------- #
# ============================================================= #
class ReservedBlock(Block):
    """ Class to handle reserved schedule time blocks.
        
        :param time_min:
            Starting time of the reserved time window.
        :type time_min:
            :class:`~astropy.time.Time`
        :param time_max:
            Ending time of the reserved time window.
        :type time_max:
            :class:`~astropy.time.Time`

        .. rubric:: Methods Summary

        .. autosummary::

            ~ReservedBlock.from_VCR

        .. rubric:: Attributes and Methods Documentation
    """

    def __init__(self, time_min, time_max):
        # These atrributes are filled once the ReservedBlk
        # has been inserted over a time range
        self.time_min = time_min
        self.time_max = time_max

        self.blockIdx = 0
        self.isBooked = False
        
        super().__init__(self)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def from_VCR(cls, file_name):
        """ Instantiates a :class:`~nenupy.schedule.obsblocks.ReservedBlock` object from the `Virtual Control Room <https://gui-nenufar.obs-nancay.fr/Planning/>`_ current booking list.

            :param file_name:
                CSV file describing the VCR current booking.
            :type file_name:
                `str`

            :returns:
                Reserved slots from the VCR active bookings.
            :rtype:
                :class:`~nenupy.schedule.obsblocks.ReservedBlock`
            
            .. warning::
                Only users with 'administrator' status may download the booking files.

        """
        reserved = []
        # with open(file_name, 'r') as rfile:
        #     for line in rfile.readlines():
        #         words = line.split(',')
        #         print(words[0])
        #         reserved.append(
        #             cls(
        #                 time_min=Time(words[0]),
        #                 time_max=Time(words[1])
        #             )
        #         )
        bookings = np.loadtxt(
            file_name,
            skiprows=1,
            delimiter=',',
            dtype={
                'names': ('start', 'stop', 'kp', 'comment'),
                'formats': ('U19', 'U19', 'U4', 'U50')
            }
        )
        starts = Time(bookings["start"])
        stops = Time(bookings["stop"])
        for start, stop in zip(starts, stops):
            reserved.append(
                cls(
                    time_min=start,
                    time_max=stop
                )
            )
        return functools.reduce(
            lambda x, y: x+y,
            reserved
        )
    # @classmethod
    # def fromBookingFile(cls, fileName):
    #     """ Parse VCR booking.
    #     """
    #     blocks = []
    #     return cls()


    def is_within(self, start: Time, stop: Time) -> bool:
        """ """
        return (self.time_min >= start)*(self.time_max < stop)\
            + (self.time_min < start)*(self.time_max > start)*(self.time_max < stop)\
            + (self.time_max > stop)*(self.time_min > start)*(self.time_min < stop)\
            + (self.time_min < start)*(self.time_max > stop)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _display(self, ax):
        """
        """
        # Show the block rectangle
        ax.axvspan(
            self.time_min.datetime,
            self.time_max.datetime,
            facecolor='0.8',
            edgecolor='black',
            hatch='//'
        )
# ============================================================= #
# ============================================================= #