#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********
    Schedule
    ********
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_TimeSlots',
    'Schedule'
]


from astropy.time import Time, TimeDelta
from astropy.table import Table
import numpy as np
from copy import deepcopy, copy

from nenupy.schedule.obsblocks import (
    Block,
    ObsBlock,
    ReservedBlock,
)
from nenupy.schedule.targets import SSTarget
from nenupy.schedule.geneticalgo import GeneticAlgorithm

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ============================================================= #
def _isTime(*args):
    """
    """
    return all([isinstance(arg, Time) for arg in args])


def _isTDelta(*args):
    """
    """
    return all([isinstance(arg, TimeDelta) for arg in args])


STATUS = {
    'good': '#8CC152',
    'medium': '#ECB23C',
    'bad': '#E9573F'
}


randGen = np.random.default_rng()
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ScheduleBlock ----------------------- #
# ============================================================= #
class ScheduleBlock(ObsBlock):
    def __init__(self,
        name,
        program,
        target,
        constraints,
        duration,
        dt,
        processing_delay
    ):
        super().__init__(
            name=name,
            program=program,
            target=target,
            constraints=constraints,
            duration=duration,
            processing_delay=processing_delay
        )

        self.dt = dt
        self.startIdx = None
        self.time_min = None
        self.time_max = None
        self.nSlots = int(np.ceil(self.duration/self.dt))
        if self.processing_delay is None:
            self.n_delay_slots = 0
        else:
            self.n_delay_slots = int(np.ceil(self.processing_delay/self.dt))


    def __del__(self):
        pass


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def isBooked(self):
        """
        """
        return self.startIdx is not None


    @property
    def indices(self):
        """
        """
        if self.startIdx is not None:
            idxRange = np.arange(self.nSlots)
            if np.isscalar(self.startIdx):
                return idxRange + self.startIdx
            else:
                # Useful for genetic algorithm
                return self.startIdx[:, None] + idxRange[None, :]
        else:
            return None


    @property
    def score(self):
        """
        """
        if self.indices is None:
            return 0.
        else:
            scores = []
            weights = []
            for constraint in self.constraints:
                scores.append(constraint.get_score(self.indices))
                weights.append(constraint.weight)
            #return np.nanmean(scores, axis=0)
            return np.average(scores, weights=weights, axis=0)


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


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def evaluate_score(self, time, sun_elevation):
        """
        """
        # Evaluate the target positions over time and compute the
        # score (product of each contraint score)
        # If it has already been evaluated do not do twice
        if (self.target is not None) and (self.target._lst is None):
            self.target.computePosition(time)

        if self.constraints.score is None:
            # Compute the weighted mean score for all constraints
            self.constraints.evaluate(
                target=self.target,
                time=time,
                nslots=self.nSlots,
                sun_elevation=sun_elevation
            )

            log.debug(
                f"<ObsBlock> #{self.blockIdx} named '{self.name}': "
                "Constraint score evaluated."
            )

    def is_within(self, start: Time, stop: Time) -> bool:
        """ """
        return (self.time_min >= start)*(self.time_max < stop)\
            + (self.time_min < start)*(self.time_max > start)*(self.time_max < stop)\
            + (self.time_max > stop)*(self.time_min > start)*(self.time_min < stop)\
            + (self.time_min < start)*(self.time_max > stop)


    def plot(self, **kwargs):
        """
            kwargs
                figsize
                nPoints
                figname
        """
        import matplotlib.pyplot as plt

        # Check if the obsblock has been booked in the schedule
        if not self.isBooked:
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
        figname = kwargs.get('figname', '')
        if figname != '':
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close('all')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
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
# ---------------------- ScheduleBlocks ----------------------- #
# ============================================================= #
class ScheduleBlocks(object):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, dt):
        self.dt = dt
        self.blocks = []
        self._nSlots = []
        self._n_delay_slots = []
        self._indices = []
        self._idxCounter = 0


    def __len__(self):
        """
        """
        return len(self.blocks)


    def __getitem__(self, n):
        """
        """
        return self.blocks[n]
    

    def __del__(self):
        for block in self.blocks:
            del block


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def size(self):
        """
        """
        return len(self.blocks)


    @property
    def nSlots(self):
        """
        """
        return np.array(self._nSlots, dtype=int)


    @property
    def n_delay_slots(self):
        """
        """
        return np.array(self._n_delay_slots, dtype=int)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def insert(self, block):
        """
        """
        if not isinstance(block, (ObsBlock, Block)):
            raise TypeError(
                f'<block> should be an {ObsBlock} or {Block} '
                f'instance, got object of type {type(block)} instead.'
            )
        for blk in block:
            sb = ScheduleBlock(
                    name=blk.name,
                    program=blk.program,
                    target=blk.target,
                    constraints=blk.constraints,
                    duration=blk.duration,
                    dt=self.dt,
                    processing_delay=blk.processing_delay
                )
            sb.blockIdx = self._idxCounter
            self.blocks.append(sb)
            self._nSlots.append(sb.nSlots)
            self._n_delay_slots.append(sb.n_delay_slots)
            self._indices.append(0)
            self._idxCounter += 1
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ _TimeSlots ------------------------- #
# ============================================================= #
class _TimeSlots(object):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        time_min,
        time_max,
        dt=TimeDelta(3600, format='sec')
    ):
        if not (_isTime(time_min, time_max) and _isTDelta(dt)):
            raise TypeError(
                f'Wrong types in `_TimeSlots({type(time_min)}, '
                f'{type(time_max)}, {type(dt)})`.'
            )

        self.dt = dt
        
        # Compute the time slots
        self.starts, self.stops = self._compute_slots_time_range(
            time_min=time_min,
            time_max=time_max,
            dt=dt
        )
        self._startsJD = self.starts.jd
        self._stopsJD = self.stops.jd

        self.size = self.starts.size

        # Initialize array of free time slots
        self.freeSlots = np.ones(self.size, dtype=bool)
        # Initialize array of free processing time slots
        self.free_processing_slots = np.ones(self.size, dtype=bool)
        # Initialize array of slot indices
        self.idxSlots = np.arange(self.size, dtype=int)
        self._freeIndices = self.idxSlots

        log.debug(
            f'{self.__class__} instance created with '
            f'{self.size} time slots.'
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time_min(self):
        """ Start time of the shedule.

            :getter: Start time.
            
            :type: :class:`~astropy.time.Time`
        """
        return self.starts[0]


    @property
    def time_max(self):
        """ Stop time of the shedule.

            :getter: Stop time.
            
            :type: :class:`~astropy.time.Time`
        """
        return self.stops[-1]
    

    @property
    def dt(self):
        """ Schedule granularity (time slot duration).

            :setter: Schedule granularity.
            
            :getter: Schedule granularity.
            
            :type: :class:`~astropy.time.TimeDelta`
        """
        return self._dt
    @dt.setter
    def dt(self, d):
        self._dt = d


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def time2idx(self, *time):
        """ Converts times to schedule indices.
            This method expects either one or two :class:`~astropy.time.Time` object(s).
        """
        if not _isTime(*time):
            raise TypeError(
                f'<time2idx> expects a {Time} object.'
            )

        if len(time) == 1:
            # Find the unique corresponding slot index
            mask = (self._startsJD <= time[0].jd) &\
                (self._stopsJD > time[0].jd)
        elif len(time) == 2:
            if time[1] < time[0]:
                raise ValueError(
                    f'start>stop ({time[1].isot} < {time[0].isot})'
                    ' in <time2idx(start, stop)>.'
                )
            # Find all indices within the time range
            mask = (self._stopsJD > time[0].jd) &\
                (self._startsJD < time[1].jd)
        else:
            raise Exception(
                '<time2idx()> not properly called.'
            )

        return self.idxSlots[mask]


    def add_booking(self, time_min: Time, time_max: Time) -> None:
        """ Changes the status of the time slots comprised
            between ``time_min`` and ``time_max`` to 'booked' (i.e., not
            available anymore).
            In particular, the attribute 
            :attr:`~nenupy.schedule._TimeSlots.freeSlots` is set
            to ``False`` at the corresponding indices.

            :param time_min:
                Start time.
            :type time_min:
                :class:`~astropy.time.Time`
            :param time_max:
                Stop time.
            :type time_max:
                :class:`~astropy.time.Time`
        """
        indices = self.time2idx(time_min, time_max)
        if any(~self.freeSlots[indices]):
            log.warning(
                f"Booking on reserved slots from '{time_min.isot}' "
                f"to '{time_max.isot}'."
            )
        self.freeSlots[indices] = False
        self._freeIndices = self.idxSlots[self.freeSlots]


    def remove_booking(self, time_min: Time, time_max: Time) -> None:
        """
        """
        indices = self.time2idx(time_min, time_max)
        self.freeSlots[indices] = True
        self._freeIndices = self.idxSlots[self.freeSlots]


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _compute_slots_time_range(time_min: Time, time_max: Time, dt: TimeDelta):
        """
        """
        if time_max <= time_min:
            raise ValueError(
                f'Schedule time_max={time_max.isot} <= '
                f'time_min={time_min.isot}.'
            )

        period = time_max - time_min
        time_steps = int( np.ceil(period/dt) )
        dt_shifts = np.arange(time_steps)*dt

        slot_starts = time_min + dt_shifts
        slot_stops = slot_starts + dt

        return slot_starts, slot_stops
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- Schedule -------------------------- #
# ============================================================= #
class Schedule(_TimeSlots):
    """ Foundation class of observation scheduling.

        :param time_min:
            Schedule time range lower edge.
        :type time_min:
            :class:`~astropy.time.Time`
        :param time_max:
            Schedule time range upper edge.
        :type time_max:
            :class:`~astropy.time.Time`
        :param dt:
            Schedule granularity (or time slot duration).
            Default is ``1 hour``.
        :type dt:
            :class:`~astropy.time.TimeDelta`
        
        :Example:
            >>> from nenupy.schedule import Schedule
            >>> from astropy.time import Time, TimeDelta
            >>> schedule = Schedule(
            >>>     time_min=Time('2021-01-11 00:00:00'),
            >>>     time_max=Time('2021-01-15 00:00:00'),
            >>>     dt=TimeDelta(3600, format='sec')
            >>> )
        
        .. seealso::
            :ref:`scheduling_doc`

        .. rubric:: Attributes Summary

        .. autosummary::

            ~nenupy.schedule.schedule._TimeSlots.time_min
            ~nenupy.schedule.schedule._TimeSlots.time_max
            ~nenupy.schedule.schedule._TimeSlots.dt
            ~Schedule.observation_blocks
            ~Schedule.reserved_blocks

        .. rubric:: Methods Summary

        .. autosummary::

            ~Schedule.set_free_slots
            ~Schedule.match_booking
            ~Schedule.insert
            ~Schedule.plot
            ~Schedule.plot_range
            ~Schedule.book
            ~Schedule.export

        .. rubric:: Attributes and Methods Documentation

    """

    def __init__(
        self,
        time_min,
        time_max,
        dt=TimeDelta(3600, format='sec')
    ):
        super().__init__(
            time_min=time_min,
            time_max=time_max,
            dt=dt
        )

        # self.observation_blocks = None
        self.observation_blocks = ScheduleBlocks(dt=self.dt)
        self.reserved_blocks = None

        # Store the Sun's elevation
        sun = SSTarget.fromName('Sun')
        sun.computePosition(
            Time(
                np.append(
                    self._startsJD,
                    self._startsJD[-1] + self.dt.jd
                ),
                format='jd'
            )
        )
        elevation = sun.elevation.deg
        self.sun_elevation = (elevation[1:] + elevation[:-1])/2


    def __getitem__(self, n):
        """
        """
        return self.observation_blocks[n]


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def observation_blocks(self):
        """ Observation blocks included in the schedule.

            :setter: Observation blocks.
            
            :getter: Observation blocks.
            
            :type: :class:`~nenupy.schedule.schedule.ScheduleBlocks`
        """
        return self._observation_blocks
    @observation_blocks.setter
    def observation_blocks(self, obs):
        self._observation_blocks = obs


    @property
    def reserved_blocks(self):
        """ Reserved blocks included in the schedule.
            No observation can be planned on these time slots.

            :setter: Reserved blocks.
            
            :getter: Reserved blocks.
            
            :type: :class:`~nenupy.schedule.obsblocks.ReservedBlock`
        """
        return self._reserved_blocks
    @reserved_blocks.setter
    def reserved_blocks(self, res):
        self._reserved_blocks = res


    @property
    def score(self) -> float:
        """ Returns the current schedule score. """
        return np.mean([block.score for block in self.observation_blocks])

    # @property
    # def scheduledBlocks(self):
    #     """
    #     """
    # list(map(schedule.observation_blocks.__getitem__, [i for i, blk in enumerate(schedule.observation_blocks) if blk.isBooked]))
    #     return self._scheduledBlocks
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def insert(self, *blocks):
        r""" Inserts observation blocks or reserved blocks in the schedule.
        
            :param \*blocks:
                Observation blocks or reserved blocks.
            :type \*blocks:
                :class:`~nenupy.schedule.obsblocks.Block`, :class:`~nenupy.schedule.obsblocks.ObsBlock` or :class:`~nenupy.schedule.obsblocks.ReservedBlock`

            :Example:
                .. code-block:: python

                    from nenupy.schedule import Schedule, ESTarget, ObsBlock
                    from astropy.time import Time, TimeDelta

                    schedule = Schedule(
                        time_min=Time('2021-01-11 00:00:00'),
                        time_max=Time('2021-01-15 00:00:00'),
                        dt=TimeDelta(3600, format='sec')
                    )

                    cas_a = ObsBlock(
                        name="Cas A",
                        program="ES00",
                        target=ESTarget.fromName("Cas A"),
                        duration=TimeDelta(2*3600, format='sec'),
                    )

                    schedule.insert(cas_a)

        """
        for blocks_i in blocks:
            # if not isinstance(blocks_i, Block):
            #     raise TypeError(
            #         f'Expected input of type {Block}>'
            #         f', got <{blocks_i.__class__}> instead.'
            #     )
            if all([blk.__class__ is ObsBlock for blk in blocks_i]):
                self.observation_blocks.insert(blocks_i)
            elif all([blk.__class__ is ReservedBlock for blk in blocks_i]):
                for blk in blocks_i:
                    self.add_booking(blk.time_min, blk.time_max)
                    log.debug(f'ReservedBlock added from {blk.time_min.isot} to {blk.time_max.isot}.')
                if self.reserved_blocks is None:
                    self.reserved_blocks = blocks_i
                else:
                    self.reserved_blocks += blocks_i
            else:
                raise Exception(
                    f'Not supposed to happen for type {blocks_i.__class__}!'
                )

        self._toSchedule = np.ones(self.observation_blocks.size, dtype=bool)


    def set_free_slots(self, start_times: Time, stop_times: Time) -> None:
        """ Assign a :class:`~nenupy.schedule.obsblocks.ReservedBlock`
            to every Schedule time interval that is not comprised bewteen
            ``start_times`` and ``stop_times``.
        """
        if start_times.size != stop_times.size:
            raise ValueError(
                "start_times and stop_times should have the same length."
            )
        elif start_times.isscalar and stop_times.isscalar:
            start_times = start_times.reshape((1,))
            stop_times = stop_times.reshape((1,))
        elif (start_times.ndim != 1) or (stop_times.ndim != 1):
            raise ValueError(
                f"start_times ({start_times.ndim}D) and "
                f"stop_times ({stop_times.ndim}D) should be 1D."
            )
        # Treat the first booking
        if start_times[0] > self.time_min:
            self.insert(
                ReservedBlock(
                    time_min=self.time_min,
                    time_max=start_times[0]
                )
            )
        # Treat the rest
        for blk_start, blk_stop in zip(stop_times[:-1], start_times[1:]):
            self.insert(
                ReservedBlock(
                    time_min=blk_start,
                    time_max=blk_stop
                )
            )
        # Treat the last booking
        if stop_times[-1] < self.time_max:
            self.insert(
                ReservedBlock(
                    time_min=stop_times[-1],
                    time_max=self.time_max
                )
            )


    def match_booking(self, booking_file: str, key_program: str) -> None:
        """ """
        bookings = np.loadtxt(
            booking_file,
            skiprows=1,
            delimiter=',',
            dtype={
                'names': ('start', 'stop', 'kp', 'comment'),
                'formats': ('U19', 'U19', 'U4', 'U50')
            }
        )

        # Check that the selected Key program is valid
        kp_names = np.unique(bookings['kp'])
        if key_program not in kp_names:
            raise ValueError(
                f'Key program {key_program} does not have any entry in {booking_file}, select a value in {kp_names}.'
            )
        valid_booking = bookings[bookings['kp'] == key_program]
        booking_starts = Time(valid_booking['start'])
        booking_stops = Time(valid_booking['stop'])

        # Find each time period not in the valid booking range
        within_schedule_mask = (booking_starts >= self.time_min)*(booking_stops <= self.time_max)\
            + (booking_starts < self.time_min)*(booking_stops > self.time_min)\
            + (booking_starts < self.time_max)*(booking_stops > self.time_max)
        log.debug(f'{np.sum(within_schedule_mask)} bookings are within the schedule.')

        # Insert corresponding reserved blocks in between the booking periods
        starts = booking_starts[within_schedule_mask]
        stops = booking_stops[within_schedule_mask]
        # # Treat the first booking
        # if starts[0] > self.time_min:
        #     self.insert(
        #         ReservedBlock(
        #             time_min=self.time_min,
        #             time_max=starts[0]
        #         )
        #     )
        # for reserve_block_start, reserve_block_stop in zip(stops[:-1], starts[1:]):
        #     self.insert(
        #         ReservedBlock(
        #             time_min=reserve_block_start,
        #             time_max=reserve_block_stop
        #         )
        #     )
        # # Treat the first booking
        # if stops[-1] < self.time_max:
        #     self.insert(
        #         ReservedBlock(
        #             time_min=stops[-1],
        #             time_max=self.time_max
        #         )
        #     )
        self.set_free_slots(start_times=starts, stop_times=stops)


    def book(self, optimize=False, **kwargs):
        r""" Distributes the :attr:`~nenupy.schedule.schedule.Schedule.observation_blocks` over the schedule time slots.
            The observing constraints are evaluated over the whole schedule.

            :param optimize:
                If set to ``False`` an heuristic algorithm is used to perfom the booking.
                However faster, this could result in sub-optimized time slots filling.
                If set to ``True``, a genetic algorithm is used (see below the configuration keywords).
                This method delivers more optimized scheduling solutions.
                However, due to the random nature of genetic algorithm behaviors, the results may not be reproducible and may vary from one run to the other.
                Default is ``False``.
            :type optimize:
                `bool`
            
            .. rubric:: Genetic algorithm configuration keywords
            
            :param population_size:
                Size of the solution population, i.e., how many distinct soluttions are evolving at the same time.
                Default is 20.
            :type population_size:
                `int`
            :param random_individuals:
                At each generation, the population of solutions will replace its lower-score ``random_individuals`` solutions by random genome ones.
                This allows for genetic diversity.
                Default is ``1``.
            :type random_individuals:
                `int`
            :param score_threshold:
                Score value (between ``0`` and ``1``) at which the evolution may stop.
                Default is ``0.8``.
            :type score_threshold:
                `float`
            :param generation_max:
                Maximum generation number at which the evolution has to stop (even if ``score_threshold`` is not reached).
                Default is ``1000``.
            :type generation_max:
                `int`
            :param max_stagnating_generations:
                If the score does not evolve after ``max_stagnating_generations`` generations, the evolution is stopped.
                Default is ``100``.
            :type max_stagnating_generations:
                `int`
            :param selection:
                At each generation, pairs of individuals (solutions) are selected to give birth to new children with a mix of their genomes (see ``crossover``).
                This selection can be perfomed according to three methods.
                ``'FPS'`` (Fitness Proportionate Selection): parents are as likely to be picked as their score is high.
                ``'TNS'`` (Tournament Selection): :math:`k` individuals from the population are selected, the best becomes a parent (where :math:`k = \max( 2, \rm{population\ size}/10 )`).
                ``'RKS'`` (Rank Selection): individuals are ranked according to their scores, and are more likely to be picked according to their rank. The amplitude of score differences doesn't count.
                Default is ``'FPS'``.
            :type selection:
                `str`
            :param crossover:
                Whenever parents give birth to children, their genome (i.e., time slot indices for each observation block) is mixed.
                This cross-over can be performed according to three methods.
                ``'SPCO'`` (Single-point crossover): genes from 'parent1' and 'parent2' are interverted after a random index.
                ``'TPCO'`` (Two-point crossover): genes from 'parent1' and 'parent2' are interverted between two random indices.
                ``'UNCO'`` (Uniform crossover): every genes from 'parent1' and 'parent2' are interverted with a uniform probability.
                Default is ``'SPCO'``.
            :type crossover:
                `str`

                selectionMethod (default FPS)
                crossoverMethod (default SPCO)
                elitism
            
            :returns:
                If ``optimize`` is set to ``False``, nothing is returned.
                If ``optimize`` is set to ``True``, the :class:`~nenupy.schedule.geneticalgo.GeneticAlgorithm` instance is returned.
                The method :class:`~nenupy.schedule.geneticalgo.GeneticAlgorithm.plot` can then be called to visualize the evolution of the solution populations.
            :rtype:
                ``None`` or :class:`~nenupy.schedule.geneticalgo.GeneticAlgorithm`

        """
        # Ré-initialize the booking, in case the user calls this method several times
        self._reset_observation_block_bookings()

        # Pre-compute the constraints scores
        self._compute_block_constraint_scores(**kwargs)

        n_unscheduled_blocks = 0

        # Raise warning if the observation block are impossible to fit
        if sum(self._toSchedule) == 0:
            log.warning(
                'Required observation blocks have constraints '
                'unfitted for this schedule.'
            )
            return
        else:
            log.info(
                f'Fitting {sum(self._toSchedule)} observation blocks...'
            )

        if optimize:
            # All the observation blocks are booked at
            # once with the genetic algorithm.
            ga = GeneticAlgorithm(
                populate=self._populate,
                fitness=self._fitness,
                mutation=self._mutation,
                populationSize=kwargs.get('population_size', 20)
            )

            # Find out the best observation schedule
            ga.evolve(
                **kwargs
            )

            # Book the observation_blocks on the schedule
            k = 0
            for i, blk in enumerate(self.observation_blocks):
                if not self._toSchedule[i]:
                    # The observation had a null score over the schedule
                    continue
                best_start_index = ga.bestGenome[k]
                best_stop_index = best_start_index + blk.nSlots - 1
                k += 1
                if all(self._cnstScores[i, best_start_index:best_stop_index + 1] == 0):
                    n_unscheduled_blocks += 1
                    log.warning(
                        f"<ObsBlock> #{blk.blockIdx} '{blk.name}' cannot be scheduled."
                    )
                    continue
                blk.startIdx = best_start_index
                blk.time_min = self.starts[best_start_index]
                blk.time_max = self.stops[best_start_index + blk.nSlots-1]

            score = self.fine_tune(max_it=kwargs.get("fine_tune_max_iterations", 1000))

            # Remove blocks that are overlapping
            # Lowest scores are removed in priority
            scores = [blk.score for blk in self.observation_blocks]
            for _, blk in sorted(zip(scores, self.observation_blocks), key=lambda pair: pair[0]):
                if blk.startIdx is None: continue
                start_idx = blk.startIdx
                stop_idx = blk.startIdx + blk.nSlots
                if not all(self.freeSlots[start_idx:stop_idx]):
                    n_unscheduled_blocks += 1
                    blk.startIdx = None # unbook the block
                    log.warning(f"<ObsBlock> #{blk.blockIdx} is overlapping other blocks.")
                    continue
                self.freeSlots[start_idx:stop_idx] = False

            log.info(
                f'{sum(self._toSchedule) - n_unscheduled_blocks}/'
                f'{sum(self._toSchedule)} observation blocks scheduled '
                f'({sum(~self._toSchedule)} impossible to fit).'
            )

            return ga

        else:
            block_indices = np.arange(self.observation_blocks.size)
            if kwargs.get('sort_by_difficulty', False):
                sort_idx = np.argsort(np.sum(self._cnstScores, axis=1))
                block_indices = block_indices[sort_idx]

            # Block are booked iteratively 
            # for i, blk in enumerate(self.observation_blocks):
            for i in block_indices:
                blk = self.observation_blocks[i]

                if (not self._toSchedule[i]) or (blk.isBooked):
                    continue
                # Construct a mask to avoid setting the obsblock where it cannot fit
                if blk.n_delay_slots != 0:
                    freeSlotsShifted = self.freeSlots.copy() * self.free_processing_slots
                else:
                    freeSlotsShifted = self.freeSlots.copy()
                
                # for j in range(blk.nSlots):
                #     freeSlotsShifted *= np.roll(self.freeSlots, -j)
                # freeSlotsShifted[-blk.nSlots:] = self.freeSlots[-blk.nSlots:]
                mask_idx = (np.where(~freeSlotsShifted)[0][:, None] - np.arange(blk.nSlots)[None, ::-1]).ravel()
                mask_idx = np.unique(mask_idx)
                mask_idx = mask_idx[mask_idx >= 0]
                freeSlotsShifted[mask_idx] = False
                
                # Find the best spot
                score = self._cnstScores[i, :] * self.freeSlots * freeSlotsShifted
                if all(score==0):
                    n_unscheduled_blocks += 1
                    log.warning(
                        f"<ObsBlock> #{i} '{blk.name}' cannot be scheduled."
                    )
                    continue
                bestStartIdx = np.argmax(
                    score
                )
                # Assign a start and stop time to the block and update the free slots
                blk.startIdx = bestStartIdx
                bestStopIdx = bestStartIdx + blk.nSlots - 1
                self.freeSlots[bestStartIdx:bestStopIdx + 1] = False
                if blk.n_delay_slots != 0:
                    # Update the processing reserved slots
                    start_proc = bestStartIdx - blk.n_delay_slots
                    start_proc = 0 if start_proc < 0 else start_proc
                    stop_proc = bestStopIdx + blk.n_delay_slots + 1
                    stop_proc = self.size if stop_proc > self.size else stop_proc
                    self.free_processing_slots[start_proc:stop_proc] = False

                blk.time_min = self.starts[bestStartIdx]
                blk.time_max = self.stops[bestStopIdx]

            log.info(
                f'{sum(self._toSchedule) - n_unscheduled_blocks}/'
                f'{sum(self._toSchedule)} observation blocks scheduled '
                f'({sum(~self._toSchedule)} impossible to fit).'
            )


    def fine_tune(self, max_it: int = 1000) -> None:
        """ 
        """
        log.info("(Fine tunning) Launching...")

        scores = []

        # Loop until the score drops
        while len(scores) < max_it:
            # Gather starting index and size of each schedule obsblock
            start_indices = []
            nslots = []
            delay_nslots = []
            indices = []
            for block in self.observation_blocks:
                if not block.isBooked:
                    continue
                start_indices.append(block.startIdx)
                nslots.append(block.nSlots)
                indices.append(block.blockIdx)
                delay_nslots.append(block.n_delay_slots)

            # Sort the by increasing index
            start_indices_sorted = np.argsort(start_indices)
            start_indices = np.array(start_indices, dtype=int)[start_indices_sorted]
            nslots = np.array(nslots, dtype=int)[start_indices_sorted]
            indices = np.array(indices, dtype=int)[start_indices_sorted]
            delay_nslots = np.array(delay_nslots, dtype=int)[start_indices_sorted]

            # Associate each observation blocks with its score before and after
            max_scores = np.zeros((len(indices), 2)) # (number of booked blocks, score just before/score just after)
            for i in range(start_indices_sorted.size):
                block_index = indices[i]
                block_score = self._cnstScores[block_index, start_indices[i]]
                id1 = i - 1 # index of previous before
                gap1 = np.arange(start_indices[id1] + nslots[id1] if id1 > 0 else 0, start_indices[i])
                score1 = -1 if gap1.size == 0 else self._cnstScores[block_index, gap1[-1]] - block_score
                id2 = i + 1 # index of next block
                gap2 = np.arange(start_indices[i] + 1, start_indices[id2] - nslots[i] + 1 if id2 <= (len(indices) - 1) else self.size - 1)
                score2 = -1 if gap2.size == 0 else self._cnstScores[block_index, gap2[0]] - block_score

                if delay_nslots[i] != 0:
                    # If this is requires a processing delay, we must check for the previous and next similar
                    # observation, which may not be the very previous or very next.
                    # Override immediate left/right slot if a similar observation is too close.
                    delay_obs = np.argwhere(delay_nslots != 0)
                    # Left
                    prev_delay_idx = delay_obs[np.roll(delay_obs == i, -1)][0]
                    if prev_delay_idx > i:
                        # This is the first observation block with a delay constraint
                        pass
                    elif start_indices[prev_delay_idx] + nslots[prev_delay_idx] + delay_nslots[prev_delay_idx] >= start_indices[i]:
                        score1 = -1
                    # Right
                    next_delay_idx = delay_obs[np.roll(delay_obs == i, +1)][0]
                    if next_delay_idx < i:
                        # This is the last observation block with delay constraint
                        pass
                    elif start_indices[i] + nslots[i] + delay_nslots[i] >= start_indices[next_delay_idx]:
                        score2 = -1

                # Score 'gradients'
                max_scores[i, :] = np.array([
                    score1,
                    score2
                ])

            # Find out the maximal score difference
            left_right_max = np.argmax(max_scores, axis=1)
            max_score_diff = np.max(max_scores[np.arange(left_right_max.size), left_right_max])
            max_score_block_idx = np.argmax(max_scores[np.arange(left_right_max.size), left_right_max])
            if max_score_diff <= 0:
                log.info("(Fine tunning) No positive gradient available. End of fine-tunning loop.")
                break

            # Shift the block 1dt left or right
            block = self.observation_blocks[indices[max_score_block_idx]]
            block.startIdx += -1 if left_right_max[max_score_block_idx] == 0 else 1
            block.time_min = self.starts[block.startIdx]
            block.time_max = self.stops[block.startIdx + block.nSlots - 1]

            direction = 'left' if left_right_max[max_score_block_idx] == 0 else 'right'
            log.debug(f'Shifting block #{indices[max_score_block_idx]} to the {direction}.')

            # Compute the score
            scores.append(self.score)

        else:
            log.info("(Fine tunning) Maximum number of iterations reached.")

        log.info(f"(Fine tunning) End after {len(scores)} iterations.")

        return scores


    def export(self, score_min=0):
        """ Exports the current schedule once :meth:`~nenupy.schedule.schedule.Schedule.book` has been called.
            Every observation with a ``score`` higher than ``score_min`` is included in the export.

            :param score_min:
                Minimal :attr:`~nenupy.schedule.schedule.Schedule.obervation_blocks`'s score to be exported.
                Default is ``0``.
            :type scoremin:
                `float`

            :returns:
                Scheduled observation blocks as a :class:`~astropy.table.Table`.
                This object can further be converted to any user requirements using
                the :meth:`astropy.io.ascii.write` method
                (see the list of supported `formats <https://docs.astropy.org/en/stable/io/ascii/index.html#supported-formats>`_).
            :rtype:
                :class:`~astropy.table.Table`

        """
        if not isinstance(score_min, (float, int)):
            raise TypeError(
                '<score_min> should be a number.'
            )
        elif score_min < 0:
            raise ValueError(
                '<score_min> should be a positive.'
            )
        elif score_min > 1:
            raise ValueError(
                '<score_min> cannot be greater than 1.'
            )

        names = []
        index = []
        starts = []
        stops = []
        programs = []
        scores = []
        for blk in self.observation_blocks:
            if not blk.isBooked:
                continue
            score = blk.score
            if score < score_min:
                continue
            scores.append(score)
            names.append(blk.name)
            index.append(blk.blockIdx)
            starts.append(blk.time_min.isot)
            stops.append(blk.time_max.isot)
            programs.append(blk.program)
        start_array = Time(starts, format='isot')
        chron = np.argsort(start_array)

        tab = Table()
        tab['obsid'] = np.array(index, dtype=int)[chron]
        tab['name'] = np.array(names, dtype=str)[chron]
        tab['program'] = np.array(programs, dtype=str)[chron]
        tab['start'] = start_array[chron]
        tab['stop'] = Time(stops, format='isot')[chron]
        tab['score'] = np.array(scores)[chron]

        return tab

    
    def plot_range(self, start_time: Time, stop_time: Time, **kwargs) -> None:
        """ Plots the current schedule.

            .. rubric:: Data display keywords

            :param start_time:
                Minimal time to display.
            :type start_time:
                :class:`~astropy.time.Time`
            :param stop_time:
                Maximal time to display.
            :type stop_time:
                :class:`~astropy.time.Time`

            .. rubric:: Plotting layout keywords

            :param grid:
                If set to ``True``, the time slots are separated by vertical lines.
                Default is ``True``.
            :type grid:
                `bool`
            :param figname:
                Name of the file (absolute or relative path) to save the figure.
                Default is ``''`` (i.e., only show the figure).
            :type figname:
                `str`
            :param figsize:
                Set the figure size.
                Default is ``(15, 3)``.
            :type figsize:
                `tuple`

        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Initialize the figure
        _, ax = plt.subplots(
            figsize=kwargs.get("figsize", (15, 3))
        )

        # Display granularity
        if kwargs.get("grid", True):
            time_mask = (self._startsJD >= start_time.jd)*(self._stopsJD <= stop_time.jd)
            for slot_start in self.starts[time_mask]:
                ax.axvline(
                    slot_start.datetime,
                    color="gray",
                    linestyle="-",
                    linewidth=0.5
                )

        # Display unavailable slots
        if self.reserved_blocks is not None:
            for booked_block in self.reserved_blocks:
                if not booked_block.is_within(start_time, stop_time):
                    continue
                booked_block._display(ax=ax)

        # Display observation blocks
        if self.observation_blocks is not None:
            for obs_block in self.observation_blocks:
                if not obs_block.isBooked:
                    continue
                if not obs_block.is_within(start_time, stop_time):
                    continue
                obs_block._display(ax=ax)

        ax.set_xlim(
                left=start_time.datetime,
                right=stop_time.datetime
            )

        # Formating
        ax.yaxis.set_visible(False)
        h_fmt = mdates.DateFormatter("%y-%m-%d\n%H")
        ax.xaxis.set_major_formatter(h_fmt)

        # Save or show the figure
        figname = kwargs.get("figname", "")
        if figname != "":
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches="tight",
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close("all")


    def plot(self, days_per_line=1, **kwargs):
        """ Plots the current schedule.

            .. rubric:: Data display keywords

            :param days_per_line:
                Number of days to plots per line.
                Default is ``1``.
            :type days_per_line:
                `int`

            .. rubric:: Plotting layout keywords

            :param grid:
                If set to ``True``, the time slots are separated by vertical lines.
                Default is ``True``.
            :type grid:
                `bool`
            :param figname:
                Name of the file (absolute or relative path) to save the figure.
                Default is ``''`` (i.e., only show the figure).
            :type figname:
                `str`
            :param figsize:
                Set the figure size.
                Default is ``(15, 3*lines)``.
            :type figsize:
                `tuple`

        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Set relevant variables for the plot
        time_min = Time(self.starts[0].isot.split('T')[0])
        time_max = self.stops[-1]

        # Find the number of sub-plots to display the schedule
        if not isinstance(days_per_line, TimeDelta):
            days_per_line = TimeDelta(
                days_per_line,
                format='jd'
            )
        n_subplots = int(np.ceil((time_max - time_min)/days_per_line))

        # Initialize the figure
        fig, axs = plt.subplots(
            nrows=n_subplots,
            ncols=1,
            figsize=kwargs.get('figsize', (15, 3*n_subplots))
        )

        # Iterate over the sub-plots
        try:
            isIterable = iter(axs)
        except:
            axs = [axs]
        for i, ax in enumerate(axs):
            # Plot time limits
            tiMin = time_min + i*days_per_line
            tiMax = time_min + (i + 1)*days_per_line
            ax.set_xlim(
                left=tiMin.datetime,
                right=tiMax.datetime
            )

            # Display granularity
            if kwargs.get('grid', True):
                tMask = (self._startsJD >= tiMin.jd) *\
                    (self._stopsJD <= tiMax.jd)
                for slot_start in self.starts[tMask]:
                    ax.axvline(
                        slot_start.datetime,
                        color='gray',
                        linestyle='-',
                        linewidth=0.5
                    )

            # Display unavailable slots
            if self.reserved_blocks is not None:
                for bookedBlock in self.reserved_blocks:
                    bookedBlock._display(ax=ax)

            # Display observation blocks
            if self.observation_blocks is not None:
                for obsBlock in self.observation_blocks:
                    if not obsBlock.isBooked:
                        continue
                    obsBlock._display(ax=ax)

            # Formating
            ax.yaxis.set_visible(False)
            h_fmt = mdates.DateFormatter('%y-%m-%d\n%H')
            ax.xaxis.set_major_formatter(h_fmt)

        # Save or show the figure
        figname = kwargs.get('figname', '')
        if figname != '':
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close('all')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _compute_block_constraint_scores(self, **kwargs):
        """
        """
        log.info(
            'Evaluating observation block constraints over '
            'the schedule...'
        )
        self._cnstScores = np.zeros(
            (self.observation_blocks.size, self.size)
        )
        self._maxScores = np.zeros(self.observation_blocks.size)

        # Compute the constraint score over the schedule if it
        # has not been previously been done
        times = Time(
            np.append(
                self._startsJD,
                self._startsJD[-1] + self.dt.jd
            ),
            format='jd'
        )
        for i, blk in enumerate(self.observation_blocks):
            if blk.constraints.score is None:
                # If != 1, the constraint score has already been computed
                # Append the last time slot to get the last slot top
                blk.evaluate_score(time=times, sun_elevation=self.sun_elevation)

            # Set other attributes relying on the schedule
            blk.nSlots = int(np.ceil(blk.duration/self.dt))

            # Get the constraint and maximum possible score
            sliding_indices = np.arange(blk.nSlots)[None, :] +\
                np.arange(self.size - blk.nSlots)[:, None]
            self._cnstScores[i, :-blk.nSlots] = np.mean(
                blk.constraints.score[sliding_indices],
                axis=1
            )
            # Set constraint score to zero if forbidden index
            forbidden_mask = np.any(~self.freeSlots[sliding_indices], axis=1)
            self._cnstScores[i, :-blk.nSlots][forbidden_mask] *= 0
            self._maxScores[i] = np.max(self._cnstScores[i, :])

            # Check that the constraints are non zero
            if self._maxScores[i] == 0:
                self._toSchedule[i] = False
                log.warning(
                    f"<ObsBlock> #{blk.blockIdx} '{blk.name}'"
                    " has a null score over the schedule."
                )

        log.info(
            f'{self.observation_blocks.size} observation blocks have been '
            'successfully evaluated.'
        )        


    def _reset_observation_block_bookings(self) -> None:
        """ Loops through all observation_blocks,
            if they are booked, set them as brand new and
            release the corresponding schedule booking slots.
        """
        for block in self.observation_blocks:
            if block.startIdx is None:
                # The observation block has not been booked, skip
                continue
            # Free the schedule slots
            self.freeSlots[block.startIdx:block.startIdx + block.nSlots] = True
            if block.n_delay_slots != 0:
                self.free_processing_slots[block.startIdx - block.n_delay_slots - 1: block.startIdx + block.nSlots + block.n_delay_slots] = True
            # Reset the observation block as un-booked
            block.startIdx = None


    def _bound_to_schedule(self, indices: np.ndarray) -> np.ndarray:
        """
            startIndices = (schedule_size)
            
            return startIndices = (schedule_size)
        """
        indices[indices < 0] = 0
        beyond = (indices + self.observation_blocks.nSlots[self._toSchedule]) >= self.idxSlots.size
        indices[beyond] = self.idxSlots.size - 1
        indices -= beyond*self.observation_blocks.nSlots[self._toSchedule]
        return indices


    def _populate(self, n):
        """
        """
        # return self._bounds(
        #     randGen.integers(
        #         low=0,
        #         high=self.idxSlots.size,
        #         size=(n, self.observation_blocks.size),
        #         dtype=np.int64
        #     )
        # )
        n_obs_blocks = np.sum(self._toSchedule)
        # population = np.zeros(
        #     (n, self.observation_blocks.size),
        #     dtype=int
        # )
        population = np.zeros(
            (n, n_obs_blocks),
            dtype=int
        )
        for i in range(n_obs_blocks):
            population[:, i] = randGen.choice(
                np.where(self._cnstScores[self._toSchedule][i] != 0)[0],
                # replace=False,
                size=n,
            )

        return self._bound_to_schedule(population)


    def _fitness(self, population):
        """
            population : np.array((children, genome))
        """
        # Evaluate the fitness of all the observation blocks over
        # the cumulative (on nSlots) constraint scores and normalize
        # by the maximum available score within the schedule
        genome_fitness = np.diagonal(
            self._cnstScores[self._toSchedule].T[population],
            axis1=1,
            axis2=2
        )/self._maxScores[self._toSchedule]
        #has_zeros = np.any(genome_fitness==0., axis=1)
        fitness = np.mean(
            genome_fitness,
            axis=1
        )
        #fitness[has_zeros] = 0.
        # Find out which individual in the population contains blocks
        # that overlaps with one another.
        # Sort the population block indices because we don't care
        # about the last one 
        # sortedIdx = np.argsort(population)
        # overlaps = np.any(
        #     np.diff(
        #         np.take_along_axis(
        #             population,
        #             sortedIdx,
        #             axis=1
        #         ),
        #         axis=1
        #     ) - self.observation_blocks.nSlots[self._toSchedule][sortedIdx][:, :-1] < 0,
        #     axis=1
        # )
        # # The fitness is the product of the constraint score and
        # # is set to 0 whenever there are overlapping blocks.
        # return fitness * ~overlaps

       
        # Reduce the fitness if blocks with n_delay_slots != 0 overlap
        are_delay_constrained = self.observation_blocks.n_delay_slots[self._toSchedule] > 0
        population_delay = population[:, are_delay_constrained]
        sorted_indices = np.argsort(population_delay)
        delays = self.observation_blocks.n_delay_slots[self._toSchedule][sorted_indices][:, :-1]
        nslots = self.observation_blocks.nSlots[self._toSchedule][sorted_indices][:, :-1]
        overlaps_delay = np.sum(
            np.diff(
                np.take_along_axis(
                    population_delay,
                    sorted_indices,
                    axis=1
                ),
                axis=1
            ) - (delays + nslots) < 0,
            axis=1
        )        

        # Reduce the fitness if observations temporally overlap
        sortedIdx = np.argsort(population)
        overlaps = np.sum(
            np.diff(
                np.take_along_axis(
                    population,
                    sortedIdx,
                    axis=1
                ),
                axis=1
            ) - self.observation_blocks.nSlots[self._toSchedule][sortedIdx][:, :-1] < 0,
            axis=1
        )
        # The fitness is the product of the constraint score and
        # is reduced whenever there are overlapping blocks.
        return fitness/(overlaps + overlaps_delay + 1)


    def _mutation(self, genome):#, scaleOnFitness=False):
        """
        """
        idx = randGen.integers(
            low=0,
            high=genome.size
        )
        genome[idx] = randGen.integers(
            low=0,
            high=self.idxSlots.size
        )
        return self._bound_to_schedule(genome)
        # if scaleOnFitness:
        #     fitness = self._fitness(genome[None, :])[0]
        #     nMutations = int(np.ceil(genome.size * (1 - fitness)))
        #     idx = randGen.choice(
        #         np.arange(genome.size),
        #         replace=False,
        #         size=nMutations
        #     )
        #     genome[idx] = randGen.choice(
        #         self._freeIndices,
        #         size=nMutations
        #     )
        # else:
        #     idx = randGen.integers(
        #         low=0,
        #         high=genome.size
        #     )
        #     genome[idx] = randGen.integers(
        #         low=0,
        #         high=self.idxSlots.size
        #     )
        # return self._bounds(genome)
# ============================================================= #
# ============================================================= #

