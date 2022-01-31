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
    ReservedBlock
)
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
        dt
    ):
        super().__init__(
            name=name,
            program=program,
            target=target,
            constraints=constraints,
            duration=duration
        )

        self.dt = dt
        self.startIdx = None
        self.time_min = None
        self.time_max = None
        self.nSlots = int(np.ceil(self.duration/self.dt))


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
            for constraint in self.constraints:
                scores.append(constraint.get_score(self.indices))
            return np.nanmean(scores, axis=0)


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
    def evaluateScore(self, time):
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
                nslots=self.nSlots
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
                    dt=self.dt 
                )
            sb.blockIdx = self._idxCounter
            self.blocks.append(sb)
            self._nSlots.append(sb.nSlots)
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
        time_min, #tMin
        time_max, #tMax,
        dt=TimeDelta(3600, format='sec')
    ):
        if not (_isTime(time_min, time_max) and _isTDelta(dt)):
            raise TypeError(
                f'Wrong types in `_TimeSlots({type(time_min)}, '
                f'{type(time_max)}, {type(dt)})`.'
            )

        self.dt = dt
        
        # Compute the time slots
        self.starts, self.stops = self._computeTimeSlots(
            time_min=time_min,
            time_max=time_max,
            dt=dt
        )
        self._startsJD = self.starts.jd
        self._stopsJD = self.stops.jd

        self.size = self.starts.size

        # Initialize array of free time slots
        self.freeSlots = np.ones(self.size, dtype=bool)
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
        """
        """
        if not _isTime(*time):
            raise TypeError(
                f'<time2idx> expects a {Time} object.'
            )

        if len(time) == 1:
            # Find the unique corresponding slot index
            mask = (self._startsJD <= time.jd) &\
                (self._stopsJD > time.jd)
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


    def addBooking(self, time_min, time_max):
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


    def delBooking(self, time_min, time_max):
        """
        """
        indices = self.time2idx(time_min, time_max)
        self.freeSlots[indices] = True
        self._freeIndices = self.idxSlots[self.freeSlots]


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _computeTimeSlots(time_min, time_max, dt):
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

            ~Schedule.insert
            ~Schedule.plot
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
                >>> from nenupy.schedule import Schedule, ESTarget, ObsBlock
                >>> from astropy.time import Time, TimeDelta
                >>> schedule = Schedule(
                >>>     time_min=Time('2021-01-11 00:00:00'),
                >>>     time_max=Time('2021-01-15 00:00:00'),
                >>>     dt=TimeDelta(3600, format='sec')
                >>> )
                >>> cas_a = ObsBlock(
                >>>     name="Cas A",
                >>>     program="ES00",
                >>>     target=ESTarget.fromName("Cas A"),
                >>>     duration=TimeDelta(2*3600, format='sec'),
                >>> )
                >>> schedule.insert(cas_a)

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
                    self.addBooking(blk.time_min, blk.time_max)
                if self.reserved_blocks is None:
                    self.reserved_blocks = blocks_i
                else:
                    self.reserved_blocks += blocks_i
            else:
                raise Exception(
                    f'Not supposed to happen for type {blocks_i.__class__}!'
                )

        self._toSchedule = np.ones(self.observation_blocks.size, dtype=bool)


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
        self._evaluateCnst(**kwargs)

        nBlocksUnScheduled = 0

        if sum(self._toSchedule) == 0:
            log.warning(
                'Required observation blocks have constraints '
                'unfitted for this schedule.'
            )
        else:
            log.info(
                f'Fitting {sum(self._toSchedule)} observation blocks...'
            )

        if optimize:
            # All the observation blocks are booked at
            # once with the genetic agorithm.
            ga = GeneticAlgorithm(
                populate=self._populate,
                fitness=self._fitness,
                mutation=self._mutation,
                populationSize=kwargs.get('population_size', 20)
            )

            ga.evolve(
                **kwargs
            )

            k = 0
            for i, blk in enumerate(self.observation_blocks):
                if not self._toSchedule[i]:
                    continue
                bestStartIdx = ga.bestGenome[k]
                bestStopIdx = bestStartIdx + blk.nSlots - 1
                k += 1
                if all(self._cnstScores[i, bestStartIdx:bestStopIdx + 1]==0):
                    nBlocksUnScheduled += 1
                    log.warning(
                        f"<ObsBlock> #{blk.blockIdx} '{blk.name}' cannot be scheduled."
                    )
                    continue
                blk.startIdx = bestStartIdx
                blk.time_min = self.starts[bestStartIdx]
                blk.time_max = self.stops[bestStartIdx + blk.nSlots-1]

            log.info(
                f'{sum(self._toSchedule) - nBlocksUnScheduled}/'
                f'{sum(self._toSchedule)} observation blocks scheduled '
                f'({sum(~self._toSchedule)} impossible to fit).'
            )

            return ga

        else:
            # Block are booked iteratively 
            for i, blk in enumerate(self.observation_blocks):
                if not self._toSchedule[i]:
                    continue
                # Construct a mask to avoid setting the obsblock where it cannot fit
                freeSlotsShifted = self.freeSlots.copy()
                for j in range(blk.nSlots):
                    freeSlotsShifted *= np.roll(self.freeSlots, -j)
                freeSlotsShifted[-blk.nSlots:] = self.freeSlots[-blk.nSlots:]
                # Find the best spot
                score = self._cnstScores[i, :] * self.freeSlots * freeSlotsShifted
                if all(score==0):
                    nBlocksUnScheduled += 1
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
                blk.time_min = self.starts[bestStartIdx]
                blk.time_max = self.stops[bestStopIdx]

            log.info(
                f'{sum(self._toSchedule) - nBlocksUnScheduled}/'
                f'{sum(self._toSchedule)} observation blocks scheduled '
                f'({sum(~self._toSchedule)} impossible to fit).'
            )


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
        startArray = Time(starts, format='isot')
        chron = np.argsort(startArray)

        tab = Table()
        tab['obsid'] = np.array(index, dtype=int)[chron]
        tab['name'] = np.array(names, dtype=str)[chron]
        tab['program'] = np.array(programs, dtype=str)[chron]
        tab['start'] = startArray[chron]
        tab['stop'] = Time(stops, format='isot')[chron]
        tab['score'] = np.array(scores)[chron]

        return tab


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
        nSubPlots = int(np.ceil((time_max - time_min)/days_per_line))

        # Initialize the figure
        fig, axs = plt.subplots(
            nrows=nSubPlots,
            ncols=1,
            figsize=kwargs.get('figsize', (15, 3*nSubPlots))
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
                for slotStart in self.starts[tMask]:
                    ax.axvline(
                        slotStart.datetime,
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
    def _evaluateCnst(self, **kwargs):
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
        for i, blk in enumerate(self.observation_blocks):
            if blk.constraints.score is None:
                # If != 1, the constraint score has already been computed
                # Append the last time slot to get the last slot top
                blk.evaluateScore(
                    time=Time(
                        np.append(
                            self._startsJD,
                            self._startsJD[-1] + self.dt.jd
                        ),
                        format='jd'
                    )
                )

            # Set other attributes relying on the schedule
            blk.nSlots = int(np.ceil(blk.duration/self.dt))

            # Get the constraint and maximum possible score
            slidingIdx = np.arange(blk.nSlots)[None, :] +\
                np.arange(self.size - blk.nSlots)[:, None]
            self._cnstScores[i, :-blk.nSlots] = np.mean(
                blk.constraints.score[slidingIdx],
                axis=1
            )
            # Set constraint score to zero if forbidden index
            forbiddenMask = np.any(~self.freeSlots[slidingIdx], axis=1)
            self._cnstScores[i, :-blk.nSlots][forbiddenMask] *= 0
            self._maxScores[i] = np.max(self._cnstScores[i, :])

            # Check that the constraints are non zero
            if self._maxScores[i] == 0:
                self._toSchedule[i] = False
                log.info(
                    f"<ObsBlock> #{blk.blockIdx} '{blk.name}'"
                    " has a null score over the schedule."
                )

        log.info(
            f'{self.observation_blocks.size} observation blocks have been '
            'successfully evaluated.'
        )        


    def _bounds(self, indices):
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
        nBlocks = np.sum(self._toSchedule)
        # population = np.zeros(
        #     (n, self.observation_blocks.size),
        #     dtype=int
        # )
        population = np.zeros(
            (n, nBlocks),
            dtype=int
        )
        # for i in range(self.observation_blocks.size):
        for i in range(nBlocks):
            population[:, i] = randGen.choice(
                np.where(self._cnstScores[self._toSchedule][i] != 0)[0],
                # replace=False,
                size=n,
            )

        return self._bounds(population)


    def _fitness(self, population):
        """
            population : np.array((children, genome))
        """
        # Evaluate the fitness of all the observation blocks over
        # the cumulative (on nSlots) constraint scores and noormalize
        # by the maximum available score within the schedule
        fitness = np.mean(
            np.diagonal(
                self._cnstScores[self._toSchedule].T[population],
                axis1=1,
                axis2=2
            )/self._maxScores[self._toSchedule],
            axis=1
        )
        # Find out which inidicual in the population contains blocks
        # that overlaps with one another.
        # Sort the population block indices because we don't care
        # about the last one 
        sortedIdx = np.argsort(population)
        overlaps = np.any(
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
        # is set to 0 whenever there are overlapping blocks.
        return fitness * ~overlaps


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
        return self._bounds(genome)
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

