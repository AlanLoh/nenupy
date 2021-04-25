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
from copy import deepcopy

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
        self.tMin = None
        self.tMax = None
        self.nSlots = int(np.ceil(self.duration/self.dt))


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
                scores.append(constraint.getScore(self.indices))
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
        if self.target._lst is None:
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
                figName
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
        dt = (self.tMax - self.tMin)/nPoints
        times = self.tMin + np.arange(nPoints + 1)*dt
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
            self.tMin.datetime,
            color='black',
            linestyle='-.'
        )
        ax1.axvline(
            self.tMax.datetime,
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
    def _display(self, ax):
        """
        """
        import matplotlib.dates as mdates

        if self.tMin is None:
            return

        # Show the block rectangle
        ax.axvspan(
            self.tMin.datetime,
            self.tMax.datetime,
            facecolor=self.kpColor,
            edgecolor='black',
            alpha=0.6
        )

        # Indicate the status
        ax.axvspan(
            self.tMin.datetime,
            self.tMax.datetime,
            ymin=0.9,
            facecolor=self.statusColor,
            edgecolor='black',
        )
        ax.axvspan(
            self.tMin.datetime,
            self.tMax.datetime,
            ymax=0.1,
            facecolor=self.statusColor,
            edgecolor='black',
        )

        # Show the observation block title
        xMin, xMax = ax.get_xlim()
        textPos = (self.tMin + (self.tMax - self.tMin)/2)
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
        tMin,
        tMax,
        dt=TimeDelta(3600, format='sec')
    ):
        if not (_isTime(tMin, tMax) and _isTDelta(dt)):
            raise TypeError(
                f'Wrong types in `_TimeSlots({type(tMin)}, '
                f'{type(tMax)}, {type(dt)})`.'
            )

        self.dt = dt
        
        # Compute the time slots
        self.starts, self.stops = self._computeTimeSlots(
            tMin=tMin,
            tMax=tMax,
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
    def tMin(self):
        """
        """
        return self.starts[0]


    @property
    def tMax(self):
        """
        """
        return self.stops[-1]


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


    def addBooking(self, tMin, tMax):
        """ Changes the status of the time slots comprised
            between ``tMin`` and ``tMax`` to 'booked' (i.e., not
            available anymore).
            In particular, the attribute 
            :attr:`~nenupy.schedule._TimeSlots.freeSlots` is set
            to ``False`` at the corresponding indices.

            :param tMin:
                Start time.
            :type tMin:
                :class:`~astropy.time.Time`
            :param tMax:
                Stop time.
            :type tMax:
                :class:`~astropy.time.Time`
        """
        indices = self.time2idx(tMin, tMax)
        if any(~self.freeSlots[indices]):
            log.warning(
                f"Booking on reserved slots from '{tMin.isot}' "
                f"to '{tMax.isot}'."
            )
        self.freeSlots[indices] = False
        self._freeIndices = self.idxSlots[self.freeSlots]


    def delBooking(self, tMin, tMax):
        """
        """
        indices = self.time2idx(tMin, tMax)
        self.freeSlots[indices] = True
        self._freeIndices = self.idxSlots[self.freeSlots]


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _computeTimeSlots(tMin, tMax, dt):
        """
        """
        if tMax <= tMin:
            raise ValueError(
                f'Schedule tMax={tMax.isot} <= '
                f'tMin={tMin.isot}.'
            )

        period = tMax - tMin
        timeSteps = int( np.ceil(period/dt) )
        dtShifts = np.arange(timeSteps)*dt

        slotStarts = tMin + dtShifts
        slotStops = slotStarts + dt

        return slotStarts, slotStops
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- Schedule -------------------------- #
# ============================================================= #
class Schedule(_TimeSlots):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        tMin,
        tMax,
        dt=TimeDelta(3600, format='sec')
    ):
        super().__init__(
            tMin=tMin,
            tMax=tMax,
            dt=dt
        )

        # self.obsBlocks = None
        self.obsBlocks = ScheduleBlocks(dt=self.dt)
        self.reservedBlocks = None


    def __getitem__(self, n):
        """
        """
        return self.obsBlocks[n]

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    # @property
    # def obsBlocks(self):
    #     """
    #     """
    #     return self._obsBlocks
    # @obsBlocks.setter
    # def obsBlocks(self, obs):
    #     #self._blkIndices = np.zeros(obs.size, dtype=int)
    #     self._obsBlocks = obs


    # @property
    # def scheduledBlocks(self):
    #     """
    #     """
    # list(map(schedule.obsBlocks.__getitem__, [i for i, blk in enumerate(schedule.obsBlocks) if blk.isBooked]))
    #     return self._scheduledBlocks
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def insert(self, *blocks):
        """
        """
        for blocks_i in blocks:
            if not isinstance(blocks_i, Block):
                raise TypeError(
                    f'Expected input of type {Block}>'
                    f', got <{blocks_i.__class__}> instead.'
                )
            elif all([blk.__class__ is ObsBlock for blk in blocks_i]):
                self.obsBlocks.insert(blocks_i)
            elif all([blk.__class__ is ReservedBlock for blk in blocks_i]):
                for blk in blocks_i:
                    self.addBooking(blk.tMin, blk.tMax)
                if self.reservedBlocks is None:
                    self.reservedBlocks = blocks_i
                else:
                    self.reservedBlocks += blocks_i
            else:
                raise Exception(
                    f'Not supposed to happen for type {blocks_i.__class__}!'
                )

        self._toSchedule = np.ones(self.obsBlocks.size, dtype=bool)


    def book(self, optimize=True, **kwargs):
        """
            kwargs
                nChildren
                scoreMin
                maxGen

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
                populationSize=kwargs.get('nChildren', 20)
            )

            ga.evolve(
                **kwargs
            )

            k = 0
            for i, blk in enumerate(self.obsBlocks):
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
                blk.tMin = self.starts[bestStartIdx]
                blk.tMax = self.stops[bestStartIdx + blk.nSlots-1]

            log.info(
                f'{sum(self._toSchedule) - nBlocksUnScheduled}/'
                f'{sum(self._toSchedule)} observation blocks scheduled '
                f'({sum(~self._toSchedule)} impossible to fit).'
            )

            return ga

        else:
            # Block are booked iteratively 
            for i, blk in enumerate(self.obsBlocks):
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
                blk.tMin = self.starts[bestStartIdx]
                blk.tMax = self.stops[bestStopIdx]

            log.info(
                f'{sum(self._toSchedule) - nBlocksUnScheduled}/'
                f'{sum(self._toSchedule)} observation blocks scheduled '
                f'({sum(~self._toSchedule)} impossible to fit).'
            )


    def export(self, scoreMin=0):
        """ https://docs.astropy.org/en/stable/io/ascii/index.html#supported-formats
        """
        if not isinstance(scoreMin, (float, int)):
            raise TypeError(
                '<scoreMin> should be a number.'
            )
        elif scoreMin < 0:
            raise ValueError(
                '<scoreMin> should be a positive.'
            )
        elif scoreMin > 1:
            raise ValueError(
                '<scoreMin> cannot be greater than 1.'
            )

        names = []
        index = []
        starts = []
        stops = []
        programs = []
        scores = []
        for blk in self.obsBlocks:
            if not blk.isBooked:
                continue
            score = blk.score
            if score < scoreMin:
                continue
            scores.append(score)
            names.append(blk.name)
            index.append(blk.blockIdx)
            starts.append(blk.tMin.isot)
            stops.append(blk.tMax.isot)
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


    def plot(self, daysPerSubplot=1, **kwargs):
        """
            kwargs:
                figName
                figsize
                grid
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Set relevant variables for the plot
        tMin = Time(self.starts[0].isot.split('T')[0])
        tMax = self.stops[-1]

        # Find the number of sub-plots to display the schedule
        if not isinstance(daysPerSubplot, TimeDelta):
            daysPerSubplot = TimeDelta(
                daysPerSubplot,
                format='jd'
            )
        nSubPlots = int(np.ceil((tMax - tMin)/daysPerSubplot))

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
            tiMin = tMin + i*daysPerSubplot
            tiMax = tMin + (i + 1)*daysPerSubplot
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
            if self.reservedBlocks is not None:
                for bookedBlock in self.reservedBlocks:
                    bookedBlock._display(ax=ax)

            # Display observation blocks
            if self.obsBlocks is not None:
                for obsBlock in self.obsBlocks:
                    obsBlock._display(ax=ax)

            # Formating
            ax.yaxis.set_visible(False)
            h_fmt = mdates.DateFormatter('%y-%m-%d\n%H')
            ax.xaxis.set_major_formatter(h_fmt)

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
    def _evaluateCnst(self, **kwargs):
        """
        """
        log.info(
            'Evaluating observation block constraints over '
            'the schedule...'
        )
        self._cnstScores = np.zeros(
            (self.obsBlocks.size, self.size)
        )
        self._maxScores = np.zeros(self.obsBlocks.size)

        # Compute the constraint score over the schedule if it
        # has not been previously been done
        for i, blk in enumerate(self.obsBlocks):
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
            f'{self.obsBlocks.size} observation blocks have been '
            'successfully evaluated.'
        )        


    def _bounds(self, indices):
        """
            startIndices = (schedule_size)
            
            return startIndices = (schedule_size)
        """
        indices[indices < 0] = 0
        beyond = (indices + self.obsBlocks.nSlots[self._toSchedule]) >= self.idxSlots.size
        indices[beyond] = self.idxSlots.size - 1
        indices -= beyond*self.obsBlocks.nSlots[self._toSchedule]
        return indices


    def _populate(self, n):
        """
        """
        # return self._bounds(
        #     randGen.integers(
        #         low=0,
        #         high=self.idxSlots.size,
        #         size=(n, self.obsBlocks.size),
        #         dtype=np.int64
        #     )
        # )
        nBlocks = np.sum(self._toSchedule)
        # population = np.zeros(
        #     (n, self.obsBlocks.size),
        #     dtype=int
        # )
        population = np.zeros(
            (n, nBlocks),
            dtype=int
        )
        # for i in range(self.obsBlocks.size):
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
            ) - self.obsBlocks.nSlots[self._toSchedule][sortedIdx][:, :-1] < 0,
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

