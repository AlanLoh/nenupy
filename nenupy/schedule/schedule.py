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


randGen = np.random.default_rng()
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
                f'<time2idx> expects an {type(astropy.Time)} object.'
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


    def delBooking(self, tMin, tMax):
        """
        """
        indices = self.time2idx(tMin, tMax)
        self.freeSlots[indices] = True


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

        #self._blkIndices = None
        self.obsBlocks = None
        self.reservedBlocks = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def obsBlocks(self):
        """
        """
        return self._obsBlocks
    @obsBlocks.setter
    def obsBlocks(self, obs):
        #self._blkIndices = np.zeros(obs.size, dtype=int)
        self._obsBlocks = obs
        

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def insert(self, *blocks):
        """
        """
        for blocks_i in blocks:
            if not isinstance(blocks_i, Block):
                raise TypeError(
                    f'Expected input of type <{Block.__class__}>'
                    f', got <{blocks_i.__class__}> instead.'
                )
            elif all([blk.__class__ is ObsBlock for blk in blocks_i]):
                if self.obsBlocks is None:
                    self.obsBlocks = blocks_i
                else:
                    self.obsBlocks += blocks_i
            elif all([blk.__class__ is ReservedBlock for blk in blocks_i]):
                for blk in blocks_i:
                    self.addBooking(blk.tMin, blk.tMax)
                if self.reservedBlocks is None:
                    self.reservedBlocks = blocks_i
                else:
                    self.reservedBlocks += blocks_i
            else:
                raise Exception(
                    f'Not supposed to happen for type {type(blocks_i)}!'
                )


    def book(self, optimize=True, **kwargs):
        """
            kwargs
                nChildren
                scoreMin
                maxGen

        """
        self._evaluateCnst(**kwargs)

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

            for i, blk in enumerate(self.obsBlocks):
                bestStartIdx = ga._bestGenome[i]
                bestStopIdx = bestStartIdx + blk.nSlots - 1
                if all(self._cnstScores[i, bestStartIdx:bestStopIdx + 1]==0):
                    log.warning(
                        f"<ObsBlock> #{i} '{blk.name}' cannot be scheduled."
                    )
                    continue
                blk.startIdx = bestStartIdx
                blk.isBooked = True
                blk.tMin = self.starts[bestStartIdx]
                blk.tMax = self.stops[bestStartIdx + blk.nSlots-1]

            return ga

        else:
            # Block are booked iteratively 
            for i, blk in enumerate(self.obsBlocks):
                # Construct a mask to avoid setting the obsblock where it cannot fit
                freeSlotsShifted = self.freeSlots.copy()
                for j in range(blk.nSlots):
                    freeSlotsShifted *= np.roll(self.freeSlots, -j)
                freeSlotsShifted[-blk.nSlots:] = self.freeSlots[-blk.nSlots:]
                # Find the best spot
                score = self._cnstScores[i, :] * self.freeSlots * freeSlotsShifted
                if all(score==0):
                    log.warning(
                        f"<ObsBlock> #{i} '{blk.name}' cannot be scheduled."
                    )
                    continue
                bestStartIdx = np.argmax(
                    score
                )
                # Assign a start and stop time to the block and update the free slots
                blk.startIdx = bestStartIdx
                blk.isBooked = True
                bestStopIdx = bestStartIdx + blk.nSlots - 1
                self.freeSlots[bestStartIdx:bestStopIdx + 1] = False
                blk.tMin = self.starts[bestStartIdx]
                blk.tMax = self.stops[bestStopIdx]


    def export(self, scoreMin=0):
        """
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
            starts.append(blk.tMin.isot)
            stops.append(blk.tMax.isot)
            programs.append(blk.program)

        tab = Table()
        tab['name'] = np.array(names, dtype=str)
        tab['program'] = np.array(programs, dtype=str)
        tab['start'] = Time(starts)
        tab['stop'] = Time(stops)
        tab['score'] = np.array(scores)

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
            (self.obsBlocks.size, self.idxSlots.size)
        )
        self._nSlotsArr = np.ones(self.obsBlocks.size, dtype=int)
        self._maxScores = np.zeros(self.obsBlocks.size)

        # Compute the constraint score over the schedule if it
        # has not been previously been done
        for i, blk in enumerate(self.obsBlocks):
            if blk.constraints.score is 1:
                # If != 1, te constraint score has already been computed
                # Append the last time slot to get the last slot top
                blk.evaluateScore(
                    time=Time(
                        np.append(
                            self._startsJD,
                            self._startsJD[-1] + self.dt.jd
                        ),
                        format='jd'
                    ),
                    **kwargs
                )
            # Set other attributes relying on the schedule
            blk.nSlots = int(np.ceil(blk.duration/self.dt))
            self._nSlotsArr[i] = blk.nSlots

            # Get the constraint and maximum possible score
            slidingIdx = np.arange(blk.nSlots)[None, :] +\
                np.arange(self.idxSlots.size - blk.nSlots)[:, None]
            self._cnstScores[i, :-blk.nSlots] = np.mean(
                blk.constraints.score[slidingIdx],
                axis=1
            )
            # Set constraint score to zero if forbidden index
            forbiddenMask = np.any(~self.freeSlots[slidingIdx], axis=1)
            self._cnstScores[i, :-blk.nSlots][forbiddenMask] *= 0
            self._maxScores[i] = np.max(self._cnstScores[i, :])
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
        beyond = (indices + self._nSlotsArr) >= self.idxSlots.size
        indices[beyond] = self.idxSlots.size - 1
        indices -= beyond*self._nSlotsArr
        return indices


    def _populate(self, n):
        """
        """
        return self._bounds(
            randGen.integers(
                low=0,
                high=self.idxSlots.size,
                size=(n, self.obsBlocks.size),
                dtype=np.int64
            )
        )


    def _fitness(self, population):
        """
            population : np.array((children, genome))
        """
        # Evaluate the fitness of all the observation blocks over
        # the cumulative (on nSlots) constraint scores and noormalize
        # by the maximum available score within the schedule
        fitness = np.mean(
            np.diagonal(
                self._cnstScores.T[population],
                axis1=1,
                axis2=2
            )/self._maxScores,
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
            ) - self._nSlotsArr[sortedIdx][:, :-1] < 0,
            axis=1
        )
        # The fitness is the product of the constraint score and
        # is set to 0 whenever there are overlapping blocks.
        return fitness * ~overlaps


    def _mutation(self, genome):
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
# ============================================================= #
# ============================================================= #

