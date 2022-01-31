#! /usr/bin/python3
# -*- coding: utf-8 -*-


r"""
    .. _schedule_constraints:

    ***********************
    Observation constraints
    ***********************

    
    In the following, an instance of :class:`~nenupy.schedule.targets.ESTarget`
    is initialized for the source *Cygnus A* and stored in the variable
    ``target``. The method :meth:`~nenupy.schedule.targets._Target.computePosition`
    is called to compute all astronomical properties at the
    location of `NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_
    for the time-range ``times``.

    .. code-block:: python
        :emphasize-lines: 3,8,9

        >>> from astropy.time import Time, TimeDelta
        >>> import numpy as np
        >>> from nenupy.schedule import ESTarget

        >>> dt = TimeDelta(3600, format='sec')
        >>> times = Time('2021-01-01 00:00:00') + np.arange(24)*dt

        >>> target = ESTarget.fromName('Cas A')
        >>> target.computePosition(times)


    .. seealso::
        :class:`~nenupy.schedule.targets.ESTarget` or 
        :class:`~nenupy.schedule.targets.ESTarget` objects are
        described in :ref:`schedule_targets` in more details.


    Single constraint
    -----------------

    There are several constraints that could be defined (see
    :ref:`constraints_summary`). The simplest and probably most
    basic one is the 'elevation constraint' (embedded in the 
    :class:`~nenupy.schedule.constraints.ElevationCnst` class) since
    observing at :math:`e > 0^\circ` is a necessary requirement
    for a ground-based observatory.
    Once the constraint has been evaluated on ``target``,
    a normalized 'score' is returned and can be plotted using the
    :meth:`~nenupy.schedule.constraints.Constraint.plot` method.
    
    .. code-block:: python

        >>> from nenupy.schedule import ElevationCnst

        >>> c = ElevationCnst()
        >>> score = c(target)
        >>> c.plot()


    .. image:: ./_images/elevconstraint_casa0deg.png
        :width: 800

    If the required elevation to perform a good-quality observation
    needs to be greater than a given value, it could be specified
    to the :attr:`~nenupy.schedule.constraints.ElevationCnst.elevationMin`
    attribute.

    .. code-block:: python

        >>> c = ElevationCnst(elevationMin=40)
        >>> score = c(target)
        >>> c.plot()


    .. image:: ./_images/elevconstraint_casa40deg.png
        :width: 800

    The :class:`~nenupy.schedule.constraints.ElevationCnst`'s score is
    now at :math:`0` whenever the ``target`` elevation is lower than
    :attr:`~nenupy.schedule.constraints.ElevationCnst.elevationMin`.


    Multiple constraints
    --------------------
    
    Several constraints are often needed to select an appropriate
    time window for a given observation. The
    :class:`~nenupy.schedule.constraints.Constraints` class is by
    default initialized with ``ElevationCnst(elevationMin=0)``, but
    any other constraint may be passed as arguments:
    
    .. code-block:: python

        >>> from nenupy.schedule import (
                ESTarget,
                Constraints,
                ElevationCnst,
                MeridianTransitCnst,
                LocalTimeCnst
            )
        >>> from astropy.time import Time, TimeDelta
        >>> from astropy.coordinates import SkyCoord, Angle

        >>> dts = np.arange(24+1)*TimeDelta(3600, format='sec')
        >>> times = Time('2021-01-01 00:00:00') + dts
        >>> target = ESTarget.fromName('Cas A')
        >>> target.computePosition(times)

        >>> cnst = Constraints(
                ElevationCnst(elevationMin=20, weight=3),
                MeridianTransitCnst(),
                LocalTimeCnst(Angle(12, 'hour'), Angle(4, 'hour'))
            )
        
        >>> cnst.evaluate(target, times)
        
        >>> cnst.plot()

    .. image:: ./_images/sch_constraints.png
        :width: 800

    .. _constraints_summary:

    Constraint classes
    ------------------

    .. autosummary::

        ~nenupy.schedule.constraints.Constraints
        ~nenupy.schedule.constraints.ElevationCnst
        ~nenupy.schedule.constraints.MeridianTransitCnst
        ~nenupy.schedule.constraints.AzimuthCnst
        ~nenupy.schedule.constraints.LocalTimeCnst
        ~nenupy.schedule.constraints.TimeRangeCnst


    .. inheritance-diagram:: nenupy.schedule.constraints
        :parts: 3


"""

# TO DO : CENTRER SUR L AZIMUTH / transit au meridien

__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Constraint',
    'TargetConstraint',
    'ScheduleConstraint',
    'ElevationCnst',
    'MeridianTransitCnst',
    'AzimuthCnst',
    'LocalTimeCnst',
    'TimeRangeCnst',
    'Constraints'
]


from abc import ABC, abstractmethod
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle
import pytz
import matplotlib.pyplot as plt

from nenupy.schedule.targets import _Target

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------------ Constraint ------------------------- #
# ============================================================= #
class Constraint(ABC):
    """ Base class for all the constraint definitions.

        .. versionadded:: 1.2.0
    """

    def __init__(self, weight=1):
        self.score = 1
        self.weight = weight


    def __call__(self, *arg):
        """ Test de docstring
        """
        if not hasattr(self, '_evaluate'):
            raise AttributeError(
                '<Constraint> should not be used on its own.'
            )
        return self._evaluate(*arg)

    
    def __str__(self):
        """
        """
        return f'{self.__class__}'


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def weight(self):
        """
        """
        return self._weight
    @weight.setter
    def weight(self, w):
        if not isinstance(w, (float, int)):
            raise TypeError(
                '<weight> should be a number.'
            )
        elif w <= 0:
            raise ValueError(
                '<weight> should be > 0.'
            )
        self._weight = w
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, **kwargs):
        """ Plots the constraint's score previously evaluated.

            :param figsize:
                Size of the figure. Default: ``(10, 5)``.
            :type figsize:
                `tuple`
            :param figname:
                Name of the figure to be stored. Default: ``''``,
                the figure is only displayed.
            :type figname:
                `str`
            :param marker:
                Plot marker type (see :func:`matplotlib.pyplot.plot`).
                Default: ``'.'``.
            :type marker:
                `str`
            :param linestyle:
                Plot line style (see :func:`matplotlib.pyplot.plot`).
                Default: ``':'``
            :type linestyle:
                `str`
            :param linewidth:
                Plot line width (see :func:`matplotlib.pyplot.plot`).
                Default: ``1``
            :type linewidth:
                `int` or `float`
        """
        fig = plt.figure(
            figsize=kwargs.get('figsize', (10, 5))
        )
        self._plot_constraint(**kwargs)
        plt.xlabel('Time index')
        plt.ylabel('Constraint score')
        plt.title(f'{self.__class__}')

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
    def _plot_constraint(self, **kwargs):
        """ Internal method to plot a single constraint without
            initializing the figure object.
        """
        plt.plot(
            np.where(
                np.isnan(self.score),
                0,
                self.score
            ),
            marker=kwargs.get('marker', '.'),
            linestyle=kwargs.get('linestyle', ':'),
            linewidth=kwargs.get('linewidth', 1),
            label=f'{self.__class__}'
        )


    @staticmethod
    def _is_numpy_instance(arr):
        """ Check that arr is a genuine numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f'{np.ndarray} object expected.'
            )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- TargetConstraint ---------------------- #
# ============================================================= #
class TargetConstraint(Constraint):
    """ Base class for constraints involving target propertiy checks.

        .. warning::
            :class:`~nenupy.schedule.constraints.TargetConstraint`
            should not be used on its own.

        .. versionadded:: 1.2.0
    """

    def __init__(self, weight):
        super().__init__(weight=weight)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _is_target_instance(target):
        """
        """
        if not isinstance(target, _Target):
            raise TypeError(
                f'{_Target} object expected.'
            )


    @staticmethod
    def _pass_angle_instance(angle):
        """
        """
        if not isinstance(angle, Angle):
            angle = Angle(angle, unit='deg')
        return angle
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- ScheduleConstraint --------------------- #
# ============================================================= #
class ScheduleConstraint(Constraint):
    """ Base class for constraints involving time range checks.

        .. warning::
            :class:`~nenupy.schedule.constraints.ScheduleConstraint`
            should not be used on its own.

        .. versionadded:: 1.2.0
    """

    def __init__(self, weight):
        super().__init__(weight=weight)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _is_time_instance(time):
        """
        """
        if not isinstance(time, Time):
            raise TypeError(
                f'{Time.__class__} object expected.'
            )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ElevationCnst ----------------------- #
# ============================================================= #
class ElevationCnst(TargetConstraint):
    """ Elevation constraint

        :param elevationMin:
            Target's elevation below which the constraint score
            is null. If provided as a dimensionless quantity,
            the value is interpreted as degrees.
        :type elevationMin:
            `int`, `float`, or :class:`~astropy.coordinates.Angle`
        :param weight:
            Weight of the constraint. Allows to ponderate each
            constraint with respect to each other if
            :class:`~nenupy.schedule.constraint.ElevationCnst`
            is included in :class:`~nenupy.schedule.constraint.Constraints`
            for instance.
        :type weight:
            `int` or `float`

        .. versionadded:: 1.2.0

        :Example:
            >>> from astropy.time import Time, TimeDelta
            >>> from nenupy.schedule.targets import ESTarget
            >>> from nenupy.schedule.constraints import ElevationCnst
            >>> dt = TimeDelta(3600, format='sec')
            >>> times = Time('2021-01-01 00:00:00') + np.arange(24)*dt
            >>> cas_a = ESTarget.fromName('Cas A')
            >>> cas_a.computePosition(times)
            >>> elevation_constraint = ElevationCnst()
            >>> score = elevation_constraint(target, None)
            >>> c.plot()

    """

    def __init__(self, elevationMin=0., scale_elevation=True, weight=1):
        super().__init__(weight=weight)
        self.elevationMin = elevationMin
        self.scale_elevation = scale_elevation


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def elevationMin(self):
        """ Minimal elevation required to perform an observation.

            :type: `float` or :class:`~astropy.coordinates.Angle`
        """
        return self._elevationMin
    @elevationMin.setter
    def elevationMin(self, emin):
        emin = self._pass_angle_instance(emin)
        if (emin.deg < 0.) or (emin.deg > 90):
            raise ValueError(
                f'`elevationMin`={emin.deg} deg must fall '
                'between 0 and 90 degrees.'
            )
        self._elevationMin = emin


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get_score(self, indices):
        r""" Computes the :class:`~nenupy.schedule.constraint.ElevationCnst`'s
            score for the given ``indices``.

            The score is computed as:

            .. math::
                {\rm score} = \left\langle \frac{\mathbf{e}(t)}{{\rm max}(\mathbf{e})} \right\rangle_{\rm indices}
            
            where :math:`\mathbf{e}(t)` is the elevation of the
            target (set to :math:`0` whenever it is lower than
            :attr:`~nenupy.schedule.constraints.ElevationCnst.elevationMin`).

            :param indices:
                Indices of :class:`~nenupy.schedule.constraint.Constraint.score`
                on which the score will be evaluated.
            :type indices:
                :class:`~numpy.ndarray`

            :returns:
                Constraint score.
            :rtype: `float`
        """
        # aboveMin = self.score[indices] > 0
        # return np.mean(self.score[indices][aboveMin])
        # return np.mean(self.score[indices])
        # return np.mean(
        #     np.where(
        #         np.isnan(self.score[indices]),
        #         0,
        #         self.score[indices]
        #     )
        # )
        return np.mean(
            np.where(
                self.score[indices]>0.,
                1,
                0
            )
        )


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _evaluate(self, target, nslots):
        """ Evaluates the constraint :class:`~nenupy.schedule.constraint.ElevationCnst`
            on the ``target`` which astronomical positions need
            to be computed first (using :meth:`~nenupy.schedule.targets._Target.computePosition`).

            :param target:
                Target for which :class:`~nenupy.schedule.constraint.ElevationCnst`
                should be evaluated.
            :type target: :class:`~nenupy.schedule.targets._Target`

            :returns:
                Constraint's score.
            :rtype: :class:`~numpy.ndarray`
        """
        self._is_target_instance(target)

        elevation = target.elevation.deg
        elevMean = (elevation[1:] + elevation[:-1])/2
        # elevMean[elevMean <= self.elevationMin.deg] = 0.
        elevMean[elevMean <= self.elevationMin.deg] = np.nan
        # if elevMax == 0.:
        if all(np.isnan(elevMean)):
            log.warning(
                "Constraint <ElevationConstraint(elevationMin="
                f"{self.elevationMin})> evaluated for target "
                f"'{target.target}' cannot be satisfied over the "
                "given time range."
            )
            # self.score = elevation[1:]*0.
            self.score = elevation[1:]*np.nan
        elif not self.scale_elevation:
            self.score = elevMean/elevMean
        else:
            elevMax = np.nanmax(elevMean)
            self.score = elevMean/elevMax
        return self.score
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- MeridianTransitCnst -------------------- #
# ============================================================= #
class MeridianTransitCnst(TargetConstraint):
    """ Meridian Transit constraint

        .. versionadded:: 1.2.0
    """

    def __init__(self, weight=1):
        super().__init__(weight=weight)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get_score(self, indices):
        r""" Computes the :class:`~nenupy.schedule.constraint.MeridianTransitCnst`'s
            score for the given ``indices``.

            Returns 1 if the merdian transit is within the indices

            The score is computed as:

            .. math::
                {\rm score} = \begin{cases}
                    1, t_{\rm transit} \in \mathbf{t}({\rm indices})\\
                    0, t_{\rm transit} \notin \mathbf{t}({\rm indices})
                \end{cases}
            
            where :math:`t_{\rm transit}` is the meridian transit time
            and :math:`\mathbf{t}` is the time range on which the
            target positions are computed.

            :param indices:
                Indices of :class:`~nenupy.schedule.constraint.Constraint.score`
                on which the score will be evaluated.
            :type indices:
                :class:`~numpy.ndarray`

            :returns:
                Constraint score.
            :rtype: `float`
        """
        self._is_numpy_instance(indices)
        #return np.sum(self.score[indices], axis=-1)
        return int((self.score[indices]>0.7).any())


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _evaluate(self, target, nslots):
        self._is_target_instance(target)

        hourAngle = target.hourAngle
        transitIdx = np.where(
            (np.roll(hourAngle, -1) - hourAngle)[:-1] < 0
        )[0]
        scores = np.zeros(hourAngle.size)
        # Set neighbor slots to non-zero to maximize centering
        slotShifts = np.arange(nslots) - int(np.floor(nslots/2))
        neighborIdx = (transitIdx[:, None] + slotShifts[None, :]).ravel()
        outOfBounds = (neighborIdx < 0) + (neighborIdx >= scores.size)
        neighborIdx = np.delete(
            neighborIdx,
            np.argwhere(outOfBounds)
        )
        scores = np.where(
            np.isin(
                np.arange(scores.size, dtype=int),
                neighborIdx
            ),
            0.5,
            scores
        )
        # Set transit slots to maximal score
        scores[transitIdx] = 1.
        self.score = scores[:-1]
        return self.score
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ AzimuthCnst ------------------------ #
# ============================================================= #
class AzimuthCnst(TargetConstraint):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, azimuth, weight=1):
        super().__init__(weight=weight)
        self.azimuth = azimuth


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def azimuth(self):
        """
        """
        return self._azimuth
    @azimuth.setter
    def azimuth(self, az):
        az = self._pass_angle_instance(az)
        if (az.deg < 0.) or (az.deg > 360):
            raise ValueError(
                f'`azimuth`={az.deg} deg must fall '
                'between 0 and 360 degrees.'
            )
        self._azimuth = az


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get_score(self, indices):
        """
        """
        self._is_numpy_instance(indices)
        # return np.sum(self.score[indices], axis=-1)
        return int((self.score[indices]>0.7).any())


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _evaluate(self, target, nslots):
        self._is_target_instance(target)
        
        azimuths = target.azimuth.rad
        az = self.azimuth.rad
        
        if target.isCircumpolar:
            az = np.angle(np.cos(az) + 1j*np.sin(az))
            complexAzStarts = np.angle(
                np.cos(azimuths[:-1]) + 1j*np.sin(azimuths[:-1])
            )
            complexAzStops = np.angle(
                np.cos(azimuths[1:]) + 1j*np.sin(azimuths[1:])
            )

            mask = (az >= complexAzStarts) &\
                (az <= complexAzStops)
            mask |= (az <= complexAzStarts) &\
                (az >= complexAzStops)
        else:
            mask = (az >= azimuths[:-1]) &\
                (az <= azimuths[1:])

        scores = np.zeros(mask.size)
        azIndices = np.where(mask)[0]
        # Set neighbor slots to non-zero to maximize centering
        slotShifts = np.arange(nslots) - int(np.floor(nslots/2))
        neighborIdx = (azIndices[:, None] + slotShifts[None, :]).ravel()
        outOfBounds = (neighborIdx < 0) + (neighborIdx >= scores.size)
        neighborIdx = np.delete(
            neighborIdx,
            np.argwhere(outOfBounds)
        )
        scores = np.where(
            np.isin(
                np.arange(scores.size, dtype=int),
                neighborIdx
            ),
            0.5,
            scores
        )

        # Set azimuth found slots to maximal score
        scores[azIndices] = 1.
        self.score = scores
        # self.score = mask.astype(float)
        return self.score
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- LocalTimeCnst ----------------------- #
# ============================================================= #
class LocalTimeCnst(ScheduleConstraint):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, hMin, hMax, weight=1):
        super().__init__(weight=weight)
        self.hMin = hMin
        self.hMax = hMax


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def hMin(self):
        """
        """
        return self._hMin
    @hMin.setter
    def hMin(self, h):
        if not isinstance(h, Angle):
            raise TypeError(
                f'{h} should be of type {type(Angle)}.'
            )
        self._hMin = h


    @property
    def hMax(self):
        """
        """
        return self._hMax
    @hMax.setter
    def hMax(self, h):
        if not isinstance(h, Angle):
            raise TypeError(
                f'{h} should be of type {type(Angle)}.'
            )
        self._hMax = h


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get_score(self, indices):
        """
        """
        self._is_numpy_instance(indices)
        return np.mean(self.score[indices], axis=-1)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _evaluate(self, time, nslots):
        """
        """
        self._is_time_instance(time)
        
        # Convert time to France local time (take into account
        # daylight savings)
        tz = pytz.timezone('Europe/Paris')
        timezoneTime = map(tz.localize, time.datetime)
        utcOffset = np.array(
            [tt.utcoffset().total_seconds() for tt in timezoneTime]
        )
        localTime = time + TimeDelta(utcOffset, format='sec')
        
        # Convert the 'hour' part in decimal 'angle' values
        hours = np.array([tt.split()[1] for tt in localTime.iso])
        localHours = Angle(hours, unit='hour').hour

        # Selection
        if self.hMin > self.hMax:
            # If 'midnight' is in the range
            mask = (localHours <= self.hMin.hour) &\
                (localHours >= self.hMax.hour)
            mask = ~mask
        else:
            mask = (localHours >= self.hMin.hour) &\
                (localHours <= self.hMax.hour)
        score = mask[:-1].astype(float)
        self.score = np.where(score==0, np.nan, score)
        return self.score
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- TimeRangeCnst ----------------------- #
# ============================================================= #
class TimeRangeCnst(ScheduleConstraint):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, time_min, time_max, weight=1):
        super().__init__(weight=weight)
        self.time_min = time_min
        self.time_max = time_max


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def time_min(self):
        """
        """
        return self._time_min
    @time_min.setter
    def time_min(self, t):
        if not isinstance(t, Time):
            raise TypeError(
                f'{t} should be of type {type(Time)}.'
            )
        self._time_min = t


    @property
    def time_max(self):
        """
        """
        return self._time_max
    @time_max.setter
    def time_max(self, t):
        if not isinstance(t, Time):
            raise TypeError(
                f'{t} should be of type {type(Time)}.'
            )
        self._time_max = t


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get_score(self, indices):
        """
        """
        self._is_numpy_instance(indices)
        return np.mean(self.score[indices], axis=-1)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _evaluate(self, time, nslots):
        """ time: dim + 1
        """
        self._is_time_instance(time)
        
        jds = time.jd
        
        if self.time_min.isscalar:
            mask = (jds >= self.time_min.jd) & (jds <= self.time_max.jd)
        else:
            mask = np.sum(
                (jds[:, None] >= self.time_min.jd) & (jds[:, None] <= self.time_max.jd),
                axis=1,
                dtype=bool
            )
        score = mask[:-1].astype(float)
        self.score = np.where(score==0, np.nan, score)
        return self.score
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ Constraints ------------------------ #
# ============================================================= #
class Constraints(object):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, *constraints):
        self._default_el = False
        self.constraints = constraints
        self.score = None
        
        # Check they are all of unique type
        unique, count = np.unique(
            [str(cons.__class__) for cons in self.constraints],
            return_counts=True
        )
        if any(count > 1):
            message = (
                'There can only be one constraint type per '
                'observation:'
            )
            for typ, n in zip(unique, count):
                if n > 1:
                    message += f"\n\t* '{typ}': {n} instances."
            raise ValueError(message)

        # Add the elevation constraint by default
        #if not str(ElevationCnst) in unique:
        if not np.any(np.isin(np.array([str(ElevationCnst)]), unique)):
            self.constraints += (ElevationCnst(0.),)
            self._default_el = True


    def __add__(self, other):
        """
        """
        if not isinstance(other, Constraint):
            raise TypeError('')

        if isinstance(other, ElevationCnst):
            if self._default_el:
                # Remove the default elevation constraint
                # if a new one is added
                cs = np.array([str(c.__class__) for c in self.constraints])
                elCnst_idx = np.where(cs == str(ElevationCnst))[0][0]
                listCnst = list(self.constraints) 
                listCnst.pop(elCnst_idx) 
                self.constraints = tuple(listCnst)
                self._default_el = False
        
        constraints = self.constraints + (other,)
        cts = Constraints(*constraints)
        cts._default_el = self._default_el
        return cts


    def __getitem__(self, n):
        """
        """
        return self.constraints[n]


    def __len__(self):
        """
        """
        return len(self.constraints)


    def __del__(self):
        for constraint in self.constraints:
            del constraint


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def size(self):
        """
        """
        return len(self)


    @property
    def weights(self):
        """
        """
        return np.array([cnt.weight for cnt in self])
    


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def evaluate(self, target, time, nslots=1):
        """
        """
        cnts = np.zeros((self.size, time.size - 1))
        unEvaluatedCnst = 0
        for i, cnt in enumerate(self):
            if isinstance(cnt, TargetConstraint) and (target is not None):
                cnts[i, :] = cnt(target, nslots)
            elif isinstance(cnt, ScheduleConstraint):
                cnts[i, :] = cnt(time, nslots)
            else:
                unEvaluatedCnst += 1
        if unEvaluatedCnst == self.size:
            cnts += 1.
            log.debug(
                'No defined constraint could be used. Schedule '
                'slot scores have been set to 1...'
            )
        score = np.average(cnts, weights=self.weights, axis=0)
        self.score = np.where(
            np.isnan(score),
            0,
            score
        )
        self.score[np.where(np.prod(cnts, axis=0)==0)[0]] = 0
        return self.score


    def plot(self, **kwargs):
        """
            kwargs:
                figsize
                figname

        """
        fig = plt.figure(
            figsize=kwargs.get('figsize', (10, 5))
        )
        for cnt in self:
            # Overplot each constraint
            cnt._plot_constraint(**kwargs)

        plt.plot(
            self.score,
            label='Total'
        )

        plt.xlabel('Time index')
        plt.ylabel('Constraint score')
        plt.legend()

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
# ============================================================= #
# ============================================================= #


