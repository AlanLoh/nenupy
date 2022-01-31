#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *************************
    Observation configuration
    *************************

    `NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_
    is a versatile low-frequency radio telescope. Several
    observing modes are available, as represented by the 
    diversity of its `receivers <https://nenufar.obs-nancay.fr/en/astronomer/#receivers>`_.
    Observations are configured thanks to text files
    called *parsets* in which the details of the instrumental set-up,
    the observation mode(s),
    the data sampling parameters,
    the operations applied on the data, etc. are listed.

    The :mod:`~nenupy.observation.obs_config` module aims at
    handling these different observing configurations as well
    as providing estimation on the data volume output by
    *NenuFAR* in one or several given set-up(s).

    NenuFAR Receiver setup
    ----------------------
    
    Manual setting
    ^^^^^^^^^^^^^^

    The various `receivers <https://nenufar.obs-nancay.fr/en/astronomer/#receivers>`_
    configurations may be set 'manually'. In such case, the user
    needs to fill in the different parameters relevant to
    characterize the observation with the desired receiver.

    For instance, say one is interested in performing an observation
    and wants to estimate the volume of the most basic *NenuFAR* data
    output: the *Beamlet Statistics* (or *BST*) FITS files. Instanciating
    an 'empty' :class:`~nenupy.observation.obs_config.BSTConfig` object
    and printing it gives a quicklook of all the properties one may want
    to modify, as well as their current values that are set by default:

    .. code-block:: python

        >>> from nenupy.observation import BSTConfig
        >>> bstconf = BSTConfig()
        >>> print(bstconf)
        Backend configuration of type 'BSTConfig'
            Properties: 'nSubBands=768', 'nPolars=2', 'durationSec=0'

    .. note::
        At any time, the user may query the receiver parameters
        by printing the corresponding instance.

    .. seealso::
        The other receivers dedicated classes are listed in :ref:`obs_config_class_summary`.

    Attribute values can be directly set to the user preferences
    to update the status of the current :class:`~nenupy.observation.obs_config.BSTConfig`
    instance:

    .. code-block:: python

        >>> bstconf.durationSec = 1800

    Alternatively, the object can be initialized with specific
    property values given as keyword arguments:

    .. code-block:: python

        >>> bstconf = BSTConfig(durationSec=1800)

    Finally, to compute an estimation of the data volume 
    (returned as a :class:`~astropy.units.Quantity` object):

    .. code-block:: python

        >>> bstconf.volume
        10.546875 Mibyte

        >>> vol = bstconf.volume
        >>> vol.to('Gibyte')
        0.010299683Gibyte

    .. warning::
        The *beamformer* receivers allow for multi-beams observations. These
        properties cannot be set manually in a straightforward way.
        Instead, it is recommended to either treat each individual beam
        separately or to instanciate the relevant objects with a parset file.

    Setting from Parset file
    ^^^^^^^^^^^^^^^^^^^^^^^^

    The most convenient way to set a given receiver's properties
    associated to as specific observation is to initialize the
    corresponding object instance from the observation *parset*:

    .. code-block:: python

        >>> from nenupy.observation import BSTConfig
        >>> bstconf = BSTConfig.fromParset('/path/to/observation.parset')

    Calling the class method ``fromParset`` automatically loads the given file
    as a :class:`~nenupy.observation.parset.Parset`. The contained instrumental
    information is parsed and the properties relevant to the receiver class
    are used to initialize the object instance.

    If an observation is configured to use the multi-beams capability of
    *NenuFAR*, the receiver properties will take that into account 
    and the data volume estimation will then be computed accordingly.

    .. warning::
        At the current stage of development, the *NenuFAR* configuration files
        called *parset user*
        (ending with ``'.parset_user'``) are not supported.

    Observation setup
    -----------------

    Rather than configuring each receiver individually, one might be
    interested in setting all of the *NenuFAR* receivers at once, from one
    or several *parset* file(s).
    This is achieved using the :class:`~nenupy.observation.obs_config.ObsConfig`
    class which stores information on all available receivers and update their
    configuration parameters according to what is described in the *parset* file(s).

    Single observation
    ^^^^^^^^^^^^^^^^^^

    In the case of a single observation, described by a unique *parset* file
    (namely ``'/path/to/observation.parset'`` in the following example),
    an instance of :class:`~nenupy.observation.obs_config.ObsConfig` is
    simply created using the class method
    :meth:`~nenupy.observation.obs_config.ObsConfig.fromParset`:

    .. code-block:: python

        >>> from nenupy.observation import ObsConfig
        >>> obsconf = ObsConfig.fromParset('/path/to/observation.parset')

    The variable called ``obsconf`` of type :class:`~nenupy.observation.obs_config.ObsConfig`
    now contains attributes named after
    the various *NenuFAR* receivers. Every one of these attributes is a
    list (of only one element in this case) of corresponding configuration
    class instances:

    .. code-block:: python

        >>> type(obsconf.tf[0])
        nenupy.observation.obs_config.TFConfig

        >>> type(obsconf.nickel[0])
        nenupy.observation.obs_config.NICKELConfig

    Querying :attr:`~nenupy.observation.obs_config.ObsConfig.volume` returns
    a dictionnary composed of the *NenuFAR* receivers as keys and their
    corresponding raw data volume estimations for the current observation:

    .. code-block:: python

        >>> obsconf.volume
        {'nickel': <Quantity 0. Gibyte>,
         'raw': <Quantity 0. Gibyte>,
         'tf': <Quantity 56.57784641 Gibyte>,
         'bst': <Quantity 9.4921875 Mibyte>,
         'pulsar_fold': <Quantity 0. Gibyte>,
         'pulsar_waveolaf': <Quantity 0. Gibyte>,
         'pulsar_single': <Quantity 0. Gibyte>}


    List of observations
    ^^^^^^^^^^^^^^^^^^^^

    Conveniently, it is also possible to initialize an
    :class:`~nenupy.observation.obs_config.ObsConfig` object
    from a list of several *parset* files.
    In order to do that, one simply needs to call the
    :meth:`~nenupy.observation.obs_config.ObsConfig.fromParsetList`
    class method:

    .. code-block:: python

        >>> from nenupy.observation import ObsConfig
        >>> obsconf = ObsConfig.fromParsetList(
                [
                    '/path/to/observation_1.parset',
                    '/path/to/observation_2.parset',
                    '/path/to/observation_3.parset'
                ]
            )
    
    Querying the :attr:`~nenupy.observation.obs_config.ObsConfig.volume`
    attribute returns a dictionnary with the summed estimated raw
    data volumes for all the *NenuFAR* receivers over all the
    observations described by the *parset* files:

    .. code-block:: python

        >>> obsconf.volume
        {'nickel': <Quantity 726.41601562 Gibyte>,
         'raw': <Quantity 282.88923204 Gibyte>,
         'tf': <Quantity 1093.13987195 Gibyte>,
         'bst': <Quantity 264.4921875 Mibyte>,
         'pulsar_fold': <Quantity 11.50373708 Gibyte>,
         'pulsar_waveolaf': <Quantity 558.79404545 Gibyte>,
         'pulsar_single': <Quantity 61.29266694 Gibyte>}

    To get the total estimated raw data volume for a specific
    receiver, and convert its unit to *Terabytes* for instance, on can
    do:

    .. code-block:: python

        >>> obsconf.volume['tf'].to('Tibyte')
        1.0675194 Tibyte

    Assuming ``dec2020_parset_list`` is a list of parsets asociated with
    the past observations done in December 2020, plotting the cumulative
    estimated raw data volume is also eased by the method
    :meth:`~nenupy.observation.obs_config.ObsConfig.plotCumulativeVolume`:

    .. code-block:: python

        >>> from nenupy.observation import ObsConfig
        >>> obsconf = ObsConfig.fromParsetList(dec2020_parset_list)
        >>> obsconf.plotCumulativeVolume(
                title='NenuFAR observations, December 2020',
                scale='log'
            )

    .. image:: ./_images/volume_december20_log.png
        :width: 800

    .. _obs_config_class_summary:

    Classes summary
    ---------------

    .. autosummary::
        :nosignatures:

        ~nenupy.observation.obs_config.ObsConfig
        ~nenupy.observation.obs_config.BSTConfig
        ~nenupy.observation.obs_config.NICKELConfig
        ~nenupy.observation.obs_config.TFConfig
        ~nenupy.observation.obs_config.RAWConfig
        ~nenupy.observation.obs_config.PulsarFoldConfig
        ~nenupy.observation.obs_config.PulsarWaveConfig
        ~nenupy.observation.obs_config.PulsarSingleConfig

"""


__author__ = 'Alan Loh, Baptiste Cecconi'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'BSTConfig',
    'NICKELConfig',
    'TFConfig',
    'RAWConfig',
    'PulsarFoldConfig',
    'PulsarWaveConfig',
    'PulsarSingleConfig',
    'ObsConfig'
]


import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np

from nenupy.observation import Parset
from nenupy.miscellaneous import accepts

import logging
log = logging.getLogger(__name__)


backendProperties = {
    'nickel': {
        'nSubBands': {
            'min': 1,
            'max': 384,
            'default': 384,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        },
        'nChannels': {
            'min': 1,
            'max': 64,
            'default': 64,
            'type': '`int`',
            'desc': 'Number of channels'
        },
        'nPolars': {
            'min': 1,
            'max': 4,
            'default': 4,
            'type': '`int`',
            'desc': 'Number of polarizations'
        },
        'nMAs': {
            'min': 2,
            'max': 96 + 6,
            'default': 96,
            'type': '`int`',
            'desc': 'Number of Mini-Arrays'
        },
        'timeRes': {
            'min': 0.,
            'max': 10.,
            'default': 1.,
            'type': '`float` or :class:`~astropy.time.TimeDelta`',
            'desc': 'Time resolution in seconds'
        }
    },
    'raw': {
        'nPolars': {
            'min': 1,
            'max': 4,
            'default': 4,
            'type': '`int`',
            'desc': 'Number of polarizations'
        },
        'nSubBands': {
            'min': 1,
            'max': 192,
            'default': 192,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        },
        'nBits': {
            'min': 8,
            'max': 16,
            'default': 8,
            'type': '`int`',
            'desc': 'Number of bits on which are recorded data elements'
        }
    },
    'tf': {
        'nPolars': {
            'min': 1,
            'max': 4,
            'default': 4,
            'type': '`int`',
            'desc': 'Number of polarizations'
        },
        'timeRes': {
            'min': (0.30*u.ms).to(u.s).value,
            'max': (83.89*u.ms).to(u.s).value,
            'default': (5.00*u.ms).to(u.s).value,
            'type': '`float` or :class:`~astropy.time.TimeDelta`',
            'desc': 'Time resolution in seconds'
        },
        'freqRes': {
            'min': (0.10*u.kHz).to(u.Hz).value,
            'max': (12.21*u.kHz).to(u.Hz).value,
            'default': (6.10*u.kHz).to(u.Hz).value,
            'type': '`float` or :class:`~astropy.units.Quantity`',
            'desc': 'Frequency resolution in Hz'
        },
        'nSubBands': {
            'min': 1,
            'max': 768,
            'default': 768,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        },
    },
    'bst': {
        'nSubBands': {
            'min': 1,
            'max': 768,
            'default': 768,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        },
        'nPolars': {
            'min': 1,
            'max': 2,
            'default': 2,
            'type': '`int`',
            'desc': 'Number of polarizations'
        }
    },
    # 'sst': {},
    'pulsar_fold': {
        'nSubBands': {
            'min': 1,
            'max': 192,
            'default': 192,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        },
        'nPolars': {
            'min': 1,
            'max': 4,
            'default': 4,
            'type': '`int`',
            'desc': 'Number of polarizations'
        },
        'tFold': {
            'min': 5.36870912,
            'max': 21.47483648,
            'default': 10.73741824,
            'type': '`float` or :class:`~astropy.time.TimeDelta`',
            'desc': 'Pulsar time fold in seconds'
        },
        'nBins': {
            'min': 16,
            'max': 8096,
            'default': 2048,
            'type': '`int`',
            'desc': 'Number of bins'
        }
    },
    'pulsar_waveolaf': {
        'nSubBands': {
            'min': 1,
            'max': 192,
            'default': 192,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        }
    },
    'pulsar_single': {
        'nSubBands': {
            'min': 1,
            'max': 192,
            'default': 192,
            'type': '`int`',
            'desc': 'Number of sub-bands'
        },
        'nPolars': {
            'min': 1,
            'max': 4,
            'default': 4,
            'type': '`int`',
            'desc': 'Number of polarizations'
        },
        'dsTime': {
            'min': 1,
            'max': 4096,
            'default': 128,
            'type': '`int`',
            'desc': 'Downsampling'
        },
        'nBits': {
            'min': 8,
            'max': 64,
            'default': 32,
            'type': '`int`',
            'desc': 'Number of bits on which are recorded data elements'
        }
    }
}


complex64 = np.complex64().itemsize * u.byte
float32 = np.float32().itemsize * u.byte


def doc(docstring, backend):
    prop = backendProperties[backend]
    paramDoc = ''
    for key in prop.keys():
        param = prop[key]
        paramDoc += f"""
            :param {key}:
                {param['desc']} (min: ``{param['min']}``,
                max: ``{param['max']}``, default: ``{param['default']}``).
            :type {key}: {param['type']}
        """
    paramDoc += """
            :param durationSec:
                Observation duration in seconds (default: ``0``).
            :type durationSec: `int` or :class:`~astropy.time.TimeDelta`

            .. versionadded:: 1.2.0
    """
    def document(func):
        func.__doc__ = docstring + '\n' + paramDoc
        return func
    return document


# ============================================================= #
# ---------------------- _BackendConfig ----------------------- #
# ============================================================= #
class _BackendConfig(object):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, backend, **kwargs):
        self.startTime = kwargs.get(
            'startTime',
            Time.now()
        )
        self.durationSec = kwargs.get(
            'durationSec',
            0
        )
        self._backend = backend

        # Catch the irrelevant kwargs
        for attr in kwargs.keys():
            if attr.startswith('_'):
                raise AttributeError(
                    "Attribute '{}' starting with '_' cannot be set.".format(
                        attr
                    )
                )
            if attr not in dir(self):
                raise AttributeError(
                    "'{}' object has no attribute '{}'".format(
                        self.__class__.__name__,
                        attr
                    )
                )

        # Fill the relevant kwargs
        for attr in backendProperties[self._backend].keys():
            setattr(
                self,
                attr,
                kwargs.get(
                    attr,
                    backendProperties[self._backend][attr]['default']
                )
            )


    def __str__(self):
        className = self.__class__.__name__
        title = "Backend configuration of type '{}'\n".format(className)
        attributes = backendProperties[self._backend].keys()
        properties = '\tProperties: '
        for att in attributes:
            properties += "'{}={}', ".format(att, getattr(self, att))
        properties += "'{}={}', ".format('durationSec', getattr(self, 'durationSec'))
        properties = properties[:-2] # remove the last coma
        return title + properties


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def nSubBands(self):
        """
        """
        return self._nSubBands
    @nSubBands.setter
    @accepts(object, int)
    def nSubBands(self, nsb):
        self._nSubBands = self._checkAttr(
            key='nSubBands',
            value=nsb,
            name='sub-bands'
        )


    @property
    def nChannels(self):
        """
        """
        return self._nChannels
    @nChannels.setter
    @accepts(object, int)
    def nChannels(self, chan):
        self._nChannels = self._checkAttr(
            key='nChannels',
            value=chan,
            name='channels'
        )


    @property
    def nBins(self):
        """
        """
        return self._nBins
    @nBins.setter
    @accepts(object, int)
    def nBins(self, bins):
        self._nBins = self._checkAttr(
            key='nBins',
            value=bins,
            name='bins'
        )


    @property
    def nPolars(self):
        """
        """
        return self._nPolars
    @nPolars.setter
    @accepts(object, int)
    def nPolars(self, np):
        self._nPolars = self._checkAttr(
            key='nPolars',
            value=np,
            name='polarizations'
        )


    @property
    def timeRes(self):
        """
        """
        return self._timeRes
    @timeRes.setter
    @accepts(object, (float, int, TimeDelta))
    def timeRes(self, dt):
        if isinstance(dt, TimeDelta):
            dt = dt.sec
        self._timeRes = self._checkAttr(
            key='timeRes',
            value=dt,
            name='time resolution'
        )


    @property
    def tFold(self):
        """
        """
        return self._tFold
    @tFold.setter
    @accepts(object, (float, int, TimeDelta))
    def tFold(self, tfold):
        if isinstance(tfold, TimeDelta):
            tfold = tfold.sec
        self._tFold = self._checkAttr(
            key='tFold',
            value=tfold,
            name='pulsar time fold'
        )


    @property
    def dsTime(self):
        """ Downsampling, can take values in [1, 2, 4, 8, 16, 32, 64, 128]
        """
        return self._dsTime
    @dsTime.setter
    @accepts(object, int)
    def dsTime(self, ds):
        # Check that ds is a power of 2:
        is2pow = (ds & (ds-1) == 0) and ds != 0
        if not is2pow:
            raise ValueError(
                "`dsTime` takes only power of two integer values."
            )
        self._dsTime = self._checkAttr(
            key='dsTime',
            value=ds,
            name='pulsar downsampling'
        )


    @property
    def nBits(self):
        """
        """
        return self._nBits
    @nBits.setter
    @accepts(object, int)
    def nBits(self, n):
        # Check that n is a power of 2:
        is2pow = (n & (n-1) == 0) and n != 0
        if not is2pow:
            raise ValueError(
                "`nBits` takes only power of two integer values."
            )
        self._nBits = self._checkAttr(
            key='nBits',
            value=n,
            name='bits'
        )


    @property
    def freqRes(self):
        """
        """
        return self._freqRes
    @freqRes.setter
    @accepts(object, (float, int, u.Quantity))
    def freqRes(self, df):
        if isinstance(df, u.Quantity):
            df = df.to(u.Hz).value
        self._freqRes = self._checkAttr(
            key='freqRes',
            value=df,
            name='frequency resolution'
        )


    @property
    def nMAs(self):
        """
        """
        return self._nMAs
    @nMAs.setter
    @accepts(object, int)
    def nMAs(self, ma):
        self._nMAs = self._checkAttr(
            key='nMAs',
            value=ma,
            name='Mini-Arrays'
        )


    @property
    def durationSec(self):
        """
        """
        return self._durationSec
    @durationSec.setter
    @accepts(object, (float, int, TimeDelta), strict=False)
    def durationSec(self, s):
        if isinstance(s, TimeDelta):
            s = s.sec
        self._durationSec = s


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _checkAttr(self, key, value, name):
        attrProp = backendProperties[self._backend][key]
        minVal = attrProp['min']
        maxVal = attrProp['max']
        defaultVal = attrProp['default']
        if value > maxVal:
            log.warning(
                "Maximal value of {0} is {1}. Setting to default '{2}={3}'.".format(
                    name,
                    maxVal,
                    key,
                    defaultVal
                )
            )
            value = defaultVal
        elif value < minVal:
            log.warning(
                "Minimal value for {0} is {1}. Setting to default '{2}={3}'.".format(
                    name,
                    minVal,
                    key,
                    defaultVal
                )
            )
            value = defaultVal
        return value
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- BSTConfig ------------------------- #
# ============================================================= #
@doc('*Beamlet Statistics* observation configuration.', 'bst')
class BSTConfig(_BackendConfig):
    
    def __init__(self, **kwargs):
        super().__init__(backend='bst', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of a *BST*
            FITS file.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import BSTConfig
                >>> bstconf = BSTConfig(
                        durationSec=3600
                    )
                >>> bstconf.volume
                21.09375 Mibyte

                >>> from nenupy.observation import BSTConfig
                >>> bstconf = BSTConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> bstconf.volume
                XXX Mibyte

            .. warning::
                The data volume estimation does not handle
                specificities of the FITS file in which the *BST*
                are stored (in particular metadata and FITS 
                architecture). Therefore, the volume may be
                underestimated by a few MB.

        """
        log.debug(str(self))
        nElements = self.nPolars * self.durationSec * self.nSubBands
        return (nElements * float32).to(u.Mibyte)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.BSTConfig`
            instance in which *BST* observation configuration properties
            are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *BST* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.BSTConfig`

            :Example:
                >>> from nenupy.observation import BSTConfig
                >>> bstconf = BSTConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        dbeams = parset.digibeams
        
        # Find out the total duration of observation
        # Loop over the digibeams, as they can be simultaneous
        totalTimes = np.array([])
        for db in dbeams.keys():
            dts = TimeDelta(
                np.arange(dbeams[db]['duration']),
                format='sec'
            )
            dbTimes = dbeams[db]['startTime'] + dts
            totalTimes = np.union1d(totalTimes, dbTimes.jd)
        
        return BSTConfig(
            durationSec=totalTimes.size,
            startTime=parset.observation['startTime']
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- NICKELConfig ------------------------ #
# ============================================================= #
@doc('*NICKEL* correlator observation configuration.', 'nickel')
class NICKELConfig(_BackendConfig):

    def __init__(self, **kwargs):
        super().__init__(backend='nickel', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of a *NICKEL*
            Measurement Set.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import NICKELConfig
                >>> nriconf = NICKELConfig(
                        nMAs=56,
                        nSubBands=244,
                        nChannels=64,
                        timeRes=1,
                        durationSec=3600
                    )
                >>> nriconf.volume.to('Tibyte')
                2.6112914 Tibyte

                >>> from nenupy.observation import NICKELConfig
                >>> nriconf = NICKELConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> nriconf.volume
                XXX Gibyte

            .. warning::
                The data volume estimation does not handle
                specificities of the Measurement Set.
                Therefore, the volume may be
                underestimated.

        """
        log.debug(str(self))
        nBaselines = self.nMAs * (self.nMAs - 1)/2 + self.nMAs
        visVolume = self.nPolars * nBaselines * complex64
        ratePerSB = visVolume * self.nChannels / self.timeRes
        ratePerObs = ratePerSB * self.nSubBands
        return (ratePerObs * self.durationSec).to(u.Gibyte)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.NICKELConfig`
            instance in which *NICKEL* observation configuration properties
            are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *NICKEL* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.NICKELConfig`

            :Example:
                >>> from nenupy.observation import NICKELConfig
                >>> nriconf = NICKELConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        out = parset.output
        anabeams = parset.anabeams

        if 'nri_receivers' not in out.keys():
            # Nickel receiver has not been used
            return NICKELConfig(
                startTime=parset.observation['startTime']
            )
        elif 'nickel' not in out['nri_receivers']:
            # Nickel receiver has not been used
            return NICKELConfig(
                startTime=parset.observation['startTime']
            )

        # Hypothesis that only one analog beam is used!
        return NICKELConfig(
            durationSec=anabeams[0]['duration'],
            timeRes=out['nri_dumpTime'],
            nSubBands=len(out['nri_subbandList']),
            nChannels=out['nri_channelization'],
            nMAs=len(anabeams[0]['maList']),
            startTime=parset.observation['startTime']
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- _UnDySPuTeDConfig --------------------- #
# ============================================================= #
class _UnDySPuTeDConfig(_BackendConfig):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)


    def __str__(self):
        className = self.__class__.__name__
        title = "Backend configuration of type '{}'\n".format(className)
        properties = ''
        for i, beam in enumerate(self._beamConfigs):
            attributes = backendProperties[beam._backend].keys()
            properties += '\tBeam {} Properties: '.format(i)
            for att in attributes:
                properties += "'{}={}', ".format(att, getattr(beam, att))
            properties += "'{}={}', \n".format('durationSec', getattr(beam, 'durationSec'))
        properties = properties[:-4] # remove the last line skip and coma
        return title + properties


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _parseParameters(self, parameters, pulsar=False):
        """ Parse values from the digital beam 'parameters'
            entry.
            E.g. 'TF: DF=3.05 DT=10.0 HAMM'
        """
        parameters = parameters.lower()
        mode = parameters.split(':')[0]
        if pulsar:
            configs = {
                param.split('=')[0]: param.split('=')[1]\
                for param in parameters.split('--')\
                if '=' in param
            }
            configs.update({
                param.rstrip(): True\
                for param in parameters.split('--')\
                if '=' not in param
            })
        else:
            configs = {
                param.split('=')[0]: param.split('=')[1]\
                for param in parameters.split()\
                if '=' in param
            }
            configs.update({
                param.rstrip(): True\
                for param in parameters.split('--')\
                if '=' not in param
            })
        return mode, configs


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    @accepts((float, int, u.Quantity), (float, int, u.Quantity))
    def _checkDFvsDT(dt, df):
        """ Short time resolutions are impossible for narrow
            frequency resolutions. Some df/dt combinations are
            therefore not allowed.
        """
        if isinstance(dt, u.Quantity):
            dt = dt.to(u.s).value
        if isinstance(df, u.Quantity):
            df = df.to(u.Hz).value

        allowedFftlen = 2**( np.arange(8) + 4 )
        allowedNfft2int = 2**(np.arange(9) + 2)
        
        # Find the closest fftlen to the desired df value
        fftlen = allowedFftlen[
            np.argmin(
                np.abs(allowedFftlen - 1.0 / 5.12e-6 / df) # 1/(5.12e-6 s) = 195312.5 Hz
            )
        ]

        # Find the closest nfft2int to the desired df value
        nfft2int = allowedNfft2int[
            np.argmin(
                np.abs(allowedNfft2int - dt / (5.12e-6 * fftlen))
            )
        ]

        dtEff = 5.12e-6 * fftlen * nfft2int
        dfEff = 1.0 / 5.12e-6 / fftlen

        log.debug(
            "'freqRes={0:.2f}', 'timeRes={1:.2f}' <--> 'df={2:.2f}', 'dt={3:.2f}' ('fftlen={4}', 'nfft2int={5}')".format(
                (df*u.Hz).to(u.kHz),
                (dt*u.s).to(u.ms),
                (dfEff*u.Hz).to(u.kHz),
                (dtEff*u.s).to(u.ms),
                fftlen,
                nfft2int
            )
        )

        return dtEff, dfEff
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- _TFBeamConfig ----------------------- #
# ============================================================= #
class _TFBeamConfig(_BackendConfig):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, **kwargs):
        super().__init__(backend='tf', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """
        """
        log.debug(str(self))
        print(self.nPolars, self.freqRes, self.timeRes, self.nSubBands, self.durationSec)
        ratePerSB = self.nPolars * float32 * (200.e6/1024./self.freqRes) / self.timeRes
        rateObs = ratePerSB * self.nSubBands
        return (rateObs * self.durationSec).to(u.Gibyte)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- TFConfig -------------------------- #
# ============================================================= #
@doc('*UnDySPuTeD Time-Frequency* mode observation configuration.', 'tf')
class TFConfig(_UnDySPuTeDConfig):

    def __init__(self, _setFromParset=False, **kwargs):
        if not _setFromParset:
            super().__init__(backend='tf', **kwargs)
            self.timeRes, self.freqRes = self._checkDFvsDT(
                dt=self.timeRes,
                df=self.freqRes
            )
            kwargs['timeRes'] = self.timeRes
            kwargs['freqRes'] = self.freqRes
            self._beamConfigs = [
                _TFBeamConfig(**kwargs)
            ]
        else:
            self._beamConfigs = []


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of an
            *UnDySPuTeD-TF* observation file.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import TFConfig
                >>> tfconf = TFConfig(
                        nSubBands=500,
                        timeRes=42e-3,
                        freqRes=200,
                        durationSec=3600
                    )
                >>> tfconf.volume
                654.83619 Gibyte

                >>> from nenupy.observation import TFConfig
                >>> tfconf = TFConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> tfconf.volume
                XXX Gibyte

            .. note::
                Combinations of ``timeRes`` and ``freqRes`` pairs
                are limited to that available within the
                *UnDySPuTeD* receiver. If set otherwise, the closest
                allowed values will be filled instead.

                One can check the corresponding attributes
                after setting up the desired configuration:
                
                    >>> tfconf = TFConfig(
                            nSubBands=500,
                            timeRes=1e-3,
                            freqRes=200,
                            durationSec=3600
                        )
                    >>> tfconf.timeRes
                    0.02097152
                    >>> tfconf.freqRes
                    190.73486328125

                Altenatively, one can set the log to ``DEBUG``
                in order to print conversion details:

                    >>> import logging
                    >>> logging.getLogger('nenupy').setLevel(logging.DEBUG)
                    >>> nriconf = TFConfig(
                            nSubBands=500,
                            timeRes=1e-3,
                            freqRes=200,
                            durationSec=3600
                        )
                    2020-12-16 17:28:52 -- DEBUG: 'freqRes=0.20 kHz', 'timeRes=1.00 ms' <--> 'df=0.19 kHz', 'dt=20.97 ms' ('fftlen=1024', 'nfft2int=4')

        """
        vol = 0 * u.Gibyte
        for bc in self._beamConfigs:
            vol += bc.volume
        return vol


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (str, Parset))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.TFConfig`
            instance in which *UnDySPuTeD-TF* observation configuration
            properties are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *UnDySPuTeD-TF* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.TFConfig`

            :Example:
                >>> from nenupy.observation import TFConfig
                >>> tfconf = TFConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        out = parset.output
        digibeams = parset.digibeams

        tf = TFConfig(_setFromParset=True)
        tf.startTime = parset.observation['startTime']
        beamConfigs = []
        
        if 'undysputed' not in out['hd_receivers']:
            # UnDySPuTeD receiver has not been used
            pass
        else:
            for db in digibeams.keys():
                if digibeams[db]['toDo'].lower() != 'dynamicspectrum':
                    continue
                try:
                    mode, parameters = tf._parseParameters(digibeams[db]['parameters'])
                except KeyError:
                    log.warning(
                        "Parset '{}' has no 'parameters' key.".format(parset.parset)
                    )
                    continue
                if mode != 'tf':
                    continue
                dt, df = tf._checkDFvsDT(
                    dt=(float(parameters['dt'])*u.ms).to(u.s).value,
                    df=(float(parameters['df'])*u.kHz).to(u.Hz).value
                )
                beamConfigs.append(
                    _TFBeamConfig(
                        timeRes=dt,
                        freqRes=df,
                        durationSec=digibeams[db]['duration'],
                        nSubBands=len(digibeams[db]['subbandList'])
                    )
                )
        
        tf._beamConfigs = beamConfigs
        return tf
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- _RawBeamConfig ----------------------- #
# ============================================================= #
class _RawBeamConfig(_BackendConfig):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, **kwargs):
        super().__init__(backend='raw', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """
        """
        log.debug(str(self))
        duration = self.durationSec - 60 # Burning time at start
        nBytes = self.nBits / 8 * u.byte
        rateSB = self.nPolars * nBytes / 5.12e-6
        rateObs = rateSB * self.nSubBands
        return (rateObs * duration).to(u.Gibyte)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------- RAWConfig ------------------------- #
# ============================================================= #
@doc('*UnDySPuTeD Waveform* mode observation configuration.', 'raw')
class RAWConfig(_UnDySPuTeDConfig):

    def __init__(self, _setFromParset=False, **kwargs):
        if not _setFromParset:
            super().__init__(backend='raw', **kwargs)
            self._beamConfigs = [
                _RawBeamConfig(**kwargs)
            ]
        else:
            self._beamConfigs = []


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of an
            *UnDySPuTeD-RAW* observation file.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import RAWConfig
                >>> rawconf = RAWConfig(
                        durationSec=3600
                    )
                >>> rawconf.volume
                494.53229 Gibyte

                >>> from nenupy.observation import RAWConfig
                >>> rawconf = RAWConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> rawconf.volume
                XXX Gibyte

        """
        vol = 0 * u.Gibyte
        for bc in self._beamConfigs:
            vol += bc.volume
        return vol


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.RAWConfig`
            instance in which *UnDySPuTeD-RAW* observation configuration
            properties are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *UnDySPuTeD-RAW* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.RAWConfig`

            :Example:
                >>> from nenupy.observation import RAWConfig
                >>> rawconf = RAWConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        out = parset.output
        digibeams = parset.digibeams

        raw = RAWConfig(_setFromParset=True)
        raw.startTime = parset.observation['startTime']

        beamConfigs = []
        
        if 'undysputed' not in out['hd_receivers']:
            # UnDySPuTeD receiver has not been used
            pass
        else:
            for db in digibeams.keys():
                if digibeams[db]['toDo'].lower() != 'waveform':
                    continue
                beamConfigs.append(
                    _RawBeamConfig(
                        durationSec=digibeams[db]['duration'],
                        nSubBands=len(digibeams[db]['subbandList'])
                    )
                )
        
        raw._beamConfigs = beamConfigs
        return raw
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- _FoldBeamConfig ---------------------- #
# ============================================================= #
class _FoldBeamConfig(_BackendConfig):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, **kwargs):
        super().__init__(backend='pulsar_fold', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """
        """
        log.debug(str(self))
        duration = self.durationSec - 60 # Burning time at start
        duration = 0 if duration < 0 else duration
        rateObs = self.nSubBands * self.nPolars * float32 * self.nBins / self.tFold
        return (rateObs * duration).to(u.Gibyte)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- PulsarFoldConfig --------------------- #
# ============================================================= #
@doc('*UnDySPuTeD Pulsar-FOLD* mode observation configuration.', 'pulsar_fold')
class PulsarFoldConfig(_UnDySPuTeDConfig):

    def __init__(self, _setFromParset=False, **kwargs):
        if not _setFromParset:
            super().__init__(backend='pulsar_fold', **kwargs)
            self._beamConfigs = [
                _FoldBeamConfig(**kwargs)
            ]
        else:
            self._beamConfigs = []


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of an
            *UnDySPuTeD Pulsar-FOLD* observation file.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import PulsarFoldConfig
                >>> foldconf = PulsarFoldConfig(
                        durationSec=3600
                    )
                >>> foldconf.volume
                1.9317667 Gibyte

                >>> from nenupy.observation import PulsarFoldConfig
                >>> foldconf = PulsarFoldConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> foldconf.volume
                XXX Gibyte

        """
        vol = 0 * u.Gibyte
        for bc in self._beamConfigs:
            vol += bc.volume
        return vol


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.PulsarFoldConfig`
            instance in which *UnDySPuTeD Pulsar-FOLD* observation configuration
            properties are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *UnDySPuTeD Pulsar-FOLD* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.PulsarFoldConfig`

            :Example:
                >>> from nenupy.observation import PulsarFoldConfig
                >>> foldconf = PulsarFoldConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        out = parset.output
        digibeams = parset.digibeams

        fold = PulsarFoldConfig(_setFromParset=True)
        fold.startTime = parset.observation['startTime']

        beamConfigs = []
        
        if 'undysputed' not in out['hd_receivers']:
            # UnDySPuTeD receiver has not been used
            pass
        else:
            for db in digibeams.keys():
                if digibeams[db]['toDo'].lower() != 'pulsar':
                    continue
                try:
                    mode, parameters = fold._parseParameters(
                        digibeams[db]['parameters'],
                        pulsar=True
                    )
                except KeyError:
                    log.warning(
                        "Parset '{}' has no 'parameters' key.".format(parset.parset)
                    )
                    continue
                if mode != 'fold':
                    continue
                
                props = backendProperties['pulsar_fold']
                
                beamConfigs.append(
                    _FoldBeamConfig(
                        nSubBands=len(digibeams[db]['subbandList']),
                        nPolars=1 if 'onlyi' in parameters else 4,
                        tFold=float(parameters.get('tfold', props['tFold']['default'])),
                        durationSec=digibeams[db]['duration'],
                        nBins=int(parameters.get('nbin', props['nBins']['default']))
                    )
                )

        fold._beamConfigs = beamConfigs
        return fold
# ============================================================= #
# ============================================================= #


# ============================================================= #
# -------------------- _WaveolafBeamConfig -------------------- #
# ============================================================= #
class _WaveolafBeamConfig(_BackendConfig):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, **kwargs):
        super().__init__(backend='pulsar_waveolaf', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """
        """
        log.debug(str(self))
        duration = self.durationSec - 60 # Burning time at start
        duration = 0 if duration < 0 else duration
        rateObs = 451590 * u.byte * self.nSubBands
        return (rateObs * duration).to(u.Gibyte)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- PulsarWaveConfig --------------------- #
# ============================================================= #
@doc('*UnDySPuTeD Pulsar-WAVEOLAF* mode observation configuration.', 'pulsar_waveolaf')
class PulsarWaveConfig(_UnDySPuTeDConfig):

    def __init__(self, _setFromParset=False, **kwargs):
        if not _setFromParset:
            super().__init__(backend='pulsar_waveolaf', **kwargs)
            self._beamConfigs = [
                _WaveolafBeamConfig(**kwargs)
            ]
        else:
            self._beamConfigs = []


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of an
            *UnDySPuTeD Pulsar-WAVEOLAF* observation file.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import PulsarWaveConfig
                >>> waveconf = PulsarWaveConfig(
                        durationSec=3600
                    )
                >>> waveconf.volume
                285.85707 Gibyte

                >>> from nenupy.observation import PulsarWaveConfig
                >>> waveconf = PulsarWaveConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> waveconf.volume
                XXX Gibyte

        """
        vol = 0 * u.Gibyte
        for bc in self._beamConfigs:
            vol += bc.volume
        return vol


   # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.PulsarWaveConfig`
            instance in which *UnDySPuTeD Pulsar-WAVEOLAF* observation configuration
            properties are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *UnDySPuTeD Pulsar-WAVEOLAF* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.PulsarWaveConfig`

            :Example:
                >>> from nenupy.observation import PulsarWaveConfig
                >>> waveconf = PulsarWaveConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        out = parset.output
        digibeams = parset.digibeams

        wave = PulsarWaveConfig(_setFromParset=True)
        wave.startTime = parset.observation['startTime']

        beamConfigs = []
        
        if 'undysputed' not in out['hd_receivers']:
            # UnDySPuTeD receiver has not been used
            pass
        else:
            for db in digibeams.keys():
                if digibeams[db]['toDo'].lower() != 'pulsar':
                    continue
                try:
                    mode, parameters = wave._parseParameters(
                        digibeams[db]['parameters'],
                        pulsar=True
                    )
                except KeyError:
                    log.warning(
                        "Parset '{}' has no 'parameters' key.".format(parset.parset)
                    )
                    continue
                if mode != 'waveolaf':
                    continue
                
                props = backendProperties['pulsar_waveolaf']
                
                beamConfigs.append(
                    _WaveolafBeamConfig(
                        nSubBands=len(digibeams[db]['subbandList']),
                        durationSec=digibeams[db]['duration']
                    )
                )

        wave._beamConfigs = beamConfigs
        return wave
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- _SingleBeamConfig --------------------- #
# ============================================================= #
class _SingleBeamConfig(_BackendConfig):
    """
        .. versionadded:: 1.2.0
    """

    def __init__(self, **kwargs):
        super().__init__(backend='pulsar_single', **kwargs)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """
        """
        log.debug(str(self))
        duration = self.durationSec - 60 # Burning time at start
        duration = 0 if duration < 0 else duration
        nBytes = self.nBits / 8 * u.byte
        rateObs = self.nSubBands * self.nPolars * nBytes /5.12e-6 / self.dsTime
        return (rateObs * duration).to(u.Gibyte)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- PulsarSingleConfig -------------------- #
# ============================================================= #
@doc('*UnDySPuTeD Pulsar-SINGLE* mode observation configuration.', 'pulsar_single')
class PulsarSingleConfig(_UnDySPuTeDConfig):

    def __init__(self, _setFromParset=False, **kwargs):
        if not _setFromParset:
            super().__init__(backend='pulsar_single', **kwargs)
            self._beamConfigs = [
                _SingleBeamConfig(**kwargs)
            ]
        else:
            self._beamConfigs = []



    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume of an
            *UnDySPuTeD Pulsar-SINGLE* observation file.

            :getter: Data volume.

            :type: :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import PulsarSingleConfig
                >>> singleconf = PulsarSingleConfig(
                        durationSec=3600
                    )
                >>> singleconf.volume
                15.454134 Gibyte

                >>> from nenupy.observation import PulsarSingleConfig
                >>> singleconf = PulsarSingleConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> singleconf.volume
                XXX Gibyte

        """
        vol = 0 * u.Gibyte
        for bc in self._beamConfigs:
            vol += bc.volume
        return vol


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.PulsarSingleConfig`
            instance in which *UnDySPuTeD Pulsar-SINGLE* observation configuration
            properties are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *UnDySPuTeD Pulsar-SINGLE* configuration as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.PulsarSingleConfig`

            :Example:
                >>> from nenupy.observation import PulsarSingleConfig
                >>> waveconf = PulsarSingleConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        out = parset.output
        digibeams = parset.digibeams

        single = PulsarSingleConfig(_setFromParset=True)
        single.startTime = parset.observation['startTime']

        beamConfigs = []
        
        if 'undysputed' not in out['hd_receivers']:
            # UnDySPuTeD receiver has not been used
            pass
        else:
            for db in digibeams.keys():
                if digibeams[db]['toDo'].lower() != 'pulsar':
                    continue
                try:
                    mode, parameters = single._parseParameters(
                        digibeams[db]['parameters'],
                        pulsar=True
                    )
                except KeyError:
                    log.warning(
                        "Parset '{}' has no 'parameters' key.".format(parset.parset)
                    )
                    continue
                if mode != 'single':
                    continue
                
                props = backendProperties['pulsar_single']

                beamConfigs.append(
                    _SingleBeamConfig(
                        nSubBands=len(digibeams[db]['subbandList']),
                        nPolars=1 if 'onlyi' in parameters else 4,
                        dsTime=int(parameters.get('dstime', props['dsTime']['default'])),
                        durationSec=digibeams[db]['duration'],
                        nBits=int(parameters.get('nbits', props['nBits']['default']))
                    )
                )

        single._beamConfigs = beamConfigs
        return single
# ============================================================= #
# ============================================================= #


backendClasses = {
    'nickel': NICKELConfig,
    'raw': RAWConfig,
    'tf': TFConfig,
    'bst': BSTConfig,
    # 'sst': '',
    # 'xst': '',
    'pulsar_fold': PulsarFoldConfig,
    'pulsar_waveolaf': PulsarWaveConfig,
    'pulsar_single': PulsarSingleConfig
}


# ============================================================= #
# ------------------------- ObsConfig ------------------------- #
# ============================================================= #
class ObsConfig(object):
    """ Main observation configuration class.

        .. versionadded:: 1.2.0
    """

    def __init__(self):
        for key in backendClasses:
            setattr(self, key, [])


    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('{} expected.'.format(self.__class__))

        new = ObsConfig()
        for key in backendClasses:
            summedVal = getattr(self, key) + getattr(other, key)
            setattr(new, key, summedVal)
        return new


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def volume(self):
        """ Computes an estimation of the data volume output for
            all the *NenuFAR* receivers. If the object
            :class:`~nenupy.observation.obs_config.ObsConfig` has been
            set with several parset files (with the method
            :meth:`~nenupy.observation.obs_config.ObsConfig.fromParsetList`),
            the volumes are summed over all observations.

            :getter: Data volume.

            :type: `dict` of :class:`~astropy.units.Quantity`

            :Example:
                >>> from nenupy.observation import ObsConfig
                >>> obsconf = ObsConfig.fromParset(
                        'nenufar_obs.parset'
                    )
                >>> obsconf.volume
                {'nickel': <Quantity 0. Gibyte>,
                 'raw': <Quantity 0. Gibyte>,
                 'tf': <Quantity 0. Gibyte>,
                 'bst': <Quantity 20.625 Mibyte>,
                 'pulsar_fold': <Quantity 3.7763691 Gibyte>,
                 'pulsar_waveolaf': <Quantity 558.79404545 Gibyte>,
                 'pulsar_single': <Quantity 0. Gibyte>}

        """
        volumes = {}
        for key in backendClasses.keys():
            volumes[key] = sum([subconf.volume for subconf in getattr(self, key)])
        return volumes


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    @accepts(type, (Parset, str))
    def fromParset(cls, parset):
        """ Returns a :class:`~nenupy.observation.obs_config.ObsConfig`
            instance in which all *NenuFAR* receiver configuration
            properties are set as defined by the ``parset``.

            :param parset:
                Observation parset file.
            :type parset: `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                Full *NenuFAR* receiver configurations as defined by the ``parset`` file.
            :rtype: :class:`~nenupy.observation.obs_config.ObsConfig`

            :Example:
                >>> from nenupy.observation import ObsConfig
                >>> obsconf = ObsConfig.fromParset('nenufar_obs.parset')

        """
        if isinstance(parset, str):
            parset = Parset(parset)

        dv = ObsConfig()
        for key in backendClasses.keys():
            #setattr(dv, key, backendClasses[key].fromParset(parset))
            getattr(dv, key).append(backendClasses[key].fromParset(parset))
        return dv


    @classmethod
    @accepts(type, list)
    def fromParsetList(cls, parsetList):
        """ Returns a :class:`~nenupy.observation.obs_config.ObsConfig`
            instance in which all *NenuFAR* receiver configuration
            properties are set as defined by each parset file conained
            in ``parsetList``.

            :param parsetList:
                List of observation parset files.
            :type parsetList: `list` of `str` or :class:`~nenupy.observation.parset.Parset`

            :returns:
                *NenuFAR* receiver configurations for all observations defined
                by the parset files listed in ``parsetList``.
            :rtype: :class:`~nenupy.observation.obs_config.ObsConfig`

            :Example:
                >>> from nenupy.observation import ObsConfig
                >>> obsconf = ObsConfig.fromParsetList(
                        ['nenufar_obs_1.parset', 'nenufar_obs_2.parset']
                    )

        """
        if not isinstance(parsetList, list):
            raise TypeError(
                "`parsetList` should be a `list`."
            )

        tot = ObsConfig()
        for parset in parsetList:
            obs = ObsConfig.fromParset(parset)
            tot += obs

        return tot


    @accepts(object, str, (str, u.Quantity), strict=False)
    def getCumulativeVolume(self, receiver, unit='Tibyte'):
        """ Gets an estimation of the cumulative raw data volume
            over time
            computed from the observations listed in the current
            :class:`~nenupy.observation.obs_config.ObsConfig`
            instance for the given ``receiver``.

            :param receiver:
                Name of the receiver from which the cumulative data
                volume is estimated.
            :type receiver: `str`
            :param unit:
                Data volume unit in which the cumulative volume
                will be expressed (see also
                `binary unit prefixes <https://docs.astropy.org/en/stable/units/standard_units.html#prefixes>`_).
            :type unit: `str` or :class:`~astropy.units.Quantity`

            :returns: 
                Observation start times and cumulative data volumes.
            :rtype: (:class:`~astropy.time.Time`, :class:`~numpy.ndarray`)

            :Example:
                >>> from nenupy.observation import ObsConfig
                >>> obsconf = ObsConfig.fromParsetList(
                        ['nenufar_obs_1.parset', 'nenufar_obs_2.parset']
                    )
                >>> times, volumes = obsconf.getCumulativeVolume(
                        receiver='nickel',
                        unit='Gibyte'
                    )

        """
        if receiver not in backendClasses.keys():
            raise ValueError(
                f"Receiver '{receiver}' not in '{backendClasses.keys()}'"
            )
        obs_list = getattr(self, receiver)
        times = Time([obs.startTime for obs in obs_list])
        indices = np.argsort(times.mjd)
        times = times[indices]
        volumes = np.array([obs.volume.to(unit).value for obs in obs_list])
        cumVol = np.cumsum(volumes[indices])
        return times, cumVol


    def plotCumulativeVolume(self, figname='', **kwargs):
        """ Plots the cumulative raw data volume estimation.

            :param figname:
                Figure name to store. If set to ``''`` (by default),
                the figure is only displayed.
            :type figname: `str`
            :param figsize:
                Figure size in inches (default: ``(10, 5)``).
            :type figsize: `tuple`
            :param unit:
                Data volume unit in which the cumulative
                volume will be expressed (see also
                `binary unit prefixes <https://docs.astropy.org/en/stable/units/standard_units.html#prefixes>`_).
                Default is ``'Tibyte'``.
            :type unit: `str` or :class:`~astropy.units.Quantity`
            :param receivers:
                List of receivers whose cumulative data
                volumes must be plotted. Default: all
                available *NenuFAR* receivers.
            :type receivers: `list` of `str`
            :param scale:
                y-axis scaling (``'linear'`` or ``'log'``).
            :type scale: `str`
            :param title:
                Title of the plot.
            :type title: `str`
            :param grid:
                Add a grid to help the visualization. Default is ``True``.
            :type grid: `bool`
            :param tMin:
                Minimum time to represent.
            :type tMin: `str` or :class:`~astropy.time.Time`
            :param tMax:
                Maximum time to represent.
            :type tMax: `str` or :class:`~astropy.time.Time`

        """
        import matplotlib.pylab as plt
        from itertools import cycle
        
        fig = plt.figure(
            figsize=kwargs.get('figsize', (10, 5))
        )
        unit = kwargs.get('unit', 'Tibyte')

        receivers = kwargs.get('receivers', list(backendClasses.keys()))
        if not isinstance(receivers, list):
            raise TypeError(
                "`receivers` must be set as a list."
            )
        lStyles = [
            'solid',
            'dotted',
            'dashed',
            'dashdot',
            (0, (5, 5)), # loose dashed
            (0, (3, 5, 1, 5, 1, 5)), # dashdotteddotted
            (0, (3, 1, 1, 1, 1, 1)) # dense dashdotteddotted
        ]
        linecycler = cycle(lStyles)
        
        volCumDico = {}
        for receiver in receivers:
            times, cumVol = self.getCumulativeVolume(
                receiver=receiver,
                unit=unit
            )
            volCumDico[receiver] = {
                'times': times,
                'cumulative_sum': cumVol
            }
            plt.plot(
                times.datetime,
                cumVol,
                label=receiver,
                linewidth=1,
                linestyle=next(linecycler)
            )

        plt.plot(
            times.datetime,
            sum([volCumDico[k]['cumulative_sum'] for k in volCumDico.keys()]),
            label='Total',
            color='black',
            linestyle='solid',
            linewidth=2,
            )

        plt.yscale(kwargs.get('scale', 'linear'))
        plt.legend()
        plt.xlabel('UTC Time')
        plt.ylabel(f'Raw data volume ({unit})')
        plt.title(kwargs.get('title', ''))
        plt.xlim(
            (
                Time(kwargs.get('tMin', times[0])).datetime,
                Time(kwargs.get('tMax', times[-1])).datetime
            )
        )
        if kwargs.get('grid', True):
            plt.grid()
        
        if figname == '':
            plt.show()
        else:
            plt.savefig(
                figname,
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
# ============================================================= #
# ============================================================= #

