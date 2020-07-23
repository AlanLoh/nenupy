#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ********************
    Observation Database
    ********************

    The amount of `NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_
    observations is rising, since the start of the commissioning
    period up to now, the early science phase period.
    This results in a huge quantity of available observations,
    with many different array configurations, which could be used
    for scientific or commissioning purposes.
    However, with the increase of recorded observations, the
    difficulty to find one particular observation that suits
    specific needs is becoming more and more tricky.

    Each NenuFAR observation is associated with the production of
    a Beamlet Statistics FITS file (or `BST 
    <https://nenufar.obs-nancay.fr/en/astronomer/#data-products>`_,
    see also :class:`~nenupy.beamlet.bstdata.BST_Data`)
    containing the beamformed low-rate data from the *LaNewBa*
    receiver, aiming at:

    * commissioning;
    * providing a quick-look of the observation;
    * providing the observation context/metadata in their header.

    These BST observations are recorded in the `Nançay database
    <http://vogate.obs-nancay.fr/__system__/dc_tables/show/tableinfo/nenufar.bst>`_
    as a `TAP <http://www.ivoa.net/documents/TAP/>`_ service, an
    `IVOA <http://ivoa.net/>`_ standard which can be queried
    using the `ADQL <http://cdsportal.u-strasbg.fr/adqltuto/>`_
    (Astronomical Data Query Language ADQL ) language.


    The class :class:`~nenupy.observation.tapdatabase.ObsDatabase`
    is designed to ease such queries. It uses the TAP access
    capabilities of `PyVO <https://pyvo.readthedocs.io/en/latest/>`_
    to search and return NenuFAR BST observation queries as 
    :class:`~astropy.table.Table` instances. A limited number of
    parameters can be filled to prepare the queries yet, however,
    the class is flexible enough to eventually accepts more
    parameters. Those are:

    * :attr:`~nenupy.observation.tapdatabase.ObsDatabase.time_range`: Query NenuFAR observations made within a time period.
    * :attr:`~nenupy.observation.tapdatabase.ObsDatabase.freq_range`: Query NenuFAR observations made within a frequency range.
    * :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_radius`: Angular radius used to search for observations around specific sky coordinates.
    * :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_center`: Center (in sky coordinates) of the search for NenuFAR observation targetting coordinates within :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_radius`.
    
    The logger may help following building up of the query.

    >>> from nenupy.observation import ObsDatabase
    >>> import logging
    >>> logging.getLogger('nenupy').setLevel(logging.INFO)
    >>> db = ObsDatabase()
    2020-04-01 12:00:00 -- INFO: TAP service http://vogate.obs-nancay.fr/tap accessed.
    >>> db.query
    'SELECT target_name, obs_creator_did, s_ra, s_dec, t_min, t_max, em_min, em_max from nenufar.bst WHERE ()'
    >>> db.meta_names = ['obs_title']
    >>> db.query
    'SELECT obs_title from nenufar.bst WHERE ()'
    >>> db.time_range = ['2019-04-01 12:00:00', '2019-04-02 12:00:00']
    2020-04-01 12:00:00 -- INFO: time_range set to [<Time object: scale='utc' format='iso' value=2019-04-01 12:00:00.000>, <Time object: scale='utc' format='iso' value=2019-04-02 12:00:00.000>].
    >>> db.query
    'SELECT obs_title from nenufar.bst WHERE ((t_min >= 58574.5 AND t_max <= 58575.5))'
    >>> observations = db.search()
    2020-04-01 12:00:00 -- INFO: TAP service columns to return: ['obs_title'].
    >>> len(observations)
    5

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'ObsDatabase'
]


from pyvo.dal import TAPService
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np

from nenupy.astro import wavelength

import logging
log = logging.getLogger(__name__)


nancay_tap = 'http://vogate.obs-nancay.fr/tap'

colnames = [
    'owner',
    'embargo',
    'mime',
    'accsize',
    'dataproduct_type',
    'dataproduct_subtype',
    'calib_level',
    'obs_collection',
    'obs_id',
    'obs_title',
    'obs_publisher_did',
    'obs_creator_did',
    'access_url',
    'access_format',
    'access_estsize',
    'target_name',
    'target_class',
    's_ra',
    's_dec',
    's_fov',
    's_region',
    's_resolution',
    't_min',
    't_max',
    't_exptime',
    't_resolution',
    'em_min',
    'em_max',
    'em_res_power',
    'o_ucd',
    'pol_states',
    'facility_name',
    'instrument_name',
    's_xel1',
    's_xel2',
    't_xel',
    'em_xel',
    'pol_xel',
    's_pixel_scale',
    'em_ucd',
    'preview',
    'source_table'
]


# ============================================================= #
# ------------------------ ObsDatabase ------------------------ #
# ============================================================= #
class ObsDatabase(object):
    """ Class to access NenuFAR BST TAP service.
    """

    def __init__(self):
        self.service = TAPService(nancay_tap)
        log.info(
            f'TAP service {nancay_tap} accessed.'
        )
        self.meta_names = [
            'target_name',
            'obs_creator_did',
            's_ra',
            's_dec',
            't_min',
            't_max',
            'em_min',
            'em_max'
        ]
        self._conditions = {
            'time': '',
            'freq': '',
            'pos': '',
        }
        self.time_range = [None, None]
        self.freq_range = [None, None]
        self.fov_radius = 180*u.deg
        self.fov_center = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def meta_names(self):
        """ Column names (observation properties) to return using
            :meth:`~nenupy.observation.tapdatabase.ObsDatabase.search`.

            :setter: `list` of column names.

            :getter: Properties to query.
            
            :type: `str`

            :seealso:
                `Database description
                <http://vogate.obs-nancay.fr/__system__/dc_tables/show/tableinfo/nenufar.bst>`_
        
        """
        return ', '.join(self._meta_names)
    @meta_names.setter
    def meta_names(self, m):
        if not isinstance(m, list):
            raise TypeError(
                'meta_names must be a list'
            )
        unknown_mask = ~np.isin(m, colnames)
        if unknown_mask.any():
            unknown = np.array(m)[unknown_mask]
            raise ValueError(
                'Unknown meta_names: {}, available: {}'.format(
                    unknown,
                    colnames
                )
            )
        self._meta_names = m
        return


    @property
    def time_range(self):
        """ Time range selection for the ADQL query using
            :meth:`~nenupy.observation.tapdatabase.ObsDatabase.search`.
            Default is ``[None, None]`` which means that no
            condition based on observation time will be applied.

            :setter: Length-2 list of ``[start, stop]``. ``start``
                and ``stop`` may be passed as :class:`~astropy.time.Time`
                instances or as ISO/ISOT `str`.

            :getter: ``[start, stop]`` list.
            
            :type: `list`
        """
        return self._time_range
    @time_range.setter
    def time_range(self, t):
        if t == [None, None]:
            self._conditions['time'] = ''
            self._time_range = t
            return
        if not isinstance(t, list):
            raise TypeError(
                'time_range must be a list'
            )
        if not len(t) == 2:
            raise ValueError(
                'time_range must be of length 2'
            )
        if not all([isinstance(ti, Time) for ti in t]):
            t = [Time(ti) for ti in t]        
        self._conditions['time'] = f'(t_min >= {t[0].mjd} '\
            f'AND t_max <= {t[1].mjd})'
        log.info(f'time_range set to {t}.')
        self._time_range = t
        return


    @property
    def freq_range(self):
        """ Frequency range selection for the ADQL query using
            :meth:`~nenupy.observation.tapdatabase.ObsDatabase.search`.
            Default is ``[None, None]`` which means that no
            condition based on observation frequencies will be applied.

            :setter: Length-2 list of ``[fmin, fmax]``. ``fmin``
                and ``fmax`` may be passed as :class:`~astropy.units.Quantity`
                instances or as `float` (assumed to be expressed
                in MHz).

            :getter: ``[fmin, fmax]`` list.
            
            :type: `list`
        """
        return self._freq_range
    @freq_range.setter
    def freq_range(self, f):
        if f == [None, None]:
            self._conditions['freq'] = ''
            self._freq_range = f
            return
        if not isinstance(f, list):
            raise TypeError(
                'freq_range must be a list'
            )
        if not len(f) == 2:
            raise ValueError(
                'freq_range must be of length 2'
            )
        if not all([isinstance(fi, u.Quantity) for fi in f]):
            f = [fi*u.MHz for fi in f]
        lmax = wavelength(f[0]).to(u.m).value
        lmin = wavelength(f[1]).to(u.m).value
        self._conditions['freq'] = f'(em_min >= {lmin} AND '\
            f'em_max <= {lmax})'
        log.info(f'freq_range set to {f}.')
        self._freq_range = f
        return


    @property
    def fov_radius(self):
        """ Radius of the query, in combination with the query
            center :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_center`.
    
            :setter: Radius (in degrees if no unit is provided). Default is ``180 deg``.

            :getter: Radius in degrees.
            
            :type: `float` or :class:`~astropy.units.Quantity`

            .. warning::
                Must be set **before**
                :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_center`.
        """
        return self._fov_radius
    @fov_radius.setter
    def fov_radius(self, r):
        if r is None:
            r = 180*u.deg
        if not isinstance(r, u.Quantity):
            r *= u.deg
        if not r.isscalar:
            raise ValueError(
                'FOV radius must be a scalar.'
            )
        self._fov_radius = r
        return    


    @property
    def fov_center(self):
        """ Center of the field of view queried, in comination
            with the radius :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_radius`.

            :setter: Center of the field of view.
            
            :getter: Center of the field of view.

            :type: :class:`~astropy.coordinates.SkyCoord`

            .. warning::
                Must be set **after**
                :attr:`~nenupy.observation.tapdatabase.ObsDatabase.fov_radius`.
        """
        return self._fov_center
    @fov_center.setter
    def fov_center(self, f):
        if f is None:
            self._conditions['pos'] = ''
            self._fov_center = f
            return
        if not isinstance(f, SkyCoord):
            raise TypeError(
                'fov_center must be a SkyCoord instance.'
            )
        if not f.isscalar:
            raise ValueError(
                'fov_center must be scalar.'
            )
        radius = self.fov_radius.to(u.deg).value
        self._conditions['pos'] = f'1 = CONTAINS'\
            f"(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', "\
            f'{f.ra.deg}, {f.dec.deg}, {radius}))'
        log.info(f'fov_center set to {f}, radius={radius} deg.')
        self._fov_center = f
        return


    @property
    def conditions(self):
        """ Conditions summary of the query.

            :getter: Query conditions.

            :type: `str`
        """
        conds = []
        for key in self._conditions.keys():
            if self._conditions[key] != '':
                conds.append(self._conditions[key])
        conditions = ' AND '.join(conds)
        return conditions
    

    @property
    def query(self):
        """ Full query, combining returned parameteres
            :attr:`~nenupy.observation.tapdatabase.ObsDatabase.meta_names`
            and the conditions
            :attr:`~nenupy.observation.tapdatabase.ObsDatabase.conditions`.

            :getter: Query.

            :type: `str`
        """
        q = f'SELECT {self.meta_names} from nenufar.bst '\
            f'WHERE ({self.conditions})'
        return q
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def search(self):
        """ Run the TAP :attr:`~nenupy.observation.tapdatabase.ObsDatabase.query`
            on the `NenuFAR BST service <http://vogate.obs-nancay.fr/tap>`_.

            :returns:
                NenuFAR observation properties resulting from the
                ADQL query.
            :rtype: :class:`~astropy.table.Table`
        """
        if self.conditions == '':
            raise ValueError(
                'Empty query, fill out some attributes first.'
            )
        log.info(
            f'TAP service columns to return: {self._meta_names}.'
        )
        log.debug(
            f"Querying: '{self.query}'"
        )
        result = self.service.search(self.query)
        return result.to_table()


    def reset(self):
        """ Reset query parameters to default values.
        """
        self.meta_names = [
            'target_name',
            'obs_creator_did',
            's_ra',
            's_dec',
            't_min',
            't_max',
            'em_min',
            'em_max'
        ]
        self.time_range = [None, None]
        self.freq_range = [None, None]
        self.fov_radius = 180*u.deg
        self.fov_center = None
        log.info(
            'Query parameters reset to default values.'
        )
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #
# ============================================================= #

