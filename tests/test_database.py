#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import pytest

# from nenupy.observation import ObsDatabase

def faketest():
    assert True

# ============================================================= #
# ------------------ test_database_attribute ------------------ #
# ============================================================= #
# def test_database_attribute():
#     db = ObsDatabase()
#     # meta_names
#     with pytest.raises(TypeError):
#         db.meta_names = 2
#     with pytest.raises(ValueError):
#         db.meta_names = ['wrong']
#     db.meta_names = ['s_ra', 's_dec']
#     assert db.meta_names == 's_ra, s_dec'
#     # time_range
#     with pytest.raises(TypeError):
#         db.time_range = 2
#     with pytest.raises(ValueError):
#         db.time_range = [1, 2, 3]
#     db.time_range = [None, None]
#     assert db._conditions['time'] == ''
#     db.time_range = ['2019-01-02', '2019-01-03']
#     assert db._conditions['time'] == '(t_min >= 58485.0 AND t_max <= 58486.0)'
#     # freq_range
#     with pytest.raises(TypeError):
#         db.freq_range = 2
#     with pytest.raises(ValueError):
#         db.freq_range = [1, 2, 3]
#     db.freq_range = [None, None]
#     assert db._conditions['freq'] == ''
#     db.freq_range = [10, 50]
#     assert db._conditions['freq'] == '(em_min >= 5.99584916 AND em_max <= 29.9792458)'
#     # fov_radius
#     with pytest.raises(ValueError):
#         db.fov_radius = [20]
#     db.fov_radius = 20
#     assert isinstance(db.fov_radius, u.Quantity)
#     assert db.fov_radius.to(u.deg).value == 20.
#     # fov_center
#     with pytest.raises(TypeError):
#         db.fov_center = 'wrong'
#     db.fov_center = SkyCoord(300*u.deg, 50*u.deg)
#     assert isinstance(db.fov_center, SkyCoord)
#     assert db.fov_center.ra.deg == 300.
#     assert db._conditions['pos'] == "1 = CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 300.0, 50.0, 20.0))"
#     # Conditions build up
#     db.fov_center = None
#     assert db.conditions == '(t_min >= 58485.0 AND t_max <= 58486.0) AND (em_min >= 5.99584916 AND em_max <= 29.9792458)'
# ============================================================= #


# ============================================================= #
# -------------------- test_database_query -------------------- #
# ============================================================= #
# def test_database_query():
#     db = ObsDatabase()
#     with pytest.raises(ValueError):
#         # Empty query
#         db.search()
#     db.meta_names = ['s_ra']
#     db.time_range = ['2019-01-02', '2019-01-03']
#     assert db.query == 'SELECT s_ra from nenufar.bst WHERE ((t_min >= 58485.0 AND t_max <= 58486.0))'
#     result = db.search()
#     assert isinstance(result, Table)
#     assert len(result) == 21
#     with pytest.raises(ValueError):
#         # Emptying query
#         db.reset()
#         db.search()
# ============================================================= #

