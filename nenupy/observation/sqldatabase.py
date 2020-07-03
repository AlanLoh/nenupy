#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ************
    SQL Database
    ************

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    '_ObservationTable',
    '_AnalogBeamTable',
    '_DigitalBeamTable',
    'ParsetDataBase'
]


import numpy as np
from os.path import abspath, isfile
from astropy.time import TimeDelta

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    create_engine
)
from sqlalchemy.orm import sessionmaker, relationship

from nenupy.instru import sb2freq

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


Base = declarative_base()


# ============================================================= #
# --------------------- _ObservationTable --------------------- #
# ============================================================= #
class _ObservationTable(Base):
    """
    """

    __tablename__ = 'observation'

    id = Column(Integer, primary_key=True)

    obsName = Column(String, nullable=False)
    contactName = Column(String, nullable=False)
    contactEmail = Column(String, nullable=False)
    keyProjectCode = Column(String, nullable=False)
    keyProjectName = Column(String, nullable=False)
    startTime = Column(DateTime, nullable=False)
    stopTime = Column(DateTime, nullable=False)
    nAnaBeams = Column(Integer, nullable=False)
    nDigiBeams = Column(Integer, nullable=False)
    setiON = Column(Boolean, nullable=False)
    undysputedON = Column(Boolean, nullable=False)
    nickelON = Column(Boolean, nullable=False)
    sstON = Column(Boolean, nullable=False)
    bstON = Column(Boolean, nullable=False)
    xstON = Column(Boolean, nullable=False)
    parsetFile = Column(String, nullable=False)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- _AnalogBeamTable ---------------------- #
# ============================================================= #
class _AnalogBeamTable(Base):
    """
    """

    __tablename__ = 'analogbeam'

    id = Column(Integer, primary_key=True)
    observation_id = Column(Integer, ForeignKey('observation.id'))
    observation = relationship(_ObservationTable)

    angle1 = Column(Float, nullable=False)
    angle2 = Column(Float, nullable=False)
    coordType = Column(String, nullable=False)
    pointingType = Column(String, nullable=False)
    startTime = Column(DateTime, nullable=False)
    stopTime = Column(DateTime, nullable=False)
    nMiniArrays = Column(Integer, nullable=False)
    miniArrays = Column(String, nullable=False)
    nAntennas = Column(Integer, nullable=False)
    antennas = Column(String, nullable=False)
    beamSquintFreq = Column(Float, nullable=False)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def _miniArrays(self):
        return list(map(int, self.miniArrays.split(',')))
    @_miniArrays.setter
    def _miniArrays(self, m):
        if not isinstance(m, (list, np.ndarray)):
            raise TypeError(
                'miniarrays should be a list-like object'
            )
        self.miniArrays = ','.join([str(mi) for mi in m])


    @property
    def _antennas(self):
        return list(map(int, self.antennas.split(',')))
    @_antennas.setter
    def _antennas(self, a):
        if not isinstance(a, (list, np.ndarray)):
            raise TypeError(
                'antennas should be a list-like object'
            )
        self.antennas = ','.join([str(ai) for ai in a])
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- _DigitalBeamTable --------------------- #
# ============================================================= #
class _DigitalBeamTable(Base):
    """
    """

    __tablename__ = 'digitalbeam'

    id = Column(Integer, primary_key=True)
    anabeam_id = Column(Integer, ForeignKey('analogbeam.id'))
    anabeam = relationship(_AnalogBeamTable)

    angle1 = Column(Float, nullable=False)
    angle2 = Column(Float, nullable=False)
    coordType = Column(String, nullable=False)
    pointingType = Column(String, nullable=False)
    startTime = Column(DateTime, nullable=False)
    stopTime = Column(DateTime, nullable=False)
    subBands = Column(String, nullable=False)
    fMin = Column(Float, nullable=False)
    fMax = Column(Float, nullable=False)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def _subBands(self):
        return list(map(int, self.subBands.split(',')))
    @_subBands.setter
    def _subBands(self, s):
        if not isinstance(s, (list, np.ndarray)):
            raise TypeError(
                'subBands should be a list-like object'
            )
        self.subBands = ','.join([str(si) for si in s])
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ParsetDataBase ---------------------- #
# ============================================================= #
class ParsetDataBase(object):
    """
    """

    def __init__(self, dataBaseName):
        self.engine = ''
        self.name = dataBaseName
        self.parset = None
        Base.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()
        self.obsid = None
        self.anaid = {}

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, n):
        if not isinstance(n, str):
            raise TypeError(
                'name should be a string'
            )
        if not n.endswith('.db'):
            raise ValueError(
                'name should end with .db'
            )
        self._name = abspath(n)
        self.engine = create_engine(
            'sqlite:///' + self._name
        )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def addTable(self, parsetProperty, desc):
        """
        """
        pProp = parsetProperty

        if desc.lower() == 'observation':
            newRow = _ObservationTable(
                obsName=pProp['name'],
                contactName=pProp['contactName'],
                contactEmail=pProp['contactEmail'],
                keyProjectCode=pProp['topic'].split(' ', 1)[0],
                keyProjectName=pProp['topic'].split(' ', 1)[1],
                startTime=pProp['startTime'].datetime,
                stopTime=pProp['stopTime'].datetime,
                nAnaBeams=pProp['nrAnaBeams'],
                nDigiBeams=pProp['nrBeams'],
                setiON='seti' in pProp.get('hd_receivers', []),
                undysputedON='undysputed' in pProp.get('hd_receivers', []),
                nickelON='nickel' in pProp.get('nri_receivers', []),
                sstON=pProp['sst_userfile'],
                bstON=pProp['bst_userfile'],
                xstON=pProp['xst_userfile'],
                parsetFile=self.parset
            )
            self.obsid = newRow

        elif desc.lower() == 'anabeam':
            duration = TimeDelta(pProp['duration'] , format='sec')
            if pProp['directionType'] not in ['J2000', 'AZELGEO']:
                # This is a Solar System observation
                pProp['angle1'] = '999'
                pProp['angle2'] = '999'
            newRow = _AnalogBeamTable(
                angle1 = pProp['angle1'].value,
                angle2 = pProp['angle2'].value,
                coordType = pProp['directionType'],
                pointingType = 'TRANSIT' if pProp['directionType'] == 'AZELGEO' else 'TRACKING',
                startTime = pProp['startTime'].datetime,
                stopTime = (pProp['startTime'] + duration).datetime,
                nMiniArrays = len(pProp['maList']),
                _miniArrays = pProp['maList'],
                nAntennas = len(pProp['antList']),
                _antennas = pProp['antList'],
                beamSquintFreq = pProp['optFrq'] if pProp['beamSquint'] else 0,
                observation = self.obsid
            )
            self.anaid[pProp['anaIdx']] = newRow

        elif desc.lower() == 'digibeam':
            duration = TimeDelta(pProp['duration'] , format='sec')
            if pProp['directionType'] not in ['J2000', 'AZELGEO']:
                # This is a Solar System observation
                pProp['angle1'] = '999'
                pProp['angle2'] = '999'
            newRow = _DigitalBeamTable(
                angle1 = pProp['angle1'].value,
                angle2 = pProp['angle2'].value,
                coordType = pProp['directionType'],
                pointingType = 'TRANSIT' if pProp['directionType'] == 'AZELGEO' else 'TRACKING',
                startTime = pProp['startTime'].datetime,
                stopTime = (pProp['startTime'] + duration).datetime,
                _subBands = pProp['subbandList'],
                fMin = sb2freq(min(pProp['subbandList'])).value, 
                fMax = sb2freq(max(pProp['subbandList'])).value,
                anabeam = self.anaid[pProp['noBeam']]
            )

        else:
            raise ValueError(
                'desc should be observation/anabeam/digibeam'
            )

        self.session.add(newRow)
        self.session.commit()
# ============================================================= #
# ============================================================= #

