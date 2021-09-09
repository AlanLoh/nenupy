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

    obsName = Column(String(40), nullable=False)
    contactName = Column(String(50), nullable=False)
    contactEmail = Column(String(100), nullable=False)
    keyProjectCode = Column(String(4), nullable=False)
    keyProjectName = Column(String(100), nullable=False)
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
    parsetFile = Column(String(300), nullable=False)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- _MiniArrayTable ---------------------- #
# ============================================================= #
class _MiniArrayAssociation(Base):
    """
    """
    __tablename__ = 'mini_array_association'

    analog_beam_id = Column(ForeignKey("analogbeam.id"), primary_key=True)
    mini_array_id = Column(ForeignKey("miniarray.id"), primary_key=True)
    
    extra_data = Column(String(50))
    mini_array = relationship("_MiniArrayTable", back_populates="analog_beams")
    analog_beam = relationship("_AnalogBeamTable", back_populates="mini_arrays")


class _MiniArrayTable(Base):
    """
    """
    __tablename__ = 'miniarray'

    id = Column(Integer, primary_key=True)
    analog_beams = relationship("_MiniArrayAssociation", back_populates='mini_array')
    name = Column(String(2), nullable=False)
    antennas = relationship("_AntennaAssociation", back_populates='mini_array')
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- _AntennaTable ----------------------- #
# ============================================================= #
class _AntennaAssociation(Base):
    """
    """
    __tablename__ = 'antenna_association'

    mini_array_id = Column(ForeignKey("miniarray.id"), primary_key=True)
    antenna_id = Column(ForeignKey("antenna.id"), primary_key=True)
    
    extra_data = Column(String(50))
    antenna = relationship("_AntennaTable", back_populates="mini_arrays")
    mini_array = relationship("_MiniArrayTable", back_populates="antennas")


class _AntennaTable(Base):
    """
    """
    __tablename__ = 'antenna'

    id = Column(Integer, primary_key=True)
    name = Column(String(2), nullable=False)
    mini_arrays = relationship("_AntennaAssociation", back_populates='antenna')
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- _SubBandTable ----------------------- #
# ============================================================= #
class _SubBandAssociation(Base):
    """
    """
    __tablename__ = 'subband_association'

    digital_beam_id = Column(ForeignKey("digitalbeam.id"), primary_key=True)
    subband_id = Column(ForeignKey("subband.id"), primary_key=True)
    
    extra_data = Column(String(50))
    subband = relationship("_SubBandTable", back_populates="digital_beams")
    digital_beam = relationship("_DigitalBeamTable", back_populates="subbands")


class _SubBandTable(Base):
    """
    """
    __tablename__ = 'subband'

    id = Column(Integer, primary_key=True)
    digital_beams = relationship("_SubBandAssociation", back_populates='subband')
    index = Column(String(3), nullable=False)
    frequency_mhz = Column(Float, nullable=False)
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
    coordType = Column(String(50), nullable=False)
    pointingType = Column(String(50), nullable=False)
    startTime = Column(DateTime, nullable=False)
    stopTime = Column(DateTime, nullable=False)
    nMiniArrays = Column(Integer, nullable=False)
    # miniArrays = Column(String(500), nullable=False)
    mini_arrays = relationship("_MiniArrayAssociation", back_populates='analog_beam')
    nAntennas = Column(Integer, nullable=False)
    #antennas = Column(String(200), nullable=False)
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
    coordType = Column(String(50), nullable=False)
    pointingType = Column(String(50), nullable=False)
    startTime = Column(DateTime, nullable=False)
    stopTime = Column(DateTime, nullable=False)
    # subBands = Column(String(500), nullable=False)
    subbands = relationship("_SubBandAssociation", back_populates='digital_beam')
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

    def __init__(self, dataBaseName, engine=None):
        self.name = dataBaseName
        self.engine = engine
        self.parset = None
        Base.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()
        self.obsid = None
        self.anaid = {}

        # Initialize the Mini-Array Table
        self.session.add_all([
            _MiniArrayTable(name=str(miniarray_name))
            for miniarray_name in range(96)
        ])
        self.session.commit()

        # Initialize the Antenna Table
        self.session.add_all([
            _AntennaTable(name=str(antenna_name))
            for antenna_name in range(19)
        ])
        self.session.commit()

        # Initialize the SubBand Table
        self.session.add_all([
            _SubBandTable(index=str(subband), frequency_mhz=sb2freq(subband)[0].value)
            for subband in range(512)
        ])
        self.session.commit()


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


    @property
    def engine(self):
        return self._engine
    @engine.setter
    def engine(self, e):
        if e is None:
            e = create_engine(
                'sqlite:///' + self.name
            )
        self._engine = e

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def done(self):
        """
        """
        self.engine.dispose()


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
            
            # Link to Antenna
            # antennas = self.session.query(_AntennaTable).filter(_AntennaTable.name.in_(pProp['antList'])).all()

            # Link to Mini-Arrays
            miniarrays = self.session.query(_MiniArrayTable).filter(_MiniArrayTable.name.in_(pProp['maList'])).all()

            newRow = _AnalogBeamTable(
                angle1 = pProp['angle1'].value,
                angle2 = pProp['angle2'].value,
                coordType = pProp['directionType'],
                pointingType = 'TRANSIT' if pProp['directionType'] == 'AZELGEO' else 'TRACKING',
                startTime = pProp['startTime'].datetime,
                stopTime = (pProp['startTime'] + duration).datetime,
                nMiniArrays = len(pProp['maList']),
                # _miniArrays = pProp['maList'],
                mini_arrays = [_MiniArrayAssociation(mini_array=ma, extra_data='extra') for ma in miniarrays],#[ma for ma in miniarrays],
                nAntennas = len(pProp['antList']),
                #_antennas = pProp['antList'],
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
            
            # Link to Sub-Bands
            subbands = self.session.query(_SubBandTable).filter(_SubBandTable.index.in_(pProp['subbandList'])).all()

            newRow = _DigitalBeamTable(
                angle1 = pProp['angle1'].value,
                angle2 = pProp['angle2'].value,
                coordType = pProp['directionType'],
                pointingType = 'TRANSIT' if pProp['directionType'] == 'AZELGEO' else 'TRACKING',
                startTime = pProp['startTime'].datetime,
                stopTime = (pProp['startTime'] + duration).datetime,
                # _subBands = pProp['subbandList'],
                subbands = [_SubBandAssociation(subband=sb) for sb in subbands],
                fMin = sb2freq(min(pProp['subbandList']))[0].value, 
                fMax = sb2freq(max(pProp['subbandList']))[0].value,
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

