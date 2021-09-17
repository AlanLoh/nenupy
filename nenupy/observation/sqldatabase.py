#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    ************
    SQL Database
    ************

    Query obs containing 19 antennas in database:

    SELECT * 
    FROM observation o 
        inner join analogbeam a
        on o.id = a.observation_id
    where a.nAntennas = 19;

    Query obs containing MA 55 in database:

    select * 
    from observation o
    inner join analogbeam a
        on o.id = a.observation_id
        inner join mini_array_association aa
            on a.id = aa.analog_beam_id
            inner join miniarray ma
                on ma.id = aa.mini_array_id
    where ma.name = 55;

    from nenupy.observation import ParsetDataBase
    from nenupy.observation import Parset
    from sqlalchemy import create_engine
    import os

    os.remove('/Users/aloh/Desktop/ma_base.db')
    db = ParsetDataBase(dataBaseName='/Users/aloh/Desktop/ma_base.db')#, engine=create_engine('mysql:///'))
    parset = Parset('/Users/aloh/Desktop/es11-2021-06-04-crab.parset')
    parset.addToDatabase(data_base=db)
    parset2 = Parset('/Users/aloh/Desktop/parset/test_alan.parset')
    parset2.addToDatabase(data_base=db)
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'SchedulingTable',
    'AnalogBeamTable',
    'DigitalBeamTable',
    'ParsetDataBase'
]


import numpy as np
from os.path import abspath, isfile
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz, ICRS, solar_system_ephemeris, get_body

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.inspection import inspect
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    BigInteger,
    String,
    Float,
    Boolean,
    DateTime,
    create_engine,
)
from sqlalchemy.orm import Session, sessionmaker, relationship

from nenupy.instru import sb2freq
from nenupy import nenufar_position

import logging
log = logging.getLogger(__name__)


Base = declarative_base()

# ============================================================= #
# ------------------------- Constants ------------------------- #
# ============================================================= #
MINI_ARRAYS = np.concatenate(
    (np.arange(96, dtype=int), np.arange(100, 107, dtype=int))
)

ANTENNAS = np.arange(1, 20, dtype=int)

SUB_BANDS = np.arange(512, dtype=int)

RECEIVERS = np.array(['undysputed', 'xst', 'nickel', 'seti', 'radiogaga'])
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- SchedulingTable ---------------------- #
# ============================================================= #
class SchedulingTable(Base):
    """
    """

    __tablename__ = 'scheduling'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    fileName = Column(String(255), nullable=False)
    path = Column(String(255), nullable=True)
    comments = Column(String(255), nullable=True)
    checkTime = Column(DateTime, nullable=True)
    startTime = Column(DateTime, nullable=False)
    endTime = Column(DateTime, nullable=False)
    abortTime = Column(DateTime, nullable=True)
    state = Column(String(30), nullable=False)
    status = Column(String(30), nullable=False, default="unknown")
    other_error = Column(String(150), nullable=True)
    type = Column(String(30), nullable=False, default="unknown")
    topic = Column(String(255), nullable=False, default="debug")
    tags = Column(String(255), nullable=True)
    submitTime = Column(DateTime, nullable=False, default=Time.now().datetime)
    token = Column(String(255), nullable=True)
    username = Column(String(255), nullable=False, default="testobs")
    checker_username = Column(String(255), nullable=True)
    # PRIMARY KEY (`id`),
    # UNIQUE KEY `fileName` (`fileName`),
    # KEY `username` (`username`),
    # KEY `topic` (`topic`),
    # CONSTRAINT `contrainte_topic` FOREIGN KEY (`topic`) REFERENCES `key_projects` (`value`) ON DELETE NO ACTION ON UPDATE NO ACTION,
    # CONSTRAINT `contrainte_username` FOREIGN KEY (`username`) REFERENCES `nenufar_users` (`username`)
    # ) ENGINE=InnoDB AUTO_INCREMENT=51947 DEFAULT CHARSET=latin1

    # obs_name = Column(String(40), nullable=False)
    # contact_name = Column(String(255), nullable=False)
    # contact_email = Column(String(255), nullable=False)
    # key_project_code = Column(String(4), nullable=False)
    # key_project_name = Column(String(100), nullable=False)
    # start_time = Column(DateTime, nullable=False)
    # stop_time = Column(DateTime, nullable=False)
    # parset_file = Column(String(300), nullable=False)
    receivers = relationship("ReceiverAssociation", back_populates='scheduling', cascade="all, delete, delete-orphan")
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- _MiniArrayTable ---------------------- #
# ============================================================= #
class ReceiverAssociation(Base):
    """ """
    __tablename__ = "receiver_association"

    scheduling_id = Column(BigInteger, ForeignKey("scheduling.id", ondelete="CASCADE"), primary_key=True)
    receiver_id = Column(ForeignKey("receivers.id", ondelete="CASCADE"), primary_key=True)
    
    receiver = relationship("ReceiverTable", back_populates="schedulings", cascade="all, delete")
    scheduling = relationship("SchedulingTable", back_populates="receivers", cascade="all, delete")


class ReceiverTable(Base):
    """ """
    __tablename__ = 'receivers'

    id = Column(Integer, primary_key=True)
    schedulings = relationship("ReceiverAssociation", back_populates='receiver', cascade="all, delete, delete-orphan")
    name = Column(String(20), nullable=False)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- MiniArrayTable ----------------------- #
# ============================================================= #
class MiniArrayAssociation(Base):
    """
    """
    __tablename__ = 'mini_array_association'

    analog_beam_id = Column(ForeignKey("analogbeam.id", ondelete="CASCADE"), primary_key=True)
    mini_array_id = Column(ForeignKey("miniarray.id", ondelete="CASCADE"), primary_key=True)
    antenna_id = Column(ForeignKey("antenna.id", ondelete="CASCADE"), primary_key=True)
    
    mini_array = relationship("MiniArrayTable", back_populates="analog_beams", cascade="all, delete")
    analog_beam = relationship("AnalogBeamTable", back_populates="mini_arrays", cascade="all, delete")
    antenna = relationship("AntennaTable", back_populates="mini_arrays", cascade="all, delete")

class MiniArrayTable(Base):
    """
    """
    __tablename__ = 'miniarray'

    id = Column(Integer, primary_key=True)
    analog_beams = relationship("MiniArrayAssociation", back_populates='mini_array', cascade="all, delete, delete-orphan")
    name = Column(String(3), nullable=False)
    # antennas = relationship("_AntennaAssociation", back_populates='mini_array')
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- AntennaTable ------------------------ #
# ============================================================= #
# class _AntennaAssociation(Base):
#     """
#     """
#     __tablename__ = 'antenna_association'

#     mini_array_id = Column(ForeignKey("miniarray.id"), primary_key=True)
#     antenna_id = Column(ForeignKey("antenna.id"), primary_key=True)
#     antenna = relationship("_AntennaTable", back_populates="mini_arrays")
#     mini_array = relationship("_MiniArrayTable", back_populates="antennas")


class AntennaTable(Base):
    """
    """
    __tablename__ = 'antenna'

    id = Column(Integer, primary_key=True)
    name = Column(String(2), nullable=False)
    mini_arrays = relationship("MiniArrayAssociation", back_populates='antenna', cascade="all, delete, delete-orphan")
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- SubBandTable ------------------------ #
# ============================================================= #
class SubBandAssociation(Base):
    """
    """
    __tablename__ = 'subband_association'

    digital_beam_id = Column(ForeignKey("digitalbeam.id", ondelete="CASCADE"), primary_key=True)
    subband_id = Column(ForeignKey("subband.id", ondelete="CASCADE"), primary_key=True)
    
    extra_data = Column(String(50))
    subband = relationship("SubBandTable", back_populates="digital_beams", cascade="all, delete")
    digital_beam = relationship("DigitalBeamTable", back_populates="subbands", cascade="all, delete")


class SubBandTable(Base):
    """
    """
    __tablename__ = 'subband'

    id = Column(Integer, primary_key=True)
    digital_beams = relationship("SubBandAssociation", back_populates='subband', cascade="all, delete, delete-orphan")
    index = Column(String(3), nullable=False)
    frequency_mhz = Column(Float, nullable=False)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- AnalogBeamTable ---------------------- #
# ============================================================= #
class AnalogBeamTable(Base):
    """
    """

    __tablename__ = 'analogbeam'

    id = Column(Integer, primary_key=True)
    scheduling_id = Column(BigInteger, ForeignKey('scheduling.id', ondelete="CASCADE"))
    scheduling = relationship(SchedulingTable, cascade="all, delete")

    angle1 = Column(Float, nullable=False)
    angle2 = Column(Float, nullable=False)
    coord_type = Column(String(50), nullable=False)
    pointing_type = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    stop_time = Column(DateTime, nullable=False)
    # nMiniArrays = Column(Integer, nullable=False)
    # miniArrays = Column(String(500), nullable=False)
    mini_arrays = relationship("MiniArrayAssociation", back_populates='analog_beam')
    # nAntennas = Column(Integer, nullable=False)
    #antennas = Column(String(200), nullable=False)
    beam_squint_freq_mhz = Column(Float, nullable=False)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    # @property
    # def _miniArrays(self):
    #     return list(map(int, self.miniArrays.split(',')))
    # @_miniArrays.setter
    # def _miniArrays(self, m):
    #     if not isinstance(m, (list, np.ndarray)):
    #         raise TypeError(
    #             'miniarrays should be a list-like object'
    #         )
    #     self.miniArrays = ','.join([str(mi) for mi in m])


    # @property
    # def _antennas(self):
    #     return list(map(int, self.antennas.split(',')))
    # @_antennas.setter
    # def _antennas(self, a):
    #     if not isinstance(a, (list, np.ndarray)):
    #         raise TypeError(
    #             'antennas should be a list-like object'
    #         )
    #     self.antennas = ','.join([str(ai) for ai in a])
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- DigitalBeamTable ---------------------- #
# ============================================================= #
class DigitalBeamTable(Base):
    """
    """

    __tablename__ = 'digitalbeam'

    id = Column(Integer, primary_key=True)
    anabeam_id = Column(Integer, ForeignKey('analogbeam.id', ondelete="CASCADE"))
    anabeam = relationship(AnalogBeamTable, cascade="all, delete")

    angle1 = Column(Float, nullable=False)
    angle2 = Column(Float, nullable=False)
    coord_type = Column(String(50), nullable=False)
    pointing_type = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    stop_time = Column(DateTime, nullable=False)
    # subBands = Column(String(500), nullable=False)
    subbands = relationship("SubBandAssociation", back_populates='digital_beam', cascade="all, delete, delete-orphan")
    freq_min_mhz = Column(Float, nullable=False)
    freq_max_mhz = Column(Float, nullable=False)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    # @property
    # def _subBands(self):
    #     return list(map(int, self.subBands.split(',')))
    # @_subBands.setter
    # def _subBands(self, s):
    #     if not isinstance(s, (list, np.ndarray)):
    #         raise TypeError(
    #             'subBands should be a list-like object'
    #         )
    #     self.subBands = ','.join([str(si) for si in s])
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ParsetDataBase ---------------------- #
# ============================================================= #
class DuplicateParsetEntry(Exception):
    pass

class ParsetDataBase(object):
    """
    """

    def __init__(self, database_name, engine=None, session=None):
        self.name = database_name
        self.engine = engine
        self.session = session

        log.info(f"Session started on {self.engine.url}.")

        self.parset = None
        self.current_scheduling = None
        self.anaid = {}


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def parset(self):
        return self._parset
    @parset.setter
    def parset(self, p):
        # if p is not None:
        #     # Check if an entry already exists
        #     if inspect(self.engine).has_table("scheduling"):
        #         entry_exists = self.session.query(SchedulingTable).filter_by(fileName=p).first() is not None
        #         if entry_exists:
        #             log.info(f"Parset {p} already in {self.name}. Skipping it.")
        #             raise DuplicateParsetEntry(f"Duplicated parset {p}.")
        
        if p is not None:
            parset_entry = self.session.query(SchedulingTable).filter_by(fileName=p).first()
            if parset_entry is not None:
                if inspect(self.engine).has_table("receiver_association"):
                    scheduling_id = parset_entry.id
                    entry = self.session.query(ReceiverAssociation).filter_by(scheduling_id=scheduling_id).first()
                    if entry is not None:
                        log.info(f"Parset {p} already in {self.name}. Skipping it.")
                        raise DuplicateParsetEntry(f"Duplicated parset {p}.")
        self._parset = p


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def new(cls, database_name='my_base.db', engine=None):
        """ """
        if engine is None:
            # Create a default engine
            engine = create_engine(
                'sqlite:///' + database_name
            )
        Base.metadata.create_all(engine)
        DBSession = sessionmaker(bind=engine)
        session = DBSession()
        return cls(database_name=database_name, engine=engine, session=session)


    @classmethod
    def from_existing_database(cls, engine):
        """ """
        Base = automap_base()
        Base.prepare(engine, reflect=True)
        session = Session(engine)
        name = engine.url.database
        return cls(database_name=name, engine=engine, session=session)


    def create_configuration_tables(self):
        """ Creates the tables 'mini_arrays', 'antennas', 'sub-bands' and 'receivers'. """

        existing_tables = inspect(self.engine).get_table_names()

        if ("miniarray" in existing_tables) and (self.session.query(MiniArrayTable).first() is not None):
            log.warning("'miniarrays' table already exists.")
        else:
            # Initialize the Mini-Array Table (96 core + 6 remote Mini-Arrays)
            log.debug("Generating the 'miniarrays' table.")
            self.session.add_all([
                MiniArrayTable(name=str(miniarray_name))
                for miniarray_name in MINI_ARRAYS
            ])

        if ("antenna" in existing_tables) and (self.session.query(AntennaTable).first() is not None):
            log.warning("'antenna' table already exists.")
        else:
            # Initialize the Antenna Table
            log.debug("Generating the 'antennas' table.")
            self.session.add_all([
                AntennaTable(name=str(antenna_name))
                for antenna_name in ANTENNAS
            ])

        if ("subband" in existing_tables) and (self.session.query(SubBandTable).first() is not None):
            log.warning("'subband' table already exists.")
        else:
            # Initialize the SubBand Table
            log.debug("Generating the 'sub-bands' table.")
            self.session.add_all([
                SubBandTable(index=str(subband), frequency_mhz=sb2freq(subband)[0].value)
                for subband in SUB_BANDS
            ])

        if ("receivers" in existing_tables) and (self.session.query(ReceiverTable).first() is not None):
            log.warning("'receivers' table already exists.")
        else:
            # Initialize the Receiver Table
            log.debug("Generating the 'receivers' table.")
            self.session.add_all([
                ReceiverTable(name=receiver)
                for receiver in RECEIVERS
            ])

        # Commit the changes
        self.session.commit()
        log.info("Tables 'mini-arrays', 'antennas', 'sub-bands' and 'receivers' ready.")


    def create_association_tables(self):
        """ """


    def done(self):
        """
        """
        self.engine.dispose()
    

    def delete_row(self, scheduling_id):
        """ """
        self.session.query(SchedulingTable).filter_by(id=scheduling_id).delete()
        self.session.commit()


    def add_row(self, parset_property, desc):
        """
        """
        pProp = parset_property

        if desc.lower() == 'observation':
            new_row, is_new = self._create_scheduling_row(parset_property)

            # Keep track of current scheduling row
            self.current_scheduling = new_row

        elif desc.lower() == 'anabeam':
            new_row, is_new = self._create_analog_beam_row(parset_property)

            # Keep track of analog beam rows
            self.anaid[pProp['anaIdx']] = new_row

        elif desc.lower() == 'digibeam':
            new_row, is_new = self._create_digital_beam_row(parset_property)

        else:
            raise ValueError(
                'desc should be observation/anabeam/digibeam'
            )

        if is_new:
            self.session.add(new_row)
        self.session.commit()


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _normalize_beam_pointing(parset_property) -> dict:
        """ Returns a RA, Dec whatever the pointing type is. """

        # Sort out the beam start and stop times
        duration = TimeDelta(parset_property['duration'] , format='sec')
        start_time = parset_property['startTime']
        stop_time = (parset_property['startTime'] + duration)

        # Deal with coordinates and pointing types
        direction_type = parset_property['directionType'].lower()
        if direction_type == "j2000":
            # Nothing else to do
            log.debug(f"'{direction_type}' beam direction type.")
            right_ascension = parset_property['angle1'].value
            declination = parset_property['angle2'].value

        elif direction_type == "azelgeo":
            # This is a transit observation, compute the mean RA/Dec
            log.debug(f"'{direction_type}' beam direction type, taking the mean RA/Dec.")
            # Convert AltAz to RA/Dec
            radec = SkyCoord(
                parset_property['angle1'],
                parset_property['angle2'],
                frame=AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
            right_ascension = radec.ra.deg
            declination = radec.dec.deg

        else:
            # Dealing with a Solar System source
            log.debug(f"'{direction_type}' beam direction type, taking the mean RA/Dec.")
            with solar_system_ephemeris.set('builtin'):
                source = get_body(
                    body=direction_type,
                    time=start_time + duration/2.,
                    location=nenufar_position
                )
            radec = source.transform_to(ICRS)
            right_ascension = radec.ra.deg
            declination = radec.dec.deg

        return {
            "ra": right_ascension,
            "dec": declination,
            "start_time": start_time.datetime,
            "stop_time": stop_time.datetime
        }


    def _create_scheduling_row(self, parset_property):
        """ """
        # Link to receivers
        receivers_on = parset_property.get("hd_receivers", [])
        receivers_on += parset_property.get("nri_receivers", [])
        if parset_property["xst_userfile"]:
            # Add the xst option, which is not a proper receiver
            receivers_on.append("xst")
        if not np.all(np.isin(receivers_on, RECEIVERS)):
            log.warning(f"One of the receiver listed ({receivers_on}) does not belong to the predefined list ({RECEIVERS}).")

        receivers = self.session.query(ReceiverTable).filter(ReceiverTable.name.in_(receivers_on)).all()

        # scheduling_row = SchedulingTable(
        #     name=parset_property['name'],
        #     contact_name=parset_property['contactName'],
        #     contact_email=parset_property['contactEmail'],
        #     key_project_code=parset_property['topic'].split(' ', 1)[0],
        #     key_project_name=parset_property['topic'].split(' ', 1)[1],
        #     start_time=parset_property['startTime'].datetime,
        #     stop_time=parset_property['stopTime'].datetime,
        #     receivers=[ReceiverAssociation(receiver=receiver) for receiver in receivers],
        #     parset_file=self.parset
        # )

        scheduling_row = self.session.query(SchedulingTable).filter_by(fileName=self.parset).first()
        if scheduling_row is None:
            # Create the new row
            scheduling_row = SchedulingTable(
                name=parset_property['name'],
                fileName=self.parset,
                startTime=parset_property["startTime"].datetime,
                endTime=parset_property["stopTime"].datetime,
                state="default_value",
                topic=parset_property["topic"].split(" ", 1)[1],
                username=parset_property["contactName"],
                receivers=[ReceiverAssociation(receiver=receiver) for receiver in receivers]
            )
            is_new = True
        else:
            # Only add the receiver association, the row already exist in scheduling table
            [ReceiverAssociation(receiver=receiver, scheduling=parset_entry) for receiver in receivers]
            is_new = False

        log.debug(f"Row of table 'scheduling' created for '{scheduling_row.name}'.")

        return scheduling_row, is_new


    def _create_analog_beam_row(self, parset_property):
        """ """

        pointing = self._normalize_beam_pointing(parset_property)
        # duration = TimeDelta(parset_property['duration'] , format='sec')
        # if parset_property['directionType'] not in ['J2000', 'AZELGEO']:
        #     # This is a Solar System observation
        #     parset_property['angle1'] = '999'
        #     parset_property['angle2'] = '999'
        
        # Link to Antenna
        antennas = self.session.query(AntennaTable).filter(AntennaTable.name.in_(parset_property['antList'])).all()
        #antennas_assoc = [_AntennaAssociation(antenna=ant) for ant in antennas]
        # Link to Mini-Arrays
        miniarrays = self.session.query(MiniArrayTable).filter(MiniArrayTable.name.in_(parset_property['maList'])).all()
        #for ma in miniarrays:
        #    ma.antennas = antennas_assoc

        analog_beam_row = AnalogBeamTable(
            angle1 = pointing["ra"],
            angle2 = pointing["dec"],
            coord_type = parset_property['directionType'],
            pointing_type = 'TRANSIT' if parset_property['directionType'] == 'AZELGEO' else 'TRACKING',
            start_time = pointing["start_time"],
            stop_time = pointing["stop_time"],
            # nMiniArrays = len(pProp['maList']),
            # _miniArrays = pProp['maList'],
            mini_arrays = [MiniArrayAssociation(mini_array=ma, antenna=ant) for ma in miniarrays for ant in antennas],
            # nAntennas = len(pProp['antList']),
            #_antennas = pProp['antList'],
            beam_squint_freq_mhz = parset_property['optFrq'] if parset_property['beamSquint'] else 0,
            scheduling = self.current_scheduling
        )

        log.debug(f"Row of table 'analogbeam' (index {parset_property['anaIdx']}) created for '{self.current_scheduling.name}'.")

        return analog_beam_row, True


    def _create_digital_beam_row(self, parset_property):
        """ """

        pointing = self._normalize_beam_pointing(parset_property)

        # duration = TimeDelta(pProp['duration'] , format='sec')
        # if pProp['directionType'] not in ['J2000', 'AZELGEO']:
        #     # This is a Solar System observation / compute the mean radec?
        #     pProp['angle1'] = '999'
        #     pProp['angle2'] = '999'
        
        # Link to Sub-Bands
        subbands = self.session.query(SubBandTable).filter(SubBandTable.index.in_(parset_property['subbandList'])).all()

        digital_beam_row = DigitalBeamTable(
            angle1 = pointing["ra"],
            angle2 = pointing["dec"],
            coord_type = parset_property["directionType"],
            pointing_type = 'TRANSIT' if parset_property["directionType"] == 'AZELGEO' else 'TRACKING',
            start_time = pointing["start_time"],
            stop_time = pointing["stop_time"],
            # _subBands = pProp['subbandList'],
            subbands = [SubBandAssociation(subband=sb) for sb in subbands],
            freq_min_mhz = sb2freq(min(parset_property['subbandList']))[0].value, 
            freq_max_mhz = sb2freq(max(parset_property['subbandList']))[0].value,
            anabeam = self.anaid[parset_property['noBeam']]
        )

        log.debug(f"Row of table 'digitalbeam' (index {parset_property['digiIdx']}) created for '{self.current_scheduling.name}'.")

        return digital_beam_row, True
# ============================================================= #
# ============================================================= #

