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
from os.path import abspath, isfile, basename, dirname
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, ICRS, solar_system_ephemeris, get_body

from sqlalchemy.ext.declarative import DeferredReflection, declarative_base
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

RECEIVERS = np.array(['undysputed', 'xst', 'nickel', 'seti', 'radiogaga', 'codalema'])
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------------- SchedulingTable ---------------------- #
# ============================================================= #
# class NenufarUserTable(DeferredReflection, Base):
#     """
#         Fake class for NenuFAR User Table
#     """

#     __tablename__ = 'nenufar_users'


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

    # scheduling_id = Column(BigInteger, ForeignKey('scheduling.id', ondelete="CASCADE"))
    # scheduling = relationship(SchedulingTable, cascade="all, delete")
    # obs_name = Column(String(40), nullable=False)
    # contact_name = Column(String(255), nullable=False)
    # contact_email = Column(String(255), nullable=False)
    # key_project_code = Column(String(4), nullable=False)
    # key_project_name = Column(String(100), nullable=False)
    # start_time = Column(DateTime, nullable=False)
    # stop_time = Column(DateTime, nullable=False)
    # parset_file = Column(String(300), nullable=False)
    receivers = relationship("ReceiverAssociation", back_populates='scheduling', cascade="all, delete, delete-orphan")
    nickel_subbands = relationship("SubBandNickelAssociation", back_populates='scheduling', cascade="all, delete, delete-orphan")
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- ReceiverTable ----------------------- #
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
    
    # extra_data = Column(String(50))
    subband = relationship("SubBandTable", back_populates="digital_beams", cascade="all, delete")
    digital_beam = relationship("DigitalBeamTable", back_populates="subbands", cascade="all, delete")


class SubBandNickelAssociation(Base):
    """
    """
    __tablename__ = 'subband_nickel_association'

    scheduling_id = Column(BigInteger, ForeignKey("scheduling.id", ondelete="CASCADE"), primary_key=True)
    subband_id = Column(ForeignKey("subband.id", ondelete="CASCADE"), primary_key=True)
    subband = relationship("SubBandTable", back_populates="scheduling", cascade="all, delete")
    scheduling = relationship("SchedulingTable", back_populates="nickel_subbands", cascade="all, delete")


class SubBandTable(Base):
    """
    """
    __tablename__ = 'subband'

    id = Column(Integer, primary_key=True)
    digital_beams = relationship("SubBandAssociation", back_populates='subband', cascade="all, delete, delete-orphan")
    scheduling = relationship("SubBandNickelAssociation", back_populates='subband', cascade="all, delete, delete-orphan")
    index = Column(String(3), nullable=False)
    frequency_mhz = Column(Float, nullable=False)


# class SubBandNickelTable(Base):
#     """
#     """
#     __tablename__ = 'subband_nickel'

#     id = Column(Integer, primary_key=True)
#     scheduling = relationship("SubBandNickelAssociation", back_populates='subband', cascade="all, delete, delete-orphan")
#     index = Column(String(3), nullable=False)
#     frequency_mhz = Column(Float, nullable=False)
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

    ra_j2000 = Column(Float, nullable=True)
    dec_j2000 = Column(Float, nullable=True)
    observed_coord_type = Column(String(50), nullable=False)
    observed_pointing_type = Column(String(50), nullable=False)
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

    ra_j2000 = Column(Float, nullable=True)
    dec_j2000 = Column(Float, nullable=True)
    observed_coord_type = Column(String(50), nullable=False)
    observed_pointing_type = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    stop_time = Column(DateTime, nullable=False)
    # subBands = Column(String(500), nullable=False)
    subbands = relationship("SubBandAssociation", back_populates='digital_beam', cascade="all, delete, delete-orphan")
    freq_min_mhz = Column(Float, nullable=False)
    freq_max_mhz = Column(Float, nullable=False)
    processing = Column(String(255), nullable=True)


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

class UserNameNotFound(Exception):
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
            parset_entry = self.session.query(SchedulingTable).filter_by(fileName=basename(p)).first()
            if parset_entry is not None:
                if inspect(self.engine).has_table("analogbeam"):
                    scheduling_id = parset_entry.id
                    entry = self.session.query(AnalogBeamTable).filter_by(scheduling_id=scheduling_id).first()
                    if entry is not None:
                        log.info(f"Parset {basename(p)} already in {self.name}. Skipping it.")
                        raise DuplicateParsetEntry(f"Duplicated parset {basename(p)}.")
        self._parset = p


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    @classmethod
    def new(cls, database_name='my_base.db', engine=None):
        """ """
        if engine is None:
            # Create a default engine
            engine = create_engine(
                'sqlite:///' + database_name,
                pool_pre_ping=True
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

        # if ("subband_nickel" in existing_tables) and (self.session.query(SubBandNickelTable).first() is not None):
        #     log.warning("'subband_nickel' table already exists.")
        # else:
        #     # Initialize the SubBand Table
        #     log.debug("Generating the 'subband_nickel' table.")
        #     self.session.add_all([
        #         SubBandNickelTable(index=str(subband), frequency_mhz=sb2freq(subband)[0].value)
        #         for subband in SUB_BANDS
        #     ])

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
            decal_ra = float(parset_property.get("decal_ra", 0.0))*u.deg
            decal_dec = float(parset_property.get("decal_dec", 0.0))*u.deg
            right_ascension = (parset_property['angle1'].to(u.deg) + decal_ra).value
            declination = (parset_property['angle2'].to(u.deg) + decal_dec).value

        elif direction_type == "azelgeo":
            # This is a transit observation, compute the mean RA/Dec
            log.debug(f"'{direction_type}' beam direction type, taking the mean RA/Dec.")
            # Convert AltAz to RA/Dec
            radec = SkyCoord(
                parset_property['angle1'] + float(parset_property.get("decal_az", 0.0))*u.deg,
                parset_property['angle2'] + float(parset_property.get("decal_el", 0.0))*u.deg,
                frame=AltAz(
                    obstime=start_time + duration/2.,
                    location=nenufar_position
                )
            ).transform_to(ICRS)
            right_ascension = radec.ra.deg
            declination = radec.dec.deg
        
        elif direction_type == "natif":
            # This is a test observation, unable to parse the RA/Dec
            log.debug(f"'{direction_type}' beam direction type, RA/Dec fields will be empty.")
            right_ascension = None
            declination = None

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
            decal_ra = float(parset_property.get("decal_ra", 0.0))*u.deg
            decal_dec = float(parset_property.get("decal_dec", 0.0))*u.deg
            right_ascension = radec.ra.deg + decal_ra.value
            declination = radec.dec.deg + decal_dec.value

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
        if parset_property.get("xst_userfile", False):
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

        scheduling_row = self.session.query(SchedulingTable).filter_by(fileName=basename(self.parset)).first()
        if scheduling_row is None:
            username = "testobs" if parset_property["contactName"]=="" else parset_property["contactName"]

            # Check if 'username' exists
            if inspect(self.engine).has_table("nenufar_users"):
                class NenufarUserTable(DeferredReflection, Base):
                    """
                        Fake class for NenuFAR User Table
                    """
                    __tablename__ = 'nenufar_users'

                DeferredReflection.prepare(self.engine)
                username_entry = self.session.query(NenufarUserTable).filter_by(username=username).first()
                if username_entry is None:
                    log.warning(f"Username '{username}' not found in 'nenufar_users' table, skipping it.")
                    raise UserNameNotFound(f"'{username}'")

            # Sort out the topic
            topic = parset_property.get("topic", "ES00 DEBUG")
            if topic.lower().strip() == 'maintenance':
                topic = "MAINTENANCE"
            else:
                # We have something like "ES00 DEBUG"
                topic = topic.split(" ", 1)[1]

            # Create the new row
            scheduling_row = SchedulingTable(
                name=parset_property['name'],
                fileName=basename(self.parset),
                path=dirname(self.parset),
                startTime=parset_property["startTime"].datetime,
                endTime=parset_property["stopTime"].datetime,
                state="default_value",
                topic=topic,
                username=username,
                receivers=[ReceiverAssociation(receiver=receiver) for receiver in receivers]
            )
            is_new = True
            log.debug(f"Row of table 'scheduling' created for '{scheduling_row.name}'.")
        else:
            # Only add the receiver association, the row already exists in scheduling table
            [ReceiverAssociation(receiver=receiver, scheduling=scheduling_row) for receiver in receivers]
            is_new = False
            log.debug(f"Row of table 'scheduling' updated for '{scheduling_row.name}'.")
        
        if "nickel" in receivers_on:
            log.debug("Adding the association to NICKEL subbands.")
            nickel_subbands = self.session.query(SubBandTable).filter(SubBandTable.index.in_(parset_property.get("nri_subbandList", []))).all()
            [SubBandNickelAssociation(subband=sb, scheduling=scheduling_row) for sb in nickel_subbands]

        return scheduling_row, is_new


    def _create_analog_beam_row(self, parset_property):
        """ """

        log.debug(f"Treating 'analogbeam' (index {parset_property['anaIdx']})...")

        pointing = self._normalize_beam_pointing(parset_property)
        
        # Link to Antenna
        antennas = self.session.query(AntennaTable).filter(AntennaTable.name.in_(parset_property['antList'])).all()
        #antennas_assoc = [_AntennaAssociation(antenna=ant) for ant in antennas]
        # Link to Mini-Arrays
        miniarrays = self.session.query(MiniArrayTable).filter(MiniArrayTable.name.in_(parset_property['maList'])).all()
        #for ma in miniarrays:
        #    ma.antennas = antennas_assoc

        analog_beam_row = AnalogBeamTable(
            ra_j2000 = pointing["ra"],
            dec_j2000 = pointing["dec"],
            observed_coord_type = parset_property['directionType'],
            observed_pointing_type = 'TRANSIT' if parset_property['directionType'] == 'AZELGEO' else 'TRACKING',
            start_time = pointing["start_time"],
            stop_time = pointing["stop_time"],
            # nMiniArrays = len(pProp['maList']),
            # _miniArrays = pProp['maList'],
            mini_arrays = [MiniArrayAssociation(mini_array=ma, antenna=ant) for ma in miniarrays for ant in antennas],
            # nAntennas = len(pProp['antList']),
            #_antennas = pProp['antList'],
            beam_squint_freq_mhz = parset_property['optFrq'] if parset_property.get("beamSquint", False) else 0,
            scheduling = self.current_scheduling
        )

        log.debug(f"Row of table 'analogbeam' (index {parset_property['anaIdx']}) created for '{self.current_scheduling.name}'.")

        return analog_beam_row, True


    def _create_digital_beam_row(self, parset_property):
        """ """

        log.debug(f"Treating 'digitalbeam' (index {parset_property['digiIdx']})...")

        pointing = self._normalize_beam_pointing(parset_property)
        
        # Link to Sub-Bands
        subbands = self.session.query(SubBandTable).filter(SubBandTable.index.in_(parset_property['subbandList'])).all()

        digital_beam_row = DigitalBeamTable(
            ra_j2000 = pointing["ra"],
            dec_j2000 = pointing["dec"],
            observed_coord_type = parset_property["directionType"],
            observed_pointing_type = 'TRANSIT' if parset_property["directionType"] == 'AZELGEO' else 'TRACKING',
            start_time = pointing["start_time"],
            stop_time = pointing["stop_time"],
            # _subBands = pProp['subbandList'],
            subbands = [SubBandAssociation(subband=sb) for sb in subbands],
            freq_min_mhz = sb2freq( max(min(parset_property['subbandList']), 0) )[0].value, 
            freq_max_mhz = sb2freq( min(max(parset_property['subbandList']), 511) )[0].value,
            anabeam = self.anaid[parset_property['noBeam']],
            processing = parset_property["toDo"]
        )

        log.debug(f"Row of table 'digitalbeam' (index {parset_property['digiIdx']}) created for '{self.current_scheduling.name}'.")

        return digital_beam_row, True
# ============================================================= #
# ============================================================= #

