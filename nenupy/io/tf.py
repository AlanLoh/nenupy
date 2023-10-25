"""

    Example :

    from nenupy.io.tf import Spectra
    sp = Spectra("/Users/aloh/Documents/Work/NenuFAR/Undisputed/JUPITER_TRACKING_20230527_083737_0.spectra")

"""

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from typing import Union, Tuple, List
import logging
log = logging.getLogger(__name__)

import nenupy.io.tf_utils as utils
from nenupy.beamlet import SData


BLOCK_HEADER = [
    ("idx", "uint64"), # first effective sample index used for this block
    ("TIMESTAMP", "uint64"), # TIMESTAMP of first sample used for this block
    ("BLOCKSEQNUMBER", "uint64"), # BLOCKSEQNUMBER of first sample used for this block
    ("fftlen", "int32"),  # fftlen : FFT length -> freq resolution= 0.1953125/fftlen MHz
    ("nfft2int", "int32"), # nfft2int : Nber of integrated FFTs -> time resolution= fftlen*nfft2int*5.12 us
    ("fftovlp", "int32"), # fftovlp : FFToverlap : 0 (for none) or fftlen/2
    ("apodisation", "int32"), # apodisation with 0 : none (01 to 99 : a generic K=1 cosine-sum window with coefficient 0.xx, 101 : hamming window, 102 : hann window)
    ("nffte", "int32"), # nffte : nber of FFTs to be read per beamlet within this block (so we have 'nffte' spectra/timesample per block)
    ("nbchan", "int32"), # nbchan : nber of subsequent beamlets/channels or nber of NenuFAR/LOFAR 195kHz beamlets, set to a NEGATIVE value when data packets lost ! (so we have 'fftlen*nbchan' frequencies for each spectra)
]

SUBBAND_WIDTH = 195312.5 * u.Hz

# ============================================================= #
# ----------------- _ProcessingConfiguration ------------------ #
class _ProcessingConfiguration:
    
    def __init__(self,
            time_min: Time,
            time_max: Time,
            frequency_min: u.Quantity,
            frequency_max: u.Quantity,
            available_beams: np.ndarray
        ):
        self._available_beams = available_beams
        self._time_min = time_min
        self._time_max = time_max
        self._frequency_min = frequency_min
        self._frequency_max = frequency_max

        self.time_range = Time([time_min.isot, time_max.isot], precision=max(time_min.precision, time_max.precision))
        self.frequency_range = [frequency_min.to_value(u.Hz), frequency_max.to_value(u.Hz)]*u.Hz
        self.beam = 0
        self.dispersion_measure = None
        self.rotation_measure = None
        self.rebin_dt = None
        self.rebin_df = None
        # self.jump_correction = False
        self.dreambeam = (None, None, None)
        self.correct_bandpass = True
        self.edge_channels_to_remove = 0

    def __repr__(self) -> str:
        return "\n".join([f"{attr}: {getattr(self, attr)}" for attr in dir(self) if not attr.startswith("_")])

    @property
    def beam(self) -> int:
        return self._beam
    @beam.setter
    def beam(self, selected_beam: int) -> None:
        if not isinstance(selected_beam, int):
            raise TypeError("Selected beam is expected as an integer value.")
        elif selected_beam not in self._available_beams:
            raise IndexError(f"Requested beam #{selected_beam} not found among available beam indices {self.available_beams}.")
        self._beam = selected_beam
        log.info(f"\tBeam #{self._beam} selected.")

    @property
    def time_range(self) -> Time:
        return self._time_range
    @time_range.setter
    def time_range(self, selected_range: Time):
        if not isinstance(selected_range, Time):
            raise TypeError("time_range expects an astropy.time.Time object.")
        if selected_range.size != 2:
            raise ValueError("time_range should be a length-2 Time array.")
        if selected_range[0] >= selected_range[1]:
            raise ValueError("time_range start >= stop.")
        if (selected_range[1] < self._time_min) or (selected_range[0] > self._time_max):
            log.warning("Requested time_range outside availaible data!")
        self._time_range = selected_range
        log.info(f"\tTime range: {selected_range[0].isot} to {selected_range[1].isot}")

    @property
    def frequency_range(self) -> u.Quantity:
        return self._frequency_range
    @frequency_range.setter
    def frequency_range(self, selected_range: u.Quantity):
        if not isinstance(selected_range, u.Quantity):
            raise TypeError("frequency_range expects an astropy.units.Quantity object.")
        if selected_range.size != 2:
            raise ValueError("frequency_range should be a length-2 Quantity array.")
        if selected_range[0] >= selected_range[1]:
            raise ValueError("frequency_range min >= max.")
        if (selected_range[1] < self._frequency_min) or (selected_range[0] > self._frequency_max):
            log.warning("Requested time_range outside availaible data!")
        self._frequency_range = selected_range
        log.info(f"\tFrequency range: {selected_range[0].to(u.MHz)} to {selected_range[1].to(u.MHz)}")

    @property
    def edge_channels_to_remove(self) -> Union[int, Tuple[int, int]]:
        return self._edge_channels_to_remove
    @edge_channels_to_remove.setter
    def edge_channels_to_remove(self, channels: Union[int, Tuple[int, int]]):
        if isinstance(channels, tuple):
            if not len(channels) == 2:
                raise IndexError("If a `tuple` is given to the edge_channels_to_remove argument, it must be of length 2: (lower_edge_channels_to_remove, higher_edge_channels_to_remove).")
            elif not np.all([isinstance(chan, int) for chan in channels]):
                raise TypeError("Edge channels to remove must be integers.")
        elif not isinstance(channels, int):
            raise TypeError("Edge channels to remove muste be integers.")
        self._edge_channels_to_remove = channels
        log.info(f"\tEdge channels to remove set: {channels}.")

    @property
    def correct_bandpass(self) -> bool:
        return self._correct_bandpass
    @correct_bandpass.setter
    def correct_bandpass(self, correct: bool):
        if not isinstance(correct, bool):
            raise TypeError("test")
        self._correct_bandpass = correct
        log.info(f"\tBandpass correction: {correct}.")

    @property
    def dreambeam(self) -> Tuple[float, SkyCoord, bool]:
        return self._dreambeam
    @dreambeam.setter
    def dreambeam(self, db_inputs: Tuple[float, SkyCoord, bool]):
        if len(db_inputs) != 3:
            raise IndexError("dreambeam inputs should be a length 3 tuple.")
        if db_inputs != (None,)*3:
            if not isinstance(db_inputs[0], float):
                raise TypeError("First element of dreambeam must be the time resolution in sec (float).")
            if not isinstance(db_inputs[1], SkyCoord):
                raise TypeError("Second element of dreambeam must be the tracked coordinates (astropy.SkyCoord).")
            if not isinstance(db_inputs[2], bool):
                raise TypeError("Third element of dreambeam must be the parallactic angle correction (bool).")
            log.info(f"\tDreamBeam correction set (time_res, coord, parallactic)={db_inputs}")
        self._dreambeam = db_inputs

    @property
    def rotation_measure(self) -> u.Quantity:
        return self._rotation_measure
    @rotation_measure.setter
    def rotation_measure(self, rm: u.Quantity):
        if rm is None:
            pass
        elif isinstance(rm, u.Quantity):
            log.info(f"\tRotation Measure set: {rm}")
        else:
            raise TypeError("RM should be an astropy.units.Quantity object.")
        self._rotation_measure = rm

    @property
    def dispersion_measure(self) -> u.Quantity:
        return self._dispersion_measure
    @dispersion_measure.setter
    def dispersion_measure(self, dm: u.Quantity):
        if dm is None:
            pass
        elif isinstance(dm, u.Quantity):
            log.info(f"\tDispersion Measure set: {dm}")
        else:
            raise TypeError("DM should be an astropy.units.Quantity object.")
        self._dispersion_measure = dm

# ============================================================= #
# -------------------------- Spectra -------------------------- #
class Spectra:

    def __init__(self, filename: str):
        self.filename = filename

        # Decode the main header and lazy load the data
        self._n_time_per_block = 0
        self.n_channels = 0
        self.n_subbands = 0
        self.dt = None
        self.df = None
        data = self._lazy_load_data()

        # Compute the boolean mask of bad blocks
        bad_block_mask = self._get_bad_data_mask(data)

        # Compute the main data block descriptors (time / frequency / beam)
        self._block_start_unix = data["TIMESTAMP"][~bad_block_mask] + data["BLOCKSEQNUMBER"][~bad_block_mask] / SUBBAND_WIDTH.to_value(u.Hz)
        self._subband_start_hz = data["data"]["channel"][0, :] * SUBBAND_WIDTH.to_value(u.Hz) # Assumed constant over time
        self.beam_indices_dict = utils.sort_beam_edges(
            beam_array=data["data"]["beam"][0], # Asummed same for all time step
            n_channels=self.n_channels,
        )

        # Transform the data in Dask Array, once correctly reshaped
        self.data = self._assemble_to_tf(data=data, mask=bad_block_mask)

        log.info("Setting up default configuration:")
        self.configuration = _ProcessingConfiguration(
            time_min=self.time_min,
            time_max=self.time_max,
            frequency_min=self.frequency_min,
            frequency_max=self.frequency_max,
            available_beams=np.array(list(self.beam_indices_dict.keys())).astype(int)
        )

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def filename(self) -> str:
        return self._filename
    @filename.setter
    def filename(self, name: str):
        log.info(f"Reading {name}...")
        if not name.endswith(".spectra"):
            raise ValueError("A file whose extension is '.spectra' is expected.")
        self._filename = name

    @property
    def time_min(self) -> Time:
        return Time(self._block_start_unix[0], format="unix",precision=7)

    @property
    def time_max(self) -> Time:
        return Time(self._block_start_unix[-1] + self._n_time_per_block * self.dt.to_value(u.s), format="unix", precision=7)

    @property
    def frequency_min(self) -> u.Quantity:
        freq_mins = []
        for _, boundaries in self.beam_indices_dict.items():
            sb_index = int(boundaries[0]/self.n_channels)
            freq_mins.append(self._subband_start_hz[sb_index])
        return np.min(freq_mins) * u.Hz
    
    @property
    def frequency_max(self) -> u.Quantity:
        freq_maxs = []
        for _, boundaries in self.beam_indices_dict.items():
            sb_index = int((boundaries[1] + 1)/self.n_channels - 1)
            freq_maxs.append(self._subband_start_hz[sb_index])
        return np.max(freq_maxs) * u.Hz + SUBBAND_WIDTH

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def info(self) -> None:
        """ Display informations about the file. """
        message = "\n".join([
            f"filename: {self.filename}",
            f"time_min: {self.time_min.isot}",
            f"time_max: {self.time_max.isot}",
            f"dt: {self.dt.to(u.ms)}",
            f"frequency_min: {self.frequency_min.to(u.MHz)}",
            f"frequency_max: {self.frequency_max.to(u.MHz)}",
            f"df: {self.df.to(u.kHz)}",
            f"Available beam indices: {list(self.beam_indices_dict.keys())}"
        ])
        print(message)

    def get(self, stokes: Union[str, List[str]] = "I"):
        """ """

        # Select the data in time and frequency (the beam selection is implicit on the frequency idexing)
        frequency_hz, time_unix, data = self._select_data()

        # Stop the pipeline if the data is empty
        if data is None:
            return

        # Correct for the bandpass
        if self.configuration.correct_bandpass:
            data = utils.correct_bandpass(data=data, n_channels=self.n_channels)

        # Remove subband edge channels
        edge_chans = self.configuration.edge_channels_to_remove
        if edge_chans not in [0, (0, 0)]:
            data = utils.crop_subband_edges(
                data=data,
                n_channels=self.n_channels,
                lower_edge_channels=edge_chans[0] if isinstance(edge_chans, tuple) else edge_chans,
                higher_edge_channels=edge_chans[1] if isinstance(edge_chans, tuple) else edge_chans,
            )

        # DreamBeam correction (beam gain + parallactic angle)
        db_dt, db_coord, db_par = self.configuration.dreambeam
        if not ((db_dt is None) or (db_coord is None) or (db_par is None)):
            data = utils.apply_dreambeam_corrections(
                time_unix=time_unix,
                frequency_hz=frequency_hz,
                data=data,
                dt_sec=self.dt.to_value(u.s),
                time_step_sec=db_dt,
                n_channels=self.n_channels,
                skycoord=db_coord,
                parallactic=db_par
            )

        # De-faraday
        if not (self.configuration.rotation_measure is None):
            data = utils.de_faraday_data(
                data=data,
                frequency=frequency_hz*u.Hz,
                rotation_measure=self.configuration.rotation_measure
            )

        # De-disperse array
        if not (self.configuration.dispersion_measure is None):
            tmp_chuncks = data.chunks
            data = utils.de_disperse_array(
                data=data.compute(), # forced to leave Dask
                frequencies=frequency_hz*u.Hz,
                time_step=self.dt,
                dispersion_measure=self.configuration.dispersion_measure,
            )
            data = da.from_array(data, chunks=tmp_chuncks)

        # Rebin the data
        if self.configuration.rebin_dt is not None:
            log.info("Rebinning in time...")
            time_unix, data = utils.rebin_along_dimension(
                data=data,
                axis_array=time_unix,
                axis=0,
                dx=self.dt.to_value(u.s),
                new_dx=self.configuration.rebin_dt.to_value(u.s)
            )
        if self.configuration.rebin_df is not None:
            log.info("Rebinning in frequency...")
            frequency_hz, data = utils.rebin_along_dimension(
                data=data,
                axis_array=frequency_hz,
                axis=1,
                dx=self.df.to_value(u.Hz),
                new_dx=self.configuration.rebin_df.to_value(u.Hz)
            )
        log.info(f"Shape of rebinned data: {data.shape}.")

        # Compute the selected Stokes parameters
        data = utils.compute_stokes_parameters(data_array=data, stokes=stokes)

        log.info("Computing the data...")
        with ProgressBar():
            data = data.compute()
        log.info(f"\tData of shape (time, frequency, stokes) = {data.shape} produced.")

        return SData(
            data=data,
            time=Time(time_unix, format="unix", precision=7),
            freq=frequency_hz*u.Hz,
            polar=stokes,
        )

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _lazy_load_data(self) -> np.ndarray:

        # Read the header of the first block
        with open(self.filename, "rb") as rf:
            header_dtype = np.dtype(BLOCK_HEADER)
            header = np.frombuffer(
                rf.read(header_dtype.itemsize),
                count=1,
                dtype=header_dtype,
            )[0]
        self._n_time_per_block = header["nffte"]
        self.n_channels = header["fftlen"]
        self.n_subbands = np.abs(header["nbchan"]) # this could be negative

        # Fill in global attributes
        self.dt = (self.n_channels * header["nfft2int"] / SUBBAND_WIDTH).to(u.s)#* 5.12e-6 * u.s
        self.df = SUBBAND_WIDTH / self.n_channels #0.1953125 / self.n_channels * u.MHz

        # Deduce the structure of the full file
        beamlet_data_structure = (
            self._n_time_per_block,
            self.n_channels,
            2
        )
        beamlet_dtype = np.dtype(
            [
                ("lane", "int32"),
                ("beam", "int32"),
                ("channel", "int32"),
                ("fft0", "float32", beamlet_data_structure),
                ("fft1", "float32", beamlet_data_structure),
            ]
        )
        global_struct = BLOCK_HEADER + [("data", beamlet_dtype, (self.n_subbands))]

        # Open the file as a memory-mapped object
        with open(self.filename, "rb") as rf:
            tmp = np.memmap(rf, dtype="int8", mode="r")

        log.info(f"\t{self.filename} has been correctly parsed.")

        return tmp.view(np.dtype(global_struct))

    @staticmethod
    def _get_bad_data_mask(data: np.ndarray) -> np.ndarray:
        """ """

        log.info("Checking for missing data...")

        # Either the TIMESTAMP is set to 0, the first idx, or the SB number is negative
        # which indicates missing data. In all those cases we will ignore the associated data
        # since we cannot properly reconstruct the time ramp or the data themselves.
        block_timestamp_mask = data["TIMESTAMP"] < 0.1 # it should be 0, but to be sure...
        block_start_idx_mask = data["idx"] == 0
        block_nsubbands_mask = data["nbchan"] < 0

        # Computing the mask, setting the first index at non-zero since it should be the normal value.
        block_start_idx_mask[0] = False # Fake value, just to trick the mask
        bad_block_mask = block_timestamp_mask + block_start_idx_mask + block_nsubbands_mask

        log.info(f"\tThere are {np.sum(bad_block_mask)}/{block_timestamp_mask.size} blocks containing missing data and/or wrong time information.")

        return bad_block_mask

    def _assemble_to_tf(self, data: np.ndarray, mask:  np.ndarray) -> da.Array:
        """ """
        # Transform the array in a Dask array, one chunk per block
        # Filter out the bad blocks
        data = da.from_array(
            data,
            chunks=(1,)
        )[~mask]

        # Convert the data to cross correlation electric field matrix
        # The data will be shaped like (n_block, n_subband, n_time_per_block, n_channels, 2, 2)
        data = utils.spectra_data_to_matrix(data["data"]["fft0"], data["data"]["fft1"])

        # Reshape the data into (time, frequency, 2, 2)
        data = utils.blocks_to_tf_data(
            data=data,
            n_block_times=self._n_time_per_block,
            n_channels=self.n_channels
        )

        return data

    def _select_data(self) -> Tuple[np.ndarray, np.ndarray, da.Array]:
        """ """
        
        log.info("Computing the time selection...")
        tmin, tmax = self.configuration.time_range.unix

        # Find out which block indices are at the edges of the desired time range
        block_idx_min = int(np.argmin(np.abs(np.ceil(self._block_start_unix - tmin))))# n_blocks - np.argmax(((self._block_start_unix - tmin) <= 0)[::-1]) - 1
        block_idx_max = int(np.argmin(np.abs(np.ceil(self._block_start_unix - tmax))))# n_blocks - np.argmax(((self._block_start_unix - tmax) <= 0)[::-1]) - 1
        log.info(f"\tClosest time block from requested range are #{block_idx_min} and #{block_idx_max}.")

        # Get the closest time index within each of the two bounding blocks
        dt_sec = self.dt.to_value(u.s)
        # Compute the time index within the block and bound it between 0 and the number of spectra in each block
        time_idx_min_in_block = int(np.round((tmin - self._block_start_unix[block_idx_min])/dt_sec))
        time_idx_min_in_block = max(0, min(self._n_time_per_block - 1, time_idx_min_in_block)) # bound the value between in between channels indices
        time_idx_min = block_idx_min * self._n_time_per_block + time_idx_min_in_block
        # Do the same for the higher edge of the desired time range
        time_idx_max_in_block = int(np.round((tmax - self._block_start_unix[block_idx_max])/dt_sec))
        time_idx_max_in_block = max(0, min(self._n_time_per_block - 1, time_idx_max_in_block))
        time_idx_max = block_idx_max * self._n_time_per_block + time_idx_max_in_block
        log.info(f"\t{time_idx_max - time_idx_min + 1} time samples selected.")

        # Raise warnings if the time selection results in no data selected
        if time_idx_min == time_idx_max:
            if (time_idx_min > 0) and (time_idx_min < self._block_start_unix.size * self._n_time_per_block - 1):
                log.warning("Desired time selection encompasses missing data.")
                if tmin < self._block_start_unix[block_idx_min]:
                    # The found block is just after the missing data
                    closest_tmin = Time(self._block_start_unix[block_idx_min - 1] + self._n_time_per_block * dt_sec, format='unix').isot
                    closest_tmax = Time(self._block_start_unix[block_idx_min], format='unix').isot
                else:
                    # The found block is just before the missing data
                    closest_tmin = Time(self._block_start_unix[block_idx_min] + self._n_time_per_block * dt_sec, format='unix').isot
                    closest_tmax = Time(self._block_start_unix[block_idx_min + 1], format='unix').isot
                log.info(f"Time selection lies in the data gap between {closest_tmin} and {closest_tmax}.")
            log.warning("Time selection leads to empty dataset.")
            return (None, None, None)

        # Compute the time ramp between those blocks
        time_unix = utils.compute_spectra_time(
            block_start_time_unix=self._block_start_unix[block_idx_min:block_idx_max + 1],
            ntime_per_block=self._n_time_per_block,
            time_step_s=self.dt.to_value(u.s)
        )
        # Cut down the first and last time blocks
        time_unix = time_unix[time_idx_min_in_block:time_unix.size - (self._n_time_per_block - time_idx_max_in_block) + 1]

        log.info("Computing the frequency selection...")
        fmin, fmax = self.configuration.frequency_range.to_value(u.Hz)
        beam_idx_start, beam_idx_stop = self.beam_indices_dict[str(self.configuration.beam)]

        # Find out the subband edges covering the selected frequency range
        subbands_in_beam = self._subband_start_hz[int(beam_idx_start/self.n_channels):int((beam_idx_stop + 1)/self.n_channels)]
        sb_idx_min = int(np.argmin(np.abs(np.ceil(subbands_in_beam - fmin))))
        sb_idx_max = int(np.argmin(np.abs(np.ceil(subbands_in_beam - fmax))))
        log.info(f"\tClosest beamlet indices from requested range are #{sb_idx_min} and #{sb_idx_max}.")

        # Select frequencies at the subband granularity at minimum
        # Later, we want to correct for bandpass, edge channels and so on...
        frequency_idx_min = sb_idx_min * self.n_channels
        frequency_idx_max = (sb_idx_max + 1) * self.n_channels
        frequency_hz = utils.compute_spectra_frequencies(
            subband_start_hz=subbands_in_beam[sb_idx_min:sb_idx_max + 1],
            n_channels=self.n_channels,
            frequency_step_hz=self.df.to_value(u.Hz)
        )
        log.info(f"\t{frequency_idx_max - frequency_idx_min} frequency samples selected.")

        selected_data = self.data[:, beam_idx_start:beam_idx_stop + 1, ...][time_idx_min:time_idx_max + 1, frequency_idx_min:frequency_idx_max, ...]
        log.debug(f"Data of shape {selected_data.shape} selected.")

        return frequency_hz.compute(), time_unix.compute(), selected_data
