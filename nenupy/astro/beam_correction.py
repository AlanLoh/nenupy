"""
"""

__all__ = [
   "compute_jones_matrices",
   "convert_to_mueller",
   "matrices_to_hdf5"
]


from nenupy.astro.jones_mueller import JonesMatrix, MuellerMatrix
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import h5py
from typing import Union, Tuple
import logging
log = logging.getLogger("nenupy")
log.setLevel(logging.INFO)

try:
    from dreambeam.rime.scenarios import on_pointing_axis_tracking
except ImportError:
    log.error(
        "DreamBeam is not installed. "
        "See installation instructions https://github.com/2baOrNot2ba/dreamBeam"
    )
    raise


# ============================================================= #
# ------------------ compute_jones_matrices ------------------- #
def compute_jones_matrices(
        start_time: Time,
        time_step: TimeDelta,
        duration: TimeDelta,
        skycoord: SkyCoord,
        parallactic: bool = True
    ) -> Tuple[Time, u.Quantity, JonesMatrix]:
    """
    """
    log.info("Computing Jones matrices using DreamBeam...")
    times, frequencies, Jn, _ = on_pointing_axis_tracking(
        telescopename="NenuFAR",
        stnid="NenuFAR",
        band="LBA",
        antmodel="Hamaker-NEC4_Charrier_v1r1",
        obstimebeg=start_time.datetime,
        obsdur=duration.datetime,
        obstimestp=time_step.datetime,
        pointingdir=(skycoord.ra.rad, skycoord.dec.rad, "J2000"),
        do_parallactic_rot=parallactic
    )
    # import numpy as np
    # times = start_time + TimeDelta(3600, format="sec")*np.arange(12)
    # frequencies = np.array([30e6, 50e6])*u.MHz
    # Jn = np.tile(np.array([[1, 1], [0, 1]]), (times.size, frequencies.size, 1, 1))
    return Time(times, format="datetime"), frequencies*u.Hz, JonesMatrix(Jn)
# ============================================================= #


# ============================================================= #
# -------------------- convert_to_mueller --------------------- #
def convert_to_mueller(jones_matrices: JonesMatrix) -> MuellerMatrix:
    """ """
    log.info(f"Converting {jones_matrices.shape} Jones matrices to Mueller matrices...")
    return jones_matrices.to_mueller()
# ============================================================= #


# ============================================================= #
# --------------------- matrices_to_hdf5 ---------------------- #
def matrices_to_hdf5(
        times: Time, frequencies: u.Quantity,
        matrices: Union[JonesMatrix, MuellerMatrix],
        file_name: str
    ) -> None:
    """ """
    if not file_name.endswith(".hdf5"):
        raise ValueError("The name of the HDF5 file must end with '.hdf5'!")

    if matrices.shape[:-2] != frequencies.shape +  times.shape:
        raise IndexError(
            "Inconsistent shapes detected. "
            f"times.shape={times.shape}, "
            f"frequencies.shape={frequencies.shape}, "
            f"matrices.shape={matrices.shape}, "
        )
  
    log.info(f"Writing the results in {file_name}...")
    
    with h5py.File(file_name, "w") as f:
        f["data"] = matrices
        f["data"].dims[0].label = "frequency"
        f["data"].dims[1].label = "time"
        f["time"] = times.jd
        f["time"].make_scale("Time (JD)")
        f["frequency"] = frequencies.to(u.MHz).value
        f["frequency"].make_scale("Frequency (MHz)")
        f["data"].dims[0].attach_scale(f["frequency"])
        f["data"].dims[1].attach_scale(f["time"])
    
    log.info(f"{file_name} written!")
# ============================================================= #
# ============================================================= #
