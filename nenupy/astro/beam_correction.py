"""
"""

__all__ = [
   "compute_jones_matrices",
   "compute_projection_corrections",
   "convert_to_mueller",
   "matrices_to_hdf5"
]


from nenupy.astro.jones_mueller import JonesMatrix, MuellerMatrix
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np
import h5py
from typing import Union, Tuple
import logging
log = logging.getLogger("nenupy")
log.setLevel(logging.INFO)

try:
    from dreambeam.rime.scenarios import on_pointing_axis_tracking
    from dreambeam.rime.jones import DualPolFieldPointSrc, PJones
    from dreambeam.telescopes.rt import load_mountedfeed
except ModuleNotFoundError:
    # This will raise an error eventually with an appropriate message
    pass

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
    log.info("\tComputing Jones matrices using DreamBeam...")

    if time_step.sec <= 1.:
        raise ValueError("DreamBeam does not allow for time intervals lesser than 1 sec.")

    try:
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
    except NameError:
        log.error(
            "DreamBeam is not installed. "
            "See installation instructions https://github.com/2baOrNot2ba/dreamBeam"
        )
        raise
    # import numpy as np
    # times = start_time + TimeDelta(3600, format="sec")*np.arange(12)
    # frequencies = np.array([30e6, 50e6])*u.MHz
    # Jn = np.tile(np.array([[1, 1], [0, 1]]), (times.size, frequencies.size, 1, 1))
    return Time(times, format="datetime"), frequencies*u.Hz, JonesMatrix(Jn)
# ============================================================= #

# ============================================================= #
# -------------- compute_projection_corrections --------------- #
def compute_projection_corrections(
        start_time: Time,
        time_step: TimeDelta,
        duration: TimeDelta,
        skycoord: SkyCoord,
        parallactic: bool = True
    ) -> Tuple[Time, u.Quantity, JonesMatrix]:
    """_summary_
    Took apart on_pointing_axis_tracking method from Dreambeam to only take into account
    parallactic angle and projection effects.

    Parameters
    ----------
    start_time : Time
        _description_
    time_step : TimeDelta
        _description_
    duration : TimeDelta
        _description_
    skycoord : SkyCoord
        _description_
    parallactic : bool, optional
        _description_, by default True

    Returns
    -------
    Tuple[Time, u.Quantity, JonesMatrix]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    log.info("\tComputing Jones projection matrices using DreamBeam...")

    if time_step.sec <= 1.:
        raise ValueError("DreamBeam does not allow for time intervals lesser than 1 sec.")

    try:
        stnfeed = load_mountedfeed(
            tscopename="NenuFAR",
            station="NenuFAR",
            band="LBA",
            modelname="Hamaker-NEC4_Charrier_v1r1"
        )
        stnrot = stnfeed.stnRot
        freqs = stnfeed.getfreqs() # list

        pointingdir = (skycoord.ra.rad, skycoord.dec.rad, "J2000")
        srcfld = DualPolFieldPointSrc(pointingdir)

        timespy = []
        nrTimSamps = int((duration.datetime.total_seconds() / time_step.datetime.seconds)) + 1
        for ti in range(0, nrTimSamps):
            timespy.append(start_time.datetime + ti * time_step.datetime)
        pjones = PJones(timespy, np.transpose(stnrot), do_parallactic_rot=parallactic)
        pjonesOfSrc = pjones.op(srcfld)
        jones = pjonesOfSrc.getValue()
        jones = np.repeat(jones[None, ...], len(freqs), axis=0)

    except NameError:
        log.error(
            "DreamBeam is not installed. "
            "See installation instructions https://github.com/2baOrNot2ba/dreamBeam"
        )
        raise

    return Time(timespy, format="datetime"), freqs * u.Hz, JonesMatrix(jones)
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
