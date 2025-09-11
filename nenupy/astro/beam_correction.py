"""
"""

__all__ = [
   "compute_jones_matrices",
   "compute_projection_corrections",
   "convert_to_mueller",
   "matrices_to_hdf5",
   "pointing_correction_factor"
]


from nenupy.astro.jones_mueller import JonesMatrix, MuellerMatrix
from nenupy.astro.pointing import Pointing
from nenupy.instru import NenuFAR, NenuFAR_Configuration, Polarization
from nenupy.astro.sky import Sky
from nenupy.instru.instrument_tools import pointing_correction

from astropy.coordinates import SkyCoord, ICRS
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import copy
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
# ---------------- pointing_correction_factor ----------------- #
def pointing_correction_factor(
        digital_pointing: Pointing, analog_pointing: Pointing,
        times: Time, frequencies: u.Quantity,
        nenufar: NenuFAR = NenuFAR(),
        polarization: Union[np.ndarray, Polarization] = Polarization.NW,
        nenufar_config: NenuFAR_Configuration = NenuFAR_Configuration(),
        correction_year: str = "2022",
        return_interpolation: bool = False
    ) -> Union[np.ndarray, RegularGridInterpolator]:
    """Compute the factor needed to correct the intensity of the sources observed when NenuFAR (in beamformed mode) suffered from a pointing offset (i.e., before 2025 June 17).
    The output correction factor is an array shaped like (``times``, ``frequencies``, ``polarization``).
    There's no need in computing it on a fine time-frequency grid since, the evolution is quite smooth over a few minutes and can be interpolated to match the original dataset.
    Tis function simulates NenuFAR's beam and its effective offset in order to propose a correction factor.
    ``digital_pointing`` and ``analog_pointing`` need to reflect the desired pointing coordinates (i.e., no beamsquint correction nor empirical correction).
    
    Parameters
    ----------
    digital_pointing : :class:`~nenupy.astro.pointing.Pointing`
        Digital pointing orders, that would typically follow an astrophysical source across time.
    analog_pointing : :class:`~nenupy.astro.pointing.Pointing`
        Analog pointing orders, given every 6 minutes to the NenuFAR Mini-Arrays.
    times : :class:`~astropy.time.Time`
        _description_
    frequencies : :class:`~astropy.units.Quantity`
        _description_
    nenufar : :class:`~nenupy.instru.nenufar.NenuFAR`, optional
        _description_, by default NenuFAR()
    polarization : :class:`~numpy.ndarray` | :class:`~nenupy.instru.nenufar.Polarization`, optional
        _description_, by default Polarization.NW
    nenufar_config : :class:`~nenupy.instru.nenufar.NenuFAR_Configuration`, optional
        _description_, by default NenuFAR_Configuration()
    correction_year : `str`, optional
        _description_ (see :func:`~nenupy.instru.instrument_tools.pointing_correction`), by default "2022"
    return_interpolation : `bool`, optional
        Return the interpolation function. If this option is selected the time, frequency and polarization axes are converted to Julian days / MHz / (0=`~nenupy.instru.nenufar.Polarization.NW`, 1=`~nenupy.instru.nenufar.Polarization.NE`) values for interpolation purposes (see example), by default `False`

    Returns
    -------
    :class:`~numpy.ndarray` or :class:`~scipy.interpolate._rgi.RegularGridInterpolator`
        _description_

    Examples
    --------
    .. code-block:: python

        >>> from nenupy.astro.pointing import Pointing
        >>> import numpy as np
        >>> import astropy.units as u

        >>> pp_digi = Pointing.from_file("<file_name>.altazB", include_corrections=False)
        >>> pp_ana = Pointing.from_file("<file_name>.altazA", include_corrections=False)

        >>> t_steps = 50
        >>> f_steps = 30
        >>> times = pp_digi.time[0] + np.arange(t_steps) * (pp_digi.time[-1] - pp_digi.time[0])/t_steps
        >>> freqs = np.linspace(10, 80, f_steps) * u.MHz

        >>> factor = pointing_correction_factor(
                digital_pointing=pp_digi,
                analog_pointing=pp_ana,
                times=times,
                frequencies=freqs,
                correction_year="2022",
                return_interpolation=True
            )

        >>> X, Y, Z = np.meshgrid(times.jd, freqs.to_value(u.MHz), [0], indexing='ij')
        >>> interpolated_factor = factor((X, Y, Z))

    Warning
    -------
    The correction factor is solely intended to be applied on the target astrophyiscal source's emission.
    The underlying (though most of the time dominant) background emission is not affected at the same scale by the pointing offset.

    See Also
    --------
    <link to the report>.    
    
    """
 

    # Compute the pointing orders given to NenuFAR
    altaz_orders = pointing_correction(
        altaz_coordinates=digital_pointing.horizontal_coordinates,
        correction_year=correction_year
    )

    # Apply the Lambert93 -> geo rotation to determine the effective coordinates
    rotation = 0.58 * u.deg
    real_altaz_orders = SkyCoord(
        altaz_orders.az - rotation,
        altaz_orders.alt,
        frame=altaz_orders.frame
    )

    # Simulate the value of the numerical beam in comparison with old and new positions
    real_pointing = Pointing(
        coordinates=real_altaz_orders.transform_to(ICRS),
        time=real_altaz_orders.obstime,
        # duration=digital_pointing.duration
        duration=TimeDelta(np.diff(real_altaz_orders.obstime.jd, append=(real_altaz_orders.obstime[-1] + TimeDelta(1, format="jd")).jd), format="jd")
    )
    beam_desired = nenufar.beam(
        sky=Sky(
            coordinates=copy.deepcopy(digital_pointing)[times].coordinates,
            time=times,
            frequency=frequencies,
            polarization=polarization
        ),
        pointing=copy.deepcopy(real_pointing),
        analog_pointing=copy.deepcopy(analog_pointing),
        configuration=nenufar_config
    )
    beam_real = nenufar.beam(
        sky=Sky(
            coordinates=copy.deepcopy(real_pointing)[times].coordinates,
            time=times,
            frequency=frequencies,
            polarization=polarization
        ),
        pointing=copy.deepcopy(real_pointing),
        analog_pointing=copy.deepcopy(analog_pointing),
        configuration=nenufar_config
    )

    # Compute the correction factor
    desired_val = np.diagonal(beam_desired.value, offset=0, axis1=0, axis2=3)
    real_val = np.diagonal(beam_real.value, offset=0, axis1=0, axis2=3)
    factor = (real_val / desired_val).compute()

    if return_interpolation:
        # Convert the polarization to index
        pol_size = beam_desired.polarization.size
        polars_int = np.zeros(pol_size)
        for i in range(pol_size):
            polars_int[i] = 0 if beam_desired.polarization[i] == Polarization.NW else 1
        return RegularGridInterpolator(
            (times.jd, frequencies.to_value(u.MHz), polars_int),
            np.moveaxis(factor, 2, 0),
            bounds_error=False
        )
    else:
        return np.moveaxis(factor, 2, 0) # (time, freq, polar)
# ============================================================= #
# ============================================================= #
