#! /usr/bin/python3
# -*- coding: utf-8 -*-


r"""
    ************
    HEALPix Beam
    ************

    NenuFAR beam simulations are required for many applications:
    observation simulations, calibration, imaging process... It
    is a critical step done in stages, while trying to match as
    much as possible the NenuFAR instrument specifications.

    NenuFAR antenna (see individual response 
    :func:`~nenupy.instru.instru.nenufar_ant_gain`) are
    analogically phased, thanks to cable delays switches, by
    group of 19 called Mini-Arrays.
    The (up-to 96) Mini-Arrays are then digitally phased to get
    the beamformed NenuFAR data.
    These two types of beam (analog or digital) are described by
    the two classes :class:`~nenupy.beam.hpxbeam.HpxABeam` and
    :class:`~nenupy.beam.hpxbeam.HpxDBeam` respectively. They
    both inherit from the base class 
    :class:`~nenupy.beam.hpxbeam.HpxBeam` which is also a
    :class:`~nenupy.astro.hpxsky.HpxSky` enabling full HEALPix
    representation and functionalities.

    .. math::
        \mathcal{M}_g (\mathbf{r}, \nu, p) = \mathcal{A}_g (\nu, p) \sum_{n_{\scriptscriptstyle \rm ant}} e^{2 \pi i \frac{\nu}{c} \mathbf{r}_c (\mathbf{r}) \cdot \mathbf{r}_{n_{\scriptscriptstyle \rm ant}}}

    .. math::
        \mathcal{N}_g (\mathbf{r}, \nu, p) = \sum_{\rm{MA}_i}  \mathcal{M}_{g,\, \rm{MA}_i} (\mathbf{r}, \nu, p) e^{2 \pi i \frac{\nu}{c} \mathbf{r} \cdot \mathbf{r}_{\rm{MA}_{\scriptscriptstyle i}}}

    :math:`\mathcal{M}_g (\mathbf{r}, \nu, p)` :math:`\mathcal{N}_g (\mathbf{r}, \nu, p)` :math:`\mathcal{A}_g (\nu, p)` :math:`\mathbf{r}` :math:`\mathbf{r}_c` :math:`\mathbf{r}_{n_{\scriptscriptstyle \rm ant}}` :math:`\mathbf{r}_{\rm{MA}_{\scriptscriptstyle i}}`
    :math:`p` :math:`\nu`

    .. seealso::
        :ref:`tuto_beam_ref` tutorial.
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'HpxBeam',
    'HpxABeam',
    'HpxDBeam'
]


from nenupy.astro import (
    HpxSky,
    wavelength,
    to_radec,
    ho_coord
)
from nenupy.instru import (
    ma_antpos,
    ma_info,
    ma_pos,
    desquint_elevation,
    analog_pointing,
    nenufar_ant_gain,
    resolution
)

import numpy as np
import numexpr as ne
from multiprocessing import Pool, sharedctypes
from healpy import ud_grade
import astropy.units as u

import logging
log = logging.getLogger(__name__)


# ============================================================= #
# ------------------- Parallel computation -------------------- #
# ============================================================= #
def _init(arr_r, arr_i, coeff, delay):
    """
    """
    global arr1
    global arr2
    global coef
    global darr
    arr1 = arr_r
    arr2 = arr_i
    coef = coeff
    darr = delay
    return


def fill_per_block(args):
    indices = args.astype(int)
    tmp_r = np.ctypeslib.as_array(arr1)
    tmp_i = np.ctypeslib.as_array(arr2)
    dd = darr[:, indices]
    beam_part = ne.evaluate('exp(coef*dd)')
    tmp_r[:, indices] = beam_part.real
    tmp_i[:, indices] = beam_part.imag
    return


def mp_expo(ncpus, coeff, delay):
    block_indices = np.array_split(
        np.arange(delay.shape[1]),
        ncpus
    )
    result_r = np.ctypeslib.as_ctypes(
        np.zeros_like(delay)
    )
    result_i = np.ctypeslib.as_ctypes(
        np.zeros_like(delay)
    )
    shared_r = sharedctypes.RawArray(
        result_r._type_,
        result_r
    )
    shared_i = sharedctypes.RawArray(
        result_i._type_,
        result_i
    )
    pool = Pool(
        processes=ncpus,
        initializer=_init,
        initargs=(shared_r, shared_i, coeff, delay)
    )
    res = pool.map(fill_per_block, (block_indices))
    result_r = np.ctypeslib.as_array(shared_r)
    result_i = np.ctypeslib.as_array(shared_i)
    return result_r + 1j * result_i


# import numba
# @numba.jit(nopython=True, parallel=True,fastmath=True)
# def perfcompute(phase):
#     return np.sum(np.exp(phase), axis=0)
# ============================================================= #


# ============================================================= #
# ------------------------- HpxABeam -------------------------- #
# ============================================================= #
class HpxBeam(HpxSky):
    """
    """

    def __init__(self, resolution=1):
        super().__init__(
            resolution=resolution
        )
        self.ncpus = 1

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def azana(self):
        """
        """
        return self._azana
    @azana.setter
    def azana(self, a):
        if isinstance(a, u.Quantity):
            pass
        elif a is None:
            raise ValueError(
                'azana should not be None'
            )
        else:
            a *= u.deg
        self._azana = a
        return


    @property
    def elana(self):
        """
        """
        return self._elana
    @elana.setter
    def elana(self, e):
        if isinstance(e, u.Quantity):
            pass
        elif e is None:
            raise ValueError(
                'elana should not be None'
            )
        else:
            e *= u.deg
        self._elana = e
        return


    @property
    def azdig(self):
        """
        """
        if self._azdig is None:
            return self.azana
        return self._azdig
    @azdig.setter
    def azdig(self, a):
        if isinstance(a, u.Quantity):
            pass
        elif a is None:
            pass
        else:
            a *= u.deg
        self._azdig = a
        return


    @property
    def eldig(self):
        """
        """
        if self._eldig is None:
            return self.elana
        return self._eldig
    @eldig.setter
    def eldig(self, e):
        if isinstance(e, u.Quantity):
            pass
        elif e is None:
            pass
        else:
            e *= u.deg
        self._eldig = e
        return

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def array_factor(self, az, el, antpos, freq):
        r""" Computes the array factor :math:`\mathcal{A}` (i.e.
            the far-field radiation pattern obtained for an array
            of :math:`n_{\rm ant}` radiators).

            .. math::
                \mathcal{A} = \left| \sum_{n_{\scriptscriptstyle \rm ant}} e^{2 \pi i \frac{\nu}{c} (\varphi_0 - \varphi)} \right|^2

            .. math::
                \varphi = \underset{\scriptstyle n_{\scriptscriptstyle \rm ant}\, \times\, 3}{\mathbf{P}_{\rm ant}} \cdot \pmatrix{
                    \cos(\phi)\cos(\theta)\\
                    \sin(\phi)\cos(\theta)\\
                    \sin(\theta)
                }

            .. math::
                \varphi_0 = \underset{\scriptstyle n_{\scriptscriptstyle \rm ant}\, \times\, 3}{\mathbf{P}_{\rm ant}} \cdot \pmatrix{
                    \cos(\phi_0)\cos(\theta_0)\\
                    \sin(\phi_0)\cos(\theta_0)\\
                    \sin(\theta_0)
                }

            :math:`\mathbf{P}_{\rm ant}` is the antenna position
            matrix, :math:`\phi` and :math:`\theta` are the sky
            local coordinates (azimuth and elevation respectively)
            gridded on a HEALPix representation, whereas 
            :math:`\phi_0` and :math:`\theta_0` are the pointing
            direction in local coordinates.

            :param az:
                Pointing azimuth (in degrees if `float`)
            :type az: `float` or :class:`~astropy.units.Quantity`
            :param el:
                Pointing elevation (in degrees if `float`)
            :type el: `float` or :class:`~astropy.units.Quantity`
            :param antpos:
                Antenna positions shaped as (n_ant, 3)
            :type antpos: :class:`~numpy.ndarray`
            :param freq:
                Frequency (in MHz if `float`)
            :type freq: `float` or :class:`~astropy.units.Quantity`

            :returns: Array factor
            :rtype: :class:`~numpy.ndarray`

        """
        def get_phi(az, el, antpos):
            """ az, el in radians
            """
            xyz_proj = np.array(
                [
                    np.cos(az) * np.cos(el),
                    np.sin(az) * np.cos(el),
                    np.sin(el)
                ]
            )
            antennas = np.array(antpos)
            phi = np.dot(antennas, xyz_proj)
            return phi

        if not isinstance(az, u.Quantity):
            az *= u.deg
        if not isinstance(el, u.Quantity):
            el *= u.deg

        self.phase_center = to_radec(
            ho_coord(
                az=az,
                alt=el,
                time=self.time
            )
        )

        phi0 = get_phi(
            az=[az.to(u.rad).value],
            el=[el.to(u.rad).value],
            antpos=antpos
        )
        phi_grid = get_phi(
            az=self.ho_coords.az.rad,
            el=self.ho_coords.alt.rad,
            antpos=antpos
        )
        nt = ne.set_num_threads(ne._init_num_threads())
        delay = ne.evaluate('phi_grid-phi0')
        coeff = 2j * np.pi / wavelength(freq).value
        
        if self.ncpus == 1:
            # Normal
            af = ne.evaluate('sum(exp(coeff*delay),axis=0)')
        # elif self.ncpus == 'numba':
        #     af = perfcompute(coeff * delay)
        else:
            # Multiproc
            af = np.sum(mp_expo(self.ncpus, coeff, delay), axis=0)

        #return np.abs(af * af.conjugate())
        return np.real(af * af.conjugate())


    def radial_profile(self, da=1.):
        """ Computes the beam radial profile from the center
            defined by the pointing direction while computing
            :func:`~nenupy.beam.hpxbeam.HpxBeam.array_factor`.
            This is done by averaging concentric rings of width 
            ``da``.
            
            :param da:
                Width of the concentric rings to be averaged from
                the center.
            :type da: `float` or :class:`~astropy.units.Quantity`

            :returns:
                (separation,  radial profile)
            :rtype: `tuple` of (:class:`~astropy.units.Quantity`, :class:`~numpy.ndarray`)
            
            :Example:
                >>> from nenupy.beam import HpxABeam
                >>> import astropy.units as u
                >>> ana = HpxABeam(resolution=1)
                >>> ana.beam(
                    freq=80*u.MHz,
                    azana=180*u.deg,
                    elana=90*u.deg,
                    time='2020-04-01 12:00:00'
                    )
                >>> separation, profile = ana.radial_profile(da=0.5)
                
                .. image:: ./_images/beam_profile.png
                    :width: 600
                
        """
        if not hasattr(self, 'phase_center'):
            raise Exception('run array_factor first')
        sep = self.phase_center.separation(
            self._eq_coords[self._is_visible]
        ).deg
        if isinstance(da, u.Quantity):
            da = da.to(u.deg).value
        min_seps = np.arange(
            sep.min(),
            sep.max(),
            da
            )
        max_seps = min_seps + da
        separations = np.zeros(min_seps.size)
        profile = np.zeros(min_seps.size)
        for i, (min_s, max_s) in enumerate(zip(min_seps, max_seps)):
            mask = (sep >= min_s) & (sep < max_s)
            separations[i] = np.mean([min_s, max_s])
            profile[i] = np.mean(self.skymap[self._is_visible][mask])
        return separations*u.deg, profile

# ============================================================= #


# ============================================================= #
# ------------------------- HpxABeam -------------------------- #
# ============================================================= #
class HpxABeam(HpxBeam):
    """
    """

    def __init__(self, resolution=1, **kwargs):
        super().__init__(
            resolution=resolution
        )
        self._fill_attr(kwargs)
        self._correct_beamsquint = True

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def beam(self, **kwargs):
        """
        """
        self._fill_attr(kwargs)

        # Compensate beam squint
        if self._correct_beamsquint:
            el = desquint_elevation(
                elevation=self.elana,
                opt_freq=self.squintfreq
            )
        else:
            el = self.elana.copy()
        # Real pointing
        az, el = analog_pointing(self.azana, el)
        log.debug(
            'ABeam asked=({}; {}) effective=({}, {})'.format(
                self.azana,
                self.elana,
                az,
                el
            )
        )

        arrfac = self.array_factor(
            az=az,
            el=el,
            antpos=ma_antpos(
                rot=ma_info['rot'][ma_info['ma'] == self.ma][0]
            ),
            freq=self.freq
        )
        # Antenna Gain
        antgain = nenufar_ant_gain(
            freq=self.freq,
            polar=self.polar,
            nside=self.nside,
            time=self.time
        )[self._is_visible]

        # Make sure to have an empty skymap to begin with
        self.skymap[:] = 0.
        # Udpate the pixels that can be seen
        self.skymap[self._is_visible] = arrfac * antgain
        return


    def grating_lobes(self):
        """
        """
        af = self.array_factor(
            az=self.azana,
            el=self.elana,
            antpos=ma_antpos(
                rot=ma_info['rot'][ma_info['ma'] == self.ma][0]
            ),
            freq=self.freq
        )
        pix_idx = np.arange(af.size)
        mask = np.isclose(
            af,
            af.max(),
            rtol=3e-1,
            atol=3e-01,
            equal_nan=False
        )
        gl_indices = []
        while np.any(mask):
            altaz_sel = self.ho_coords[mask]
            seps = altaz_sel[0].separation(altaz_sel).deg
            beamwidth = resolution(
                freq=self.freq,
                miniarrays=0
            ).to(u.deg).value * 3
            cluster_mask = seps < beamwidth
            max_id = np.argmax(af[mask][cluster_mask])
            gl_indices.append(
                pix_idx[mask][cluster_mask][max_id]
            )
            # radec.append(
            #     self.eq_coords[mask][cluster_mask][max_id]
            # )
            mask[pix_idx[mask][cluster_mask]] = False
        return gl_indices#altaz, radec


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _fill_attr(self, kwargs):
        """
        """
        def_vals = {
            'ma': 0,
            'freq': 50,
            'azana': 180,
            'elana': 90,
            'polar': 'NW',
            'ncpus': 1,
            'time': None,
            'squintfreq': 30,
            **kwargs
        } 
        for key, val in def_vals.items():
            if hasattr(self, key) and (key not in kwargs.keys()):
                continue
            setattr(self, key, val)
        return
# ============================================================= #


# ============================================================= #
# ------------------------- HpxDBeam -------------------------- #
# ============================================================= #
class HpxDBeam(HpxBeam):
    """
    """

    def __init__(self, resolution=1, **kwargs):
        super().__init__(
            resolution=resolution
        )
        self._fill_attr(kwargs)



    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def beam(self, **kwargs):
        """
        """
        self._fill_attr(kwargs)

        # Build the Mini-Array 'summed' response
        ana = HpxABeam(
            resolution=0.5 #self.resolution Fix it
        )
        abeams = {}
        for ma in self.ma:
            rot = ma_info['rot'][ma_info['ma'] == ma][0]
            if str(rot%60) not in abeams.keys():
                kwargs['ma'] = ma
                ana.beam(
                    **kwargs
                )
                abeams[str(rot%60)] = ud_grade(
                    ana.skymap,
                    nside_out=self.nside
                )[self._is_visible].copy()
            if not 'summa' in locals():
                summa = abeams[str(rot%60)]
            else:
                summa += abeams[str(rot%60)]
        log.debug(
            'DBeam effective=({}, {})'.format(
                self.azdig,
                self.eldig,
            )
        )
        arrfac = self.array_factor(
            az=self.azdig,
            el=self.eldig,
            antpos=ma_pos[np.isin(ma_info['ma'], self.ma)],
            freq=self.freq
        )

        beam = arrfac * summa

        # Make sure to have an empty skymap to begin with
        self.skymap[:] = 0.
        # Udpate the pixels that can be seen
        self.skymap[self._is_visible] = beam
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _fill_attr(self, kwargs):
        """
        """
        def_vals = {
            'ma': ma_info['ma'],
            'freq': 50,
            'azana': 180,
            'elana': 90,
            'azdig': 180,
            'eldig': 90,
            'polar': 'NW',
            'ncpus': 1,
            'time': None,
            **kwargs
        } 
        for key, val in def_vals.items():
            if hasattr(self, key) and (key not in kwargs.keys()):
                continue
            setattr(self, key, val)
        return
# ============================================================= #

