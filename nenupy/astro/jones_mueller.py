#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *************************
    Jones - Mueller Formalism
    *************************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'JonesMatrix',
    'MuellerMatrix',
    'PolarVector',
    'JonesVector'
]


import numpy as np
from abc import ABC, abstractmethod


pauli_matrices = np.array([
    [[1, 0], [0, 1]],
    [[1, 0], [0, -1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]]
])
unitary_matrix = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [0, 1, 1, 0],
    [0, 1j, -1j, 0]
])
unitary_matrix_inv = np.linalg.inv(unitary_matrix)


# ============================================================= #
# ------------------------ PolarMatrix ------------------------ #
# ============================================================= #
class PolarizerMatrix(np.ndarray, ABC):


    @abstractmethod
    def __new__(cls):
        raise NotImplementedError


    def __array_finalize__(self, obj):
        if obj is None:
            return


    @classmethod
    @abstractmethod
    def elliptical(cls, theta: float, delta: float):
        """ """
        raise NotImplementedError()


    @classmethod
    @abstractmethod
    def linear_retarder(cls, theta: float, delta: float):
        """ """
        raise NotImplementedError()


    @classmethod
    def right_circular(cls):
        """ """
        return cls.elliptical(theta=45., delta=90.)


    @classmethod
    def left_circular(cls):
        """ """
        return cls.elliptical(theta=45., delta=-90.)


    @classmethod
    def linear(cls, theta: float):
        r""" Linear polarizer whose principal axis subtends an angle :math:`\theta` with the horizontal.

            :param theta:
                Rotation angle (in degrees) between the horizontal plane and the fast axis.
            :type theta:
                `float`

            .. seealso::
                Theocaris, Matrix Theory of Photoelasticity, eq. 4.34, 1979

        """
        return cls.elliptical(theta=theta, delta=0.)


    @classmethod
    def quarter_waveplate_retarder(cls, theta: float):
        """ Waveplate that converts linearly polarized light into circularly polarized light and vice versa.

            :param theta:
                Rotation angle (in degrees) between the horizontal plane and the fast axis.
            :type theta:
                `float`

        """
        return cls.linear_retarder(theta=theta, delta=90.)
    

    @classmethod
    def half_waveplate_retarder(cls, theta: float):
        """ Waveplate that shifts the polarization direction of linearly polarized light.

            :param theta:
                Rotation angle (in degrees) between the horizontal plane and the fast axis.
            :type theta:
                `float`

        """
        return cls.linear_retarder(theta=theta, delta=180.)
# ============================================================= #
# ============================================================= #

# ============================================================= #
# ------------------------ JonesMatrix ------------------------ #
# ============================================================= #
class JonesMatrix(PolarizerMatrix):
    """ Class handling Jones matrices.
        The Jones matrices are operators that act on Jones vectors.

        .. rubric:: Methods Summary

        .. autosummary::

            ~JonesMatrix.to_mueller

        .. rubric:: Class Methods Summary

        .. autosummary::

            ~JonesMatrix.elliptical
            ~JonesMatrix.linear_retarder
            ~PolarizerMatrix.right_circular
            ~PolarizerMatrix.left_circular
            ~PolarizerMatrix.linear
            ~PolarizerMatrix.quarter_wave_plate
            ~PolarizerMatrix.half_wave_plate

        .. rubric:: Attributes and Methods Documentation
    """


    def __new__(cls, input_array: np.ndarray):
        if input_array.shape[-2:] != (2, 2):
            raise ValueError(
                'Jones matrix should have the dimension (2, 2). '
                f'Input array is {input_array.shape}.'
            )
        obj = np.asarray(input_array).view(cls)
        return obj


    @classmethod
    def elliptical(cls, theta: float, delta: float):
        r""" Ideal elliptical, producing elliptically polarized light with azimuth :math:`\psi` and ellipticity :math:`\omega` such that :math`\tan(2\psi)=\tan(2\theta)\cos(\delta)` and :math:`\sin(2\omega)=\sin(2\theta)\sin(\delta).
            
            :param theta: in degrees.
            :type theta: `float`
            :param delta: in degrees.
            :type delta: `float`

            Theocaris, Matrix Theory of Photoelasticity, Table 4.1, 1979
        """
        th = np.radians(theta)
        ct = np.cos(th)
        st = np.sin(th)
        de = np.radians(delta)

        el_pol = np.array([
            [              ct*ct, np.exp(-1j*de)*st*ct],
            [np.exp(1j*de)*st*ct,                st*st],
        ])
        return cls(el_pol)


    @classmethod
    def linear_retarder(cls, theta: float, delta: float):
        r""" Linear retarder with its fast axis at an angle :math:`\theta` and retardation :math:`\delta`.
            
            :param theta:
                Rotation angle (in degrees) between the horizontal plane and the fast axis.
            :type theta:
                `float`
            :param delta:
                Phase delay (in degrees) between the fast and the slow axes.
            :type delta:
                `float`

            .. seealso::
                Theocaris, Matrix Theory of Photoelasticity, Table 4.2, 1979

        """
        th = np.radians(theta)
        ct = np.cos(th)
        st = np.sin(th)
        de = np.radians(delta)
        ed = np.exp(1j*de)
        ret_pol = np.exp(-1j*de/2) * np.array([
            [ct*ct + ed*st*st,   (1 - ed)*ct*st],
            [  (1 - ed)*ct*st, st*st + ed*ct*ct]
        ])
        return cls(ret_pol)


    def to_mueller(self):
        """ """
        new_shape = self.shape[:-2] + (4, 4)
        if len(new_shape) == 2:
            einsum_transfo = "ij,kl->ikjl"
        elif len(new_shape) == 3:
            einsum_transfo = "aij,akl->aikjl"
        elif len(new_shape) == 4:
            einsum_transfo = "abij,abkl->abikjl"
        return MuellerMatrix(
            np.matmul(
                np.matmul(
                    unitary_matrix, np.einsum(einsum_transfo, self, np.conj(self)).reshape(new_shape)# np.kron(self, np.conj(self))
                ),
                unitary_matrix_inv
            )
        )
# ============================================================= #
# ============================================================= #

# ============================================================= #
# ----------------------- MuellerMatrix ----------------------- #
# ============================================================= #
class MuellerMatrix(PolarizerMatrix):
    """ Class handling Mueller matrices.
        The Mueller matrices are operators that act on Stokes vectors.

        .. rubric:: Methods Summary

        .. autosummary::

            ~MuellerMatrix.to_jones
            ~MuellerMatrix.to_hermitian

        .. rubric:: Class Methods Summary

        .. autosummary::

            ~MuellerMatrix.elliptical
            ~MuellerMatrix.linear_retarder
            ~PolarizerMatrix.right_circular
            ~PolarizerMatrix.left_circular
            ~PolarizerMatrix.linear
            ~PolarizerMatrix.quarter_wave_plate
            ~PolarizerMatrix.half_wave_plate

        .. rubric:: Attributes and Methods Documentation
    """

    def __new__(cls, input_array: np.ndarray):
        if input_array.shape[-2:] != (4, 4):
            raise ValueError(
                'Mueller matrix should have the dimension (4, 4).'
            )
        if np.all(input_array.imag == np.zeros((4, 4))):
            obj = np.asarray(input_array.real).view(cls)
            return obj
        else:
            raise ValueError(
                'Mueller matrix should be real.'
            )


    @classmethod
    def elliptical(cls, theta: float, delta: float):
        r""" Ideal elliptical, producing elliptically polarized light with azimuth :math:`\psi` and ellipticity :math:`\omega` such that :math`\tan(2\psi)=\tan(2\theta)\cos(\delta)` and :math:`\sin(2\omega)=\sin(2\theta)\sin(\delta).
            
            :param theta: in degrees.
            :type theta: `float`
            :param delta: in degrees.
            :type delta: `float`

            Theocaris, Matrix Theory of Photoelasticity, Table 4.3, 1979
        """
        th = 2*np.radians(theta)
        ct = np.cos(th)
        st = np.sin(th)
        de = np.radians(delta)
        cd = np.cos(de)
        sd = np.sin(de)

        el_pol = 0.5*np.array([
            [    1,       ct,       st*cd,       st*sd],
            [   ct,    ct*ct,    st*ct*cd,    st*ct*sd],
            [st*cd, st*ct*cd, st*st*cd*cd, st*st*sd*cd],
            [st*sd, st*ct*sd, st*st*sd*cd, st*st*sd*sd]
        ])
        return cls(el_pol)


    @classmethod
    def linear_retarder(cls, theta: float, delta: float):
        """
            Theocaris, Matrix Theory of Photoelasticity, eq. 4.44, 1979
        """
        th = 2*np.radians(theta)
        ct = np.cos(th)
        st = np.sin(th)
        de = np.radians(delta)
        cd = np.cos(de)
        sd = np.sin(de)

        ret_pol = np.array([
            [1,                0,                0,      0],
            [0, ct*ct + st*st*cd,   (1 - cd)*st*ct, -st*sd],
            [0,   (1 - cd)*st*ct, st*st + ct*ct*cd,  ct*sd],
            [0,            st*sd,           -ct*sd,     cd]
        ])
        return cls(ret_pol)


    def to_hermitian(self) -> np.ndarray:
        r"""
            .. math::
                \mathbf{H} = \frac{1}{4} \sum_{i,j=0}^{3} M_{ij}( \mathbf{sigma_i} \kron \mathbf{\sigma}_j^{\star})

            ref : https://arxiv.org/pdf/1906.11198.pdf eq. 6
        """
        def sigma_ij(i: int, j: int):
            return np.matmul(np.matmul(unitary_matrix, np.kron(pauli_matrices[i], np.conj(pauli_matrices[j]))), unitary_matrix_inv)

        return 0.25*np.sum([self[i, j]*sigma_ij(i, j) for i in range(4) for j in range(4)], axis=0)


    def to_jones(self) -> np.ndarray:
        # h = self.to_hermitian()
        # eigenvalues, eigenvectors = np.linalg.eig(h)
        # non_zero_index = np.argmax(np.absolute(eigenvalues))
        # return JonesMatrix(np.dot(eigenvectors[non_zero_index], np.moveaxis(pauli_matrices, 0, 1)))

        # # Amplitude derived from Theocaris, Matrix Theory of Photoelasticity, eq. 4.70-4.73, 1979
        # amplitude = np.sqrt(
        #     0.5*np.array([
        #         [self[0, 0] + self[0, 1] + self[1, 0] + self[1, 1], self[0, 0] - self[0, 1] + self[1, 0] - self[1, 1]],
        #         [self[0, 0] + self[0, 1] - self[1, 0] - self[1, 1], self[0, 0] - self[0, 1] - self[1, 0] + self[1, 1]]
        #     ])
        # )
        # # Polar angles derived from Theocaris, Matrix Theory of Photoelasticity, eq. 4.74-4.76, 1979, assuming theta_11 (i.e. theta_00) = 0
        # polar_angle = np.array([
        #     [0, -np.arctan2(self[0, 3] + self[1, 3],  self[0, 2] + self[1, 2])],
        #     [np.arctan2(self[3, 0] + self[3, 1], self[2, 0] + self[2, 1]), np.arctan2(self[3, 2] - self[2, 3], self[2, 2] + self[3, 3])]
        # ])
        # return JonesMatrix(amplitude*np.exp(1j*polar_angle))
    
        # Amplitude derived from Theocaris, Matrix Theory of Photoelasticity, eq. 4.70-4.73, 1979
        amplitude = np.sqrt(
            0.5*np.array([
                [self[..., 0, 0] + self[..., 0, 1] + self[..., 1, 0] + self[..., 1, 1], self[..., 0, 0] - self[..., 0, 1] + self[..., 1, 0] - self[..., 1, 1]],
                [self[..., 0, 0] + self[..., 0, 1] - self[..., 1, 0] - self[..., 1, 1], self[..., 0, 0] - self[..., 0, 1] - self[..., 1, 0] + self[..., 1, 1]]
            ])
        )
        # Polar angles derived from Theocaris, Matrix Theory of Photoelasticity, eq. 4.74-4.76, 1979, assuming theta_11 (i.e. theta_00) = 0
        polar_angle = np.array([
            [np.zeros(self.shape)[..., 0, 0], -np.arctan2(self[..., 0, 3] + self[..., 1, 3],  self[..., 0, 2] + self[..., 1, 2])],
            [np.arctan2(self[..., 3, 0] + self[..., 3, 1], self[..., 2, 0] + self[..., 2, 1]), np.arctan2(self[..., 3, 2] - self[..., 2, 3], self[..., 2, 2] + self[..., 3, 3])]
        ])
        return JonesMatrix(
            np.moveaxis(
                amplitude*np.exp(1j*polar_angle),
                (0, 1),
                (-2, -1) 
            )
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ PolarVector ------------------------ #
# ============================================================= #
class PolarVector(np.ndarray, ABC):


    @abstractmethod
    def __new__(cls):
        raise NotImplementedError


    def __array_finalize__(self, obj):
        if obj is None:
            return


    @classmethod
    @abstractmethod
    def linear(cls, angle_deg):
        """ """
        raise NotImplementedError()


    @classmethod
    @abstractmethod
    def circular(cls, angle_deg):
        """ """
        raise NotImplementedError()


    @classmethod
    def horizontal(cls):
        """ """
        return cls.linear(angle_deg=0.)


    @classmethod
    def vertical(cls):
        """ """
        return cls.linear(angle_deg=90.)


    @classmethod
    def diagonal(cls):
        """ """
        return cls.linear(angle_deg=+45.)


    @classmethod
    def anti_diagonal(cls):
        """ """
        return cls.linear(angle_deg=-45.)


    @classmethod
    def left_circular(cls):
        """ """
        return cls.circular(angle_deg=+90.)


    @classmethod
    def right_circular(cls):
        """ """
        return cls.circular(angle_deg=-90.)
# ============================================================= #
# ============================================================= #



# ============================================================= #
# ------------------------ JonesVector ------------------------ #
# ============================================================= #
class JonesVector(PolarVector):
    """ """

    def __new__(cls, input_array: np.ndarray):
        if input_array.shape != (2,):
            raise ValueError(
                'Jones vector should have the dimension (2,).'
            )
        obj = np.asarray(input_array).view(cls)
        return obj


    @classmethod
    def linear(cls, angle_deg: float = 0):
        """ """
        angle_rad = np.radians(angle_deg)
        rotation = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        horizontal = np.array([1, 0])
        return cls(np.matmul(rotation, horizontal))


    @classmethod
    @abstractmethod
    def circular(cls, angle_deg: float = 90):
        """ """
        dx = np.radians(0)
        dy = np.radians(angle_deg)
        j_vec = np.array([
            np.cos(dx) + 1j*np.sin(dx),
            np.cos(dy) + 1j*np.sin(dy)
        ])
        return cls(j_vec/np.linalg.norm(j_vec))


    def to_stokes(self):
        """ """
        ex = np.abs(self[0])
        ey = np.abs(self[1])
        phi = np.angle(self[1]) - np.angle(self[0])
        return StokesVector(
            np.array([
                ex**2 + ey**2,
                ex**2 - ey**2,
                2*ex*ey*np.cos(phi),
                2*ex*ey*np.sin(phi)
            ])
        )
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ------------------------ JonesVector ------------------------ #
# ============================================================= #
class StokesVector(PolarVector):
    """ """


    def __new__(cls, input_array: np.ndarray):
        if input_array.shape != (4,):
            raise ValueError(
                'Stokes vector should have the dimension (4,).'
            )
        obj = np.asarray(input_array).view(cls)
        return obj


    @classmethod
    def elliptical(cls, psi: float, chi: float):
        """ psi and chi are in degrees"""
        psi_rad = np.radians(2*psi)
        chi_rad = np.radians(2*chi)
        stokes_vector = np.array([
            1,
            np.cos(psi_rad)*np.cos(chi_rad),
            np.sin(psi_rad)*np.cos(chi_rad),
            np.sin(chi_rad)
        ])
        return cls(stokes_vector)


    @classmethod
    def circular(cls, angle_deg):
        return super().circular(angle_deg)


    def to_jones(self):
        """ """
        polar_intensity = np.sqrt(np.sum(self[1:]**2))
        polar_amplitude = np.sqrt(polar_intensity)
        # Normalize the Stokes parameters by the fraction of polarisez intensity
        # I would be 1
        Q, U, V = self[1:]/polar_intensity
        # Compute the 2 components of the Jones vector
        a = np.sqrt((1 + Q)/2)
        if a == 0.:
            b = 1
        else:
            b = complex(U, V)/(2*a)
        return JonesVector(polar_amplitude*np.array([a, b]))
# ============================================================= #
# ============================================================= #

