#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2022, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


from nenupy.astro.jones_mueller import (
    MuellerMatrix,
    JonesMatrix,
    JonesVector,
    StokesVector
)

import numpy as np


def test_mueller_to_hermitian():
    m = MuellerMatrix(
        np.array(
            [
                [20, -2, -10, 6],
                [10, -8, -10, 14],
                [2, 14, -2, 8],
                [-6, -2, 16, 6]
            ]
        )
    )
    h_computed = m.to_hermitian()
    h_desired = np.array(
        [
            [4, 2+2j, -2+4j, 6j],
            [2-2j, 2, 1+3j, 3+3j],
            [-2-4j, 1-3j, 5, 6-3j],
            [-6j, 3-3j, 6+3j, 9]
        ]
    )
    np.testing.assert_allclose(h_computed, h_desired, atol=1e-7)


def test_jones_to_mueller():
    """ Converts a linear polarizer"""
    j = JonesMatrix(
        np.array([
            [1, 0],
            [1, 0]
        ])
    )
    m_computed = j.to_mueller()
    m_desired = np.array([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_jones_to_mueller_to_jones():
    j_start = JonesMatrix(
        np.array([
            [1, 0],
            [1, 0]
        ])
    )
    m = j_start.to_mueller()
    j_end = m.to_jones()
    np.testing.assert_allclose(j_end, j_start, atol=1e-7)


def test_mueller_horizontal():
    m_computed = MuellerMatrix.linear(theta=0)
    m_desired = 0.5*np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_vertical():
    m_computed = MuellerMatrix.linear(theta=-90)
    m_desired = 0.5*np.array([
        [1, -1, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_diag1():
    m_computed = MuellerMatrix.linear(theta=+45)
    m_desired = 0.5*np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_diag2():
    m_computed = MuellerMatrix.linear(theta=-45)
    m_desired = 0.5*np.array([
        [1, 0, -1, 0],
        [0, 0, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_right_circular():
    m_computed = MuellerMatrix.right_circular()
    m_desired = 0.5*np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_left_circular():
    m_computed = MuellerMatrix.left_circular()
    m_desired = 0.5*np.array([
        [1, 0, 0, -1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1, 0, 0, 1]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_quarter_wave_vertical():
    m_computed = MuellerMatrix.quarter_waveplate_retarder(theta=90)
    m_desired = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_quarter_wave_horizontal():
    m_computed = MuellerMatrix.quarter_waveplate_retarder(theta=0)
    m_desired = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_mueller_half_wave_plate():
    m_computed1 = MuellerMatrix.half_waveplate_retarder(theta=0)
    m_computed2 = MuellerMatrix.half_waveplate_retarder(theta=-90)
    m_desired = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])
    np.testing.assert_allclose(m_computed1, m_desired, atol=1e-7)
    np.testing.assert_allclose(m_computed2, m_desired, atol=1e-7)


def test_jones_right_circular():
    j_computed = JonesMatrix.right_circular()
    j_desired = 0.5*np.array([
        [1, -1j],
        [1j, 1],
    ])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)


def test_jones_left_circular():
    j_computed = JonesMatrix.left_circular()
    j_desired = 0.5*np.array([
        [1, 1j],
        [-1j, 1],
    ])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)


def test_jones_quarter_wave_vertical():
    j_computed = JonesMatrix.quarter_waveplate_retarder(theta=90)
    j_desired = np.exp(np.pi*1j/4)*np.array([
        [1, 0],
        [0, -1j],
    ])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)


def test_jones_quarter_wave_horizontal():
    j_computed = JonesMatrix.quarter_waveplate_retarder(theta=0)
    j_desired = np.exp(-np.pi*1j/4)*np.array([
        [1, 0],
        [0, 1j],
    ])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)


def test_jones_half_wave_plate():
    j_computed = JonesMatrix.half_waveplate_retarder(theta=90)
    j_desired = np.exp(-np.pi*1j/2)*np.array([
        [-1, 0],
        [0, 1],
    ])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)


def test_jones_diag1():
    m_computed = JonesMatrix.linear(theta=+45)
    m_desired = 0.5*np.array([
        [1, 1],
        [1, 1]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_jones_diag2():
    m_computed = JonesMatrix.linear(theta=-45)
    m_desired = 0.5*np.array([
        [1, -1],
        [-1, 1]
    ])
    np.testing.assert_allclose(m_computed, m_desired, atol=1e-7)


def test_jones_vectors():
    np.testing.assert_allclose(
        JonesVector.horizontal(),
        np.array([1, 0]),
        atol=1e-7
    )
    np.testing.assert_allclose(
        JonesVector.vertical(),
        np.array([0, 1]),
        atol=1e-7
    )
    np.testing.assert_allclose(
        JonesVector.diagonal(),
        np.array([1, 1])/np.sqrt(2),
        atol=1e-7
    )
    np.testing.assert_allclose(
        JonesVector.anti_diagonal(),
        np.array([1, -1])/np.sqrt(2),
        atol=1e-7
    )
    np.testing.assert_allclose(
        JonesVector.right_circular(),
        np.array([1, -1j])/np.sqrt(2),
        atol=1e-7
    )
    np.testing.assert_allclose(
        JonesVector.left_circular(),
        np.array([1, 1j])/np.sqrt(2),
        atol=1e-7
    )


def test_jonesvector_to_stokes():
    # Horizontal P-state
    j = JonesVector.horizontal()
    s_computed = j.to_stokes()
    s_desired = np.array([1, 1, 0, 0])
    np.testing.assert_allclose(s_computed, s_desired, atol=1e-7)

    # Vertical P-state
    j = JonesVector.vertical()
    s_computed = j.to_stokes()
    s_desired = np.array([1, -1, 0, 0])
    np.testing.assert_allclose(s_computed, s_desired, atol=1e-7)

    # P-state at +45deg
    j = JonesVector.diagonal()
    s_computed = j.to_stokes()
    s_desired = np.array([1, 0, 1, 0])
    np.testing.assert_allclose(s_computed, s_desired, atol=1e-7)

    # P-state at -45deg
    j = JonesVector.anti_diagonal()
    s_computed = j.to_stokes()
    s_desired = np.array([1, 0, -1, 0])
    np.testing.assert_allclose(s_computed, s_desired, atol=1e-7)

    # R-state
    j = JonesVector.right_circular()
    s_computed = j.to_stokes()
    s_desired = np.array([1, 0, 0, -1]) # opposite sign in https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture17_0.pdf
    np.testing.assert_allclose(s_computed, s_desired, atol=1e-7)

    # L-state
    j = JonesVector.left_circular()
    s_computed = j.to_stokes()
    s_desired = np.array([1, 0, 0, +1]) # opposite sign in https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture17_0.pdf
    np.testing.assert_allclose(s_computed, s_desired, atol=1e-7)


def test_stokes_to_jones():
    # Horizontal P-state
    s = StokesVector(
        np.array([1, 1, 0, 0])
    )
    j_computed = s.to_jones()
    j_desired = np.array([1, 0])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)

    # Vertical P-state
    s = StokesVector(
        np.array([1, -1, 0, 0])
    )
    j_computed = s.to_jones()
    j_desired = np.array([0, 1])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)

    # P-state at +45deg
    s = StokesVector(
        np.array([1, 0, 1, 0])
    )
    j_computed = s.to_jones()
    j_desired = (1/np.sqrt(2))*np.array([1, 1])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)

    # P-state at -45deg
    s = StokesVector(
        np.array([1, 0, -1, 0])
    )
    j_computed = s.to_jones()
    j_desired = (1/np.sqrt(2))*np.array([1, -1])
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)

    # R-state
    s = StokesVector(
        np.array([1, 0, 0, -1])
    )
    j_computed = s.to_jones()
    j_desired = (1/np.sqrt(2))*np.array([1, -1j]) # opposite sign in https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture17_0.pdf
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)

    # L-state
    s = StokesVector(
        np.array([1, 0, 0, 1])
    )
    j_computed = s.to_jones()
    j_desired = (1/np.sqrt(2))*np.array([1, 1j]) # opposite sign in https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture17_0.pdf
    np.testing.assert_allclose(j_computed, j_desired, atol=1e-7)

