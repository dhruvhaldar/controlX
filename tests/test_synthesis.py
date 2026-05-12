import sys
import os
import numpy as np
import control as ct
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import synthesis

def test_lqr_design():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q = np.array([[1]])
    R = np.array([[1]])
    K, _, _ = synthesis.design_lqr(sys, Q, R)
    assert K.shape == (1, 1)

    with pytest.raises(TypeError, match="System must be a control.StateSpace object."):
        synthesis.design_lqr(ct.tf([1], [1, 1]), Q, R)

def test_lqr_invalid_matrix():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q = np.array([[-1]]) # Not positive semi-definite
    R = np.array([[1]])
    with pytest.raises(ValueError, match="Q must be positive semi-definite"):
        synthesis.design_lqr(sys, Q, R)

    Q = np.array([[1]])
    R = np.array([[np.nan]]) # Not finite
    with pytest.raises(ValueError, match="R must contain only finite numbers"):
        synthesis.design_lqr(sys, Q, R)

def test_kalman_invalid_matrix():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Qn = np.array([[1, 2], [3, 4]]) # Not symmetric
    Rn = np.array([[1]])
    with pytest.raises(ValueError, match="Qn must be symmetric|Qn must have shape"):
        synthesis.design_kalman_filter(sys, Qn, Rn)

    Qn = np.array([[1]])
    Rn = np.array([[1, 2]]) # Not square
    with pytest.raises(ValueError, match="Rn must be a square matrix"):
        synthesis.design_kalman_filter(sys, Qn, Rn)

def test_lqg_design():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q = np.array([[1]])
    R = np.array([[1]])
    Qn = np.array([[1]])
    Rn = np.array([[1]])
    ctrl = synthesis.design_lqg(sys, Q, R, Qn, Rn)
    assert ctrl.ninputs == 1
    assert ctrl.noutputs == 1
    assert ctrl.nstates == 1

def test_synthesis_invalid_dimension():
    # sys is 1-state, 1-input, 1-output
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])

    Q_invalid = np.eye(2)
    R_invalid = np.eye(2)

    with pytest.raises(ValueError, match="Q must have shape"):
        synthesis.design_lqr(sys, Q_invalid, np.eye(1))

    with pytest.raises(ValueError, match="R must have shape"):
        synthesis.design_lqr(sys, np.eye(1), R_invalid)

    Qn_invalid = np.eye(2)
    Rn_invalid = np.eye(2)

    with pytest.raises(ValueError, match="Qn must have shape"):
        synthesis.design_kalman_filter(sys, Qn_invalid, np.eye(1))

    with pytest.raises(ValueError, match="Rn must have shape"):
        synthesis.design_kalman_filter(sys, np.eye(1), Rn_invalid)

def test_lqr_too_large_system_dimensions():
    sys = ct.ss(np.zeros((501, 501)), np.zeros((501, 1)), np.zeros((1, 501)), np.zeros((1, 1)))
    Q = np.eye(501)
    R = np.eye(1)
    with pytest.raises(ValueError, match="System dimensions are too large"):
        synthesis.design_lqr(sys, Q, R)

def test_kalman_too_large_system_dimensions():
    sys = ct.ss(np.zeros((501, 501)), np.zeros((501, 1)), np.zeros((1, 501)), np.zeros((1, 1)))
    Qn = np.eye(1)
    Rn = np.eye(1)
    with pytest.raises(ValueError, match="System dimensions are too large"):
        synthesis.design_kalman_filter(sys, Qn, Rn)

def test_uncontrollable_system_lqr():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[0], [0]]) # B is zero, uncontrollable
    Q = np.eye(2)
    R = np.eye(1)
    sys_model = ct.ss(A, B, np.eye(2), np.zeros((2,1)))
    with pytest.raises(ValueError, match="Failed to solve Riccati equation"):
        synthesis.design_lqr(sys_model, Q, R)

def test_unobservable_system_kalman_filter():
    A = np.array([[1, 0], [0, 1]])
    C = np.array([[0, 0]]) # unobservable
    Qn = np.eye(2)
    Rn = np.eye(1)
    sys_model = ct.ss(A, np.eye(2), C, np.zeros((1,2)))
    with pytest.raises(ValueError, match="Failed to solve Riccati equation"):
        synthesis.design_kalman_filter(sys_model, Qn, Rn)

def test_validate_matrix_invalid_dimension():
    with pytest.raises(ValueError, match="Matrix must be a 1D or 2D array."):
        synthesis._validate_matrix(np.ones((2,2,2)))

from unittest.mock import patch
import scipy.linalg

def test_lqr_singular_matrix():
    # Provide an uncontrollable system (B=0) so Bt_S is 0, making M = R + Bt_S*B = 0
    # when R is 0. This forces the solver to encounter a singular matrix.
    sys_model = ct.ss([[-1]], [[0]], [[1]], [[0]], dt=0.1)
    Q = np.array([[1]])
    R = np.array([[0]])
    with pytest.raises(ValueError, match="Failed to compute LQR gain: Matrix is singular."):
        synthesis.design_lqr(sys_model, Q, R)

def test_kalman_singular_matrix():
    # To force the singular matrix failure after a successful Riccati solve, we mock the solver.
    sys_model = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Qn = np.array([[1]])
    Rn = np.array([[0]])

    def mock_solve_continuous_are(*args, **kwargs):
        return np.eye(1)

    with patch("scipy.linalg.solve_continuous_are", side_effect=mock_solve_continuous_are):
        with pytest.raises(ValueError, match="Failed to compute Kalman gain: Matrix is singular."):
            synthesis.design_kalman_filter(sys_model, Qn, Rn)
