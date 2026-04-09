import sys
import os
import numpy as np
import control as ct
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import mpc

def test_mpc_controller():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q = np.array([[1]])
    R = np.array([[1]])
    N = 10
    dt = 0.1
    constraints = {'umin': -1, 'umax': 1}

    controller = mpc.MPCController(sys, Q, R, N, dt, constraints)

    x0 = np.array([1])
    u0, status = controller.compute_control(x0)

    assert status in ["optimal", "optimal_inaccurate"]
    assert u0 <= 1
    assert u0 >= -1

def test_mpc_fail_securely_on_solver_error():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q = np.array([[1]])
    R = np.array([[1]])
    N = 10
    dt = 0.1
    # Introduce NaN constraint to force cvxpy solve exception
    constraints = {'umax': np.nan}

    controller = mpc.MPCController(sys, Q, R, N, dt, constraints)

    x0 = np.array([1])
    # Ensure it doesn't throw an exception, but fails gracefully
    u0, status = controller.compute_control(x0)

    assert status == "solver_error"
    assert np.allclose(u0, 0)

def test_mpc_invalid_matrix():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q = np.array([[1, 2], [3, 4]]) # Not symmetric, not correct shape
    R = np.array([[-1]]) # Not positive semi-definite
    N = 10
    dt = 0.1
    constraints = {'umin': -1, 'umax': 1}

    with pytest.raises(ValueError, match="Q must be a square matrix|Q must be symmetric|R must be positive semi-definite|Q must have shape"):
        controller = mpc.MPCController(sys, Q, R, N, dt, constraints)

    Q_valid = np.array([[1]])
    with pytest.raises(ValueError, match="R must be positive semi-definite"):
        controller = mpc.MPCController(sys, Q_valid, R, N, dt, constraints)

    Q_inf = np.array([[np.inf]])
    R_valid = np.array([[1]])
    with pytest.raises(ValueError, match="Q must contain only finite numbers"):
        controller = mpc.MPCController(sys, Q_inf, R_valid, N, dt, constraints)

def test_mpc_invalid_system_type():
    sys = ct.tf([1], [1, 1])
    Q = np.array([[1]])
    R = np.array([[1]])
    N = 10
    dt = 0.1
    constraints = {'umin': -1, 'umax': 1}

    with pytest.raises(TypeError, match="System must be a control.StateSpace object"):
        mpc.MPCController(sys, Q, R, N, dt, constraints)

def test_mpc_invalid_dimension():
    # sys is 1-state, 1-input
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    Q_invalid = np.eye(2) # Incorrect shape, should be (1, 1)
    R_invalid = np.eye(2) # Incorrect shape, should be (1, 1)
    N = 10
    dt = 0.1
    constraints = {'umin': -1, 'umax': 1}

    with pytest.raises(ValueError, match="Q must have shape"):
        mpc.MPCController(sys, Q_invalid, np.eye(1), N, dt, constraints)

    with pytest.raises(ValueError, match="R must have shape"):
        mpc.MPCController(sys, np.eye(1), R_invalid, N, dt, constraints)
