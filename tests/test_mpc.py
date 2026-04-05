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

def test_mpc_invalid_weight_matrices():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    R = np.array([[1]])
    N = 10
    dt = 0.1

    # Non-finite Q
    Q_inf = np.array([[np.inf]])
    with pytest.raises(ValueError, match="Q must be finite"):
        mpc.MPCController(sys, Q_inf, R, N, dt)

    # Non-square Q
    Q_rect = np.array([[1, 2]])
    with pytest.raises(ValueError, match="Q must be square"):
        mpc.MPCController(sys, Q_rect, R, N, dt)

    # Non-symmetric Q
    Q_nonsym = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Q must be symmetric"):
        # Create a mock 2-state system to accept a 2x2 matrix
        sys2 = ct.ss([[-1, 0], [0, -1]], [[1], [1]], [[1, 0]], [[0]])
        mpc.MPCController(sys2, Q_nonsym, R, N, dt)

    # Not positive semi-definite
    Q_neg = np.array([[1, 0], [0, -1]])
    with pytest.raises(ValueError, match="Q must be positive semi-definite"):
        sys2 = ct.ss([[-1, 0], [0, -1]], [[1], [1]], [[1, 0]], [[0]])
        mpc.MPCController(sys2, Q_neg, R, N, dt)


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
