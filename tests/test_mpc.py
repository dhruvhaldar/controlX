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

def test_mpc_controller_solver_error():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    # Q non-PSD will cause a DCPError in CVXPY
    Q = np.array([[-1]])
    R = np.array([[1]])
    N = 10
    dt = 0.1
    constraints = {'umin': -1, 'umax': 1}

    controller = mpc.MPCController(sys, Q, R, N, dt, constraints)

    x0 = np.array([1])
    u0, status = controller.compute_control(x0)

    assert status == "solver_error"
    assert np.all(u0 == 0)
