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
