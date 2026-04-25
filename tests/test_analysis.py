import sys
import os
import numpy as np
import control as ct
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import analysis

def test_calculate_poles():
    sys = ct.tf([1], [1, 2, 1]) # 1/(s+1)^2 -> poles at -1, -1
    poles = analysis.calculate_poles(sys)
    assert np.allclose(sorted(poles.real), [-1, -1])

def test_calculate_zeros():
    sys = ct.tf([1, 1], [1, 2, 1]) # (s+1)/(s+1)^2 = 1/(s+1) -> zero at -1
    zeros = analysis.calculate_zeros(sys)
    assert np.allclose(zeros.real, [-1])

def test_singular_values():
    # MIMO system
    # G = diag(2, 3)
    G = ct.tf([[2, 0], [0, 3]], [[1, 1], [1, 1]])
    sv = analysis.calculate_singular_values(G, omega=0)
    assert np.allclose(sv, [3, 2])

def test_relative_gain_array():
    # Diagonal system
    G = np.diag([2, 3])
    RGA = analysis.relative_gain_array(G)
    assert np.allclose(RGA, np.eye(2))

    # Triangular system
    G = np.array([[1, 1], [0, 1]])
    # RGA = G .* (G^-1)^T
    # G^-1 = [[1, -1], [0, 1]]
    # (G^-1)^T = [[1, 0], [-1, 1]]
    # RGA = [[1*1, 1*0], [0*-1, 1*1]] = [[1, 0], [0, 1]]
    RGA = analysis.relative_gain_array(G)
    assert np.allclose(RGA, np.eye(2))

    # Non-diagonal RGA
    G = np.array([[1, 1], [1, 2]])
    # G^-1 = 1/(2-1) * [[2, -1], [-1, 1]] = [[2, -1], [-1, 1]]
    # (G^-1)^T = [[2, -1], [-1, 1]]
    # RGA = [[1*2, 1*-1], [1*-1, 2*1]] = [[2, -1], [-1, 2]]
    RGA = analysis.relative_gain_array(G)
    expected = np.array([[2, -1], [-1, 2]])
    assert np.allclose(RGA, expected)

    # Singular matrix
    G = np.array([[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="Cannot compute RGA: System gain matrix is singular."):
        analysis.relative_gain_array(G)

    # Invalid string matrix
    with pytest.raises(ValueError, match="Gain matrix must be a numeric array."):
        analysis.relative_gain_array("invalid")

    # Invalid 1D array
    with pytest.raises(ValueError, match="Cannot compute RGA: System gain matrix is singular."):
        analysis.relative_gain_array([1, 2])

    # Matrix with NaNs
    with pytest.raises(ValueError, match="Gain matrix must contain only finite numbers."):
        analysis.relative_gain_array(np.array([[1, np.nan], [0, 1]]))

def test_calculate_singular_values_invalid_omega():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    with pytest.raises(ValueError, match="omega must be a numeric array or scalar."):
        analysis.calculate_singular_values(sys, omega="invalid")

    with pytest.raises(ValueError, match="omega must contain only finite numbers."):
        analysis.calculate_singular_values(sys, omega=np.nan)

    with pytest.raises(ValueError, match="omega must contain only finite numbers."):
        analysis.calculate_singular_values(sys, omega=[1, np.inf])

def test_system_gain_invalid_omega():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    with pytest.raises(ValueError, match="omega must be a numeric value."):
        analysis.system_gain(sys, omega="invalid")

    with pytest.raises(ValueError, match="omega must be finite."):
        analysis.system_gain(sys, omega=np.nan)

    with pytest.raises(ValueError, match="omega must be finite."):
        analysis.system_gain(sys, omega=np.inf)

def test_invalid_system_type():
    with pytest.raises(TypeError):
        analysis.calculate_poles("invalid")

    with pytest.raises(TypeError):
        analysis.calculate_zeros("invalid")

    with pytest.raises(TypeError):
        analysis.calculate_singular_values("invalid", omega=1.0)

    with pytest.raises(TypeError):
        analysis.system_gain("invalid", omega=1.0)
