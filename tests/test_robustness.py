import sys
import os
import numpy as np
import control as ct
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import robustness

def test_sensitivity_function():
    sys = ct.tf([1], [1, 1])
    K = ct.tf([1], [1])
    S = robustness.sensitivity_function(sys, K)
    assert np.allclose(ct.poles(S), [-2])

def test_incompatible_timebases():
    plant = ct.tf([1], [1, 1], inputs=1, outputs=1)
    plant.dt = 0
    K = ct.tf([1], [1, 1], inputs=1, outputs=1)
    K.dt = 1
    with pytest.raises(ValueError, match="Incompatible timebases: Plant and Controller have conflicting sampling times."):
        robustness.sensitivity_function(plant, K)
    with pytest.raises(ValueError, match="Incompatible timebases: Plant and Controller have conflicting sampling times."):
        robustness.complementary_sensitivity_function(plant, K)

def test_complementary_sensitivity_function():
    sys = ct.tf([1], [1, 1])
    K = ct.tf([1], [1])
    T = robustness.complementary_sensitivity_function(sys, K)
    assert np.allclose(ct.poles(T), [-2])
    assert np.allclose(ct.dcgain(T), [0.5])

def test_small_gain_theorem_check():
    M = ct.tf([0.5], [1, 1])
    Delta = ct.tf([0.5], [1, 1])
    stable, _ = robustness.small_gain_theorem_check(M, Delta)
    # M_inf = 0.5. Delta_inf = 0.5. Product = 0.25 < 1.
    assert stable

def test_robust_stability_margin():
    sys = ct.tf([1], [1, 1])
    K = ct.tf([1], [1])
    T = robustness.complementary_sensitivity_function(sys, K)
    # T = 1/(s+2)
    # T_inf = 0.5
    # Margin = 1/0.5 = 2.0
    margin = robustness.robust_stability_margin(T)
    assert np.isclose(margin, 2.0, atol=0.1)

def test_small_gain_theorem_check_invalid_input():
    sys = ct.tf([1], [1, 1])

    with pytest.raises(ValueError, match="M must be a control system or a numeric matrix/scalar."):
        robustness.small_gain_theorem_check("invalid", sys)

    with pytest.raises(ValueError, match="M must contain only finite numbers."):
        robustness.small_gain_theorem_check(np.nan, sys)

    with pytest.raises(ValueError, match="Delta must be a control system or a numeric matrix/scalar."):
        robustness.small_gain_theorem_check(sys, "invalid")

    with pytest.raises(ValueError, match="Delta must contain only finite numbers."):
        robustness.small_gain_theorem_check(sys, np.nan)

def test_invalid_system_type():
    with pytest.raises(TypeError):
        robustness.sensitivity_function("invalid", "invalid")

    with pytest.raises(TypeError):
        robustness.complementary_sensitivity_function("invalid", "invalid")

def test_calculate_hinf_norm_too_large_omega():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    with pytest.raises(ValueError, match="omega must be a 1D array or scalar."):
        robustness.calculate_hinf_norm(sys, omega=np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match="omega array is too large"):
        robustness.calculate_hinf_norm(sys, omega=np.arange(10001))

def test_too_large_system_dimensions():
    sys = ct.ss(np.zeros((501, 501)), np.zeros((501, 1)), np.zeros((1, 501)), np.zeros((1, 1)))
    K = ct.ss(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))

    with pytest.raises(ValueError, match="Plant dimensions are too large"):
        robustness.sensitivity_function(sys, K)

    with pytest.raises(ValueError, match="Plant dimensions are too large"):
        robustness.complementary_sensitivity_function(sys, K)

    with pytest.raises(ValueError, match="Controller dimensions are too large"):
        robustness.sensitivity_function(K, sys)

    with pytest.raises(ValueError, match="Controller dimensions are too large"):
        robustness.complementary_sensitivity_function(K, sys)

    with pytest.raises(ValueError, match="M matrix dimensions are too large"):
        robustness.small_gain_theorem_check(np.eye(501), 0.5)

    with pytest.raises(ValueError, match="Delta matrix dimensions are too large"):
        robustness.small_gain_theorem_check(0.5, np.eye(501))

def test_hinf_norm_too_large_system():
    sys = ct.ss(np.zeros((501, 501)), np.zeros((501, 1)), np.zeros((1, 501)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="System dimensions are too large"):
        robustness.calculate_hinf_norm(sys)

def test_non_square_loop_matrix():
    G = ct.ss(np.eye(2)*-1, np.ones((2, 3)), np.ones((2, 2)), np.zeros((2, 3)))
    K = ct.ss(np.eye(3)*-1, np.ones((3, 1)), np.ones((3, 3)), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="Loop transfer matrix must be square."):
        robustness.sensitivity_function(G, K)
    with pytest.raises(ValueError, match="Loop transfer matrix must be square."):
        robustness.complementary_sensitivity_function(G, K)

def test_incompatible_dimensions():
    # G expects 1 input, K provides 2 outputs
    G = ct.tf([[[1], [1]]], [[[1, 1], [1, 2]]])
    K = ct.tf([[[1], [1]]], [[[1, 1], [1, 2]]])
    with pytest.raises(ValueError, match="Incompatible dimensions: Plant inputs must match Controller outputs."):
        robustness.sensitivity_function(G, K)
    with pytest.raises(ValueError, match="Incompatible dimensions: Plant inputs must match Controller outputs."):
        robustness.complementary_sensitivity_function(G, K)

def test_tf_dos_degree_limit():
    sys_large = ct.tf([1] * 505, [1] * 505)
    sys_small = ct.tf([1], [1, 1])

    with pytest.raises(ValueError, match="TransferFunction polynomial degree is too large"):
        robustness.sensitivity_function(sys_large, sys_small)

    with pytest.raises(ValueError, match="TransferFunction polynomial degree is too large"):
        robustness.sensitivity_function(sys_small, sys_large)

    with pytest.raises(ValueError, match="TransferFunction polynomial degree is too large"):
        robustness.complementary_sensitivity_function(sys_large, sys_small)

    with pytest.raises(ValueError, match="TransferFunction polynomial degree is too large"):
        robustness.calculate_hinf_norm(sys_large)
