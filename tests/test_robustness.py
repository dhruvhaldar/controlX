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

def test_calculate_hinf_norm_invalid_omega():
    sys = ct.ss([[-1]], [[1]], [[1]], [[0]])
    with pytest.raises(ValueError, match="omega must be a numeric array or scalar."):
        robustness.calculate_hinf_norm(sys, omega="invalid")

    with pytest.raises(ValueError, match="omega must contain only finite numbers."):
        robustness.calculate_hinf_norm(sys, omega=np.nan)

    with pytest.raises(ValueError, match="omega must contain only finite numbers."):
        robustness.calculate_hinf_norm(sys, omega=[1, np.inf])
