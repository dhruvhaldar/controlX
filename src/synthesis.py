import numpy as np
import control as ct

def design_lqr(sys, Q, R):
    """
    Design an LQR controller for the system.
    u = -Kx
    Minimizes J = integral(x'Qx + u'Ru)

    Args:
        sys (control.StateSpace): The system.
        Q (np.ndarray): State weighting matrix.
        R (np.ndarray): Input weighting matrix.

    Returns:
        K (np.ndarray): State feedback gain.
        S (np.ndarray): Solution to Riccati equation.
        E (np.ndarray): Eigenvalues of the closed loop system.
    """
    # Security: Explicit type enforcement to fail securely. State weights cannot be applied
    # to arbitrary transfer function realizations, which leads to mathematically unsound logic.
    if not isinstance(sys, ct.StateSpace):
        raise TypeError("System must be a control.StateSpace object. State matrices (Q, Qn) cannot be applied to arbitrary transfer function realizations.")

    K, S, E = ct.lqr(sys, Q, R)
    return K, S, E

def design_kalman_filter(sys, Qn, Rn, G=None):
    """
    Design a Kalman Filter for the system.
    x_dot = Ax + Bu + Gw
    y = Cx + Du + v

    Args:
        sys (control.StateSpace): The system.
        Qn (np.ndarray): Process noise covariance.
        Rn (np.ndarray): Measurement noise covariance.
        G (np.ndarray, optional): Noise input matrix. If None, assumes G = B.

    Returns:
        L (np.ndarray): Kalman gain.
        P (np.ndarray): Error covariance.
        E (np.ndarray): Eigenvalues of the estimator.
    """
    # Security: Explicit type enforcement to fail securely. State weights cannot be applied
    # to arbitrary transfer function realizations, which leads to mathematically unsound logic.
    if not isinstance(sys, ct.StateSpace):
        raise TypeError("System must be a control.StateSpace object. State matrices (Q, Qn) cannot be applied to arbitrary transfer function realizations.")

    if G is None:
        G = sys.B

    # lqe takes (A, G, C, Qn, Rn)
    L, P, E = ct.lqe(sys.A, G, sys.C, Qn, Rn)
    return L, P, E

def design_lqg(sys, Q, R, Qn, Rn, G=None):
    """
    Design an LQG controller.
    Combines LQR and Kalman Filter.

    Args:
        sys (control.StateSpace): The system.
        Q, R: LQR weights.
        Qn, Rn: Kalman Filter noise covariances.
        G: Noise input matrix for KF.

    Returns:
        ctrl (control.StateSpace): The LQG controller K(s).
    """
    # Security: Explicit type enforcement to fail securely. State weights cannot be applied
    # to arbitrary transfer function realizations, which leads to mathematically unsound logic.
    if not isinstance(sys, ct.StateSpace):
        raise TypeError("System must be a control.StateSpace object. State matrices (Q, Qn) cannot be applied to arbitrary transfer function realizations.")

    # Design LQR
    K, _, _ = design_lqr(sys, Q, R)

    # Design KF
    L, _, _ = design_kalman_filter(sys, Qn, Rn, G)

    # Formulate Controller State Space
    # x_hat_dot = (A - BK - LC + LDK) x_hat + L y
    # u = -K x_hat

    A, B, C, D = sys.A, sys.B, sys.C, sys.D

    # Check dimensions
    # K is (n_inputs, n_states)
    # L is (n_states, n_outputs)

    Ac = A - B @ K - L @ C + L @ D @ K
    Bc = L
    Cc = -K
    Dc = np.zeros((sys.ninputs, sys.noutputs))

    ctrl = ct.ss(Ac, Bc, Cc, Dc)
    return ctrl
