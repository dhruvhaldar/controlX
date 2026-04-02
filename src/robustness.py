import numpy as np
import control as ct
import warnings

def sensitivity_function(G, K):
    """
    Calculate the sensitivity function S(s) = (I + G(s)K(s))^-1.

    Args:
        G (control.StateSpace or control.TransferFunction): The plant.
        K (control.StateSpace or control.TransferFunction): The controller.

    Returns:
        control.StateSpace: The sensitivity function S.
    """
    L = G * K
    # Sensitivity Function S = (I + L)^-1
    # control.feedback returns L / (1+L) if sign=-1
    # To get (1+L)^-1, we can compute 1 - T
    # Or simply feedback(1, L, sign=-1)

    # Using formula: S = (I + G*K)^-1
    # We can use feedback(I, G*K) ? No.
    # feedback(sys1, sys2) computes sys1 / (1 + sys1*sys2)
    # S = feedback(1, G*K) assuming identity feedback path?
    # If sys1 is identity (size of outputs of L), and sys2 is L.

    # Correct way using control library:
    # S = feedback(I, L) where I is identity with size equal to number of outputs

    # However, if G and K are MIMO, we need to be careful with dimensions.
    # Let's assume standard negative feedback.

    # Try using feedback(eye(n_outputs), L)

    n_outputs = G.noutputs
    I = ct.ss([], [], [], np.eye(n_outputs))
    S = ct.feedback(I, L)
    return S

def complementary_sensitivity_function(G, K):
    """
    Calculate the complementary sensitivity function T(s) = G(s)K(s)(I + G(s)K(s))^-1.
    T = I - S

    Args:
        G (control.StateSpace or control.TransferFunction): The plant.
        K (control.StateSpace or control.TransferFunction): The controller.

    Returns:
        control.StateSpace: The complementary sensitivity function T.
    """
    L = G * K
    # T = L / (1 + L)
    # Using feedback(L, I) or feedback(L, 1) if SISO
    n_inputs = L.ninputs
    I = ct.ss([], [], [], np.eye(n_inputs))
    T = ct.feedback(L, I)
    return T

def calculate_hinf_norm(sys, omega=None):
    """
    Calculate the H-infinity norm of a system by sampling frequency response.

    Args:
        sys (control.StateSpace or control.TransferFunction): The system.
        omega (array-like, optional): Frequency points. If None, generated automatically.

    Returns:
        float: The approximated H-infinity norm.
    """
    if omega is None:
        omega = np.logspace(-2, 2, 1000)

    # ⚡ Bolt Optimization: Replace slow python loop with vectorized batched SVD.
    # Calculates frequency response for all frequencies simultaneously.
    # Avoids sys.frequency_response overhead for StateSpace objects (which relies
    # on slow Horner evaluation fallback without slycot) by directly computing
    # C @ inv(sI - A) @ B + D over the frequency array.
    omega_arr = np.atleast_1d(omega)

    if isinstance(sys, ct.StateSpace):
        if sys.dt is None or sys.dt == 0:
            s = 1j * omega_arr
        else:
            s = np.exp(1j * omega_arr * sys.dt)

        I = np.eye(sys.nstates)
        sI_minus_A = s[:, np.newaxis, np.newaxis] * I - sys.A

        try:
            inv_sI_minus_A = np.linalg.inv(sI_minus_A)
            # resp_T shape: (freqs, outputs, inputs)
            resp_T = sys.C @ inv_sI_minus_A @ sys.B + sys.D

            if sys.ninputs == 1 and sys.noutputs == 1:
                max_sv = np.max(np.abs(resp_T))
            else:
                svs = np.linalg.svd(resp_T, compute_uv=False)
                max_sv = np.max(svs)
        except np.linalg.LinAlgError:
            # Fallback for pole collision
            resp = sys.frequency_response(omega_arr).complex

            if resp.ndim == 1:
                max_sv = np.max(np.abs(resp))
            else:
                resp_T = np.transpose(resp, (2, 0, 1))
                svs = np.linalg.svd(resp_T, compute_uv=False)
                max_sv = np.max(svs)
    else:
        resp = sys.frequency_response(omega_arr).complex

        if resp.ndim == 1:
            # SISO case: resp is 1D array of complex values
            max_sv = np.max(np.abs(resp))
        else:
            # MIMO case: resp is (outputs, inputs, frequencies)
            # Transpose to (frequencies, outputs, inputs) for batched svd
            resp_T = np.transpose(resp, (2, 0, 1))
            svs = np.linalg.svd(resp_T, compute_uv=False)
            max_sv = np.max(svs)

    return float(max_sv)

def small_gain_theorem_check(M, Delta, omega=None):
    """
    Check stability using the Small Gain Theorem.
    Specifically, check if ||M||_inf * ||Delta||_inf < 1.

    Args:
        M (control.StateSpace): The nominal closed-loop system seen by the uncertainty.
        Delta (control.StateSpace or float): The uncertainty.
        omega (array-like, optional): Frequency points for norm approximation.

    Returns:
        bool: True if stable, False otherwise.
        float: The product of norms.
    """
    if isinstance(M, (ct.StateSpace, ct.TransferFunction)):
        norm_M = calculate_hinf_norm(M, omega)
    else:
        norm_M = np.linalg.norm(M, 2) # Assume matrix gain

    if isinstance(Delta, (ct.StateSpace, ct.TransferFunction)):
        norm_Delta = calculate_hinf_norm(Delta, omega)
    else:
        norm_Delta = np.abs(Delta)

    product = norm_M * norm_Delta
    return product < 1.0, product

def robust_stability_margin(S, omega=None):
    """
    Calculate the robust stability margin, which is 1 / ||T||_inf for multiplicative uncertainty.

    Args:
        S (control.StateSpace): Sensitivity or Complementary Sensitivity function.
        omega (array-like, optional): Frequency points for norm approximation.

    Returns:
        float: The stability margin.
    """
    norm_S = calculate_hinf_norm(S, omega)
    if norm_S == 0:
        return float('inf')
    return 1.0 / norm_S
