import numpy as np
import cvxpy as cp
import control as ct

class MPCController:
    """
    Model Predictive Controller.
    """
    def __init__(self, sys, Q, R, N, dt=0.1, constraints=None):
        """
        Initialize the MPC controller.

        Args:
            sys (control.StateSpace): The system (continuous or discrete).
            Q (np.ndarray): State weighting matrix.
            R (np.ndarray): Input weighting matrix.
            N (int): Prediction horizon.
            dt (float): Sampling time.
            constraints (dict): Dictionary of constraints.
                'xmin': np.ndarray or float
                'xmax': np.ndarray or float
                'umin': np.ndarray or float
                'umax': np.ndarray or float
        """
        # Security: Input validation to prevent resource exhaustion
        if not isinstance(N, int) or N <= 0 or N > 1000:
            raise ValueError("Prediction horizon N must be a positive integer <= 1000")
        if dt <= 0:
            raise ValueError("Sampling time dt must be positive")

        self.dt = dt
        self.N = N
        self.Q = Q
        self.R = R
        self.constraints = constraints if constraints else {}

        # Discretize system if continuous
        if sys.dt is None or sys.dt == 0:
            self.sys_d = ct.c2d(sys, dt)
        else:
            self.sys_d = sys

        self.A = self.sys_d.A
        self.B = self.sys_d.B
        self.n_x = self.sys_d.nstates
        self.n_u = self.sys_d.ninputs

        # Compute terminal cost P using DARE (optional, often P=Q is used or solution to Riccati)
        # Solve P = A'PA - A'PB(R + B'PB)^-1 B'PA + Q
        try:
            # control.dare solves: X = A'XA - A'XB(R + B'XB)^-1 B'XA + Q
            # Returns X, L, G
            X, _, _ = ct.dare(self.A, self.B, self.Q, self.R)
            self.P = X
        except Exception:
            # Security: Do not leak exception details in console
            print("Warning: Could not compute terminal cost P. Using Q.")
            self.P = self.Q

    def compute_control(self, x0):
        """
        Compute the optimal control input for the current state x0.

        Args:
            x0 (np.ndarray): Current state vector.

        Returns:
            u0 (np.ndarray): Optimal control input to apply.
            status (str): Solver status.
        """
        # Variables
        x = cp.Variable((self.n_x, self.N + 1))
        u = cp.Variable((self.n_u, self.N))

        cost = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            cost += cp.quad_form(x[:, k], self.Q) + cp.quad_form(u[:, k], self.R)
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]

            # Input constraints
            if 'umin' in self.constraints:
                constraints += [u[:, k] >= self.constraints['umin']]
            if 'umax' in self.constraints:
                constraints += [u[:, k] <= self.constraints['umax']]

            # State constraints
            if 'xmin' in self.constraints:
                constraints += [x[:, k+1] >= self.constraints['xmin']]
            if 'xmax' in self.constraints:
                constraints += [x[:, k+1] <= self.constraints['xmax']]

        # Terminal cost
        cost += cp.quad_form(x[:, self.N], self.P)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"MPC Warning: Solver status: {prob.status}")
            return np.zeros(self.n_u), prob.status

        return u[:, 0].value, prob.status
