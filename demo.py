import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import analysis
import robustness
import synthesis
import mpc

def run_demo():
    print("Running ControlX Demo...")

    # 1. Define MIMO System (2x2)
    # G(s) = [ 4/(s+2)  1/(s+1) ]
    #        [ 2/(s+3)  3/(s+2) ]

    # Create State Space model
    # We can create individual TFs and combine them?
    # Or just create a random stable MIMO system.
    # Let's create a known system.

    # TF matrix
    num = [[[4], [1]], [[2], [3]]]
    den = [[[1, 2], [1, 1]], [[1, 3], [1, 2]]]
    G_tf = ct.tf(num, den)

    # Convert to SS
    # Since we don't have slycot, we should define SS directly or use simple diagonal + rotation
    # A simple way is to realize each TF and parallel/series connect them.
    # But let's just define A, B, C, D for a 4th order system.

    A = np.array([[-2, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -3, 0],
                  [0, 0, 0, -2]])
    B = np.array([[1, 0],
                  [0, 1],
                  [1, 0],
                  [0, 1]])
    C = np.array([[4, 1, 0, 0],
                  [0, 0, 2, 3]])
    D = np.zeros((2, 2))

    sys = ct.ss(A, B, C, D)
    print("System Defined.")

    # 2. Analysis
    poles = analysis.calculate_poles(sys)
    zeros = analysis.calculate_zeros(sys)
    print(f"Poles: {poles}")
    print(f"Zeros: {zeros}")

    # RGA at DC
    G_0 = analysis.system_gain(sys, omega=0)
    RGA = analysis.relative_gain_array(G_0)
    print(f"RGA at w=0:\n{RGA}")

    # Singular Values Plot
    omega = np.logspace(-2, 2, 100)
    # ⚡ Bolt Optimization: Use vectorized single call instead of slow Python loop
    sv_arr = analysis.calculate_singular_values(sys, omega)

    plt.figure()
    plt.loglog(omega, sv_arr)
    plt.title("Singular Value Plot (Sigma)")
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel("Singular Values")
    plt.grid(True)
    plt.savefig("ControlX/images/sigma_plot.png")
    plt.close()
    print("Saved sigma_plot.png")

    # 3. Synthesis (LQG)
    # Weights
    Q = np.eye(4)
    R = np.eye(2)
    # Qn is process noise covariance. Since G defaults to B, noise is added to inputs.
    # So Qn should be (n_inputs, n_inputs) = (2, 2).
    Qn = np.eye(2) * 0.1
    Rn = np.eye(2) * 0.1

    K_lqg = synthesis.design_lqg(sys, Q, R, Qn, Rn)

    # Closed loop system with LQG
    # Plant inputs u = -K_lqg * [y] (This is output feedback)
    # Wait, LQG controller has inputs y and outputs u.
    # Feedback loop:
    # Plant P: u -> y
    # Controller K: y -> u
    # Closed loop: y = P u = P (-K y + r) ?
    # Usually standard feedback is negative feedback.
    # K_lqg input is y (measurement). Output is u.
    # Check dimensions.
    # sys outputs: 2. K_lqg inputs: 2.
    # sys inputs: 2. K_lqg outputs: 2.

    L_lqg = sys * K_lqg
    T_lqg = ct.feedback(L_lqg, np.eye(2))

    # Step Response
    t, y = ct.step_response(T_lqg, T=10)
    # y is (n_outputs, n_inputs, n_time_steps)

    plt.figure()
    plt.plot(t, y[0, 0, :], label='y1 from r1')
    plt.plot(t, y[1, 0, :], label='y2 from r1')
    plt.title("LQG Step Response (r1 -> y)")
    plt.legend()
    plt.grid(True)
    plt.savefig("ControlX/images/lqg_step_response.png")
    plt.close()
    print("Saved lqg_step_response.png")

    # 4. MPC Simulation
    # Use discrete system for MPC
    dt = 0.1
    mpc_ctrl = mpc.MPCController(sys, Q=np.eye(4), R=np.eye(2), N=10, dt=dt,
                                 constraints={'umin': -1, 'umax': 1})

    # Simulate MPC loop
    T_sim = 5.0
    steps = int(T_sim / dt)
    x = np.zeros((4, 1))
    x[:, 0] = [1, 1, 1, 1] # Initial state

    # Lists to store history
    x_hist = [x.flatten()]
    u_hist = []
    t_hist = [0]

    curr_x = x.flatten()

    for k in range(steps):
        u_opt, status = mpc_ctrl.compute_control(curr_x)
        u_hist.append(u_opt)

        # Apply u to system (simulate one step)
        # x_next = Ad * x + Bd * u
        curr_x = mpc_ctrl.A @ curr_x + mpc_ctrl.B @ u_opt
        x_hist.append(curr_x)
        t_hist.append((k+1)*dt)

    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_hist, x_hist[:, 0], label='x1')
    plt.plot(t_hist, x_hist[:, 1], label='x2')
    plt.ylabel('States')
    plt.legend()
    plt.grid(True)
    plt.title("MPC Simulation")

    plt.subplot(2, 1, 2)
    plt.step(t_hist[:-1], u_hist[:, 0], label='u1')
    plt.step(t_hist[:-1], u_hist[:, 1], label='u2')
    plt.ylabel('Inputs')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.savefig("ControlX/images/mpc_simulation.png")
    plt.close()
    print("Saved mpc_simulation.png")

    # 5. Robustness Analysis
    # Sensitivity Function S = (I + L)^-1
    # Check Sensitivity Peak
    S = robustness.sensitivity_function(sys, K_lqg)

    # Plot Sensitivity Singular Value
    # ⚡ Bolt Optimization: Use vectorized calculate_singular_values instead of slow Python loop
    s_sv_all = analysis.calculate_singular_values(S, omega)
    if s_sv_all.ndim > 1:
        s_sv_list = np.max(s_sv_all, axis=1)
    else:
        s_sv_list = s_sv_all

    plt.figure()
    plt.loglog(omega, s_sv_list)
    plt.title("Sensitivity Singular Value Plot")
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel("Maximum Singular Value of S")
    plt.grid(True)
    plt.savefig("ControlX/images/sensitivity_plot.png")
    plt.close()
    print("Saved sensitivity_plot.png")

    margin = robustness.robust_stability_margin(S)
    print(f"Robust Stability Margin: {margin}")

if __name__ == "__main__":
    run_demo()
