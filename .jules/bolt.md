## 2024-05-19 - Vectorize `calculate_singular_values` for Multiple Frequencies
**Learning:** `control.StateSpace.evalfr` is extremely slow when used in a `for` loop over many frequencies for computing singular values (such as creating Sigma plots). In Python, `control` can compute frequency responses natively for an array of frequencies using `sys.frequency_response(omega).complex`, which returns a `(outputs, inputs, frequencies)` array (for MIMO systems). This can be fed directly to a vectorized `numpy.linalg.svd` by transposing to `(frequencies, outputs, inputs)` for significant speedup over looping.
**Action:** When computing any metric over a frequency array, prefer `sys.frequency_response(omega)` and vectorized `numpy` operations instead of python loops with `evalfr`.

## 2025-03-28 - Vectorizing CVXPY Constraints
**Learning:** In MPC problems using CVXPY, Python loops for constraints (e.g. `for k in range(N): constraints += [x[:, k+1] == A@x[:, k] + B@u[:, k]]`) create a significant overhead during both compilation and solve time.
**Action:** Always vectorize state dynamics constraints using matrix slicing (e.g., `_x[:, 1:] == A @ _x[:, :-1] + B @ _u`) which provides ~3.5x faster problem setup and ~2.5x faster solve times.

## 2025-05-15 - Vectorizing CVXPY Cost Functions using sum_squares
**Learning:** Python loops over prediction horizons (e.g. `for k in range(N)`) to build CVXPY cost functions using `cp.quad_form(x, Q)` generate massive ASTs that dramatically slow down problem compilation and solve times. The cost $\sum x_k^T Q x_k$ is mathematically equivalent to $\sum ||Q^{1/2} x_k||_2^2$.
**Action:** Always vectorize quadratic costs by pre-computing matrix square roots (`scipy.linalg.sqrtm(Q).real`) and using `cp.sum_squares(Q_sqrt @ x)`. This speeds up problem setup from ~47ms to ~1ms and solve times from ~195ms to ~18ms.

## 2025-06-05 - Bypassing wrapper overhead for Algebraic Riccati Equations
**Learning:** The `control` library wrappers for solving Algebraic Riccati Equations (such as `ct.dare`, `ct.lqr`, `ct.lqe`) introduce significant overhead (taking ~15ms per call) because they instantiate internal `StateSpace` objects and perform validation checks. Directly calling the underlying `scipy.linalg` solvers like `scipy.linalg.solve_discrete_are` takes only ~1ms per call.
**Action:** When solving Algebraic Riccati Equations natively with matrices, prefer using `scipy.linalg.solve_discrete_are` or `scipy.linalg.solve_continuous_are` directly to achieve ~15x performance improvements.

## 2025-06-15 - Vectorizing Frequency Response for StateSpace Systems
**Learning:** `control.StateSpace.frequency_response` is slow for arrays of frequencies when the `slycot` dependency is missing, as it falls back to a slow Python loop evaluating polynomials via Horner's method.
**Action:** For performance-critical frequency evaluations of StateSpace systems, manually compute the frequency response using vectorized NumPy array operations `C @ np.linalg.inv(sI - A) @ B + D`. Be careful to handle continuous vs discrete time systems correctly, and fallback to `sys.frequency_response` if `np.linalg.inv` fails due to a `LinAlgError` (e.g., when a frequency exactly hits a pole).

## 2026-04-07 - Replace np.linalg.inv with np.linalg.solve
**Learning:** Using `np.linalg.inv(A) @ B` is slower and less numerically stable than `np.linalg.solve(A, B)`. For batched frequency response evaluation, broadcasting `B` and using `np.linalg.solve` gives a measurable performance improvement.
**Action:** When computing `C @ inv(sI - A) @ B + D`, always use `X = np.linalg.solve(sI - A, B)` and then `C @ X + D`.

## 2026-04-08 - Fast Frequency Response Evaluation via Spectral Decomposition
**Learning:** The batched matrix solve `np.linalg.solve(sI - A, B)` for frequency response is $O(N^3)$ per frequency point. When evaluating a state space model over thousands of frequencies (e.g. for singular value plots or H-infinity norms), this matrix inversion bottleneck severely limits performance.
**Action:** When evaluating frequency response across many points, first diagonalize `A` (`eigvals, V = np.linalg.eig(A)`). If `V` is well-conditioned (e.g., `np.linalg.cond(V) < 1e10`), replace the batched matrix solve with an $O(N)$ scalar division: scale `(C @ V)` and `(inv(V) @ B)` by `1.0 / (s - eigvals)`. This reduces the frequency-dependent logic to vector math, speeding up the calculation substantially (by ~2.5x to ~10x depending on matrix size).
