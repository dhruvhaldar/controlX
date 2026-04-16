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

## 2025-10-24 - Fast Positive Semi-Definite Matrix Validation
**Learning:** Using `np.linalg.eigvalsh` to check if a matrix is positive semi-definite is an $O(4N^3/3)$ operation. In high-performance control loops or initialization (like MPC or LQG), this creates a significant validation overhead.
**Action:** Replace `eigvalsh` checks with a Cholesky decomposition (`np.linalg.cholesky(matrix + np.eye(N) * 1e-9)`). Cholesky decomposition is $O(N^3/3)$, providing a ~2-3x speedup for validating matrix positive semi-definiteness.

## 2026-11-20 - Fast Condition Number Validation
**Learning:** The default `np.linalg.cond(V)` computes the 2-norm condition number which requires a full Singular Value Decomposition (SVD), an $O(N^3)$ operation. In high-performance loops evaluating frequency responses, this condition check can become a bottleneck. Furthermore, `np.einsum` can replace broadcasting of intermediate arrays for final matrix summations to save time and memory.
**Action:** When validating matrix condition numbers for numerical stability (e.g., before diagonalization), always use `np.linalg.cond(V, 1)`, which computes the 1-norm much faster. Additionally, use `np.einsum('ok,fk,ki->foi', CV, inv_s_minus_eig, invVB)` instead of scaled intermediate array broadcasting.

## 2025-05-16 - Replace scipy.linalg.sqrtm with np.linalg.cholesky for CVXPY sum_squares
**Learning:** When formulating CVXPY costs using `cp.sum_squares(Q_sqrt @ x)`, computing the matrix square root `Q_sqrt` via `scipy.linalg.sqrtm` is slow (~0.59s for 100 iterations of a 100x100 matrix) because it uses the Schur decomposition. Since cost weighting matrices are positive semi-definite, `np.linalg.cholesky(Q).T` is mathematically equivalent for $x^T Q x = ||L^T x||_2^2$ and provides a massive speedup (~0.008s for 100 iterations).
**Action:** When vectorizing quadratic cost functions with `cp.sum_squares`, replace `scipy.linalg.sqrtm(Q).real` with the much faster `np.linalg.cholesky(Q + np.eye(Q.shape[0]) * 1e-9).T`.

## 2026-11-21 - Replace einsum with matmul and broadcasting for frequency response
**Learning:** While `np.einsum` is often suggested for tensor contractions to save memory and avoid broadcasting, it can be significantly slower than `np.matmul` (`@`) combined with NumPy broadcasting, especially for large arrays with complex numbers. In the frequency response evaluation where we need to compute `CV * inv_s_minus_eig @ invVB`, `np.einsum('ok,fk,ki->foi', ...)` is ~8.5x slower than `(CV * inv_s_minus_eig[:, np.newaxis, :]) @ invVB` for systems with 50 states.
**Action:** When computing batched matrix multiplications for frequency responses, prefer `np.matmul` with broadcasting over `np.einsum`.

## 2024-05-19 - Bypassing ct.feedback for StateSpace models
**Learning:** The `control.feedback` function introduces significant overhead (creating and validating objects) when computing sensitivity S = (I + L)^-1 and complementary sensitivity T = L(I + L)^-1 functions for `StateSpace` models.
**Action:** When computing sensitivity and complementary sensitivity for `StateSpace` models, directly calculate the resulting state space matrices using the algebraic formulas (e.g., A_s = A - B(I+D)^-1 C) to achieve a ~40% performance speedup.

## 2026-11-23 - Fast path for strictly proper systems in sensitivity operations
**Learning:** When computing algebraic relations for StateSpace models (like sensitivity S = (I + L)^-1 or complementary sensitivity T = L(I + L)^-1), evaluating `np.linalg.inv(I + D)` is completely unnecessary for strictly proper systems where D is a zero matrix. Creating the identity matrix, adding D, and running `np.linalg.inv` introduces a large constant overhead (~40% of the function time for typical small state spaces).
**Action:** When computing sensitivity matrix algebraic operations, always implement a fast path checking `if not np.any(L.D):` to bypass the identity matrix creation, the matrix addition, and the matrix inversion, substituting them with pre-calculated algebraic simplifications (e.g., `A_s = L.A - L.B @ L.C`). This provides an additional ~2x speedup on top of the algebraic formulas.

## 2026-11-23 - Replace np.linalg.inv with np.linalg.solve for symmetric matrices
**Learning:** Computing `L = P @ C.T @ np.linalg.inv(Rn)` explicitly calculates the inverse of `Rn`, which is slower and less numerically stable than solving a linear system. Since `Rn` is symmetric and positive semi-definite (being a covariance matrix), and `P` is symmetric, we can mathematically rearrange the equation to `L = np.linalg.solve(Rn.T, C @ P).T`.
**Action:** When calculating estimator gains or similar expressions involving a right-multiplied inverse `X = A B^-1`, rewrite it as a linear solve `X = np.linalg.solve(B.T, A.T).T`. This avoids the explicit inverse and provides a measurable speedup (~25% in benchmarks).

## 2026-11-24 - Avoiding dense identity matrices with np.eye for diagonal additions
**Learning:** Using `matrix + np.eye(N) * epsilon` to add a small value to the diagonal of a matrix creates a full dense identity matrix. For large matrices or high-frequency loops, this identity creation and matrix addition is measurably slower than copying the array and modifying its flat view `eps_matrix.flat[::N+1] += epsilon`.
**Action:** When adding small epsilon values to matrix diagonals (e.g., for numerical stability before Cholesky decompositions), avoid `np.eye`. Instead, copy the matrix and modify the flat diagonal elements directly using `.flat[::N+1] += epsilon`.
