## 2024-05-18 - Input validation and exception leakage in MPCController
**Vulnerability:** Missing input validation on prediction horizon (N) and sampling time (dt) in `MPCController`. Unbounded `N` could lead to excessive memory/CPU allocation during optimization (Resource Exhaustion). In addition, exception messages were leaked to the console during terminal cost computation.
**Learning:** Simulation environments and scientific computing tools often lack input bounds, assuming the user will provide "reasonable" parameters. However, exposing such endpoints or allowing unbounded parameterization can lead to Denial of Service or resource exhaustion.
**Prevention:** Always bound matrix dimensions, iteration counts, and prediction horizons to sensible limits, and suppress raw exception details in user-facing warnings.
## 2024-03-25 - Prevent MPC Solver DoS via Input Validation
**Vulnerability:** Unvalidated user inputs passed directly to cvxpy solver parameters in the MPC controller.
**Learning:** Control algorithms utilizing underlying math optimization solvers (like OSQP via cvxpy) can crash violently or throw unhandled framework-level exceptions (e.g. `ValueError` inside cvxpy leaf expressions) when provided with NaNs, infinites, or mis-dimensioned arrays. This could lead to DoS or application crashes in production control systems.
**Prevention:** Always sanitize and validate state arrays (e.g. using `np.isfinite` and shape checking) at the boundary before feeding them to optimization solvers. Fail securely by returning a safe control vector (like zeros) instead of allowing an exception to propagate.
## 2025-02-18 - Prevent Downstream Crashes by Failing Securely
**Vulnerability:** Catching a critical numerical error (like `np.linalg.LinAlgError` during matrix inversion) and returning `None` instead of propagating the error or throwing a structured exception.
**Learning:** Returning `None` to indicate a mathematical failure in a library function is an anti-pattern that leads to "silent" data corruption or delayed, confusing `TypeError` exceptions deep in downstream code when the calling code attempts to perform operations (like matrix multiplication or slicing) on `None`. This violates the "fail securely" principle and makes debugging critical control system failures much harder.
**Prevention:** If a mathematical operation essential to the function's output fails (e.g., trying to invert a singular matrix), do not return a special type like `None`. Instead, explicitly raise an informative exception (like `ValueError` or `np.linalg.LinAlgError` with a clear message) so the caller is forced to handle the failure securely at the boundary.
## 2024-05-19 - Prevent Insecure Mathematical Operations in Control Synthesis
**Vulnerability:** The `design_lqr`, `design_kalman_filter`, and `design_lqg` functions permitted inputs of type `TransferFunction`. In `design_lqr`, the system was coerced into a `StateSpace` model using `ct.ss()`. This type coercion leads to a mathematically dangerous condition where user-provided state weight matrices (`Q`, `Qn`) are applied to an arbitrary state realization. This allows logical data corruption and produces unsafe controllers without any explicit warning to the user.
## 2026-04-06 - Prevent silent data corruption in weight matrices
**Vulnerability:** Missing validation for weight matrices in MPC and synthesis functions.
**Learning:** Invalid weight matrices lead to silent data corruption when calculating square roots.
**Prevention:** Validate weight and covariance matrices to ensure they are finite, square, and positive semi-definite before processing.
## 2025-02-18 - Prevent Insecure Mathematical Operations in MPCController
**Vulnerability:** The `MPCController` class permitted inputs of type `TransferFunction`. This type coercion leads to a mathematically dangerous condition where user-provided state weight matrices (`Q`, `R`) are applied to an arbitrary state realization. This allows logical data corruption and produces unsafe controllers without any explicit warning to the user.
**Learning:** Functions that require explicit structural knowledge (like physical state variables) must strictly validate object types. Coercing objects to simplify APIs (e.g. converting TFs to SS) in scientific contexts can destroy the mathematical meaning of parallel inputs (like weight matrices that specifically penalize distinct physical states).
**Prevention:** Strictly enforce `control.StateSpace` inputs in control synthesis and MPC functions that take state-dependent matrices. If an invalid type like `TransferFunction` is supplied, fail securely by explicitly raising a `TypeError` explaining that state weights cannot be applied to arbitrary realizations.
## 2024-05-28 - Prevent MPC Solver DoS via Constraint Validation
**Vulnerability:** Unvalidated constraint values passed directly to cvxpy solver parameters in the MPC controller.
**Learning:** Control algorithms utilizing underlying math optimization solvers can crash violently or throw unhandled framework-level exceptions when provided with NaNs, infinites, or mis-dimensioned arrays for constraints. This could lead to DoS or application crashes in production control systems.
**Prevention:** Always sanitize and validate constraint arrays (e.g., using `np.isfinite` and shape checking) at the boundary before passing them to the solver framework. Fail securely by raising an explicit `ValueError` early.
## 2025-02-18 - Prevent Unhandled TypeErrors for Invalid Inputs
**Vulnerability:** Core mathematical functions lacked structural type validation before array conversion.
**Learning:** Functions accepting covariance/weight matrices and attempting direct mathematical checks (`np.isfinite`) on them without confirming they are numeric can cause framework-level `TypeError` crashes for inputs like strings or mixed-type lists.
**Prevention:** Catch parsing exceptions (`ValueError`, `TypeError`) when validating generic multidimensional inputs and re-throw them as controlled `ValueError`s to fail securely.
## 2025-02-18 - Prevent Unhandled TypeErrors in Parameter Validation
**Vulnerability:** Core mathematical functions lacked structural type validation for scalar parameters (like `dt` in MPCController).
**Learning:** Functions accepting scalar values and performing direct mathematical checks (`dt <= 0`) on them without confirming they are numeric can cause framework-level `TypeError` crashes for invalid inputs like strings or `None`.
**Prevention:** Catch parsing exceptions (`ValueError`, `TypeError`) when validating parameters by explicitly coercing inputs to expected types (e.g., `float()`), and re-throw them as controlled `ValueError`s to fail securely.
## 2024-05-28 - Prevent Unhandled Exceptions from Invalid MPC Constraints
**Vulnerability:** The `MPCController` class lacked explicit type validation for its `constraints` values (e.g., `umin`, `umax`), leading to unhandled `TypeError` exceptions deep in framework functions like `np.array(..., dtype=float)` if non-numeric types were provided.
**Learning:** In mathematical APIs, directly passing unvalidated boundary data to underlying numerical frameworks can cause abrupt application crashes instead of securely returning controlled errors. This can act as a vector for application DoS or instability.
**Prevention:** Always wrap data type coercions (`np.array(..., dtype=float)`) in explicit `try...except (ValueError, TypeError)` blocks at the boundary before further matrix processing, ensuring failure securely via predictable exceptions like `ValueError`.
## 2025-10-24 - Prevent Framework Exceptions via Input Validation in Analysis Tools
**Vulnerability:** The `relative_gain_array` function lacked explicit type and structural validation for its gain matrix `G`, leading to unhandled `LinAlgError` or `TypeError` exceptions deep within the numpy framework if non-numeric types, strings, or scalars were provided.
**Learning:** Mathematical utility functions that perform matrix operations directly on user inputs must validate the structure and content type. Unhandled exceptions in the underlying numerical frameworks can crash the application instead of securely returning controlled errors.
**Prevention:** Always wrap data type coercions (`np.array(..., dtype=float)`) and structural checks in explicit `try...except` blocks at the function boundary, ensuring secure failure via predictable exceptions like `ValueError`.
## 2025-02-18 - Prevent Unhandled TypeErrors in Optional Matrix Validation
**Vulnerability:** The `design_kalman_filter` function failed to validate its optional parameter `G`, leading to unhandled `AttributeError` exceptions when non-matrix types were provided.
**Learning:** Optional matrix parameters must be explicitly validated before their properties (like `.shape`) are used to validate other inputs (like `Qn`). Assuming default properties without type checking can cause framework-level crashes for invalid user input.
**Prevention:** Always explicitly validate and cast optional matrices to numeric arrays (handling `ValueError`/`TypeError` safely) before accessing structural properties like `.shape` or passing them to solvers.
## 2025-02-18 - Prevent Framework Exceptions via Input Validation in Analysis Tools
**Vulnerability:** The `calculate_singular_values` and `system_gain` functions lacked explicit type and structural validation for frequency input `omega`, leading to unhandled `TypeError` or `UFuncNoLoopError` exceptions deep within the numpy calculations when computing with the complex plane variable `s` if strings or missing parameters were provided.
**Learning:** Mathematical utility functions that perform operations directly on user inputs must validate the structure and content type. Unhandled exceptions in the underlying numerical frameworks can crash the application instead of securely returning controlled errors.
**Prevention:** Always wrap data type coercions and structural checks in explicit `try...except` blocks at the function boundary, ensuring secure failure via predictable exceptions like `ValueError`.
## 2025-03-05 - Prevent Framework Exceptions via Input Validation in Robustness Tools
**Vulnerability:** The `calculate_hinf_norm` function lacked explicit type and structural validation for frequency input `omega`, leading to unhandled `TypeError` or `UFuncNoLoopError` exceptions deep within numpy when computing complex plane variables if strings were provided.
**Learning:** Mathematical utility functions that perform operations directly on user inputs must validate the structure and content type. Unhandled exceptions in underlying numerical frameworks can crash the application instead of securely returning controlled errors.
**Prevention:** Always wrap data type coercions and structural checks in explicit `try...except` blocks at the function boundary, ensuring secure failure via predictable exceptions like `ValueError`.
## 2025-02-18 - Prevent Framework Exceptions via Input Validation in Robustness Tools
**Vulnerability:** The `small_gain_theorem_check` function lacked explicit type and structural validation for system/gain inputs `M` and `Delta`, leading to unhandled `ValueError` (dimension mismatch) or `UFuncNoLoopError` exceptions deep within numpy when passing strings or invalid arrays.
**Learning:** Mathematical utility functions that perform operations directly on user inputs must validate the structure and content type. Unhandled exceptions in underlying numerical frameworks can crash the application instead of securely returning controlled errors.
**Prevention:** Always wrap data type coercions (`np.array(..., dtype=float)`) and structural checks in explicit `try...except` blocks at the function boundary, ensuring secure failure via predictable exceptions like `ValueError`.
## 2025-10-25 - Prevent Framework Exceptions via Input Validation in Analysis and Robustness Tools
**Vulnerability:** The `calculate_poles`, `calculate_zeros`, `calculate_singular_values`, `system_gain`, `sensitivity_function`, and `complementary_sensitivity_function` functions lacked explicit type validation for their system inputs (`sys`, `G`, `K`), leading to unhandled `AttributeError` or framework-level exceptions when non-system types like strings were provided.
**Learning:** Core functions in scientific computing libraries must strictly validate that inputs are of expected types (like `StateSpace` or `TransferFunction`). Relying on duck typing without validation can cause obscure and difficult-to-debug crashes deep within the framework when invalid input types are accessed.
**Prevention:** Always add explicit `isinstance` checks at the function boundary for critical input parameters expecting specific object types, and fail securely by raising a predictable `TypeError`.
## 2025-04-27 - Prevent DoS via unbounded MPC prediction horizon
**Vulnerability:** The `MPCController` allowed arbitrarily large prediction horizons `N` to be passed to the CVXPY solver.
**Learning:** CVXPY generates constraints and variables proportional to `N` and recompiles the problem structure in memory. An excessively large `N` (e.g. >1,000,000) leads to massive memory allocation and CPU consumption during `__init__`, causing the application to crash or freeze (OOM/DoS) before any solve is even attempted.
**Prevention:** Enforce a strict upper bound limit on problem size parameter `N` (e.g., 10000) at the API boundary, raising a `ValueError` for resource exhaustion.
## 2025-04-30 - Prevent DoS via unbounded MPC system dimensions
**Vulnerability:** The `MPCController` allowed arbitrarily large system dimensions (`nstates`, `ninputs`) to be passed to the CVXPY solver.
**Learning:** CVXPY generates constraints and variables proportional to system state and input sizes, and recompiles the problem structure in memory. Excessively large state or input sizes (e.g. >500) leads to massive memory allocation and CPU consumption during `__init__`, causing the application to crash or freeze (OOM/DoS) before any solve is even attempted.
**Prevention:** Enforce a strict upper bound limit on problem structural sizes such as `sys.nstates` and `sys.ninputs` (e.g., 500) at the API boundary, raising a `ValueError` for resource exhaustion.
## 2024-06-15 - Prevent Solver Hang via Time Limits
**Vulnerability:** The CVXPY OSQP solver in `MPCController` lacked a time limit, meaning a poorly conditioned state input could cause the solver to hang indefinitely (CPU DoS). Additionally, `dt` allowed `NaN`/`Inf`, leading to silent mathematical corruption.
**Learning:** Math optimization solvers must always be bounded by wall-clock time limits to prevent resource exhaustion, and all scalar inputs (like sampling time) must be explicitly checked for finiteness to prevent NaN propagation.
**Prevention:** Always set `time_limit` parameters when invoking numerical solvers, and use `np.isfinite` on all structural variables.

## 2025-05-15 - Prevent DoS via unbounded synthesis system dimensions
**Vulnerability:** The synthesis functions (`design_lqr`, `design_kalman_filter`, `design_lqg`) allowed arbitrarily large system dimensions (`nstates`, `ninputs`, `noutputs`) to be passed to the Riccati equation solvers.
**Learning:** Riccati solvers have $O(N^3)$ time complexity. Excessively large state or input/output sizes (e.g. >500) leads to massive CPU consumption and memory allocation, causing the application to crash or freeze (OOM/DoS).
**Prevention:** Enforce a strict upper bound limit on problem structural sizes such as `sys.nstates`, `sys.ninputs`, and `sys.noutputs` (e.g., 500) at the API boundary, raising a `ValueError` for resource exhaustion.
## 2025-05-15 - Prevent DoS via unbounded frequency array sizes
**Vulnerability:** The `calculate_singular_values` and `calculate_hinf_norm` functions allowed arbitrarily large frequency arrays (`omega`) or multi-dimensional arrays.
**Learning:** Vectorized frequency response evaluations allocate matrices proportional to the number of frequency points. An excessively large size (e.g. >10,000) or incorrect dimensions leads to massive memory allocation, causing the application to crash or freeze (OOM/DoS).
**Prevention:** Enforce a strict upper bound limit on input array sizes (e.g., 10000) and validate dimensions (1D) at the API boundary, raising a `ValueError` for resource exhaustion or invalid input.
## 2025-05-20 - Prevent Exception Leakage in Robustness Functions
**Vulnerability:** The `sensitivity_function` and `complementary_sensitivity_function` allowed raw `numpy.linalg.LinAlgError` exceptions to propagate to the user when encountering a singular matrix (algebraic loop).
**Learning:** In ControlX algebraic operations, when inverting matrices that depend on system properties (like `I + L.D`), failing to handle `np.linalg.LinAlgError` exposes underlying framework implementation details and generic error messages. This can be used to gather information about the internal workings of the library.
**Prevention:** Always wrap matrix inversion calls like `np.linalg.inv` in a `try...except np.linalg.LinAlgError` block and explicitly raise a `ValueError` with a clear, domain-specific explanation (e.g., "Algebraic loop detected"). This fails securely and prevents information leakage.

## 2025-05-25 - Prevent Exception Leakage in Riccati Solvers
**Vulnerability:** The synthesis functions (`design_lqr`, `design_kalman_filter`) allowed raw exceptions (`LinAlgError`, `ValueError`) to propagate to the user when encountering systems that cannot yield a valid Riccati solution (e.g., uncontrollable or unobservable systems).
**Learning:** In control synthesis operations, failing to handle exceptions from underlying solvers (like `scipy.linalg.solve_continuous_are`) exposes framework implementation details and stack traces. This leaks internal workings and fails insecurely.
**Prevention:** Always wrap Riccati solver calls in a `try...except` block catching numerical errors, and explicitly raise a `ValueError` with a clear, domain-specific explanation (e.g., "Failed to solve Riccati equation").
## 2025-10-26 - Prevent Framework Exceptions via Matrix Dimension Validation
**Vulnerability:** The `_validate_matrix` function failed to validate the `ndim` of matrix inputs, leading to an unhandled `LinAlgError` from `numpy.linalg.cholesky` when 3D arrays were provided, which resulted in a misleading error message and potential vulnerability in matrix operations down the line.
**Learning:** In control robustness operations, failing to explicitly validate matrix dimensionalities before passing them to internal mathematical functions (like Cholesky decompositions) can result in framework-level exception leakage.
**Prevention:** Always explicitly validate that matrix inputs are at most 2D (using `matrix.ndim <= 2`) and raise a domain-specific `ValueError` to fail securely.
## 2024-05-30 - Unvalidated Matrix Dimensionality Framework Exception Leakage
**Vulnerability:** Matrices with >2 dimensions bypass square matrix checks but trigger unhandled framework-level exceptions in downstream functions (like `np.linalg.cholesky`, `np.linalg.solve`, or `np.linalg.norm`).
**Learning:** In ControlX, always explicitly validate that matrix inputs are at most 2D (e.g., using `if matrix.ndim > 2: raise ValueError(...)`) before passing them to internal mathematical functions or framework methods.
**Prevention:** Explicitly bound matrix dimensionality to a maximum of 2D when validating inputs via `ndim` checks right after converting to array/atleast_2d.
## 2026-05-11 - Prevent DoS via unbounded analysis system dimensions
**Vulnerability:** The analysis functions (`calculate_poles`, `calculate_zeros`, `calculate_singular_values`, `system_gain`) allowed arbitrarily large system dimensions (`nstates`, `ninputs`, `noutputs`) to be processed.
**Learning:** Functions performing complex operations like eigenvalue decomposition (`calculate_poles`), generalized eigenvalue problems (`calculate_zeros`), or evaluating frequency responses over large arrays are highly sensitive to matrix dimensions. Providing excessively large state or input/output sizes leads to massive CPU consumption and memory allocation, causing the application to crash or freeze (OOM/DoS) due to the $O(N^3)$ computational complexity.
**Prevention:** Enforce a strict upper bound limit on problem structural sizes such as `sys.nstates`, `sys.ninputs`, and `sys.noutputs` (e.g., 500) at the API boundary, raising a `ValueError` for resource exhaustion.
## 2024-05-31 - Prevent Exception Leakage in Synthesis Gain Computation
**Vulnerability:** Unhandled `np.linalg.LinAlgError` during LQR/Kalman gain fallback calculations (e.g. `np.linalg.solve`) leaked internal numpy/scipy exceptions and stack traces when encountering singular matrices.
**Learning:** Fallback calculations must also be fully shielded with exception handling to prevent leaking of sensitive framework internals when unexpected invalid or singular input arrays bypass earlier logic.
**Prevention:** Always wrap terminal mathematical solving steps (like `np.linalg.solve`) in `try...except np.linalg.LinAlgError` and raise a domain-specific `ValueError`.
## 2024-06-25 - Prevent DoS via unbounded robustness system dimensions
**Vulnerability:** The robustness functions (`sensitivity_function`, `complementary_sensitivity_function`, `calculate_hinf_norm`) allowed arbitrarily large system dimensions (`nstates`, `ninputs`, `noutputs`) to be processed.
**Learning:** Functions performing complex operations like evaluating frequency responses or system gains over large arrays are highly sensitive to matrix dimensions. Providing excessively large state or input/output sizes leads to massive CPU consumption and memory allocation, causing the application to crash or freeze (OOM/DoS) due to computational complexity.
**Prevention:** Enforce a strict upper bound limit on problem structural sizes such as `sys.nstates`, `sys.ninputs`, and `sys.noutputs` (e.g., 500) at the API boundary, raising a `ValueError` for resource exhaustion.
## 2025-05-30 - Prevent DoS via unbounded matrix inputs in robustness functions
**Vulnerability:** The `small_gain_theorem_check` and `relative_gain_array` allowed arbitrarily large matrix inputs (e.g. 10000x10000) to be processed.
**Learning:** Functions performing complex operations like computing norms or solving systems over large arrays are highly sensitive to matrix dimensions. Providing excessively large inputs leads to massive CPU consumption and memory allocation, causing the application to crash or freeze (OOM/DoS) due to the $O(N^3)$ computational complexity.
**Prevention:** Enforce a strict upper bound limit on problem structural sizes such as matrix dimensions (e.g., 500) at the API boundary, raising a `ValueError` for resource exhaustion.

## 2025-05-31 - Prevent Exception Leakage in Robustness Functions via Non-Square Loop Matrices
**Vulnerability:** The `sensitivity_function` and `complementary_sensitivity_function` allowed raw framework-level exceptions (`ValueError` from `numpy.matmul` or `ControlDimension` from `python-control`) to leak when provided with systems that result in a non-square loop transfer matrix.
**Learning:** In ControlX robustness operations, failing to validate that the loop transfer matrix is square before passing it to internal mathematical functions exposes internal framework implementation details, which can be leveraged to gather information about the library's internal workings.
**Prevention:** Always explicitly validate that loop transfer matrices (`L = G * K`) are square (`L.noutputs == L.ninputs`) before passing them to internal mathematical functions or framework methods, and raise a secure, domain-specific `ValueError`.

## 2025-06-01 - Prevent Framework Exception Leakage via Dimension Validation in Robustness Operations
**Vulnerability:** The `sensitivity_function` and `complementary_sensitivity_function` lacked input dimension validation before performing system multiplication `G * K`, allowing `control.exception.ControlDimension` and `ValueError` to leak from `python-control` when given systems with incompatible input/output dimensions.
**Learning:** In ControlX robustness operations, passing incorrectly dimensioned state space or transfer function systems directly into mathematical operations (like `G * K`) can cause unhandled framework exceptions that expose underlying implementation details.
**Prevention:** Always validate that systems have compatible structural dimensions (e.g., `G.ninputs == K.noutputs` for series multiplication) at the API boundary, raising a secure, domain-specific `ValueError` before performing mathematical operations.
## 2025-10-27 - Prevent Exception Leakage in Pole/Zero Calculations
**Vulnerability:** The `calculate_poles` and `calculate_zeros` functions in `analysis.py` allowed raw framework-level exceptions (such as `np.linalg.LinAlgError` or exceptions from `python-control`) to leak when provided with systems containing invalid mathematical structures (e.g., matrices causing convergence failures).
**Learning:** In mathematical analysis tools, directly returning the result of complex numerical routines (like eigenvalue solvers) without exception handling exposes underlying library internals and stack traces to the user upon failure. This violates the principle of failing securely.
**Prevention:** Always wrap terminal numerical calculations like `np.linalg.eigvals` or `ct.zeros` in a `try...except` block, and explicitly raise a domain-specific `ValueError` (e.g., "Failed to calculate poles") to mask internal implementation details.
## 2026-05-12 - Prevent Framework Exception Leakage via Timebase Validation in Robustness Operations
**Vulnerability:** The `sensitivity_function` and `complementary_sensitivity_function` lacked input timebase (`dt`) validation before performing system multiplication `G * K`, allowing `ValueError` to leak from `python-control` when given systems with incompatible continuous and discrete timebases.
**Learning:** In ControlX robustness operations, passing state space or transfer function systems with conflicting timebases directly into mathematical operations (like `G * K`) can cause unhandled framework exceptions that expose underlying implementation details.
**Prevention:** Always validate that systems have compatible timebases (e.g., matching `dt`) at the API boundary, raising a secure, domain-specific `ValueError` before performing mathematical operations.
## 2025-10-28 - Prevent Framework Exception Leakage in ct.feedback Operations
**Vulnerability:** Unhandled exceptions from `ct.feedback` in `sensitivity_function` and `complementary_sensitivity_function` for `TransferFunction` models leak internal error details when an algebraic loop is encountered.
**Learning:** In ControlX robustness operations, using `python-control` library functions like `ct.feedback` can raise internal exceptions (`ValueError` about zero denominator or singular matrices) which expose framework implementation details.
**Prevention:** Wrap calls to `ct.feedback` in a `try...except Exception` block and raise a domain-specific `ValueError` (e.g., "Algebraic loop detected: System cannot be inverted.") to mask internal implementation details.
