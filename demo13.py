import cProfile
import pstats
from src.analysis import system_gain
import control as ct
import numpy as np

# Create a sample system
sys = ct.rss(100, 10, 10)
omega_val = 10.0
s = omega_val * 1j

def run():
    for _ in range(1000):
        sI_minus_A = np.empty_like(sys.A, dtype=complex)
        sI_minus_A[...] = -sys.A
        sI_minus_A.flat[::sys.nstates + 1] += s
        res = sys.C @ np.linalg.solve(sI_minus_A, sys.B) + sys.D

def run2():
    for _ in range(1000):
        # We can optimize out the C @ inv(sI - A) @ B + D
        # Using LU or Cho if it was symmetric, but it's not.

        # What if we use np.linalg.inv?
        sI_minus_A = np.empty_like(sys.A, dtype=complex)
        sI_minus_A[...] = -sys.A
        sI_minus_A.flat[::sys.nstates + 1] += s
        inv_A = np.linalg.inv(sI_minus_A)
        res = sys.C @ inv_A @ sys.B + sys.D

cProfile.run('run()', 'stats14')
p = pstats.Stats('stats14')
p.strip_dirs().sort_stats('tottime').print_stats(10)

cProfile.run('run2()', 'stats15')
p = pstats.Stats('stats15')
p.strip_dirs().sort_stats('tottime').print_stats(10)
