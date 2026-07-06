import cProfile
import pstats
from src.analysis import relative_gain_array
import control as ct
import numpy as np

# Create a sample system
G = np.random.randn(200, 200)

def run():
    for _ in range(100):
        # ⚡ Bolt Optimization: Fast computation of (G^-1)^T.
        # RGA = G .* (G^-1)^T. np.linalg.inv is faster than np.linalg.solve for computing the full inverse.
        RGA = G * np.linalg.inv(G).T

def run2():
    for _ in range(100):
        # Let's try np.linalg.solve to see if it's faster
        RGA = G * np.linalg.solve(G.T, np.eye(G.shape[0]))

cProfile.run('run()', 'stats16')
p = pstats.Stats('stats16')
p.strip_dirs().sort_stats('tottime').print_stats(10)

cProfile.run('run2()', 'stats17')
p = pstats.Stats('stats17')
p.strip_dirs().sort_stats('tottime').print_stats(10)
