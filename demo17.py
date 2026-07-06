import cProfile
import pstats
import control as ct
import numpy as np
import scipy.linalg

# Create a sample system
G = ct.rss(10, 3, 3)

def run():
    for _ in range(500):
        # We know that for Cholesky solving we can use choose_factor / cho_solve
        # Let's verify that scipy.linalg.cho_factor doesn't throw a LinAlgError
        pass

cProfile.run('run()', 'stats21')
p = pstats.Stats('stats21')
p.strip_dirs().sort_stats('tottime').print_stats(10)
