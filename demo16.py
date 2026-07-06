import cProfile
import pstats
import control as ct
import numpy as np

# Create a sample system
G = ct.rss(10, 3, 3)

def run():
    for _ in range(500):
        matrix = G.A @ G.A.T
        try:
            # this works for PSD
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            pass

cProfile.run('run()', 'stats20')
p = pstats.Stats('stats20')
p.strip_dirs().sort_stats('tottime').print_stats(10)
