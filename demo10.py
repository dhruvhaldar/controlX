import cProfile
import pstats
import numpy as np

M = np.random.randn(500, 500)

def run():
    for _ in range(100):
        # np.allclose is slow
        # Let's see if we can do better
        np.allclose(M, M.T)

def run2():
    for _ in range(100):
        # fast check
        np.max(np.abs(M - M.T)) < 1e-8

cProfile.run('run()', 'stats10')
p = pstats.Stats('stats10')
p.strip_dirs().sort_stats('tottime').print_stats(10)

cProfile.run('run2()', 'stats11')
p = pstats.Stats('stats11')
p.strip_dirs().sort_stats('tottime').print_stats(10)
