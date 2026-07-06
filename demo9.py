import cProfile
import pstats
from src.synthesis import _validate_matrix
import numpy as np

M = np.random.randn(500, 500)
M = M @ M.T

def run():
    for _ in range(10):
        _validate_matrix(M)

cProfile.run('run()', 'stats9')
p = pstats.Stats('stats9')
p.strip_dirs().sort_stats('tottime').print_stats(30)
