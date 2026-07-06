import cProfile
import pstats
import control as ct
import numpy as np

# Create a sample system
G = ct.rss(10, 3, 3)

def run():
    for _ in range(500):
        # np.linalg.cholesky on semi-definite matrices needs + eps
        matrix = G.A @ G.A.T
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            eps_matrix = matrix.copy()
            eps_matrix.flat[::matrix.shape[0]+1] += 1e-9
            np.linalg.cholesky(eps_matrix)

def run2():
    for _ in range(500):
        matrix = G.A @ G.A.T
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            eps_matrix = matrix + np.eye(matrix.shape[0]) * 1e-9
            np.linalg.cholesky(eps_matrix)

cProfile.run('run()', 'stats18')
p = pstats.Stats('stats18')
p.strip_dirs().sort_stats('tottime').print_stats(10)

cProfile.run('run2()', 'stats19')
p = pstats.Stats('stats19')
p.strip_dirs().sort_stats('tottime').print_stats(10)
