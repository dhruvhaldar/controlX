import cProfile
import pstats
from src.synthesis import design_lqr, design_kalman_filter, design_lqg
import control as ct
import numpy as np

# Create a sample system
G = ct.rss(50, 10, 10)
Q = np.eye(50)
R = np.eye(10)
Qn = np.eye(10)
Rn = np.eye(10)

def run():
    for _ in range(50):
        # design_lqr(G, Q, R)
        design_kalman_filter(G, Qn, Rn)

cProfile.run('run()', 'stats8')
p = pstats.Stats('stats8')
p.strip_dirs().sort_stats('tottime').print_stats(30)
