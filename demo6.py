import cProfile
import pstats
from src.mpc import MPCController
import control as ct
import numpy as np

# Create a sample system
G = ct.rss(10, 3, 3)
Q = np.eye(10)
R = np.eye(3)
controller = MPCController(G, Q, R, N=20)

def run():
    for _ in range(50):
        controller.compute_control(np.random.randn(10))

cProfile.run('run()', 'stats6')
p = pstats.Stats('stats6')
p.strip_dirs().sort_stats('tottime').print_stats(30)
