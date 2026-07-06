import cProfile
import pstats
from src.analysis import system_gain, calculate_singular_values
from src.synthesis import design_lqr
from src.robustness import sensitivity_function
import control as ct
import numpy as np

# Create a sample system
G = ct.rss(4, 2, 2)
K = ct.rss(4, 2, 2)
Q = np.eye(4)
R = np.eye(2)

def run():
    for _ in range(50):
        # system_gain(G, 10.0)
        # calculate_singular_values(G, np.linspace(0, 100, 1000))
        # design_lqr(G, Q, R)
        sensitivity_function(G, K)

cProfile.run('run()', 'stats2')
p = pstats.Stats('stats2')
p.strip_dirs().sort_stats('tottime').print_stats(30)
