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

def run():
    for _ in range(50):
        # Fallback path trigger
        G.A = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float)
        calculate_singular_values(G, np.linspace(0, 100, 1000))

cProfile.run('run()', 'stats4')
p = pstats.Stats('stats4')
p.strip_dirs().sort_stats('tottime').print_stats(30)
