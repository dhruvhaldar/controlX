import cProfile
import pstats
from src.analysis import system_gain, calculate_singular_values, relative_gain_array
import control as ct
import numpy as np

# Create a sample system
G = np.random.randn(200, 200)

def run():
    for _ in range(100):
        relative_gain_array(G)

cProfile.run('run()', 'stats7')
p = pstats.Stats('stats7')
p.strip_dirs().sort_stats('tottime').print_stats(30)
