import cProfile
import pstats
from src.analysis import system_gain, calculate_singular_values
from src.robustness import calculate_hinf_norm
import control as ct
import numpy as np

# Create a sample system
G = ct.rss(50, 20, 20)

def run():
    calculate_singular_values(G, np.linspace(0, 100, 1000))

cProfile.run('run()', 'stats5')
p = pstats.Stats('stats5')
p.strip_dirs().sort_stats('tottime').print_stats(30)
