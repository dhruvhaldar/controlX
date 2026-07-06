import cProfile
import pstats
from src.analysis import system_gain
import control as ct
import numpy as np

# Create a sample system
sys = ct.rss(100, 10, 10)

def run():
    for _ in range(100):
        # We want to trigger the complex initialization code path
        system_gain(sys, 10.0)

cProfile.run('run()', 'stats13')
p = pstats.Stats('stats13')
p.strip_dirs().sort_stats('tottime').print_stats(10)
