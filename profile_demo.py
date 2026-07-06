import cProfile
import pstats
import demo

cProfile.run('demo.run_demo()', 'profile_stats')
p = pstats.Stats('profile_stats')
p.strip_dirs().sort_stats('tottime').print_stats(30)
