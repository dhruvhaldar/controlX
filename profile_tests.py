import cProfile
import pstats
import pytest

cProfile.run('pytest.main(["tests/"])', 'test_stats')
p = pstats.Stats('test_stats')
p.strip_dirs().sort_stats('tottime').print_stats(30)
