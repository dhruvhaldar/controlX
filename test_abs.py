import numpy as np
import time

N = 1000000
arr = np.random.randn(N) + 1j * np.random.randn(N)

t0 = time.time()
for _ in range(100):
    np.abs(arr)**2
t1 = time.time()

t2 = time.time()
for _ in range(100):
    arr.real**2 + arr.imag**2
t3 = time.time()

print(f"abs(arr)**2 time: {t1-t0:.4f}")
print(f"arr.real**2 + arr.imag**2 time: {t3-t2:.4f}")
