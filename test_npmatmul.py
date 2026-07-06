import numpy as np
import time

A = np.random.randn(1000, 4, 4) + 1j * np.random.randn(1000, 4, 4)
B = np.random.randn(1000, 4, 4) + 1j * np.random.randn(1000, 4, 4)

t0 = time.time()
for _ in range(1000):
    C1 = A @ B
t1 = time.time()

print(f"matmul time: {t1-t0:.4f}")
