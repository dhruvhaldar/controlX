import numpy as np
import time

F = 10000
N = 100
s = 1j * np.random.randn(F)
eigvals = np.random.randn(N) + 1j * np.random.randn(N)

t0 = time.time()
for _ in range(100):
    s_minus_eig = s[:, np.newaxis] - eigvals
    np.reciprocal(s_minus_eig, out=s_minus_eig)
t1 = time.time()

t2 = time.time()
for _ in range(100):
    s_minus_eig2 = np.empty((F, N), dtype=complex)
    s_minus_eig2[...] = -eigvals
    s_minus_eig2 += s[:, np.newaxis]
    np.reciprocal(s_minus_eig2, out=s_minus_eig2)
t3 = time.time()

print(f"Direct sub time: {t1-t0:.4f}")
print(f"In-place add time: {t3-t2:.4f}")
