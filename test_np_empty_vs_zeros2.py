import numpy as np
import time

F = 1000
N = 100
s = 1j * np.random.randn(F)
A = np.random.randn(N, N)

def method1():
    sI_minus_A = np.empty((F, N, N), dtype=complex)
    sI_minus_A[...] = -A
    sI_minus_A[:, np.arange(N), np.arange(N)] += s[:, np.newaxis]
    return sI_minus_A

def method2():
    sI_minus_A = np.empty((F, N, N), dtype=complex)
    sI_minus_A[...] = -A
    sI_minus_A.reshape(F, -1)[:, ::N+1] += s[:, np.newaxis]
    return sI_minus_A

t0 = time.time()
for _ in range(100):
    method1()
t1 = time.time()

t2 = time.time()
for _ in range(100):
    method2()
t3 = time.time()

print(f"fancy indexing time: {t1-t0:.4f}")
print(f"flat indexing time: {t3-t2:.4f}")
print(f"Equal: {np.allclose(method1(), method2())}")
