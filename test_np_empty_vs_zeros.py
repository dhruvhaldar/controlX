import numpy as np
import time

F = 1000
N = 100

t0 = time.time()
for _ in range(100):
    sI_minus_A = np.empty((F, N, N), dtype=complex)
    # sI_minus_A[...] = 0 # Not setting it to anything yet, let's say we have A
t1 = time.time()

print(f"empty time: {t1-t0:.4f}")

# test block matrix
def run1():
    A = np.random.randn(50, 50)
    B = np.random.randn(50, 10)
    C = np.random.randn(20, 50)
    D = np.random.randn(20, 10)

    n, m = 50, 10
    M1 = np.empty((n + m, n + m))
    M1[:n, :n] = A
    M1[:n, n:] = B
    M1[n:, :n] = C
    M1[n:, n:] = D

def run2():
    A = np.random.randn(50, 50)
    B = np.random.randn(50, 10)
    C = np.random.randn(20, 50)
    D = np.random.randn(20, 10)

    M1 = np.block([[A, B], [C, D]])

t2 = time.time()
for _ in range(1000):
    run1()
t3 = time.time()

t4 = time.time()
for _ in range(1000):
    run2()
t5 = time.time()
print(f"slice block time: {t3-t2:.4f}")
print(f"np.block time: {t5-t4:.4f}")
