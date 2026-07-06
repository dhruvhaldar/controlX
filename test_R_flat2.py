import numpy as np
import time

O, I, N = 10, 10, 100
CV = np.random.randn(O, N) + 1j * np.random.randn(O, N)
invVB_T = np.random.randn(I, N) + 1j * np.random.randn(I, N)

def method1():
    R = CV[:, np.newaxis, :] * invVB_T[np.newaxis, :, :]
    return R.reshape(O * I, N)

def method2():
    return np.repeat(CV, I, axis=0) * np.tile(invVB_T, (O, 1))

t0 = time.time()
for _ in range(10000):
    method1()
t1 = time.time()

t2 = time.time()
for _ in range(10000):
    method2()
t3 = time.time()

print(f"broadcast reshape time: {t1-t0:.4f}")
print(f"repeat tile time: {t3-t2:.4f}")
print(f"Equal: {np.allclose(method1(), method2())}")
