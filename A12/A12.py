import numpy as np
from scipy import linalg

Sk = np.array([2, 0.02, 0.1, 0.07])
lambdaIN = np.zeros(4)
lambdaIN[0] = 2.5
lambdaIN[1] = 2

lam = np.sum(lambdaIN)

l = lambdaIN / lam

P = np.array([
    [0.0, 0.7, 0.0, 0.0],      # From self-check
    [0.0, 0.0, 0.25, 0.45],    # From application server
    [0.0, 1.0, 0.0, 0.0],      # From storage
    [0.0, 1.0, 0.0, 0.0]       # From DBMS
])

I = np.eye(4)

vk = linalg.solve((I - P).T, l)
print("Application server visits:       ", vk[1])
print("Storage visits:                  ", vk[2])
print("DBMS visits:                     ", vk[3])

X = lam
print("Throughput [req/s]:              ", X)

Dk = vk * Sk
Uk = X * Dk

Rk = Dk / (1 - Uk)
Rk[0] = Dk[0]                   # First is a delay center
R = np.sum(Rk)
print("Response time [s]:               ", R)

Nk = Uk / (1 - Uk)
Nk[0] = Uk[0]                   # First is a delay center
N = np.sum(Nk)
print("Avg #jobs in the system:         ", N)

maxLam = 1 / np.max(Dk)
print("Max throughput [req/s]:          ", maxLam)

