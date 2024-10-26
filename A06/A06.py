import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings

srv = 1.0
arr = 0.8

M = 5000
K0 = 50
DeltaK = 10
MaxK = 1000

maxRelErr = 0.02

U1 = 0
U2 = 0
X = 0
W = 0
L = 0

K = K0
Krange = K

lambda1Hyper = 0.025
lambda2Hyper = 0.1
p1Hyper = 0.35

kWeibull = 0.333
lambdaWeibull = 2.5

while K < MaxK:
    for k in range(0, Krange):
        u = np.random.rand(M,1)
        u2 = np.random.rand(M,1)
        Xarr = np.where(u < p1Hyper, -np.log(u) / lambda1Hyper, -np.log(u2) / lambda2Hyper)

        u = np.random.rand(M,1)
        Xsrv = lambdaWeibull * (-np.log(u)) ** (1 / kWeibull)

        A = np.zeros((M, 1))
        C = np.zeros((M, 1))

        A[0,0] = Xarr[0,0]
        C[0,0] = Xarr[0,0] + Xsrv[0,0]

        for i in range(1, M):
            A[i,0] = A[i-1,0] + Xarr[i,0]
            C[i,0] = max(A[i,0], C[i-1,0]) + Xsrv[i,0]
        
        T = C[M-1,0]
        B = np.sum(Xsrv)
        Uk = B / T
        Xk = M / T
        R = np.sum(C[:, 0] - Xsrv[:, 0])
        Wk = R / M
        Lk = Xk * Wk

        U1 = U1 + Uk
        U2 = U2 + Uk*Uk
        X = X + Xk
        W = W + Wk
        L = L + Lk

    EU     = U1 / K # utilization
    EU2    = U2 / K
    VarU   = EU2 - EU*EU
    SigmaU = np.sqrt(VarU)
    DeltaU  = 1.96 * np.sqrt(VarU / K)
    Ul = EU - DeltaU
    Uu = EU + DeltaU
    RelErrU = 2 * (Uu - Ul) / (Uu + Ul)

    if RelErrU < maxRelErr:
        break

    K = K + DeltaK
    Krange = DeltaK

#print("E[U]     = ", EU)
#print("E[U^2]   = ", EU2)
#print("Var[U]   = ", VarU)
#print("Sigma[U] = ", SigmaU)
print("95% confidence interval of U:    ", Ul, Uu)
print("Relative error of U:             ", RelErrU)
print("Solution obtained in             ", K, " iterations")

print("Utilization:             ", EU)
print("Throughput:              ", X)
print("Response time:           ", W)
print("Number of customers:     ", L)

