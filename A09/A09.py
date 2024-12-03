import numpy as np
import math

D = 1.6     # avg service time
I = 2       # avg interrival time


# ******************************** M/M/1 ********************************
print("FIRST YEAR: ")
lam = 1 / I
mu = 1 / D

rho = lam / mu

pi = np.zeros(11)
pNgei = np.zeros(11)

for i in range(0, 11):
    pi[i] = (1 - rho) * (rho ** i)
    pNgei[i] = (rho ** (i+1)) + pi[i]

U = rho
print("U                    = ", U)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", 1 - pNgei[5])
Nq = (rho ** 2) / (1 - rho)
print("Avg N in queue       = ", Nq)
R = 1 / (mu - lam)
print("Avg Response time    = ", R)
print("P(R > 2)             = ", np.exp(-2 / R))
percentile95 = -np.log(1 - 95/100) * R
print("95th percentile      = ", percentile95)

print("*" * 50)



# ******************************** M/M/2 ********************************
print("SECOND YEAR: ")

lam = 1

pi = np.zeros(11)
pNlti = np.zeros(11)

rho = lam / (2*mu)

pi[0] = (2*mu - lam) / (2*mu + lam)
pNlti[0] = 0

for i in range(1, 11):
    pi[i] = 2 * pi[0] * (rho ** i)
    pNlti[i] = pNlti[i-1] + pi[i-1]

U = 2 * rho
print("U                    = ", U)
Ubar = rho
print("Ubar                 = ", Ubar)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", pNlti[5])
N = (2 * rho) / (1 - (rho ** 2))
Nq = N - U
print("Avg N in queue       = ", Nq)
R = D / (1 - (rho ** 2))
print("Avg Response time    = ", R)

print("*"*50)


# ******************************** M/M/c ********************************
print("THIRD YEAR")

lam = 4

c = 1 # num of servers
cMax = 10

pi = np.zeros(11)
pNlti = np.zeros(11)

while (c < cMax):
    # debugging
    #print("try c = ", c)

    rho = lam * D / c

    cfact = 1
    for i in range(1, c+1):
        cfact *= i

    pi[0] = (c * rho) ** c / cfact * 1 / (1 - rho)
    kfact = 1

    for k in range(0, c):
        if k > 0:
            kfact *= k
        pi[0] += (c * rho) ** k / kfact
    pi[0] = 1 / pi[0]
    pNlti[0] = 0

    for n in range(1, 11):
        if n < c:
            pi[n] = pi[n-1] * (c * rho) / n
        else:
            pi[n] = pi[n-1] * rho
        pNlti[n] = pNlti[n-1] + pi[n-1]
    
    if pi[0] > 0:
        print("FOUND C = ", c)
        break

    c += 1

U = c * rho
print("U                    = ", U)
Ubar = rho
print("Ubar                 = ", Ubar)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", pNlti[5])

NdenSum = 0
kfact = 1
for k in range(0, c):
    if k > 0:
        kfact *= k
    NdenSum += (c * rho) ** k / kfact
Nq = (rho / (1 - rho)) / (1 + (1 - rho) * (cfact / ((c * rho) ** c)) * NdenSum)
print("Avg N in queue       = ", Nq)

R = D + (D / (c * (1 - rho))) / (1 + (1 - rho) * (cfact / ((c * rho) ** c)) * NdenSum)
print("Avg Response time    = ", R)

print("*"*50)


# ******************************** M/M/inf ********************************
print("FOUTH YEAR")

lam = 10

pi = np.zeros(11)
pNlti = np.zeros(11)
ifact = 1
e = np.exp(1)

rho = lam / mu

pi[0] = e ** -rho
pNlti[0] = 0

for i in range(1, 11):
    ifact *= i
    pi[i] = (e ** -rho) * (rho ** i) / ifact
    pNlti[i] = pNlti[i-1] + pi[i-1]

U = rho
print("U                    = ", U)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", pNlti[5])
R = D
print("Avg Response time    = ", R)