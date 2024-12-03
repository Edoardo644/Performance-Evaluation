import numpy as np

lam = 240 / 60 # 240 req / min
D = 200 / 1000 # 200 ms

mu = 1 / D


# ******************************** M/M/1/K ********************************
print("FIRST YEAR: ")
K = 16

rho = lam / mu

pi = np.zeros(K+1)

pi[0] = (1 - rho) / (1 - rho ** (K+1))

for i in range(1, K+1):
    pi[i] = pi[0] * (rho ** i)

U = 1 - pi[0]
print("U                    = ", U)
pL = pi[K]
print("Probability of loss  = ", pL)
N = rho / (1 - rho) - ((K+1) * (rho ** (K+1))) / (1 - rho ** (K+1))
print("Avg #jobs in system  = ", N)
Dr = lam * pi[K] * 60 # req / min
print("Drop rate [req/min]  = ", Dr)
R = N / (lam * (1 - pi[K]))
print("Avg Response time    = ", R)
Th = R - D
print("Time in queue        = ", Th)

print("*" * 50)
# ******************************** M/M/2/K ********************************
print("SECOND YEAR: ")

lam = 360 / 60 # 360 req / min

rho = lam / (2 * mu)

pi = np.zeros(K+1)
ifact = 1

pi[0] = np.pow(((2 * rho) ** 2) / 2 * (1 - rho ** (K - 2 + 1)) / (1 - rho) + 1 + (2 * rho), -1)

for i in range(1, K+1):
    ifact *= i
    if (i < 2):
        pi[i] = pi[0] / ifact * ((lam / mu) ** i)
    else:
        pi[i] = pi[0] / (2 * (2 ** (i - 2))) * ((lam / mu) ** i)

U = 0
for i in range(1, K+1):
    if (i < 2):
        U += i * pi[i]
    else:
        U += 2 * pi[i]
print("U                    = ", U)
Ubar = U / 2
print("Ubar                 = ", Ubar)
pL = pi[K]
print("Probability of loss  = ", pL)
N = 0
for i in range(1, K+1):
    N += i * pi[i]
print("Avg #jobs in system  = ", N)
Dr = lam * pi[K] * 60 # req / min
print("Drop rate [req/min]  = ", Dr)
R = N / (lam * (1 - pi[K]))
print("Avg Response time    = ", R)
Th = R - D
print("Time in queue        = ", Th)

print("*" * 50)
# ******************************** M/M/c/K ********************************
print("THIRD YEAR: ")

lam = 960 / 60 # 960 req / min
c = 3
pLmax = 0.01
cMax = 10

pi = np.zeros(K+1)

while (c <= cMax):
    print("c = ", c)
    rho = lam / (c * mu)

    cfact = 1
    for i in range(1, c+1):
        cfact *= i

    pi[0] = ((c * rho) ** c) / cfact * (1 - rho ** (K - c + 1)) / (1 - rho) + 1
    kfact = 1
    for k in range (1, c):
        kfact *= k
        pi[0] += ((c * rho) ** k) / kfact
    pi[0] = pi[0] ** -1

    ifact = 1
    for i in range(1, K+1):
        if (i < c):
            ifact *= i
            pi[i] = pi[0] / ifact * ((lam / mu) ** i)
        else:
            pi[i] = pi[0] / (cfact * (c ** (i - c))) * ((lam / mu) ** i)
    
    if (pi[K] < pLmax):
        print("Found c = ", c)
        break
    c += 1

U = 0
for i in range(1, K+1):
    if (i < c):
        U += i * pi[i]
    else:
        U += c * pi[i]
print("U                    = ", U)
Ubar = U / c
print("Ubar                 = ", Ubar)
pL = pi[K]
print("Probability of loss  = ", pL)
N = 0
for i in range(1, K+1):
    N += i * pi[i]
print("Avg #jobs in system  = ", N)
Dr = lam * pi[K] * 60 # req / min
print("Drop rate [req/min]  = ", Dr)
R = N / (lam * (1 - pi[K]))
print("Avg Response time    = ", R)
Th = R - D
print("Time in queue        = ", Th)
    