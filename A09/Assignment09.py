import numpy as np
import math

# Define constants for service time and interarrival time
D = 1.6     # Average service time
I = 2       # Average interarrival time

# ******************************** M/M/1 ********************************
print("FIRST YEAR: ")

# Calculate arrival rate (lambda) and service rate (mu)
_lambda_ = 1 / I
mu = 1 / D

# Compute traffic intensity (utilization factor rho)
rho = _lambda_ / mu

# Initialize probabilities
pi = np.zeros(11)  # P(N = i)
pNgei = np.zeros(11)  # P(N >= i)

# Compute state probabilities and cumulative probabilities
for i in range(0, 11):
    pi[i] = (1 - rho) * (rho ** i)
    pNgei[i] = (rho ** (i + 1)) + pi[i]

#Metrics
U = rho
print("U                    = ", U)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", 1 - pNgei[5])
Nq = (rho ** 2) / (1 - rho)
print("Avg N in queue       = ", Nq)
R = 1 / (mu - _lambda_)
print("Avg Response time    = ", R)
print("P(R > 2)             = ", np.exp(-2 / R))
percentile95 = -np.log(1 - 95 / 100) * R
print("95th percentile      = ", percentile95)

print("*" * 50)

# ******************************** M/M/2 ********************************
print("SECOND YEAR: ")

_lambda_ = 1  # Update arrival rate

# Compute new traffic intensity for M/M/2
rho = _lambda_ / (2 * mu)

# Initialize again probabilities
pi = np.zeros(11)
pNlti = np.zeros(11)

pi[0] = (2 * mu - _lambda_) / (2 * mu + _lambda_)
pNlti[0] = 0

for i in range(1, 11):
    pi[i] = 2 * pi[0] * (rho ** i)
    pNlti[i] = pNlti[i - 1] + pi[i - 1]

# Calculate metrics for M/M/2
U = 2 * rho
print("U                    = ", U)
#Utilization per server
Ubar = rho
print("Ubar                 = ", Ubar)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", pNlti[5])
N = (2 * rho) / (1 - (rho ** 2))
Nq = N - U
print("Avg N in queue       = ", Nq)
R = D / (1 - (rho ** 2))
print("Avg Response time    = ", R)

print("*" * 50)

# ******************************** M/M/c ********************************
print("THIRD YEAR")

_lambda_ = 4  # Update arrival rate
c = 1  # Initial number of servers
cMax = 10  # Maximum number of servers to test

#Initialize probabilities for the third time
pi = np.zeros(11)
pNlti = np.zeros(11)

#Find the optimal number of servers
while c < cMax:
    
    # Calculate traffic intensity per server
    rho = _lambda_ * D / c

    # Calculate factorial of the number of servers for normalization
    cfact = 1
    for i in range(1, c + 1):
        cfact *= i

    # Compute the probability of zero entities in the system (P(N = 0))
    pi[0] = (c * rho) ** c / cfact * 1 / (1 - rho)
    kfact = 1  # Initialize factorial for other states

    # Sum probabilities for states with fewer than `c` servers busy
    for k in range(0, c):
        if k > 0:
            kfact *= k 
        pi[0] += (c * rho) ** k / kfact

    # Normalize to ensure probabilities sum to 1
    pi[0] = 1 / pi[0]
    pNlti[0] = 0 

    # Calculate probabilities for states N = 1 to N = 10
    for n in range(1, 11):
        if n < c:
            
            pi[n] = pi[n - 1] * (c * rho) / n
        else:
            
            pi[n] = pi[n - 1] * rho
        # Update cumulative probability P(N < i)
        pNlti[n] = pNlti[n - 1] + pi[n-1]

    # Check if P(N = 0) is valid and exit if found
    if pi[0] > 0:
        print("FOUND C = ", c)
        break

    # Increment the number of servers for the next iteration
    c += 1
    
U = c * rho
print("U                    = ", U)
Ubar = rho
print("Ubar                 = ", Ubar)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", pNlti[5])

# Initialize summation variable for the denominator of the queue length formula
NdenSum = 0
kfact = 1

# Compute the summation term for 0 <= k < c
for k in range(0, c):
    if k > 0:
        kfact *= k
    NdenSum += (c * rho) ** k / kfact
Nq = (rho / (1 - rho)) / (1 + (1 - rho) * (cfact / ((c * rho) ** c)) * NdenSum)
print("Avg N in queue       = ", Nq)

R = D + (D / (c * (1 - rho))) / (1 + (1 - rho) * (cfact / ((c * rho) ** c)) * NdenSum)
print("Avg Response time    = ", R)

print("*" * 50)

# ******************************** M/M/inf ********************************
print("FOURTH YEAR")

_lambda_ = 10  # Update arrival rate

rho = _lambda_ / mu

pi = np.zeros(11)
pNlti = np.zeros(11)
ifact = 1
e = np.exp(1)

pi[0] = e ** -rho
pNlti[0] = 0

for i in range(1, 11):
    ifact *= i
    pi[i] = (e ** -rho) * (rho ** i) / ifact
    pNlti[i] = pNlti[i - 1] + pi[i - 1]

U = rho
print("U                    = ", U)
print("P(N = 2)             = ", pi[2])
print("P(N < 5)             = ", pNlti[5])
R = D
print("Avg Response time    = ", R)
