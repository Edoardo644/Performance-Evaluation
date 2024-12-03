import numpy as np

_lambda_ = 240 / 60  # Arrival rate in req/sec 
D = 200 / 1000       # Average service time. 200 ms

mu = 1 / D  # Service rate in jobs/sec

# ******************************** M/M/1/K ********************************
print("FIRST YEAR: ")
K = 16  # System capacity 

rho = _lambda_ / mu  # Traffic intensity (utilization factor)

pi = np.zeros(K+1)  

# Calculate P(N=0) 
pi[0] = (1 - rho) / (1 - rho ** (K+1))

# Calculate P(N=i) for i from 1 to K
for i in range(1, K+1):
    pi[i] = pi[0] * (rho ** i)

U = 1 - pi[0]  # Utilization 
print("U                    = ", U)
pL = pi[K]  # Probability of loss 
print("Probability of loss  = ", pL)

# Average number of jobs in the system
N = rho / (1 - rho) - ((K+1) * (rho ** (K+1))) / (1 - rho ** (K+1))
print("Avg #jobs in system  = ", N)

# Drop rate in req/min (arrival rate * loss probability)
Dr = _lambda_ * pi[K] * 60
print("Drop rate [req/min]  = ", Dr)

# Average response time
R = N / (_lambda_ * (1 - pi[K]))
print("Avg Response time [ms]   = ", R * 1000)

# Average time spent in queue
Th = R - D
print("Time in queue [ms]       = ", Th * 1000)

print("*" * 50)

# ******************************** M/M/2/K ********************************
#
#Comments for re-used variables are specified in the first year
#
print("SECOND YEAR: ")

_lambda_ = 360 / 60  # Updated arrival rate (360 requests per minute)

rho = _lambda_ / (2 * mu) 

pi = np.zeros(K+1)  
ifact = 1  # Factorial for normalization calculations

pi[0] = np.pow(((2 * rho) ** 2) / 2 * (1 - rho ** (K - 2 + 1)) / (1 - rho) + 1 + (2 * rho), -1)

# Calculate P(N=i) for i from 1 to K
for i in range(1, K+1):
    ifact *= i  # Compute factorial 
    if (i < 2): 
        pi[i] = pi[0] / ifact * ((_lambda_ / mu) ** i)
    else:  
        pi[i] = pi[0] / (2 * (2 ** (i - 2))) * ((_lambda_ / mu) ** i)

# Calculate total utilization (U)
U_Tot = 0
for i in range(1, K+1):
    if (i < 2):  # Jobs handled by fewer than 2 servers
        U_Tot += i * pi[i]
    else:  # Jobs handled by both servers
        U_Tot += 2 * pi[i]
print("U_Tot                    = ", U_Tot)

# Average utilization per server
U_Avg = U_Tot / 2
print("U_Avg                 = ", U_Avg)

pL = pi[K]  
print("Probability of loss  = ", pL)


N = 0
for i in range(1, K+1):
    N += i * pi[i]
print("Avg #jobs in system  = ", N)


Dr = _lambda_ * pi[K] * 60
print("Drop rate [req/min]  = ", Dr)


R = N / (_lambda_ * (1 - pi[K]))
print("Avg Response time [ms]   = ", R * 1000)


Th = R - D
print("Time in queue [ms]       = ", Th * 1000)

print("*" * 50)

# ******************************** M/M/c/K ********************************
print("THIRD YEAR: ")

_lambda_ = 960 / 60  #Updated arrival rate (960 requests per minute)
c = 3  # Initial number of servers
pLmax = 0.01  # Max acceptable loss probability
cMax = 10  #Max number of servers t

pi = np.zeros(K+1)  # Array to store probabilities

# Find the optimal number of servers (c) 
while (c <= cMax):
    print("c = ", c)
    rho = _lambda_ / (c * mu)  

   
    cfact = 1
    for i in range(1, c+1):
        cfact *= i

     
    pi[0] = ((c * rho) ** c) / cfact * (1 - rho ** (K - c + 1)) / (1 - rho) + 1
    kfact = 1  # Initialize factorial f
    for k in range(1, c): 
        kfact *= k
        pi[0] += ((c * rho) ** k) / kfact
    pi[0] = pi[0] ** -1  # Normalize 

    # Calculate P(N=i) for i from 1 to K
    ifact = 1
    for i in range(1, K+1):
        if (i < c): 
            ifact *= i
            pi[i] = pi[0] / ifact * ((_lambda_ / mu) ** i)
        else:  
            pi[i] = pi[0] / (cfact * (c ** (i - c))) * ((_lambda_ / mu) ** i)

    # Check if the loss probability meets the requirement
    if (pi[K] < pLmax):
        print("Found c = ", c)
        break
    c += 1  # Increment server count and retry

# Calculate total utilization (U)
U = 0
for i in range(1, K+1):
    if (i < c):  
        U += i * pi[i]
    else: 
        U += c * pi[i]
print("U_Tot                    = ", U)

U_Avg = U / c
print("U_Avg                = ", U_Avg)

pL = pi[K] 
print("Probability of loss  = ", pL)


N = 0
for i in range(1, K+1):
    N += i * pi[i]
print("Avg #jobs in system  = ", N)


Dr = _lambda_ * pi[K] * 60
print("Drop rate [req/min]  = ", Dr)


R = N / (_lambda_ * (1 - pi[K]))
print("Avg Response time [ms]    = ", R * 1000)


Th = R - D
print("Time in queue [ms]       = ", Th * 1000)