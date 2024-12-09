import numpy as np

import math

########################################## FIRST YEAR #############################################################
_lambda_ = 20  # λ = 20 jobs/second

print("-"* 20)
print("FIRST YEAR:")

#General parameters for the erlang dist

lambdaErlang= 100  # λe = 100
kErlang = 4              # Shape parameter

#Compute average service time D
D = kErlang / lambdaErlang # D = E[X_G]

varErlang = kErlang / (lambdaErlang ** 2) #Variance(Var[X_G])
cv2E = varErlang / (D ** 2) # squared coefficient of variation

# Second moment m2
second_moment = (kErlang / (lambdaErlang ** 2)) * (kErlang + 1)  # m2 = k / λe^2 * (k + 1)

#Service rate
mu = 1 / D # μ = λe / k
#Calculate traffic intensity
rho = _lambda_ / mu

#Compute utilization
utilization = _lambda_ / mu  # ρ = λ / μ

if utilization >= 1:
    raise ValueError("System is unstable (utilization >= 1).")

#Compute the average response time with M/G/1
response_time = D + ((_lambda_ * second_moment) / (2 * (1 - rho)))

#Compute the average number of jobs in the system
#
average_jobs = rho + ((_lambda_ * _lambda_ * second_moment) / (2 * (1 - rho)))  # L = λ * R

# Output results
print(f"Utilization of the system: {utilization:.4f}")
print(f"Average response time: {response_time:.4f} seconds")
print(f"Average number of jobs in the system: {average_jobs:.4f}")
print("-"* 20)


########################################## SECOND YEAR #############################################################

print("SECOND YEAR:")

#Starting number of servers
c = 1

utilization2 = _lambda_ * D / c

#Parameters for Hyper-Exp dist
lam1Hyper = 40
lam2Hyper = 240
p1Hyper = 0.8

# 
T = (p1Hyper / lam1Hyper) + ((1 - p1Hyper) / lam2Hyper)

#New arrival rate
lam = 1 / T

#again variance and squared coefficient
varHyper = (2 * p1Hyper / (lam1Hyper ** 2)) + (2 * (1 - p1Hyper) / (lam2Hyper ** 2)) - (T ** 2)
cv2H = varHyper / (T ** 2)

#cycle to find the right amount of servers c
while True:
    utilization2 = lam * D / c
    if utilization2 < 1:
        break
    c += 1

print("Min #servers c:", c)
print("Utilization:", utilization2)

# compute theta 
cfact = 1
for i in range(1, c + 1):
    cfact *= i

rho = lam * D / c

sumTerms = 1
kfact = 1
for k in range(1, c):
    kfact *= k
    sumTerms += ((c * rho) ** k) / kfact

theta = (D / (c * (1 - rho))) / (1 + (1 - rho) * (cfact / ((c * rho) ** c)) * sumTerms)

#Avg resposne times
R = D + ((cv2E + cv2H) / 2) * theta
print("Avg Response time [ms]:", R * 1000)

#Avg number of jobs
N = R * lam
print("Avg #jobs in system:", N)
print("-"* 20)
