import numpy as np
from scipy import linalg

# Input arrival rates (in parts/hour), converted to parts/minute
lambdaInA = np.array([1.5 / 60 , 0])
lambdaInB = np.array([2.5 / 60, 0])
lambdaInC = np.array([2.0 / 60, 0])

#lamInA = lamInA / 60   # Convert in seconds
#lamInB = lamInB / 60
#lamInC = lamInC / 60

# Total arrival rates per product class
lambdaA = np.sum(lambdaInA)
lambdaB = np.sum(lambdaInB)
lambdaC = np.sum(lambdaInC)

# Service times for each product type  [minutes]
SA = np.array([8, 10])
SB = np.array([3, 2])
SC = np.array([4, 7])

#SA = SA * 60           # Convert in seconds
#SB = SB * 60
#SC = SC * 60


#Transition matrices for each product class
transition_matrix_A = np.array([
    [0.0, 1],
    [0.10, 0.0]
])
transition_matrix_B = np.array([
    [0.0, 1],
    [0.08, 0.0]
])
transition_matrix_C = np.array([
    [0.0, 1],
    [0.12, 0.0]
])

#Identity matrix for solving linear system
identity_matrix = np.eye(2)

#Solve for the visit ratios 
visits_A = linalg.solve((identity_matrix - transition_matrix_A).T, lambdaInA / lambdaA)
visits_B = linalg.solve((identity_matrix - transition_matrix_B).T, lambdaInB / lambdaB)
visits_C = linalg.solve((identity_matrix - transition_matrix_C).T, lambdaInC / lambdaC)

#Useful just to check the overall result's correctness
print("-" * 20)
print("Visits_A:", visits_A)
print("Visits_B:", visits_B)
print("Visits_C:", visits_C)
print("-" * 20)


#Calculate demands at each station for each product class
demand_A = visits_A * SA
demand_B = visits_B * SB
demand_C = visits_C * SC

#U1 = np.zeros(len(service_times))


#Calculate utilizations at each station for each product class
Utilization_1A = lambdaA * demand_A[0]
Utilization_1B = lambdaB * demand_B[0]
Utilization_1C = lambdaC * demand_C[0]
Utilization_2A = lambdaA * demand_A[1]
Utilization_2B = lambdaB * demand_B[1]
Utilization_2C = lambdaC * demand_C[1]

# Total utilizations at each station
U1 = Utilization_1A + Utilization_1B + Utilization_1C
U2 = Utilization_2A + Utilization_2B + Utilization_2C


print("U1:", U1)
print("U2:", U2)
print("-" * 20)

#Calculate response times for each product class at each station
R1A = demand_A[0] / (1 - U1)
R1B = demand_B[0] / (1 - U1)
R1C = demand_C[0] / (1 - U1)
R2A = demand_A[1] / (1 - U2)
R2B = demand_B[1] / (1 - U2)
R2C = demand_C[1] / (1 - U2)

#Total response times for each product class
RA = R1A + R2A
RB = R1B + R2B
RC = R1C + R2C

#Calculate average number of jobs for each product class at each station
N1A = lambdaA * R1A
N1B = lambdaB * R1B
N1C = lambdaC * R1C
N2A = lambdaA * R2A
N2B = lambdaB * R2B
N2C = lambdaC * R2C

#Total average number of jobs for each product class
NA = N1A + N2A
NB = N1B + N2B
NC = N1C + N2C


print("NA:                      ", NA)
print("NB:                      ", NB)
print("NC:                      ", NC)

print("RA:                      ", RA)
print("RB:                      ", RB)
print("RC:                      ", RC)
print("-" * 20)


# --------------------------------- Calculate class-independent metrics -----------------

X = lambdaA + lambdaB + lambdaC

#Total average number of jobs in the system
N = NA + NB + NC

print("# of Jobs in the system: ", N)

# Avg response times at each station and system overall Avg Response time
R1 = lambdaA / X * R1A + lambdaB / X * R1B + lambdaC / X * R1C
R2 = lambdaA / X * R2A + lambdaB / X * R2B + lambdaC / X * R2C
R = R1 + R2

print("System Response Timne:   ", R)
print("-" * 20)