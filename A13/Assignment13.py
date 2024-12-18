import numpy as np
from scipy import linalg

# Incoming request rates 
_lambda_ = np.zeros(6)
_lambda_[0] = 1  # Initialize to make the matrix invertible

N = 100 #Number of employers

# Think times
service_times = np.array([40, 0.05, 0.002, 0.08, 0.08, 0.1])

#Initialize transistion matrix
transition_matrix = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      
    [0.0, 0.0, 0.35, 0.6, 0.0, 0.0],    
    [0.0, 0.0, 0.0, 0.0, 0.65, 0.35],      
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.0, 0.0, 0.0, 0.1],
    [0.0, 0.9, 0.0, 0.0, 0.1, 0.0]
])

     # ---------------------------- VISITS ----------------------------

# Identity matrix for solving linear system
identity_matrix = np.eye(6)

# Solve for the visit ratios 
visit_ratios = linalg.solve((identity_matrix - transition_matrix).T, _lambda_)

# ---------------------------- DEMANDS ----------------------------

# Calculate demands for each station
demands = np.zeros(len(service_times))

for i in range(0, len(service_times)):
    demands[i] = visit_ratios[i] * service_times[i]
    

#Demands of the 2 disks
print("-" * 20)

print("Demands for the 1st disk [ms]:", demands[4] * 1000)
print("Demands for the 2nd disk [ms]:", demands[5] * 1000)

print("-" * 20)

# ---------------------------- MVA ANALYSIS ----------------------------

#Initialize  number of jobs at each station
N2 = 0 
N3 = 0
N4 = 0
N5 = 0
N6 = 0

for i in range (1, N+1):
    
    #response times of stations
    R2 = demands[1] * (1 + N2)
    R3 = demands[2] * (1 + N3)
    R4 = demands[3] * (1 + N4)
    R5 = demands[4] * (1 + N5)
    R6 = demands[5] * (1 + N6)
    
    # Total response time
    R_TOT = R2 + R3 + R4 + R5 + R6
    
    #throughput of the system
    X = i / (R_TOT + demands[0])
    
    #update number of jobs in stations
    N2 = R2 * X
    N3 = R3 * X
    N4 = R4 * X
    N5 = R5 * X
    N6 = R6 * X
    
    #utilization fpr AppServer - DBMS - DISKS 
    U_Application_Server = demands[1] * X
    U_DBMS = demands[3] * X
    U_Disk1 = demands[4] * X
    U_Disk2 = demands[5] * X
    
    
    
# Output average system throughput and response time
print("Throughput of the system:", X)
print("-" * 20)
print("Avg Response time of the system [ms]:", R_TOT * 1000)
print("-" * 20)

# Output utilization for specific stations
print("Utilization Application Server:", U_Application_Server)
print("Utilization DBMS:", U_DBMS)
print("Utilization Disk1:", U_Disk1)
print("Utilization Disk2:", U_Disk2)
print("-" * 20)

# Calculate station throughputs as total throughput multiplied by visit ratios
X_Disk1 = X * visit_ratios[4]
X_Disk2 = X * visit_ratios[5]

# Output throughput for each disk
print("Throughput Disk 1:", X_Disk1)
print("Throughput Disk 2:", X_Disk2)
print("-" * 20)
    
    

