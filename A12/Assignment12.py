import numpy as np
from scipy import linalg

# Service times at each component (in seconds)
service_times = np.array([2, 0.02, 0.1, 0.07])  # [Self-check, App server, Storage, DBMS]

# Incoming request rates 
_lambda_ = np.zeros(4)
_lambda_[0] = 2.5  # Requests that pass through self-check
_lambda_[1] = 2    # Requests directly arriving at the application server

# Total request rate
total_rate = np.sum(_lambda_)

# Proportion of total requests for each component
relative_arrival_rates = _lambda_ / total_rate

#Initialize transistion matrix
transition_matrix = np.array([
    [0.0, 0.7, 0.0, 0.0],      # From self-check
    [0.0, 0.0, 0.25, 0.45],    # From application server
    [0.0, 1.0, 0.0, 0.0],      # From storage
    [0.0, 1.0, 0.0, 0.0]       # From DBMS
])

# Identity matrix for solving linear system
identity_matrix = np.eye(4)

# Solve for the visit ratios 
visit_ratios = linalg.solve((identity_matrix - transition_matrix).T, relative_arrival_rates)
print("-" * 20)
print("Application server visits:", visit_ratios[1])
print("Storage visits:", visit_ratios[2])
print("DBMS visits:", visit_ratios[3])

# System throughput
X = total_rate
print("Throughput [req/s]:", X)

# Service demands at each component
service_demands = visit_ratios * service_times

# Utilization of each component
utilizations = X * service_demands


# Average number of jobs at each component
avg_jobs = utilizations / (1 - utilizations)
avg_jobs[0] = utilizations[0]  
total_jobs = np.sum(avg_jobs)
print("Avg #jobs in the system:", total_jobs)

#
# For response times and avg number of jobs the first one is a delay center
#

# Response time for each component
response_times = service_demands / (1 - utilizations)
response_times[0] = service_demands[0]  
total_response_time = np.sum(response_times)
print("Response time [ms]:", total_response_time * 1000)



# Maximum throughput (determined by the bottleneck component)
max_throughput = 1 / np.max(service_demands)
print("Max throughput [req/s]:", max_throughput)
print("-" * 20)
