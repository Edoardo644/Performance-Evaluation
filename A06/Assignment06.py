import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings

# Parameters for system performance simulation
service_rate = 1.0    # Mean service rate
arrival_rate = 0.8    # Mean arrival rate

# Simulation configuration
M = 5000              # Number of customers
initial_iterations = 50
iteration_step = 10
max_iterations = 1000
max_relative_error = 0.02  # Maximum relative error tolerance for convergence

# Hyperexponential distribution parameters
lambda_hyper1 = 0.025
lambda_hyper2 = 0.1
prob_hyper1 = 0.35

# Weibull distribution parameters
weibull_k = 0.333
weibull_lambda = 2.5

# Initialize variables to accumulate results
U1_sum = U2_sum = X_sum = W_sum = L_sum = 0
K = initial_iterations  # Start with initial number of iterations

while K < max_iterations:
    for _ in range(K):
        # Generate inter-arrival times based on a Hyperexponential distribution
        u1, u2 = np.random.uniform(0, 1, M), np.random.uniform(0, 1, M)
        inter_arrival_times = np.where(u1 < prob_hyper1, -np.log(u1) / lambda_hyper1, -np.log(u2) / lambda_hyper2)

        # Generate service times based on a Weibull distribution
        u = np.random.uniform(0, 1, M)
        service_times = weibull_lambda * (-np.log(u)) ** (1 / weibull_k)

        # Initialize arrival and completion time arrays
        arrival_times = np.zeros(M)
        completion_times = np.zeros(M)

        arrival_times[0] = inter_arrival_times[0]
        completion_times[0] = inter_arrival_times[0] + service_times[0]

        # Calculate arrival and completion times iteratively
        for i in range(1, M):
            arrival_times[i] = arrival_times[i - 1] + inter_arrival_times[i]
            completion_times[i] = max(arrival_times[i], completion_times[i - 1]) + service_times[i]

        # Calculate time in system and performance metrics
        total_time = completion_times[-1]
        total_service_time = np.sum(service_times)
        utilization = total_service_time / total_time
        throughput = M / total_time
        waiting_time = np.sum(completion_times - service_times) / M
        customers_in_system = throughput * waiting_time

        # Accumulate performance metrics
        U1_sum += utilization
        U2_sum += utilization ** 2
        X_sum += throughput
        W_sum += waiting_time
        L_sum += customers_in_system

    # Calculate statistics for utilization
    expected_utilization = U1_sum / K
    expected_utilization_squared = U2_sum / K
    variance_utilization = expected_utilization_squared - expected_utilization ** 2
    sigma_utilization = np.sqrt(variance_utilization)

    # 95% confidence interval
    delta_u = 1.96 * sigma_utilization / np.sqrt(K)
    lower_bound = expected_utilization - delta_u
    upper_bound = expected_utilization + delta_u
    relative_error = 2 * (upper_bound - lower_bound) / (upper_bound + lower_bound)

    # Check if relative error is within the acceptable range
    if relative_error < max_relative_error:
        break

    # Update iteration count if convergence criterion not met
    K += iteration_step

# Output final results
print("95% confidence interval of U:    ", lower_bound, upper_bound)
print("Relative error of U:             ", relative_error)
print("Solution obtained in             ", K, " iterations\n")
print("Utilization:                     ", expected_utilization)
print("Throughput:                      ", X_sum / K)
print("Average response time:           ", W_sum / K)
print("Average number of customers:     ", L_sum / K)
