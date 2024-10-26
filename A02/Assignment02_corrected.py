import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve

# Load CSV files (use input() or other methods to get file paths dynamically)
file_path_1 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A02\Logger1.csv'
file_path_2 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A02\Logger2.csv'

# Read CSV files
logger1_data = pd.read_csv(file_path_1, header=None)
logger2_data = pd.read_csv(file_path_2, header=None)

# Convert into arrays (lists)
inter_arrival_times = logger1_data[0].tolist()
service_times = logger2_data[0].tolist()

# Get the number of rows in each file
num_rows_logger1 = len(inter_arrival_times)
num_rows_logger2 = len(service_times)

# Calculate the cumulative sum of inter-arrival times to get arrival times of each request
arrival_times = np.cumsum(inter_arrival_times)

# Initialize completion times
completion_times = np.zeros(num_rows_logger2)
completion_times[0] = arrival_times[0] + service_times[0]

# Calculate the completion times for each request
for st in range(1, num_rows_logger2):
    completion_times[st] = max(arrival_times[st], completion_times[st-1]) + service_times[st]

# Calculate response times
response_times = completion_times - arrival_times

# Calculate average response time
R = sum(response_times) / num_rows_logger2
print(f"Original Average response time = {R}")

# More accurate approach to find alpha directly from the fact that R = 20
# Define the target response time
target_response_time_beta = 15

# Function to calculate the new response time based on beta (modifying the service times)
def calculate_new_response_time_with_beta(beta):
    adjusted_service_times = [s * beta for s in service_times]
    ST_C_T = np.zeros(num_rows_logger2)
    ST_C_T[0] = arrival_times[0] + adjusted_service_times[0]
    
    # Calculate new completion times with modified service times
    for st in range(1, num_rows_logger2):
        ST_C_T[st] = max(arrival_times[st], ST_C_T[st-1]) + adjusted_service_times[st]
    
    # Calculate new response times
    new_response_times = ST_C_T - arrival_times
    new_R_with_beta = sum(new_response_times) / num_rows_logger2
    
    return new_R_with_beta

# Function to find beta that makes the response time equal to the target (15 minutes)
def find_beta_for_response_time(beta):
    return calculate_new_response_time_with_beta(beta) - target_response_time_beta

# Initial guess for beta
beta_guess = 1.0

# Solve for beta using fsolve to meet the target response time of 15 minutes
beta_solution = fsolve(find_beta_for_response_time, beta_guess)[0]

# Calculate new response times using the beta_solution
adjusted_service_times = [s * beta_solution for s in service_times]

CT1_beta = np.zeros(num_rows_logger2)
CT1_beta[0] = arrival_times[0] + adjusted_service_times[0]
for st in range(1, num_rows_logger2):
    CT1_beta[st] = max(arrival_times[st], CT1_beta[st-1]) + adjusted_service_times[st]

# Calculate the final new response time
new_response_times_beta = CT1_beta - arrival_times
final_R_beta = sum(new_response_times_beta) / num_rows_logger2

# Print the results
print("")
print(f"Beta that gives an average response time of 15 minutes: {beta_solution}")
print(f"New Average Response Time (with Beta): {final_R_beta}")
