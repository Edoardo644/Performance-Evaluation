import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve

#Load CSV files
file_path_1 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A02\Logger1.csv'
file_path_2 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A02\Logger2.csv'

#Read  CSV files
logger1_data = pd.read_csv(file_path_1, header=None)
logger2_data = pd.read_csv(file_path_2, header=None)

#Convert into arrays (lists)
Inter_Arrival_times= logger1_data[0].tolist()
Service_times = logger2_data[0].tolist()

#Get the number of rows in each file
num_rows_logger1 = len(Inter_Arrival_times)
num_rows_logger2 = len(Service_times)


############################## FIRST PART ASSIGNMENT ##############################

#Global variables to store the last res and the last R (adjusted inter-arrival times and Average Response time)
last_res = []
last_R = []

A_T = np.cumsum(Inter_Arrival_times)

C_T = np.zeros(num_rows_logger2)
C_T[0] = A_T[0] + Service_times[0]
for st in range(1, num_rows_logger2):
    C_T[st] = max(A_T[st], C_T[st-1]) + Service_times[st]

response_times = C_T - A_T

R = sum(response_times) / num_rows_logger2
print(f"Original Average response time = {R}")

#Approach to do it by guess
"""alpha = float(input("Choose your alpha "))

res = [x * alpha for x in Inter_Arrival_times]

new_A_T = np.cumsum(res)

new_C_T = np.zeros(num_rows_logger2)
new_C_T[0] = new_A_T[0] + Service_times[0]
for st in range(1, num_rows_logger2):
    new_C_T[st] = max(new_A_T[st], new_C_T[st-1]) + Service_times[st]


new_response_times = new_C_T - new_A_T
new_R = sum(new_response_times) / num_rows_logger2
print(f"New Average Response Time: {new_R}")"""

#More accurate approach to find alpha directly from the fact R = 20

#Define the function to calculate new_R based on alpha
def calculate_new_response_time(alpha):
    global last_res 
    global last_R
    last_res= [x * alpha for x in Inter_Arrival_times]
    new_A_T = np.cumsum(last_res)
    
    #Initialize new_C_T
    new_C_T = np.zeros(num_rows_logger2)
    new_C_T[0] = new_A_T[0] + Service_times[0]
    
    #Calculate new C_T for each request
    for st in range(1, num_rows_logger2):
        new_C_T[st] = max(new_A_T[st], new_C_T[st-1]) + Service_times[st]
    
    #Calculate new response times
    new_response_times = new_C_T - new_A_T
    new_R = sum(new_response_times) / num_rows_logger2
    last_R = new_R
    
    return new_R

# Define the target response time
target_response_time = 20

# Define the function to solve for alpha where new_R = 20
def find_alpha(alpha):
    return calculate_new_response_time(alpha) - target_response_time

# Initial guess for alpha
alpha_guess = 1.0

# Solve for alpha using fsolve
alpha_solution = fsolve(find_alpha, alpha_guess)[0]

print(f"Alpha that gives an average response time of 20 minutes: {alpha_solution}")

#Average inter arrival rate and average service rate
_A_ = (sum(Inter_Arrival_times) / num_rows_logger1) 

#Original arrival rate
arrival_rate = 1 / _A_
print(f"Original Arrival Rate: {arrival_rate}")

#New Average inter-arrival-Rate
new_A_ = (sum(last_res) / num_rows_logger1)
max_arrival_rate = 1 / new_A_
print(f"Max Arrival Rate: {max_arrival_rate}")

#New Average Response Time
print(f"Average Response Time with inter arrival times modified of a factor alpha: {last_R}")


#################  SECOND PART OF ASSIGNMENT #########################


##### CALCULATE ALPHA GIVEN THE MAX ARRIVAL RATE = 1.2 ######

# Define the target arrival rate
target_arrival_rate = 1.2

# Function to calculate the arrival rate based on alpha
def calculate_arrival_rate(alpha_prime):
    # Adjust inter-arrival times with alpha_prime
    app = [x * alpha_prime for x in Inter_Arrival_times]
    
    # Calculate the new average inter-arrival time
    A1 = sum(app) / num_rows_logger1
    
    # Calculate the arrival rate (1 / average inter-arrival time)
    arrival_rate_prime = 1 / A1
    
    return arrival_rate_prime

# Function to find alpha that makes the arrival rate equal to the target (1.2 jobs/minute)
def find_alpha_for_arrival_rate(alpha_prime):
    return calculate_arrival_rate(alpha_prime) - target_arrival_rate

# Initial guess for alpha_prime
alpha_prime_guess = 1.0

# Solve for alpha_prime using fsolve to meet the target arrival rate of 1.2
alpha_prime_solution = fsolve(find_alpha_for_arrival_rate, alpha_prime_guess)[0]

# Calculate new inter-arrival times and response times using alpha_prime_solution
app = [x * alpha_prime_solution for x in Inter_Arrival_times]
AT_beta = np.cumsum(app)

CT1 = np.zeros(num_rows_logger2)
CT1[0] = AT_beta[0] + Service_times[0]
for st in range(1, num_rows_logger2):
    CT1[st] = max(AT_beta[st], CT1[st-1]) + Service_times[st]

# Calculate new response times
new_response_times = CT1 - AT_beta
new_R = sum(new_response_times) / num_rows_logger2

# Print the results
print(f"Alpha that gives an arrival rate of 1.2 jobs per minute: {alpha_prime_solution}")

# Verify the new arrival rate
A1 = (sum(app) / num_rows_logger1)
max_arrival_rate = 1 / A1
print(f"Max Arrival Rate (should be 1.2): {max_arrival_rate}")
print("\n\n\n")

from scipy.optimize import fsolve

# Define the target response time
target_response_time_beta = 15

# Function to calculate the new response time based on beta (modifying the service times)
def calculate_new_response_time_with_beta(beta):
    # Adjust the service times by multiplying by beta
    adjusted_service_times = [s * beta for s in Service_times]

    #It works but it gives a deprecating warning for some unknown reason
    # Initialize new C_T
    ST_C_T = np.zeros(num_rows_logger2)
    
   #Ensure extraction of scalar values explicitly using float conversion
    ST_C_T[0] = float(AT_beta[0]) + adjusted_service_times[0]

    for st in range(1, num_rows_logger2):
        ST_C_T[st] = max(float(AT_beta[st]), float(ST_C_T[st-1])) + adjusted_service_times[st]

        
    # Calculate new response times
    new_response_times = ST_C_T - AT_beta
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
adjusted_service_times = [s * beta_solution for s in Service_times]

CT1_beta = np.zeros(num_rows_logger2)
CT1_beta[0] = AT_beta[0] + adjusted_service_times[0]
for st in range(1, num_rows_logger2):
    CT1_beta[st] = max(AT_beta[st], CT1_beta[st-1]) + adjusted_service_times[st]

# Calculate the final new response time
new_response_times_beta = CT1_beta - AT_beta
final_R_beta = sum(new_response_times_beta) / num_rows_logger2

# Print the results
print("\n\n\n")
print(f"Beta that gives an average response time of 15 minutes: {beta_solution}")
print(f"New Average Response Time (with Beta): {final_R_beta}")