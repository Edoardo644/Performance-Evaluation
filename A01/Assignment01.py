import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

#Load CSV files
file_path_1 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A01\Logger1.csv'
file_path_2 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A01\Logger2.csv'

#Read  CSV files
logger1_data = pd.read_csv(file_path_1, header=None)
logger2_data = pd.read_csv(file_path_2, header=None)

#Convert into arrays (lists)
Arrival_times = logger1_data[0].tolist()
Completion_times = logger2_data[0].tolist()

#Get the number of rows in each file
num_rows_logger1 = len(Arrival_times)
num_rows_logger2 = len(Completion_times)

#Arrival rate
arrival_rate = num_rows_logger1 / Completion_times[-1]

#Throughput
X = num_rows_logger2 / Completion_times[-1]


Avg_Inter_arrival_time = 1 / arrival_rate
print(f"\nThroughput = {X}")
print(f"Lambda = {arrival_rate}")
print(f"Average Inter arrival time = {Avg_Inter_arrival_time}")

#Service times
service_times = [] 

service_times.append(Completion_times[0] - Arrival_times[0])
for i in range(1, num_rows_logger1):
        service_times.append(Completion_times[i] - max(Arrival_times[i], Completion_times[i-1]))

#Busy time and Utilization
B_T = sum(service_times)
U = B_T / Completion_times[-1]
Avg_Service_Time = B_T/len(service_times)

print(f"Busy time = {B_T}")
print(f"Utilization = {U}")
print(f"Average service time = {Avg_Service_Time}")

#Response times
response_times = []

for i in range(1, num_rows_logger1):
    response_times.append(Completion_times[i] - Arrival_times[i])
  
#Average response times  
R = sum(response_times) / len(response_times)
print(f"Average response times = {R}")

#Average number of jobs
N = X * R
print(f"Average number of jobs = {N}")


# ---------- Plotting Distributions ------------

# 1. Distribution of the number of cars (0 to 25)
events = [(time, 'arrival') for time in Arrival_times] + [(time, 'completion') for time in Completion_times]
events.sort()

# Create AC_T array with time and cumulative number of jobs
times = np.array([event[0] for event in events])
job_changes = np.array([1 if event[1] == 'arrival' else -1 for event in events])
cumulative_jobs = np.cumsum(job_changes)

# Combine times and cumulative number of jobs into AC_T
AC_T = np.column_stack((times, cumulative_jobs))

# Calculate the time spent with a specific number of jobs
dTN_T = (np.c_[[AC_T[1:, 0] - AC_T[:-1, 0], AC_T[:-1, 1]]]).T

# Define TforP as the difference between the time of the last departure and the time of the first arrival
TforP = AC_T[-1, 0] - AC_T[0, 0]

# Plot the queue length distribution
PN = [0] * 26
PNL = [0] * 26
for i in range(0, 26):
    PNL[i] = i
    PN[i] = np.sum(dTN_T[dTN_T[:, 1] == i, 0]) / TforP

plt.bar(PNL, PN)
plt.title('Queue Length Distribution (0 to 25)')
plt.xlabel('Number of Cars')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 2. Response time distribution (1 to 40 minutes, granularity 1 min)
response_times = np.array(response_times)
PR = [0]*41
PRT = [0]*41
for i in range(0, 41):
    t = i
    PRT[i] = t
    PR[i] = sum(response_times < t) / num_rows_logger1

plt.plot(PRT, PR)
plt.title('Response Time Distribution (1 to 40 minutes)')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# 3. Service time distribution (0.1 to 5 minutes, granularity 0.1 min)
service_times = np.array(service_times)
PS = [0]*51
PST = [0]*51
for i in range(0, 51):
    t = i / 10
    PST[i] = t
    PS[i] = sum(service_times < t) / num_rows_logger1
    
plt.plot(PST, PS)
plt.title('Service Time Distribution (0.1 to 5 minutes)')
plt.xlabel('Service Time (minutes)')
plt.ylabel('Frequency')
plt.show()