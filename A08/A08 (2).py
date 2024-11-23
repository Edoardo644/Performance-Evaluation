import os
import numpy as np
import pandas as pd
from scipy import linalg

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'Data.csv')

# Read the CSV file using pandas
data = pd.read_csv(csv_path, delimiter=';', header=None, names=[
    'length', 'extension_prob', 'skip_next_prob', 'extension_length', 'skip_next_if_extended_prob', 'royalty_fee'
])

data['extension_prob'] = data['extension_prob'] / 100
data['skip_next_prob'] = data['skip_next_prob'] / 100
data['skip_next_if_extended_prob'] = data['skip_next_if_extended_prob'] / 100

Q = np.empty((20, 20))
for i in range(10):
    Q[2*i, 2*i] = -1 / data['length'][i]
    Q[2*i, 2*i+1] = data['extension_prob'][i] / data['length'][i]
    if i == 9:
        Q[2*i, 0] = (1 - data['extension_prob'][i]) / data['length'][i]
    else:
        Q[2*i, 2*i+2] = (1 - data['extension_prob'][i] - data['skip_next_prob'][i]) / data['length'][i]
    if i < 8:
        Q[2*i, 2*i+4] = data['skip_next_prob'][i] / data['length'][i]
    
    Q[2*i+1, 2*i+1] = -1 / data['extension_length'][i]
    if i == 9:
        Q[2*i+1, 0] = 1 / data['extension_length'][i]
    else:
        Q[2*i+1, 2*i+2] = (1 - data['skip_next_if_extended_prob'][i]) / data['extension_length'][i]
    if i < 8:
        Q[2*i+1, 2*i+4] = data['skip_next_if_extended_prob'][i] / data['extension_length'][i]


Q2 = Q.copy()
Q2[:,0] = np.ones(20)
b = np.zeros(20)
b[0] = 1

p = linalg.solve(Q2.T, b)

# Compute the probability that a patron ears a specific song, when randomly entering the concert
song_probs = np.zeros(10)
for i in range(10):
    song_probs[i] = p[2*i] + p[2*i+1]

print("Probability of earing song 1: ", song_probs[0])
print("Probability of earing song 2: ", song_probs[1])
print("Probability of earing song 5: ", song_probs[4])
print("Probability of earing song 9: ", song_probs[8])
print("Probability of earing song 10: ", song_probs[9])

# Compute the average cost of the songs
average_cost = song_probs @ data['royalty_fee']
print("Average cost of the songs: ", average_cost)

# Compute the number of shows per day
xi = np.zeros((20, 20))
xi[19, 0] = 1
xi[18, 0] = 1
RTLP = ((Q * xi) @ np.ones(20)) @ p
print("Number of shows per day: ", RTLP * 60 * 60 * 24)

# Compute average duration of a concert
duration = 1 / (RTLP * 60)
print("Average show duration: ", duration, " minutes")