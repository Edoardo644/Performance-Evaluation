import numpy as np
import pandas as pd
from collections import deque

# Load song data
data = pd.read_csv(r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance_Evaluation_Repo\Performance-Evaluation\A08\A08 - CTMC modelling\Data.csv', delimiter=';', header=None, names=[
    'length', 'extension_prob', 'skip_next_prob', 'extension_length', 'skip_next_if_extended_prob', 'royalty_fee'
])

#Constants
tMax = 10000000  # Total duration for simulation 
t = 0  # Elapsed time 

s = 1  # Current song 
RTT = deque()  #Complete concert durations
currRTT = 0  #Duration of the current concert cycle

ts = np.zeros(10)  
songsPlayed = 0  
totalCost = 0  
sxt = deque()  # Tracks time and state transitions 


while t < tMax:

    # Fetch probabilities for the current song
    extension_prob = data.iloc[s-1]['extension_prob'] / 100
    skip_next_prob = data.iloc[s-1]['skip_next_prob'] / 100 #if s not in [9, 10] else 0  # Skip not allowed for song 1 and 10
    skip_if_extended_prob = data.iloc[s-1]['skip_next_if_extended_prob'] / 100
    
    p = np.random.rand()

    # Simulate durations using exponential distributions
    
    # Mean duration of the current song
    base_length = np.random.exponential(scale=data.iloc[s-1]['length'])
 
    # Mean duration of the extension
    extension_length = np.random.exponential(scale =data.iloc[s-1]['extension_length'])
    total_length = base_length + extension_length    

    # Skip, Extend, or Continue
    if p < skip_next_prob:  # Skip the next song
        ns = s + 2
        t += base_length
        currRTT += base_length
        ts[s-1] += base_length
    elif p < skip_next_prob + extension_prob:  # Extend the current song
        p_extension = np.random.rand()
        if p_extension < skip_if_extended_prob:  # Skip after extension
            ns = s + 2
        else:  # Continue to the next song
            ns = s + 1
            
        t += total_length
        currRTT += total_length
        ts[s-1] += total_length
    else:  # Continue to the next song
        ns = s + 1
        t += base_length
        currRTT += base_length
        ts[s-1] += base_length

    # Check if the last song is played
    if s == 10:
        ns = 1  # Reset to the first song
        RTT.append(currRTT)  # Store the completed concert duration
        currRTT = 0  # Reset the current round-trip time
        #totalCost += data.iloc[s-1]['royalty_fee']

    # Increment royalty fee for the current song
    totalCost += data.iloc[s-1]['royalty_fee']
    # Update state
    s = ns
    sxt.append([t, s])
    songsPlayed += 1
    

# Compute probabilities and performance metrics
print("Probability of hearing song 1: ", ts[0] / t)
print("Probability of hearing song 2: ", ts[1] / t)
print("Probability of hearing song 5: ", ts[4] / t)
print("Probability of hearing song 9: ", ts[8] / t)
print("Probability of hearing song 10: ", ts[9] / t)
print("Average cost per song: ", totalCost / songsPlayed)
avg_concert_duration = np.mean(RTT) / 60  # Average duration of a concert in minutes
print("Average concert duration: ", avg_concert_duration, "minutes")
shows_per_day = 1 / (avg_concert_duration / 60 / 24)  # Shows per day
print("Number of shows per day: ", shows_per_day)
