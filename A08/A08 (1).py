'''
A DJ has a playlist of 10 songs. Each song has a different duration. At the end of each song, she 
performs one of the following three alternatives:
    • Continue to the next song
    • Extend the current song (by looping, scratching and adding effects)
    • Skip the next song, and continue with the following one

When a song it is extended, at the end of the extension, there is again a probability of continuing to 
the next song, or skip it and jump to the following one. However, the first and the last song cannot 
be skipped, and they are always played. Each song being played, requires the payment of a royalty
fee. When the show is over, we suppose the next one starts immediately.

Compute the following performance metrics:
    • The probability that a patron ears a specific song, when randomly entering the concert (the 
        probability that, in given moment in time, a specific song is being played), for songs 1, 2, 5, 9 
        and 10.
    • The average cost of the songs.
    • The number of shows that could be played per day. Inverting this rate, also compute the 
        average duration of the concerts (a special application of Little's law) in minutes.
Use the following table to determine lengths, probabilities and costs for each song. Consider the 
length of the song being exponentially distributed, with the value considered as its mean.

Song Len.   Extension   Skip Next   Extension   Skip Next if    Royalty fee
            Prob.       Prob.       Len.        extended
240 sec.    20%         5%          30s.        10%             5€
300 sec.    10%         40%         30s.        50%             3€
210 sec.    25%         10%         60s.        30%             3€
235 sec.    20%         20%         30s.        20%             4€
350 sec.    10%         50%         20s.        50%             5€
185 sec.    40%         20%         90s.        10%             3€
220 sec.    30%         10%         30s.        10%             3€
320 sec.    10%         5%          20s.        5%              3€
260 sec.    20%         0%          60s.        0%              5€
480 sec.    50%         0%          120s.       0%              8€


Suggestion: you will very likely need 20 states. There is file called Data.csv, which contains the data 
in the previous table in a machine-readable format.
'''

import os
import numpy as np
import pandas as pd
from collections import deque

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'Data.csv')

# Read the CSV file using pandas
data = pd.read_csv(r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance_Evaluation_Repo\Performance-Evaluation\A08\A08 - CTMC modelling\Data.csv', delimiter=';', header=None, names=[
    'length', 'extension_prob', 'skip_next_prob', 'extension_length', 'skip_next_if_extended_prob', 'royalty_fee'
])

data['extension_prob'] = data['extension_prob'] / 100
data['skip_next_prob'] = data['skip_next_prob'] / 100
data['skip_next_if_extended_prob'] = data['skip_next_if_extended_prob'] / 100

tMax = 1000000

s = 1
t = 0

sxt = deque()
sxt.append([t, s])

RTT = deque()
currRTT = 0

ts = np.zeros(10)

songsPlayed = 0
totalCost = 0

while t < tMax:
    
    p = np.random.rand()
    lambdaExp = 1 / data.iloc[s-1]['length']
    # skip next song
    if p < data.iloc[s-1]['skip_next_prob']:
        ns = s + 2
        dt = -np.log(1 - np.random.rand()) / lambdaExp
        ts[s-1] += dt
        currRTT += dt
        t += dt
    # extend the current song
    elif p < data.iloc[s-1]['skip_next_prob'] + data.iloc[s-1]['extension_prob']:
        p = np.random.rand()
        lambdaExpExt = 1 / data.iloc[s-1]['extension_length']
        # skip next song after extension
        if p < data.iloc[s-1]['skip_next_if_extended_prob']:
            ns = s + 2
            dt1 = -np.log(1 - np.random.rand()) / lambdaExp
            dt2 = -np.log(1 - np.random.rand()) / lambdaExpExt
            dt = dt1 + dt2
            ts[s-1] += dt
            currRTT += dt
            t += dt
        # continue to the next song after extension
        else:
            ns = s + 1
            dt1 = -np.log(1 - np.random.rand()) / lambdaExp
            dt2 = -np.log(1 - np.random.rand()) / lambdaExpExt
            dt = dt1 + dt2            
            ts[s-1] += dt
            currRTT += dt
            t += dt
    # continue to the next song
    else:
        ns = s + 1
        dt = -np.log(1 - np.random.rand()) / lambdaExp
        ts[s-1] += dt
        currRTT += dt
        t += dt

    if s == 10:
        ns = 1
        RTT.append(currRTT)
        currRTT = 0
    
    totalCost += data.iloc[s-1]['royalty_fee']
    s = ns
    sxt.append([t, s])
    songsPlayed += 1
    

print("Probability of earing song 1: ", ts[0] / t)
print("Probability of earing song 2: ", ts[1] / t)
print("Probability of earing song 5: ", ts[4] / t)
print("Probability of earing song 9: ", ts[8] / t)
print("Probability of earing song 10: ", ts[9] / t)

print("Average cost of the songs: ", totalCost / songsPlayed)

avgDurationOfShow = np.mean(RTT) / 60 # convert to minutes
print("Average show duration: ", avgDurationOfShow, " minutes")

showsPerDay = 1 / (avgDurationOfShow / 60 / 24)
print("Number of shows per day: ", showsPerDay)