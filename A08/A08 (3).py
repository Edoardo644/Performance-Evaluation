import os
import numpy as np
import pandas as pd
from scipy import linalg

# Define the path to the CSV file
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'Data.csv')

# Load song data into a DataFrame
song_data = pd.read_csv(
    r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance_Evaluation_Repo\Performance-Evaluation\A08\A08 - CTMC modelling\Data.csv', 
    delimiter=';', 
    header=None, 
    names=['length', 'extension_prob', 'skip_next_prob', 'extension_length', 'skip_next_if_extended_prob', 'royalty_fee']
)

# Initialize the transition rate matrix for the CTMC
transition_rate_matrix = np.empty((20, 20))

# Populate the transition rate matrix
for song_index in range(10):
    
    # Fetch probabilities and lengths for the current song
    extension_probability = song_data.iloc[song_index]['extension_prob'] / 100
    skip_next_probability = song_data.iloc[song_index]['skip_next_prob'] / 100
    skip_if_extended_probability = song_data.iloc[song_index]['skip_next_if_extended_prob'] / 100

    #First song or song with song_index + 1
    transition_rate_matrix[2*song_index, 2*song_index] = -1 / song_data['length'][song_index]  # Self-transition rate
    transition_rate_matrix[2*song_index, 2*song_index+1] = extension_probability / song_data['length'][song_index]  # Transition to the extended state

    #If last song, jump back to the first otherwise go next
    if song_index == 9:
        transition_rate_matrix[2*song_index, 0] = (1 - extension_probability) / song_data['length'][song_index]
    else:
        transition_rate_matrix[2*song_index, 2*song_index+2] = (1 - extension_probability - skip_next_probability) / song_data['length'][song_index]

    # Handle transitions for skipping the next song
    if song_index < 8:  # Skipping is not allowed for the last two songs
        transition_rate_matrix[2*song_index, 2*song_index+4] = skip_next_probability / song_data['length'][song_index]

    # Transition rates for the extended song (state 2*song_index+1)
    transition_rate_matrix[2*song_index+1, 2*song_index+1] = -1 / song_data['extension_length'][song_index]  # Self-transition rate

    # Handle transitions to the next song or back to the first song
    if song_index == 9:
        transition_rate_matrix[2*song_index+1, 0] = 1 / song_data['extension_length'][song_index]
    else:
        transition_rate_matrix[2*song_index+1, 2*song_index+2] = (1 - skip_if_extended_probability) / song_data['extension_length'][song_index]

    # Handle transitions for skipping the next song from the extended state
    if song_index < 8:  # Skipping is not allowed for the last two songs
        transition_rate_matrix[2*song_index+1, 2*song_index+4] = skip_if_extended_probability / song_data['extension_length'][song_index]

# Normalization 
modified_transition_matrix = transition_rate_matrix.copy()
modified_transition_matrix[:, 0] = np.ones(20)  
normalization_vector = np.zeros(20)
normalization_vector[0] = 1  
steady_state_probabilities = linalg.solve(modified_transition_matrix.T, normalization_vector)

# Compute the probability of hearing each song when randomly entering the concert
song_probabilities = np.zeros(10)
for song_index in range(10):
    song_probabilities[song_index] = steady_state_probabilities[2*song_index] + steady_state_probabilities[2*song_index+1]

# Print probabilities for specific songs
print("Probability of hearing song 1: ", song_probabilities[0])
print("Probability of hearing song 2: ", song_probabilities[1])
print("Probability of hearing song 5: ", song_probabilities[4])
print("Probability of hearing song 9: ", song_probabilities[8])
print("Probability of hearing song 10: ", song_probabilities[9])

# Compute the average royalty cost of the songs
average_royalty_cost = song_probabilities @ song_data['royalty_fee']
print("Average royalty cost of the songs: ", average_royalty_cost)

# Compute the number of shows per day
show_transition_matrix = np.zeros((20, 20))
show_transition_matrix[19, 0] = 1  # Transition from the last song's extended state to the first song
show_transition_matrix[18, 0] = 1  # Transition from the last song to the first song
shows_per_second = ((transition_rate_matrix * show_transition_matrix) @ np.ones(20)) @ steady_state_probabilities
shows_per_day = shows_per_second * 60 * 60 * 24  # Convert to daily rate
print("Number of shows per day: ", shows_per_day)

# Compute the average duration of a concert
average_show_duration = 1 / (shows_per_second * 60)  # Convert seconds to minutes
print("Average show duration: ", average_show_duration, " minutes")
