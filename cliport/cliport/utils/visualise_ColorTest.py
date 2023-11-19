import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# location to store relevant data files
store_dir = 'data'

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(os.path.join(store_dir, 'color_test.csv'))

# Define the list of color values
colors = ['red', 'green', 'blue', 'yellow', 'brown', 'gray', 'cyan', 'orange', 'purple', 'pink', 'white']

# Create an empty matrix to store the reward values
reward_matrix = np.zeros((len(colors), len(colors)))

# Create an empty matrix to store the success rates
success_rate_matrix = np.zeros((len(colors), len(colors)))

# Create an empty matrix to store the total attempts for each pair
attempts_matrix = np.zeros((len(colors), len(colors)))

# Iterate over the DataFrame rows and update the reward and attempts matrices
for index, row in df.iterrows():
    pick_color = row['pick color']
    place_color = row['place color']
    reward = row['Total Reward']
    pick_index = colors.index(pick_color)
    place_index = colors.index(place_color)
    reward_matrix[place_index][pick_index] += reward
    attempts_matrix[place_index][pick_index] += 1

# Calculate the success rate for each pick and place color pair
for i in range(len(colors)):
    for j in range(len(colors)):
        if attempts_matrix[i][j] > 0:
            success_rate_matrix[i][j] = reward_matrix[i][j] / attempts_matrix[i][j]

# save the success_rate_matrix to map unseen color to seen color
np.savetxt(os.path.join(store_dir, 'success_rate_matrix.csv'), success_rate_matrix, delimiter=",")

# Create a matrix plot with numbers displayed
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the matrix as an image
im = ax.imshow(success_rate_matrix, cmap='coolwarm', interpolation='nearest')

# Set the x-axis and y-axis labels
ax.set_xticks(np.arange(len(colors)))
ax.set_yticks(np.arange(len(colors)))
ax.set_xticklabels(colors, rotation='vertical')
ax.set_yticklabels(colors)
ax.set_xlabel('Pick block Color')  # Add x-axis label
ax.set_ylabel('Place bowl Color')  # Add y-axis label

# Loop over data dimensions and create text annotations
for i in range(len(colors)):
    for j in range(len(colors)):
        text = ax.text(j, i, '{:.2f}'.format(success_rate_matrix[i][j]),
                       ha='center', va='center', color='w')

# Calculate the total rewards and attempts
total_rewards = np.sum(reward_matrix)
total_attempts = np.sum(attempts_matrix)

# Calculate the total success rate
total_success_rate = total_rewards / total_attempts
# Format the total success rate to show only the first three digits
formatted_success_rate = '{:.3f}'.format(total_success_rate)

# Add a title to the plot
plt.title(f'Total Success Rates: {formatted_success_rate}')

# Set the color bar for the matrix plot
cbar = plt.colorbar(im)
cbar.set_label('Success Rate')

# Show the matrix plot
plt.show()