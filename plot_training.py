import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load the CSV file
csv_filename = "training_stats.csv"
df1 = pd.read_csv(csv_filename)
# Find the number of elements to group
group_size = 5

# Group the dataframe in chunks of 'group_size' elements
grouped_df1 = [df1.iloc[i:i+group_size] for i in range(0, len(df1), group_size)]

# Initialize an empty list to store the modified data
modified_data = []

# Iterate over each group and perform the required modifications
for i, group in enumerate(grouped_df1):
    # Take the first 5 elements and sum the total reward
    total_reward_sum = group['Total Reward'].sum()

    # Create a new row with the modified values
    new_row = {
        'Episode': i * group_size,  # Set the episode based on the group index
        'Total Reward': total_reward_sum,
        'Epsilon': group.iloc[-1]['Epsilon'],  # Use the epsilon of the last element
        'Avg Reward (Last 100)': group['Avg Reward (Last 100)'].sum(),
        'Total Frames': group.iloc[-1]['Total Frames'],
        'Frame Rate': group['Frame Rate'].mean(),
        'Model Updates': group['Model Updates'].sum(),
        'Running Reward': group['Running Reward'].sum(),
        'Training Time': group['Training Time'].sum()
    }

    # Append the new row to the modified data list
    modified_data.append(new_row)

# Create a new dataframe from the modified data
df = pd.DataFrame(modified_data)

# Calculate the rolling average
rolling_average_window = 100  # You can adjust this window size
df['Rolling_Avg_Reward'] = df['Total Reward'].rolling(window=rolling_average_window).mean()

# Find the maximum total reward and its corresponding episode
max_avg_reward = df['Rolling_Avg_Reward'].max()
max_reward_frame_avg = df.loc[df['Rolling_Avg_Reward'].idxmax(), 'Total Frames']

max_reward = df['Total Reward'].max()
max_reward_frame_total = df.loc[df['Total Reward'].idxmax(), 'Total Frames']

# Plot the rewards and rolling average against Total Frames
plt.figure(figsize=(10, 6))
plt.plot(df['Total Frames'], df['Total Reward'], linestyle='--', label='Total Reward', color='b', alpha = 0.2)
plt.plot(df['Total Frames'], df['Rolling_Avg_Reward'], label=f'Rolling Avg ({rolling_average_window} episodes)')

# Use axhline for horizontal lines
plt.axhline(y=max_avg_reward, color='c', alpha=0.5, linestyle='-', label=f'Max Avg Reward ({max_avg_reward:.2f} at Frame {int(max_reward_frame_avg):,})')
plt.axhline(y=max_reward, color='purple', alpha=0.5, linestyle='-', label=f'Max Reward ({max_reward:.2f} at Frame {int(max_reward_frame_total):,})')

local_reward_line = plt.gca().lines[0]
local_reward_line.set_dashes([2, 8])  # Adjust the numbers in the list to control the dash pattern

# Set y-axis ticks to integers every two integers
plt.yticks(range(int(min(df['Total Reward'])), int(max(df['Total Reward']))+1, 5))

# Format x-axis ticks using FuncFormatter for readability
def format_ticks(value, pos):
    if value >= 1e6:
        return f'{value * 1e-6:.1f}M'
    elif value >= 1e3:
        return f'{value * 1e-3:.0f}K'
    else:
        return f'{value:.0f}'

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))

plt.xlabel('Total Frames')
plt.ylabel('Clipped Reward')
plt.legend()
plt.title('Total (Clipped) Reward and Rolling Average vs Total Frames')
plt.grid(False)
plt.savefig('reward_plot_training.png')  # Save the plot as an image
plt.show()
