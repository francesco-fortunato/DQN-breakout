import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load the CSV file
csv_filename = "evaluation_stats.csv"
df = pd.read_csv(csv_filename)

# Find the maximum total reward and its corresponding episode
max_avg_reward = df['Avg Reward'].max()
max_reward_frame_avg = df.loc[df['Avg Reward'].idxmax(), 'Frame']

max_reward = df['Max Reward'].max()
max_reward_frame_total = df.loc[df['Max Reward'].idxmax(), 'Frame']

# Plot the rewards and rolling average against Frame
plt.figure(figsize=(10, 6))
plt.plot(df['Frame'], df['Max Reward'], linestyle='--', label='Max Reward', color='b', alpha = 0.2)
plt.plot(df['Frame'], df['Avg Reward'], linestyle='-',  label=f'Avg (episodes)')

# Use axhline for horizontal lines
plt.axhline(y=max_avg_reward, color='c', alpha=0.5, linestyle='-', label=f'Max Avg Reward ({max_avg_reward:.2f} at Frame {int(max_reward_frame_avg):,})')
plt.axhline(y=max_reward, color='purple', alpha=0.5, linestyle='-', label=f'Max Reward ({max_reward:.2f} at Frame {int(max_reward_frame_total):,})')

local_reward_line = plt.gca().lines[0]
local_reward_line.set_dashes([2, 8])  # Adjust the numbers in the list to control the dash pattern

# Set y-axis ticks to integers every two integers
plt.yticks(range(int(min(df['Max Reward'])), int(max(df['Max Reward']))+1, 20))

# Format x-axis ticks using FuncFormatter for readability
def format_ticks(value, pos):
    if value >= 1e6:
        return f'{value * 1e-6:.1f}M'
    elif value >= 1e3:
        return f'{value * 1e-3:.0f}K'
    else:
        return f'{value:.0f}'

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))

plt.xlabel('Frame')
plt.ylabel('Clipped Reward')
plt.legend()
plt.title('Max Reward and Average Reward Evaluation')
plt.grid(False)
plt.savefig('reward_plot_evaluation_reward.png')  # Save the plot as an image
plt.show()


plt.figure(figsize=(10, 6))

# Plot Avg Q-value against Frame
plt.plot(df['Frame'], df['Avg Q-value'], linestyle='-', label=f'Avg Q-value')
# Set y-axis ticks to integers every two integers
# plt.yticks(range((min(df['Avg Q-value'])), (max(df['Avg Q-value']))+1, 0.2))

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))

plt.xlabel('Frame')
plt.ylabel('Avg Q-Value')
plt.legend()
plt.title('Avg Q-value')
plt.grid(False)
plt.savefig('reward_plot_evaluation_q_values.png')  # Save the plot as an image
plt.show()
