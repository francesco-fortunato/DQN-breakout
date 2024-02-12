import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load the first CSV file
csv_filename = "training_stats.csv"
df1 = pd.read_csv(csv_filename)

# Load the second CSV file
csv_filename_2 = "training_stats_0.0001.csv"
df2 = pd.read_csv(csv_filename_2)

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
df1 = pd.DataFrame(modified_data)

# Recalculate the rolling average for the modified dataframe
df1['Rolling_Avg_Reward'] = df1['Total Reward'].rolling(window=100).mean()

# Find the minimum of the last frame number between the two dataframes
min_last_frame = min(df1['Total Frames'].iloc[-1], df2['Total Frames'].iloc[-1])
print(min_last_frame)
# Filter both dataframes based on the minimum last frame
df1 = df1[df1['Total Frames'] <= min_last_frame]
df2 = df2[df2['Total Frames'] <= min_last_frame]

# Calculate the rolling average for both dataframes
rolling_average_window = 100  # You can adjust this window size
df1['Rolling_Avg_Reward'] = df1['Total Reward'].rolling(window=rolling_average_window).mean()
df2['Rolling_Avg_Reward'] = df2['Total Reward'].rolling(window=rolling_average_window).mean()

# Find the maximum total reward and its corresponding episode for both dataframes
max_avg_reward_df1 = df1['Rolling_Avg_Reward'].max()
max_reward_frame_avg_df1 = df1.loc[df1['Rolling_Avg_Reward'].idxmax(), 'Total Frames']
max_reward_df1 = df1['Total Reward'].max()
max_reward_frame_total_df1 = df1.loc[df1['Total Reward'].idxmax(), 'Total Frames']

max_avg_reward_df2 = df2['Rolling_Avg_Reward'].max()
max_reward_frame_avg_df2 = df2.loc[df2['Rolling_Avg_Reward'].idxmax(), 'Total Frames']
max_reward_df2 = df2['Total Reward'].max()
max_reward_frame_total_df2 = df2.loc[df2['Total Reward'].idxmax(), 'Total Frames']

# Plot the rewards and rolling average against Total Frames for both dataframes
plt.figure(figsize=(10, 6))
plt.plot(df1['Total Frames'], df1['Rolling_Avg_Reward'], label=f'Rolling Avg ((Frame Stacked), {rolling_average_window} episodes)')

plt.plot(df2['Total Frames'], df2['Rolling_Avg_Reward'], label=f'Rolling Avg ((Frame Not Stacked), {rolling_average_window} episodes)')

plt.plot(df1['Total Frames'], df1['Total Reward'], linestyle='--', label='(Frame Stacked) Total Reward', color='b', alpha = 0.2)
plt.plot(df2['Total Frames'], df2['Total Reward'], linestyle='--', label='(Frame Not Stacked) Total Reward', color='y', alpha = 0.2)


# Use axhline for horizontal lines
plt.axhline(y=max_avg_reward_df1, color='c', alpha=0.5, linestyle='-', label=f'Max Avg Reward ((Frame Stacked), {max_avg_reward_df1:.2f} at Frame {int(max_reward_frame_avg_df1):,})')
plt.axhline(y=max_reward_df1, color='purple', alpha=0.5, linestyle='-', label=f'Max Reward ((Frame Stacked), {max_reward_df1:.2f} at Frame {int(max_reward_frame_total_df1):,})')

plt.axhline(y=max_avg_reward_df2, color='orange', alpha=0.5, linestyle='-', label=f'Max Avg Reward ((Frame Not Stacked), {max_avg_reward_df2:.2f} at Frame {int(max_reward_frame_avg_df2):,})')
plt.axhline(y=max_reward_df2, color='red', alpha=0.5, linestyle='-', label=f'Max Reward ((Frame Not Stacked), {max_reward_df2:.2f} at Frame {int(max_reward_frame_total_df2):,})')

# Set y-axis ticks to integers every two integers
plt.yticks(range(int(min(df1['Total Reward'].min(), df2['Total Reward'].min())), int(max(df1['Total Reward'].max(), df2['Total Reward'].max()))+1, 5))

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
plt.ylabel('Reward')
plt.legend(fontsize='small', loc='best')  # Set the font size of the legend to 'small'
plt.title('Total Reward and Rolling Average vs Total Frames for Frame Stacked and not')
plt.grid(False)
plt.savefig('reward_plot_dual.png')  # Save the plot as an image
plt.show()
