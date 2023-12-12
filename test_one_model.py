import imageio
import numpy as np
import cv2
import tensorflow as tf
import gym
import os
import pandas as pd
from breakout_wrapper import make_atari_breakout, wrap

# Configuration parameters
seed = 42
num_actions = 4

# Use the Baseline Atari environment for testing
env = make_atari_breakout("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stack four frames, and scale to a smaller ratio
env = wrap(env, frame_stack=True, scale=True, clip_rewards=False, episode_life=False)
env.seed(seed)

# Get the current working directory
current_directory = os.getcwd()

episode_count = 22000
max_episodes = 10  # Set the desired number of episodes

frame_count = 0

# Load training stats CSV
training_stats_file = "training_stats.csv"  # Replace with the actual filename
training_stats_df = pd.read_csv(training_stats_file)

# Path to the saved model
model_filename = "saved_models/model_episode_{}".format(episode_count)

# Create the absolute path to the model file
absolute_model_path = os.path.join(current_directory, model_filename)

# Load the pre-trained model
loaded_model = tf.keras.models.load_model(absolute_model_path)

# Function to generate GIF from raw frames
def generate_gif(frame_number, frames_for_gif, reward, path, ep):
    imageio.mimsave(f'{path}{"ATARI_Breakout_Eval_model_{0}_reward_{1}.gif".format(ep, int(reward))}',
                    frames_for_gif, duration=1/30)
    print(f'Gif saved at {path}{"ATARI_Breakout_Eval_model_{0}_reward_{1}.gif".format(ep, int(reward))}')

# Function to choose an action based on the model's predictions with epsilon-greedy exploration
def choose_action(model, state):
    # Exploit: choose the action with the highest predicted value
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    # Take the best action
    action = tf.argmax(action_probs[0]).numpy()
    return action

# Initialize variables
highest_reward = float('-inf')  # Variable to track the highest reward
frames_highest_reward = []  # List to store frames associated with the highest reward

# Test the model in the environment
frames_for_gif = []  # List to store raw RGB frames for GIF generation
current_lives = 5
restart = True
episode_reward = 0
episode_counter = 0

# Lists to store rewards for calculating the average over 10 episodes
rewards_for_average = []

while episode_counter < max_episodes:
    frame_count += 1
    if restart:
        restart = False
        state = np.array(env.reset())
        state_next, reward, done, info = env.step(1)  # Play Fire Action
        state = np.array(state_next)

    # Capture the raw RGB frame for GIF generation
    raw_frame = env.render(mode='rgb_array')
    frames_for_gif.append(raw_frame)

    # Comment this and uncomment the other to play yourself
    action = choose_action(loaded_model, state)

    state_next, reward, done, info = env.step(action)
    state = np.array(state_next)

    num_lives = info['lives']
    print(info)
    print(done)
    print(num_lives)

    if num_lives < current_lives:
        state_next, reward, done, info = env.step(1)
        state = np.array(state_next)
        current_lives = num_lives
        print(num_lives)
    episode_reward += reward

    if done:
        # Store the maximum reward and frames with the highest rewards
        if episode_reward > highest_reward:
            highest_reward = episode_reward
            frames_highest_reward = frames_for_gif.copy()

        # Store rewards for calculating the average over 10 episodes
        rewards_for_average.append(episode_reward)

        # Get the corresponding frame count from training stats
        episode_row = training_stats_df[training_stats_df['Episode'] == episode_count]
        if not episode_row.empty:
            frame_count_training = episode_row['Total Frames'].values[0]
            # Store episode_count, frame_count, avg_reward, max_reward, avg_q_value in your evaluation CSV

        # Reset variables for the next episode
        frames_for_gif = []
        episode_reward = 0
        state = np.array(env.reset())
        env.step(1)
        current_lives = 5
        episode_reward = 0
        restart = True
        episode_counter += 1

# Optionally, generate GIF for the episode with the highest reward
generate_gif(0, frames_highest_reward, highest_reward, 'recordings/', episode_count)

# Print individual rewards for each of the 10 episodes
print("Individual Rewards for Each Episode:", rewards_for_average)

# Print the average reward over 10 episodes
avg_reward_over_10_episodes = np.mean(rewards_for_average)
print(f"Average Reward Over 10 Episodes: {avg_reward_over_10_episodes}")
