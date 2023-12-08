import imageio
import numpy as np
import cv2
import tensorflow as tf
import gym
import os

# Configuration parameters
seed = 42
num_actions = 4

# Use the Baseline Atari environment for testing
env = gym.make("ALE/Breakout-v5", render_mode="human")
env.seed(seed)

# Get the current working directory
current_directory = os.getcwd()

# Path to the saved model
model_filename = "saved_models/model_episode_3600"

# Create the absolute path to the model file
absolute_model_path = os.path.join(current_directory, model_filename)

# Load the pre-trained model
loaded_model = tf.keras.models.load_model(absolute_model_path)

# Function to generate GIF from raw frames
def generate_gif(frame_number, frames_for_gif, reward, path):
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1/30)

# Function to preprocess a frame for input to the model
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Downsample and crop to 84x84
    cropped_frame = gray[34:34+160, :160]
    resized_frame = cv2.resize(cropped_frame, (84, 84))

    # Normalize brightness to [0, 1]
    normalized_frame = resized_frame / 255.0

    normalized_frame = np.expand_dims(normalized_frame, axis=0)

    return normalized_frame

# Function to choose an action based on the model's predictions
def choose_action(model, state):
    action_probs = model.predict(state)
    return np.argmax(action_probs)

# Initialize variables
episode_count = 0
episode_reward = 0

# Test the model in the environment
state = np.array(env.reset())
frames_for_gif = []  # List to store raw RGB frames for GIF generation

while True:
    # Capture the raw RGB frame for GIF generation
    frames_for_gif.append(np.array(state))

    action = choose_action(loaded_model, preprocess_frame(state))
    state_next, reward, done, _ = env.step(action)
    state = np.array(state_next)

    episode_reward += reward

    if done:
        # Generate GIF at the end of the episode using raw RGB frames
        generate_gif(episode_count, frames_for_gif, episode_reward, 'recordings/')
        frames_for_gif = []  # Reset the list for the next episode
        episode_count += 1
        episode_reward = 0
        state = np.array(env.reset())
