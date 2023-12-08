import gym
import os
import glob
import signal
import sys
import cv2
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""
## Signal Handler
"""
# Define a flag to indicate if a termination signal is received
terminate_flag = False

# Signal handler function
def handle_sigint(signum, frame):
    global terminate_flag
    if (terminate_flag):
            print("Force arresting. . .")
            sys.exit(0)
    print("Received Ctrl+C. Waiting for episode termination. . .")
    terminate_flag = True

# Set up the signal handler
signal.signal(signal.SIGINT, handle_sigint)

"""
## Setup
"""

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# Use the Baseline Atari environment because of Deepmind helper functions
env = gym.make("ALE/Breakout-v5")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env.seed(seed)

"""
## Implement the Deep Q-Network

This network learns an approximation of the Q-table, which is a mapping between
the states and actions that an agent will take. For every state we'll have four
actions, that can be taken. The environment provides the state, and the action
is chosen by selecting the larger of the four Q-values predicted in the output layer.

"""

num_actions = 4


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 1,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

"""
## Preprocessing
"""

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Downsample and crop to 84x84
    cropped_frame = gray[34:34+160, :160]
    resized_frame = cv2.resize(cropped_frame, (84, 84))

    # Normalize brightness to [0, 1]
    normalized_frame = resized_frame / 255.0

    return normalized_frame


"""
## Train
"""
# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

csv_filename = "training_stats.csv"

# Check if the CSV file exists
if os.path.exists(csv_filename):
    with open(csv_filename, mode='r') as file:
        # CSV file already exists, read the header
        reader = csv.reader(file)
        header = next(reader)
else:
    # CSV file does not exist, create and write the header
    header = ["Episode", "Total Reward", "Epsilon", "Avg Reward (Last 100)", "Total Frames",
              "Frame Rate", "Loss", "Model Updates", "Running Reward", "Training Time"]
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


"""
## Load model
"""
upload_ep = 3504

# Check if the model file exists
model_filename = f"saved_models/model_episode_{upload_ep}"

if os.path.exists(model_filename):
    print("Loading pre-trained model.")

    # Load the pre-trained model
    model = keras.models.load_model(model_filename)
    
    # Create a new instance for the target model
    model_target = create_q_model()
    
    # Load the weights of the target model from the pre-trained model
    model_target.set_weights(model.get_weights())
    print("Loaded pre-trained model.")
    
    # Load existing training statistics from the CSV file
    csv_filename = "training_stats.csv"
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)
            # Find the last row to resume training
            for row in reader:
                pass  # The last row will be stored in the 'row' variable
            episode_count, _, epsilon, _, frame_count, _, _, _, _, _ = map(float, row)
            episode_count = int(episode_count) + 1
            print(f"Resuming training from episode {episode_count}, frame count {frame_count}, epsilon {epsilon}")

             # Read the rewards from the last 100 rows
            file.seek(0)
            last_100_rows = list(reader)[-100:]
            rewards_last_100 = [float(row[1]) for row in last_100_rows if row[0] != 'Episode']
            
            # Update episode_reward_history
            episode_reward_history.extend(rewards_last_100)
    
    print("Loading replay memory. . .")

    # List of file paths
    file_paths = [
        f"replay_memory/action_history_episode_{upload_ep}.npy",
        f"replay_memory/state_history_episode_{upload_ep}.npy",
        f"replay_memory/state_next_history_episode_{upload_ep}.npy",
        f"replay_memory/rewards_history_episode_{upload_ep}.npy",
        f"replay_memory/done_history_episode_{upload_ep}.npy"
    ]

    # Initialize lists to store loaded arrays
    loaded_arrays = []

    # Get the total size of the files
    total_size = sum(os.path.getsize(file_path) for file_path in file_paths)

    # Initialize the progress bar with a fixed total size
    with tqdm(total=total_size, desc="Loading replay memory", unit="B", unit_scale=True, colour='green') as pbar:
        for file_path in file_paths:
            loaded_array = np.load(file_path, mmap_mode='r').tolist()
            loaded_arrays.append(loaded_array)
            # Increment the progress bar by the size of the current file
            pbar.update(os.path.getsize(file_path))

    # Unpack the loaded arrays
    action_history, state_history, state_next_history, rewards_history, done_history = loaded_arrays

    del loaded_arrays
    del loaded_array

    print("Loaded replay memory.")
            
    model.compile(optimizer=optimizer, loss=loss_function)
    model_target.compile(optimizer=optimizer, loss=loss_function)
else:
    print(f"Model saved_models/model_episode_{upload_ep} not found. Starting training from zero. . .")


while True:  # Run until solved
    start_time = time.time()
    state = preprocess_frame(np.array(env.reset()))
    
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = preprocess_frame(np.array(state_next))

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        ## Show the cropped, grayscaled 4 frames in 1
        # if len(state_history) >= 4 and frame_count % 4 == 0:
        #     # Display the preprocessed frames in a 2x2 grid
        #     plt.imshow(state_history[-1].squeeze(), cmap='gray', alpha=0.25)  # Current frame
        #     plt.imshow(state_history[-2].squeeze(), cmap='gray', alpha=0.25)  # Previous frame
        #     plt.imshow(state_history[-3].squeeze(), cmap='gray', alpha=0.25)  # Frame before previous
        #     plt.imshow(state_history[-4].squeeze(), cmap='gray', alpha=0.25)  # Frame before before previous
        #     plt.show()

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # Calculate additional statistics
    avg_reward_last_100 = np.mean(episode_reward_history[-100:])
    frame_rate = frame_count / (time.time() - start_time)
    training_time = time.time() - start_time

    # Append the episode statistics to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode_count, episode_reward, epsilon, avg_reward_last_100,
                            frame_count, frame_rate, loss.numpy(), len(done_history),
                            running_reward, training_time])
    
    if (episode_count%100 == 0):
        # files = glob.glob('/replay_memory/*')
        # for f in files:
        #     os.remove(f)
        # np.save("replay_memory/action_history_episode_{}.npy".format(episode_count), np.array(action_history))
        # np.save("replay_memory/state_history_episode_{}.npy".format(episode_count), np.array(state_history))
        # np.save("replay_memory/state_next_history_episode_{}.npy".format(episode_count), np.array(state_next_history))
        # np.save("replay_memory/rewards_history_episode_{}.npy".format(episode_count), np.array(rewards_history))
        # np.save("replay_memory/done_history_episode_{}.npy".format(episode_count), np.array(done_history))
        print(f"Episode {episode_count} reached. Saving model. . .")
        model.save("saved_models/model_episode_{}".format(episode_count))
        
    if terminate_flag:
        break

    episode_count += 1
    
    # model = keras.models.load_model("model_episode_{}.h5".format(episode_count))
    # model.compile(optimizer=optimizer, loss=loss_function)

    template = "running reward: {:.2f} at episode {}, frame count {}"
    print(template.format(running_reward, episode_count, frame_count))

    if running_reward > 40:  # 40 is the avg score of human beings
        print("Solved at episode {}!".format(episode_count))
        break

print("Saving model and arrays...")

# Save the model
model.save("saved_models/model_episode_{}".format(episode_count))

chunk_size = 1000

# Convert list to np array
action_history = np.array(action_history)
state_history = np.array(state_history)
state_next_history = np.array(state_next_history)
rewards_history = np.array(rewards_history)
done_history = np.array(done_history)

files = glob.glob('replay_memory/*')
for f in files:
    os.remove(f) # delete old replay memory

# Save the history using np.save with tqdm progress bar
with tqdm(total=5, desc="Saving history", unit="file", colour='green') as pbar:
    for data, name in zip([action_history, state_history, state_next_history, rewards_history, done_history],
                          ["action_history", "state_history", "state_next_history", "rewards_history", "done_history"]):
        total_chunks = len(data) // chunk_size

        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            np.save(f"replay_memory/{name}_episode_{episode_count}_part_{i + 1}.npy", data[start_idx:end_idx])

        # Save the remaining data (if any)
        remaining_data = data[total_chunks * chunk_size:]
        np.save(f"replay_memory/{name}_episode_{episode_count}_part_{total_chunks + 1}.npy", remaining_data)

        pbar.update(1)

print("Model and arrays saved. Exiting...")

sys.exit(0)
