from breakout_wrapper import make_atari_breakout, wrap
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import time
import pickle
import gzip
import os
import signal
import sys
import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
## Signal Handler
"""
# Flag to indicate if a termination signal is received
terminate_flag = False

# Signal handler function
def handle_sigint(signum, frame):
    global terminate_flag
    # if (terminate_flag):
            # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # print(f"{current_time} - Force arresting. . .")
            # sys.exit(0)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - Received Ctrl+C. Waiting for episode termination. . .")
    terminate_flag = True

# Set up the signal handler
signal.signal(signal.SIGINT, handle_sigint)

"""
## Usage

python breakout.py <upload_ep>
"""

if len(sys.argv) > 2:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Usage: python breakout.py <upload_ep>. Exit. . .")

# Check if a command-line argument is provided for upload_ep
if len(sys.argv) > 1:
    try:
        # Attempt to convert the command-line argument to an integer
        upload_ep = int(sys.argv[1])
    except ValueError:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Invalid value for upload_ep. Usage: python breakout.py <upload_ep>. Exit. . .")
else:
    # Ask the user if they want to start from 0
    response = input("Do you want to start from episode 0? (y/n): ").lower()
    
    if response == "y":
        upload_ep = 0
    elif response == "n":
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Please provide a valid value for upload_ep.")
        print(f"{current_time} - Usage: python breakout.py <upload_ep>. Exit. . .")
        sys.exit(1)
    else:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
        print(f"{current_time} - Invalid response. Exit. . .")
        sys.exit(1)

"""
## Implement the Deep Q-Network

This network learns an approximation of the Q-table, which is a mapping between
the states and actions that an agent will take. For every state, we'll have four
actions that can be taken. The environment provides the state, and the action
is chosen by selecting the larger of the four Q-values predicted in the output layer.

The network architecture is designed based on the Deepmind paper and specifically
tailored for training on Atari Breakout. Convolutional layers are used to capture
spatial dependencies and patterns in the game frames. This is crucial for Atari
Breakout because it involves complex visual information, and convolutional layers
are effective in learning hierarchical features.

A dense neural layer, also known as a fully connected layer, wouldn't work as 
effectively for processing Atari Breakout frames because it doesn't consider the 
spatial relationships and hierarchical features present in the visual input. 
In the case of Atari Breakout, the game frames contain important spatial 
information that contributes to the understanding of the game state.

Here are a few reasons why a dense layer might not work well for processing 
Atari Breakout frames:

Loss of Spatial Information: Dense layers treat each input neuron as independent, 
disregarding the spatial arrangement of pixels. This can result in a loss of 
important spatial information present in the game frames.

High-Dimensional Input: Atari Breakout frames are high-dimensional images with 
important spatial structures. Dense layers are not designed to handle the spatial
dependencies present in such images.

Inability to Capture Local Patterns: Convolutional layers, on the other hand, 
are specifically designed to recognize local patterns and spatial hierarchies in images. 
They use weight sharing and local receptive fields to capture features at different 
spatial scales.

Computational Efficiency: Dense layers have a large number of parameters, and 
training a dense network on high-resolution images like those in Atari Breakout 
would require a massive amount of computational resources. Convolutional layers, 
by sharing weights, are more computationally efficient.

The architecture consists of three convolutional layers followed by a fully
connected layer and an output layer. Each convolutional layer uses the ReLU
activation function, and the weights are initialized using the He initialization
method with a scale factor of 2.0. The output layer has a linear activation function,
and the number of neurons corresponds to the number of actions available in the
environment (4 for the Breakout game).

Model Architecture:
- Convolutional Layer 1: 32 filters, 8x8 size, stride of 4
- Convolutional Layer 2: 64 filters, 4x4 size, stride of 2
- Convolutional Layer 3: 64 filters, each 3x3 in size, stride of 1
- Fully Connected Layer: 512 neurons with ReLU activation
- Output Layer: num_actions neurons with linear activation
"""

num_actions = 4

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Define an initializer using the He initialization method with a scale factor of 2.0
    initializer = tf.keras.initializers.variance_scaling(scale=2.0)

    # Convolutions on the frames on the screen

    # Define the first convolutional layer
    # - 32 filters, each 8x8 in size
    # - Stride of 4, meaning the filter moves 4 pixels at a time
    # - ReLU activation function is applied to the output
    # - Use the previously defined initializer for setting up weights
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu", kernel_initializer=initializer)(inputs)

    # Define the second convolutional layer
    # - 64 filters, each 4x4 in size
    # - Stride of 2
    # - ReLU activation function
    # - Use the same initializer for setting up weights
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", kernel_initializer=initializer)(layer1)

    # Define the third convolutional layer
    # - 64 filters, each 3x3 in size
    # - Stride of 1
    # - ReLU activation function
    # - Use the same initializer for setting up weights
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", kernel_initializer=initializer)(layer2)

    # Flatten the output from the convolutional layers
    layer4 = layers.Flatten()(layer3)

    # Define a fully connected layer with 512 neurons
    # - ReLU activation function
    # - Use the same initializer for setting up weights
    layer5 = layers.Dense(512, activation="relu", kernel_initializer=initializer)(layer4)

    # Output layer with num_actions neurons (4 in this case for the Breakout game)
    # - Linear activation function
    # - Use the same initializer for setting up weights
    action = layers.Dense(num_actions, activation="linear", kernel_initializer=initializer)(layer5)


    return keras.Model(inputs=inputs, outputs=action)

"""
## Setup
"""

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_final = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
epsilon_interval_2 = (
    epsilon_min - epsilon_final
)  # Rate at which to reduce chance of random action being taken after 1kk frames
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000 

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari_breakout("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap(env, frame_stack=True, scale=True)
env.seed(seed)

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

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
epsilon_random_frames = 50000.0   # Number of frames with epsilon set to 1.0
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0 # Number of frames to linearly decay epsilon from 1 to 0.1
epsilon_final_frames = 24000000.0 # Number of frames to linearly decay epsilon from 0.1 to 0.01
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

"""
## Restart from checkpoint (if any)
"""

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
              "Frame Rate", "Model Updates", "Running Reward", "Training Time"]
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# Check if the model file exists
model_filename = f"saved_models/model_episode_{upload_ep}"
model_target_filename = f"saved_models/target_model_episode_{upload_ep}"

if os.path.exists(model_filename):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Loading pre-trained models. . .")

    # Load the pre-trained model
    model = keras.models.load_model(model_filename)
    
    model_target = keras.models.load_model(model_target_filename)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Loaded pre-trained model from {model_filename}.")
    print(f"{current_time} - Loaded target model from {model_target_filename}.")
    
    # Load existing training statistics from the CSV file
    csv_filename = "training_stats.csv"
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)
            # Find the last row to resume training
            for row in reader:
                pass  # The last row will be stored in the 'row' variable
            episode_count, _, epsilon, _, frame_count, _, _, _, _ = map(float, row)
            episode_count = int(episode_count) + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
            print(f"{current_time} - Resuming training from episode {episode_count}, frame count {frame_count}, epsilon {epsilon}. . .")

            # Read the rewards from the last 100 rows
            file.seek(0)
            last_100_rows = list(reader)[-100:]
            rewards_last_100 = [float(row[1]) for row in last_100_rows if row[0] != 'Episode']
            
            # Update episode_reward_history
            episode_reward_history.extend(rewards_last_100)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Loading replay memory. . .")

    # Specify the output directory
    history_path = "experience replay"

    # Construct the load path
    load_path = os.path.join(history_path, f"history_episode_{upload_ep}.pkl.gz")

    # Create an empty dictionary to store loaded lists
    loaded_data_dict = {}

    # Use tqdm to display a progress bar while loading
    with gzip.GzipFile(load_path, 'rb') as file:
        for _ in tqdm(range(5), desc="Loading arrays", unit="array"):
            loaded_item = pickle.load(file)
            loaded_data_dict.update(loaded_item)
        file.close()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Experience replay loaded.")

    # Retrieve individual lists from the loaded dictionary
    action_history = loaded_data_dict["action_history"][-100000:]
    state_history = loaded_data_dict["state_history"][-100000:]
    state_next_history = loaded_data_dict["state_next_history"][-100000:]
    rewards_history = loaded_data_dict["rewards_history"][-100000:]
    done_history = loaded_data_dict["done_history"][-100000:]

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Cleaning cache. . .")

    del loaded_item
    del loaded_data_dict

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Cache cleaned.")


    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Starting training. . .")
            
    model.compile(optimizer=optimizer, loss=loss_function)
    model_target.compile(optimizer=optimizer, loss=loss_function)
else:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Model saved_models/model_episode_{upload_ep} not found. Starting training from zero. . .")

"""
## Train
"""

starting = datetime.datetime.now()
terminal_life_lost = False

while True:  # Run until solved
    start_time = time.time()
    state = np.array(env.reset())

    current_lives = 5
    
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
            # Take the best action
            action = tf.argmax(action_probs[0]).numpy()

        if frame_count > epsilon_random_frames: # Decay epsilon only after exploring for first 50k frames
            if epsilon > epsilon_min:
                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)
            else:
                # Continue decaying epsilon linearly over the remaining frames
                epsilon -= epsilon_interval_2 / (epsilon_final_frames)
                epsilon = max(epsilon, epsilon_final)


        # Apply the sampled action in our environment
        state_next, reward, done, info = env.step(action)
        state_next = np.array(state_next)
            
        episode_reward += reward

        # When a life is lost, we save terminal_life_lost = True in the replay memory
        # N.B. We don't modify directly done, since done is already used to break the loop
        num_lives = info['lives']

        if (num_lives < current_lives):
            terminal_life_lost = True
            current_lives = num_lives
        else:
            terminal_life_lost = False

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(terminal_life_lost if not done else done) # If the game is not terminated, if life lost add true, else add done (False or true)
        rewards_history.append(reward)
        state = state_next

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
            ) # turns True into 1.0 and False into 0.0.

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            #updated_q_values = rewards_sample + gamma * tf.reduce_max(
            #    future_rewards, axis=1
            #)

            # Correct Implementation
            # If the game is over because the agent lost or won, there is no next state and the value is simply the reward 

            updated_q_values = rewards_sample + (1- done_sample) * gamma * tf.reduce_max(future_rewards, axis=1)

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
            # print(info)
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
                            frame_count, frame_rate, len(done_history),
                            running_reward, training_time])
    
    if (episode_count%100 == 0):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        print(f"{current_time} - Episode {episode_count} reached. Saving model in saved_models/model_episode_{episode_count}. . .")
        model.save("saved_models/model_episode_{}".format(episode_count))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Model saved.")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Saving target model. . .")
        # Save the target model
        model_target.save("saved_models/target_model_episode_{}".format(episode_count))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Target model saved in saved_models/target_model_episode_{episode_count}.")

    if terminate_flag and current_lives == 0:
        break

    episode_count += 1
    if (num_lives==0):
        template = "running reward: {:.2f} at episode {}, frame count {}"
        print(template.format(running_reward, episode_count, frame_count))

    if running_reward > 40:  # 40 is the avg score of human beings
        print("Solved at episode {}!".format(episode_count))
        episode_count -= 1
        break


"""
## Save checkpoint
"""

end_time = datetime.datetime.now()
elapsed_time = end_time - starting

# Convert elapsed time to hours
elapsed_hours = elapsed_time.total_seconds() / 3600

current_time = datetime.datetime.now()

print(f"{current_time} - Stopped training. Total time trained: {elapsed_hours} hours")

if (episode_count%100 != 0):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Saving model. . .")

    # Save the model
    model.save("saved_models/model_episode_{}".format(episode_count))

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{current_time} - Model saved in saved_models/model_episode_{episode_count}.")

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"{current_time} - Saving target model. . .")

# Save the target model
model_target.save("saved_models/target_model_episode_{}".format(episode_count))

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"{current_time} - Target model saved in saved_models/target_model_episode_{episode_count}.")

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
print(f"{current_time} - Preparing to save experience replay. . .")

# Specify the output directory
output_path = "experience replay"

# Construct the save path
save_path = os.path.join(output_path, f"history_episode_{episode_count}.pkl.gz")

# Create a dictionary with the lists
data_dict = {
    "action_history": action_history,
    "state_history": state_history,
    "state_next_history": state_next_history,
    "rewards_history": rewards_history,
    "done_history": done_history
}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Print a formal message indicating the start of the process
print(f"{current_time} - Saving and compressing experience replay. . .")

# Use tqdm to display a progress bar while saving
with gzip.GzipFile(save_path, 'wb') as file:
    for key, value in tqdm(data_dict.items(), desc="Saving and compressing experience replay", unit="list"):
        pickle.dump({key: value}, file)
    file.close()    

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
print(f"{current_time} - Experience replay saved to {save_path}. Exiting. . .")

sys.exit(0)
