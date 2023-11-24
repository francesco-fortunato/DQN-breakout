import gym
import numpy as np
import random
from collections import deque
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import logging
import time

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple feedforward neural network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Breakout-v5 environment
env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=(2,3))

# Define hyperparameters
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
gamma = 0.99  # Discount factor
batch_size = 32
memory = deque(maxlen=1000000)
num_episodes = 1000
target_update_frequency = 10
learning_rate = 0.00025

# Initialize the Q-networks (target and online)
input_size = 210 * 160
output_size = 4  # Action space size
online_net = QNetwork(input_size, output_size)
target_net = QNetwork(input_size, output_size)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)

# Initialize lists for logging
episode_rewards = []
avg_rewards = []

# Training loop
for episode in range(num_episodes):
    start_time = time.time()  # Record the start time for the episode

    state = env.reset()
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = torch.tensor(state, dtype=torch.float32).view(-1)

    done = False
    total_reward = 0

    episode_q_values = []  # List to store Q-values for this episode


    while not done:
        if random.random() < epsilon:
            
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = online_net(state)
                action = torch.argmax(q_values).item()



        next_state, reward, done, _ = env.step(action)
        next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(-1)

        # Store the experience in replay memory
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            q_values = online_net(states)
            next_q_values = target_net(next_states)

            targets = q_values.clone()
            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + gamma * torch.max(next_q_values[i])

            optimizer.zero_grad()
            loss = criterion(q_values, targets)
            loss.backward()
            optimizer.step()

    # Create or open a log.txt file for writing
    with open("log.txt", "a") as log_file:

        episode_rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Update target network
        if episode % target_update_frequency == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Log and print results
        episode_rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Update target network
        if episode % target_update_frequency == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Calculate episode duration and log results
        episode_duration = time.time() - start_time
        avg_reward = np.mean(episode_rewards[-10:])
        avg_rewards.append(avg_reward)

        avg_q_value = np.mean(episode_q_values)


        # Create a string with episode-specific information
        episode_info = (f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Last 10 Avg Reward: {avg_reward}, "
                        f"Duration: {episode_duration:.2f} seconds\n")

        # Write the episode information to the log file
        log_file.write(episode_info)

# Plot the rewards
plt.plot(episode_rewards, label="Episode Reward")
plt.plot(avg_rewards, label="Avg Reward (Last 10 Episodes)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()

# Save the trained model
MODEL_FILE = "./dqn_model"
torch.save(online_net.state_dict(), MODEL_FILE)
