import os

import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

# Define the Breakout environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Constants
STATE_SHAPE = (210, 160, 3)
NUM_ACTIONS = 18
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000  # Update target network every X steps

# Initialize replay memory
replay_memory = deque(maxlen=MEMORY_SIZE)

# Define Q-Network
model = Sequential()
model.add(Dense(32, input_shape=STATE_SHAPE, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(NUM_ACTIONS, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Initialize target Q-Network with the same architecture
target_model = Sequential.from_config(model.get_config())
target_model.set_weights(model.get_weights())

# Initialize exploration rate
epsilon = EPSILON_START

# Training parameters
total_episodes = 10000  # Total number of episodes to train
max_steps_per_episode = 1000  # Maximum number of steps per episode

# Training loop
for episode in range(total_episodes):
    state = env.reset()
    state = np.reshape(state, (1, *STATE_SHAPE))
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Exploration vs. exploitation
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q_values = model.predict(state)
            action = np.argmax(Q_values)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, *STATE_SHAPE))

        total_reward += reward

        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))

        # Update state
        state = next_state

        # Sample a random minibatch from replay memory and perform gradient descent
        if len(replay_memory) >= BATCH_SIZE:
            minibatch = random.sample(replay_memory, BATCH_SIZE)

            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = np.vstack(states)
            next_states = np.vstack(next_states)

            target_Qs = model.predict(states)
            target_Qs_next = target_model.predict(next_states)

            for i in range(BATCH_SIZE):
                if dones[i]:
                    target_Qs[i, actions[i]] = rewards[i]
                else:
                    target_Qs[i, actions[i]] = rewards[i] + GAMMA * np.max(target_Qs_next[i])

            model.fit(states, target_Qs, epochs=1, verbose=0)

        # Update target network if needed
        if step % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())

        if done:
            break

    # Decay exploration rate
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Logging
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# Save the trained model
model.save("breakout_dqn_model.h5")
