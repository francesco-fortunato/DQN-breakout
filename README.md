# Deep Q-Network (DQN) for Breakout Game

Welcome to the DQN-Breakout repository! This project implements a Deep Q-Network (DQN) to play the classic Breakout game using reinforcement learning. The DQN learns to navigate the game environment, maximizing its cumulative rewards over time.

<div align="center">
  <img src="https://github.com/francesco-fortunato/DQN-breakout/blob/main/recordings/ATARI_Breakout_Eval_model_21700_reward_357-speed.gif" alt="Breakout Agent" width="200">
</div>

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Process](#training-process)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The Breakout game is a classic arcade game where the player controls a paddle to bounce a ball and break bricks. This project leverages the DQN algorithm to train an agent capable of playing Breakout autonomously. The DQN learns a policy to maximize the cumulative rewards, achieving impressive performance over time.

## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:

- Python 3.6 or later
- TensorFlow
- OpenAI Gym
- Matplotlib
- Other dependencies (check `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/DQN-breakout.git
   ```

2. Navigate to the project directory:

    ```bash
     cd DQN-breakout
     ```

4. Install dependencies:

    ```bash
     pip install -r requirements.txt
     ```

### Usage

To train the DQN agent for Breakout, run the following command:
```bash
python breakout.py
```
This will initiate the training process, and the agent will learn to play Breakout. You can customize various parameters in the script, such as epsilon values, preprocessing, learning rate and more.

## Project Structure

The organization of this project is as follows:

- **`breakout.py`**: This script serves as the main implementation of the DQN training for the Breakout game.

- **`saved_models/`**: This directory stores the saved models generated during the training process.

- **`experience replay/`**: In this directory, you can find the saved experience replay data, crucial for reinforcement learning.

- **`training_stats.csv`**: A CSV file where training statistics are logged, including episode rewards, epsilon values, average rewards, and more.

- **`requirements.txt`**: This file lists all the dependencies required for a smooth installation process.

### Training Process

The training process comprises several key steps:

1. **Environment Setup**: Initialize the Breakout game environment and set up essential parameters.

2. **Q-Network Implementation**: Implement the Deep Q-Network (DQN) using TensorFlow and Keras.

3. **Preprocessing**: Transform raw frames into a suitable format for the DQN using a preprocessing function.

4. **Training Loop**: Execute the primary training loop, involving an exploration-exploitation strategy, experience replay, and model updates.

5. **Model Saving**: Periodically save both the main model and the target model during training.

6. **Experience Replay Saving**: Save experience replay data regularly for potential future use.

7. **Training Statistics Logging**: Log various training statistics, including episode rewards, epsilon values, average rewards, and other relevant metrics.

8. **Training Completion Check**: Check for completion criteria, such as reaching a specific running reward, and print relevant information.

9. **Save Final Models and Experience Replay**: Save the final trained models and the experience replay data before concluding the training process.

### Results

The trained DQN agent demonstrates its proficiency in playing the Breakout game effectively. Detailed training statistics are logged in the `training_stats.csv` file, providing insights into the agent's performance throughout the training.

### Contributing

Contributions from the community are highly welcome. Whether you've identified a bug or have an enhancement in mind, feel free to open an issue or submit a pull request. Let's collaborate to enhance DQN-Breakout!

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.

### Acknowledgments
