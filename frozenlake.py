# frozenlake
import numpy as np
import gym 
import random 
import argparse


# Get model parameters 
parser = argparse.ArgumentParser()
parameters = parser.add_argument_group("Parameters")
parameters.add_argument('-lr', '--learning_rate', required=False, default=0.1)
parameters.add_argument('-dr', '--discount_rate', required=False, default=0.99)
parameters.add_argument('-er', '--exploration_rate', required=False, default=1)
parameters.add_argument('-maxer', '--max_exploration_rate', required=False, default=1)
parameters.add_argument('-miner', '--min_exploration_rate', required=False, default=0.01)
parameters.add_argument('-erd', '--exploration_rate_decay', required=False, default=0.001)
parameters = parser.parse_args()

# Load gym environment
# Crear el entorno desde gym
env = gym.make("FrozenLake-v1")

#INICIO Agregado KIM ----------------------------------------------------------
#Estado inicial
env.reset()          
#Imprimir estado          
env.render()
print("Action space: ", env.action_space) #posibles acciones a realizar en el entorno
print("Observation space: ", env.observation_space) #posibles estados del juego
#FIN Agregado KIM -------------------------------------------------------------


# Define Qtable dimensions (rows,cols) -> (states,actions)
action_space_size = env.action_space.n 
state_space_size = env.observation_space.n 

q_table = np.zeros((state_space_size, action_space_size))

# Q Learning algorithm parameters
num_episodes = 10000  # games the agent will play
max_steps_per_episode = 100  # limit in the moves the agent can do per game
learning_rate = parameters.learning_rate 
discount_rate = parameters.discount_rate

# Exploration parameters for random moves
exploration_rate = parameters.exploration_rate 
max_exploration_rate = parameters.max_exploration_rate
min_exploration_rate = parameters.min_exploration_rate
exploration_decay_rate = parameters.exploration_rate_decay

# Gather rewards for every episode
rewards_all_episodes = []

# Q Learning algorithm
for episode in range(num_episodes):
    # Back to square 1
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        # Pick random threshold to stop exploring
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate < exploration_rate_threshold:
            # Take a probably smart action 
            action = np.argmax(q_table[state,:])
        else: 
            # Take a random action to explore
            action = env.action_space.sample()
        # Send the action chosen to the environment and get reward
        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s,a) using the Q-Learning formula
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Move to the next state and add the reward gotten
        state = new_state
        rewards_current_episode += reward

        # Break if fail or goal were reached 
        if done == True:
            break
        
    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    # Save total reward from episode
    rewards_all_episodes.append(rewards_current_episode)

# Show results

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
final_reward_averages = []
for r in rewards_per_thousand_episodes:
    average = sum(r/1000)
    print(count, ": ", str(average))
    count += 1000
    final_reward_averages.append(average)

# Print Q-table
print(q_table)

# Save best and last reward average
# Save best and last reward average
with open("qtable.txt", "w+") as file:
    file.write(np.array_str(q_table))