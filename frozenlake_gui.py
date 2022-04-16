# frozenlake_gui
import numpy as np
import gym 
import random 
import argparse
import time
from IPython.display import clear_output
import pygame

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

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

# Common colors 
white = (194, 224, 249)
black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)

# Mapping out environment states to grid positions
state_to_position = {
    0: (10, 10),
    1: (110, 10),
    2: (210, 10),
    3: (310, 10),
    4: (10, 110),
    5: (110, 110),
    6: (210, 110),
    7: (310, 110),
    8: (10, 210),
    9: (110, 210),
    10: (210, 210),
    11: (310, 210),
    12: (10, 310),
    13: (110, 310),
    14: (210, 310),
    15: (310, 310)
}


def new_game_screen(screen):
    # Showing a blank grid with holes and goal as new game screen
    screen.fill(white)
    
    # Gridline positions
    vertical_line_positions = [
        [(100, 0), (100,400)],
        [(200, 0), (200,400)],
        [(300, 0), (300,400)]
        ]
    horizontal_line_positions = [
        [(0, 100), (400,100)],
        [(0, 200), (400,200)],
        [(0, 300), (400,300)]
        ]
    
    # Draw grid
    for i in range(len(vertical_line_positions)):
        pygame.draw.line(screen, black, vertical_line_positions[i][0], vertical_line_positions[i][1])
        pygame.draw.line(screen, black, horizontal_line_positions[i][0], horizontal_line_positions[i][1])
    
    # Draw hole positions
    pygame.draw.circle(screen, blue, (150, 150), 45)
    pygame.draw.circle(screen, blue, (350, 150), 45)
    pygame.draw.circle(screen, blue, (350, 250), 45)
    pygame.draw.circle(screen, blue, (50, 350), 45)

    # Draw goal position
    pygame.draw.circle(screen, green, (350, 350), 45)

    # Refresh screen
    pygame.display.flip()


# Initiate GUI 
pygame.init()
pygame.display.set_caption('FrozenLake-V1')

# Set up drawing window
screen = pygame.display.set_mode([500, 500])

# Load gym environment
env = gym.make("FrozenLake-v1")

# Define Qtable dimensions (rows,cols) -> (states,actions)
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

# Q Learning algorithm parameters
num_episodes = 10000  # games the agent will play
max_steps_per_episode = 1000  # limit in the moves the agent can do per game
learning_rate = parameters.learning_rate 
discount_rate = parameters.discount_rate

# Exploration parameters for random moves
exploration_rate = parameters.exploration_rate 
max_exploration_rate = parameters.max_exploration_rate
min_exploration_rate = parameters.min_exploration_rate
exploration_decay_rate = parameters.exploration_rate_decay

# Gather rewards for every episode
rewards_all_episodes = []

# Create pygame window for GUI
screen = pygame.display.set_mode([400, 400])

# Use time breaks
watch = True  

# Q Learning algorithm
for episode in range(num_episodes):
    # Back to square 1
    state = env.reset()
    done = False
    rewards_current_episode = 0
    if watch:
        time.sleep(0.3)

    # Create a new play window and draw agent at start position
    pygame.event.get()
    new_game_screen(screen)
    agent = pygame.Rect(10, 10, 80, 80)
    pygame.draw.rect(screen, red, agent)
    pygame.display.flip()

    for step in range(max_steps_per_episode):
        print("***** Episode:", episode+1, ", Step:", step+1, "*****")
        if watch:
            clear_output(wait=True)
            env.render()
            time.sleep(0.5)
        # Pick random threshold to stop exploring
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate < exploration_rate_threshold:
            # Take a probably smart action 
            action = np.argmax(q_table[state,:])
        else: 
            # Take a random action to explore
            action = env.action_space.sample()

        # Get keyboard input to change speed
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                watch = not watch

        # Erase agent from previous position
        pygame.draw.rect(screen, white, agent)

        # Send the action chosen to the environment and get reward
        new_state, reward, done, info = env.step(action)

        # Display agent position on screen
        agent = pygame.Rect(state_to_position[new_state][0], state_to_position[new_state][1], 80, 80)
        pygame.draw.rect(screen, red, agent)
        pygame.display.flip()

        # Update Q-table for Q(s,a) using the Q-Learning formula
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Move to the next state and add the reward gotten
        state = new_state
        rewards_current_episode += reward

        # Break if fail or goal were reached 
        if done:
            pygame.event.get()
            if reward == 1:
                screen.fill(green)
            else:
                screen.fill(red)
            if watch:
                    clear_output(wait=True)
                    env.render()
                    time.sleep(1)
            pygame.display.flip()
            break
        
    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    # Save total reward from episode
    rewards_all_episodes.append(rewards_current_episode)

# Close gym environment
env.close()

# Close pygame window
pygame.quit()

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
with open("qtable.txt", "w+") as file:
    file.write(q_table)


#####################################################################################################################


# code for moving by action instead of state

        # if action == 0:  # Move LEFT
        #     if agent.centerx-100 > 0:
        #         pygame.draw.rect(screen, white, agent)
        #         agent = pygame.Rect.move(agent, -100, 0)
        #         pygame.draw.rect(screen, red, agent)
        # elif action == 1:  # Move DOWN
        #     if agent.centery+100 < 400:
        #         pygame.draw.rect(screen, white, agent)
        #         agent = pygame.Rect.move(agent, 0, 100)
        #         pygame.draw.rect(screen, red, agent)
        # elif action == 2:  # Move RIGHT
        #     if agent.centerx+100 < 400:
        #         pygame.draw.rect(screen, white, agent)
        #         agent = pygame.Rect.move(agent, 100, 0)
        #         pygame.draw.rect(screen, red, agent)
        # elif action == 3:  # Move UP
        #     if agent.centery-100 > 0:
        #         pygame.draw.rect(screen, white, agent)
        #         agent = pygame.Rect.move(agent, 0, -100)
        #         pygame.draw.rect(screen, red, agent)
            
        # pygame.display.flip() 