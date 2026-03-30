"""
Q-Learning Algorithm Implementation
Reinforcement learning to find the optimal path to a goal state.
"""

import time
import sys

def typing_print(text, delay=0.01):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Loading"):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(3):
        time.sleep(0.3)
        sys.stdout.write(".")
        sys.stdout.flush()
    typing_print("\n")


import numpy as np
import random

# Reward matrix
R = np.array([
    [-1, -1, -1, -1,  0, -1],
    [-1, -1, -1,  0, -1, 100],
    [-1, -1, -1,  0, -1, -1],
    [-1,  0,  0, -1,  0, -1],
    [ 0, -1, -1,  0, -1, 100],
    [-1,  0, -1, -1,  0, 100]
])

Q = np.zeros((6, 6))
gamma = 0.8
goal_state = 5
num_episodes = 1000

# Training
for episode in range(num_episodes):
    state = random.randint(0, 5)
    while state != goal_state:
        valid_actions = np.where(R[state] != -1)[0]
        action = random.choice(valid_actions)
        next_state = action
        Q[state, action] = R[state, action] + gamma * np.max(Q[next_state])
        state = next_state

typing_print("Q-table after training:\n")
print(Q.astype(int))

# Path finder
def optimal_path(start):
    state = start
    path = [int(state)]
    while state != goal_state:
        next_state = int(np.argmax(Q[state]))
        path.append(next_state)
        state = next_state
    return path

typing_print("\nOptimal Paths:\n")
for i in range(6):
    print(f"From state {i}: {optimal_path(i)}")
