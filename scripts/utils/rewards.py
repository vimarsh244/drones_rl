import math
import numpy as np


def repulsion(d, d0=1.5):
    if d <= d0:
        return 0.5 * (((1/d) - (1/d0))**2) + 1
    return 1


# def calculate_gamma(M_e, unexplored_mask):
#     """ Calculate the gamma function for the matrix M_e. """
#     gamma = np.zeros_like(M_e)
#     for i in range(M_e.shape[0]):
#         for j in range(M_e.shape[1]):
#             if unexplored_mask[i, j]:
#                 gamma[i, j] = np.sum(1 / (M_e + 1e-6))  # Small value to avoid division by zero
#             else:
#                 gamma[i, j] = 0
#     return gamma


def calculate_entropy(occupancy_grid):
    """ Calculate the entropy of each cell in the occupancy grid. """
    epsilon = 1e-9  # Small value to prevent log(0)
    entropy_grid = -occupancy_grid * np.log(occupancy_grid + epsilon) - (1 - occupancy_grid) * np.log(1 - occupancy_grid + epsilon)
    return entropy_grid


def entropy_reward(occupancy_grid):
    """ Calculate the reward based on the entropy of the occupancy grid. """
    entropy_grid = calculate_entropy(occupancy_grid)
    unexplored_entropy = entropy_grid[occupancy_grid == 0.5]  # Unexplored cells have an initial probability of 0.5

    if unexplored_entropy.size == 0:  # If there are no unexplored cells, return a small reward
        return 0.0

    max_entropy = np.max(unexplored_entropy)
    sum_entropy = np.sum(unexplored_entropy)
    reward = max_entropy / (1 + sum_entropy)
    
    # Debug: Print intermediate values
    # print(f"Reward: {reward*30000}")
    
    return reward*30000


def compute_rewards(state, occupancy_grid, ang_vel):
    reward = 0
    for d in state:
        reward += -0.075*(repulsion(d))

    #reward += (explored * 10)
    ent_rew = entropy_reward(occupancy_grid)
    # print(ent_rew)
    reward += ent_rew

    spinnage_rew = 1/(1+(35*ang_vel*ang_vel))
    reward += spinnage_rew

    return reward
    
