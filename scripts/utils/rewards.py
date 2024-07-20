import math
import numpy as np


def repulsion(d, d0=1.5):
    # # print(d)
    # d = float(d)
    # print(d)
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

def compute_rewards_lite(state, occupancy_grid, ang_vel):
    reward = 0
    
    if isinstance(state, dict) and 'lidar' in state:
        lidar_data = state['lidar']
    elif isinstance(state, (list, np.ndarray)):
        lidar_data = state
    else:
        print(f"Unexpected state format: {type(state)}")
        print(f"State content: {state}")
        return 0  # Return a default reward if state format is unexpected
    
    for d in lidar_data:
        try:
            reward += -0.075 * (repulsion(float(d)))
        except ValueError as e:
            print(f"Error processing lidar data: {e}")
            print(f"Problematic value: {d}")
            continue  # Skip this value and continue with the next

    ent_rew = entropy_reward(occupancy_grid)
    reward += ent_rew

    spinnage_rew = 1 / (1 + (35 * ang_vel * ang_vel))
    reward += spinnage_rew

    return reward


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
    
import torch
import torch.nn.functional as F

def compute_icm_reward(icm, current_state, next_state, action):
    """
    Compute the intrinsic reward using ICM.
    
    :param icm: The ICM model
    :param current_state: Current camera image
    :param next_state: Next camera image
    :param action: Action taken
    :return: Intrinsic reward
    """
    current_state = torch.FloatTensor(current_state).unsqueeze(0)
    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    action = torch.FloatTensor(action).unsqueeze(0)
    
    with torch.no_grad():
        next_state_pred, _ = icm(current_state, action)
        intrinsic_reward = F.mse_loss(next_state_pred, next_state).item()
    
    return intrinsic_reward