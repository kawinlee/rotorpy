import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces

import math

"""
Reward functions for quadrotor tasks.
"""

def hover_reward(observation, action, writer=None, epoch_count=None, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and
    action reward.
    """

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[0:3])

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[3:6])

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(observation[10:13])

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action)

    total_reward = dist_reward + vel_reward + action_reward + ang_rate_reward

    if writer is not None and epoch_count is not None:
        writer.add_scalar('Rewards/Distance Reward', dist_reward, epoch_count)
        writer.add_scalar('Rewards/Velocity Reward', vel_reward, epoch_count)
        writer.add_scalar('Rewards/Angular Rate Reward', ang_rate_reward, epoch_count)
        writer.add_scalar('Rewards/Action Reward', action_reward, epoch_count)
        writer.add_scalar('Rewards/Total Reward', total_reward, epoch_count)


    return total_reward