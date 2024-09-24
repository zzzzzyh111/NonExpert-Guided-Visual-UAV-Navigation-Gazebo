import numpy as np
import math


def att_pot(goal_pos, curr_pos):
    k_att = 1.0 # Attraction gain
    d_goal_threshold = 2.0  # Goal Threshold
    distance = np.linalg.norm(goal_pos - curr_pos)
    if distance <= d_goal_threshold:
        att = k_att * (goal_pos - curr_pos)
    else:
        att = d_goal_threshold * k_att * (goal_pos - curr_pos) / np.linalg.norm(goal_pos - curr_pos)
    return att


def rep_pot(obs_pos, curr_pos, obs_radius):
    k_rep = 1.0  # Repulsion gain
    # obs_radius = 1.5  # Obstacle radius
    rep = 0
    min_dist = float('inf')
    min_index = None
    for i, obs in enumerate(obs_pos):
        dist = np.linalg.norm(obs - curr_pos)
        if dist < obs_radius and dist < min_dist:
            min_dist = dist
            min_index = i
    if min_index is not None:
        close_obs = obs_pos[min_index]
        rep = 0.5 * k_rep * (1.0 / min_dist - 1.0 / obs_radius) * (1.0 / min_dist) ** 2 * (curr_pos - close_obs) / min_dist
    return rep

def vel_control(goal_pos, curr_pos, obs_pos, mass, obs_radius):
    att = att_pot(goal_pos, curr_pos)
    rep = rep_pot(obs_pos, curr_pos, obs_radius)
    total_pot = att + rep
    if np.linalg.norm(total_pot) > 1:
        total_pot = total_pot / np.linalg.norm(total_pot)
    fx = total_pot[0]
    fy = total_pot[1]
    ax = fx / mass
    ay = fy / mass
    return att, rep, ax, ay


def convert_to_uav_frame(vx_world, vy_world, yaw):
    # Conversion to body coordinate system
    vx_uav = 1.8 * (vy_world * math.sin(yaw) + vx_world * math.cos(yaw))
    vy_uav = 1.8 * (vy_world * math.cos(yaw) - vx_world * math.sin(yaw))
    return vx_uav, vy_uav

def triangular_membership(x, a, b, c):
    """Triangular membership function."""
    if a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0.0


def fuzzy_map_v_triangular(v_scaled, action_space, strategy='max'):
    """Map scaled v to DQN's discrete action space using triangular membership functions."""
    # If v_scaled exceeds the endpoint value then select the corresponding endpoint value
    if v_scaled >= max(action_space):
        return len(action_space) - 1  # Returns the index of the maximum action value
    elif v_scaled <= min(action_space):
        return 0  # Returns the index of the minimum action value

    # Define the triangular membership functions for each action
    width = 0.5
    memberships = [triangular_membership(v_scaled, action - width / 2, action, action + width / 2) for action in
                   action_space]

    # Find all actions with the highest membership value
    max_indices = np.argwhere(memberships == np.amax(memberships)).flatten()

    # If there is only one maximum affiliation value, it is returned directly
    if len(max_indices) == 1:
        return max_indices[0]

    # Selection of actions with larger or smaller absolute values depending on the strategy
    if strategy == 'max':
        selected_action_index = max(max_indices, key=lambda index: abs(action_space[index]))
    elif strategy == 'min':
        selected_action_index = min(max_indices, key=lambda index: abs(action_space[index]))
    else:
        raise ValueError("Strategy must be either 'max' or 'min'.")

    return selected_action_index

