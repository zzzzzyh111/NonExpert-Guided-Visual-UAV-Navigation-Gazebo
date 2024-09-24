#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import env
import ddqn
import numpy as np
import random
import time
import torch
import rospy
import wandb
import APF_Vel_ROS

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4)


wandb.login()
wandb.init(project="Your_Project", name="Your_Run_Name")
total_episode = 10000
max_step_per_episode = 200
mass = 1.48

GazeboUAV = env.GazeboUAV(max_step=max_step_per_episode)
agent = ddqn.DQN(GazeboUAV, GazeboUAV.action_space_vx, GazeboUAV.action_space_vy, batch_size=64, memory_size=10000, target_update=4,
                gamma=0.99, learning_rate=1e-4, eps=0.95, eps_min=0.1, eps_period=5000, network='Duel') # Change the network parameter if you want to train on different network

model_path = 'Your_Model_Path'
pth_path = model_path + 'Your_Model_Path_Name'

# if os.path.exists(pth_path):
#     agent.load_model(pth_path, agent.device)
print('Action Space_vx = ', GazeboUAV.action_space_vx)
print('Action Space_vy = ', GazeboUAV.action_space_vy)
ep_reward_list = []
ep_step_list = []
ep_success_list = []
for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = GazeboUAV.reset()
    ep_reward = 0
    for t in range(max_step_per_episode):
        goal = np.array(GazeboUAV.goal)
        curr_pos = np.array(GazeboUAV.self_state[0:2])
        obs_pos = np.array(GazeboUAV.cylinder_pos)
        action_space_vx = GazeboUAV.action_space_vx
        action_space_vy = GazeboUAV.action_space_vy
        att, rep, vx_world, vy_world = APF_Vel_ROS.vel_control(goal_pos=goal, curr_pos=curr_pos, obs_pos=obs_pos, mass=mass, obs_radius=5.0)
        yaw = GazeboUAV.self_state[2]
        # -------Convert the velocity in world frame to the body frame
        vx_uav, vy_uav = APF_Vel_ROS.convert_to_uav_frame(vx_world, vy_world, yaw)
        vx_uav_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vx_uav, action_space_vx, strategy='min')
        vy_uav_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vy_uav, action_space_vy, strategy='max')
        action_vx, action_vy = agent.get_action(state1, state2, dist_normalized)
        GazeboUAV.execute_linear_velocity(action_vx, action_vy)
        ts = time.time()
        if len(agent.replay_buffer.memory) > 64:
            loss_imitation, loss_dqn = agent.learn()
            wandb.log({'Loss_Imi': loss_imitation}, step=i_episode)
            wandb.log({'Loss_DQN': loss_dqn}, step=i_episode)
        while time.time() - ts <= 0.1:
            continue
        next_state1, next_state2, terminal, reward, success = GazeboUAV.step(time_step=t+1)
        ep_reward += reward
        action = [action_vx, action_vy]
        apf_index = [vx_uav_mapped, vy_uav_mapped]
        agent.replay_buffer.add(state1, state2, action, apf_index, reward, next_state1, next_state2, terminal)
        if terminal:
            if success:
                ep_success_list.append(1)
            else:
                ep_success_list.append(0)
            break

        state1 = next_state1
        state2 = next_state2

    ep_reward_list.append(ep_reward)
    ep_step_list.append(t+1)

    if len(ep_success_list) >= 10:
        # Use the latest N results to calculate the success rate.
        recent_results = ep_success_list[-10:]
        success_rate = sum(recent_results) / 10
    else:
        success_rate = 0

    print("Episode:{} \t\t step:{} \t\t ep_reward:{:.2f} \t\t epsilon:{:.2f}".format(i_episode, t+1, ep_reward, agent.eps))
    wandb.log({'Reward': ep_reward}, step=i_episode)
    wandb.log({'Step': t+1}, step=i_episode)
    wandb.log({'Success Rate': success_rate}, step=i_episode)
    reward_file_path = model_path + 'Your_Reward_File_Name'
    step_file_path = model_path + 'Your_Step_File_Name'
    success_rate_file_path = model_path + 'Your_Success_Rate_File_Name'
    # If the file exists, it is opened in append mode; otherwise, a new file is created
    mode = 'a' if os.path.exists(success_rate_file_path) else 'w'
    with open(success_rate_file_path, mode) as f:
        f.write(f"{success_rate}\n")
    if (i_episode + 1) % 500 == 0:
        mode = 'a' if os.path.exists(reward_file_path) else 'w'
        with open(reward_file_path, mode) as f:
            for reward in ep_reward_list:
                f.write(f"{reward}\n")
        ep_reward_list = []
        mode = 'a' if os.path.exists(step_file_path) else 'w'
        with open(step_file_path, mode) as f:
            for step in ep_step_list:
                f.write(f"{step}\n")
        ep_step_list = []
        # Save the model and ONNX file
        agent.save_model(pth_path)
        onnx_file_name = f"Your_ONNX_Name_{i_episode + 1}.onnx"
        agent.save_onnx_model(model_path + onnx_file_name)

