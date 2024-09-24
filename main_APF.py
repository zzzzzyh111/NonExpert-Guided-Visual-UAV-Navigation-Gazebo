#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import env
import APF_Vel_ROS


GazeboUAV = env.GazeboUAV(max_step=300)
# -------------------------Params------------------------------------
mass = 1.48
total_episode = 100
max_step_per_episode = 300
success_num = 0
print('Action Space_vx = ', GazeboUAV.action_space_vx)
print('Action Space_vy = ', GazeboUAV.action_space_vy)


#--------------------------Path Finding with APF--------------------
for i_episode in range(total_episode):
    state1, state2, dist_normalized = GazeboUAV.reset()
    sum_vx_uav = []
    sum_vy_uav = []
    att_values = []
    rep_values = []
    for t in range(max_step_per_episode + 1):
        goal = np.array(GazeboUAV.goal)
        curr_pos = np.array(GazeboUAV.self_state[0:2])
        obs_pos = np.array(GazeboUAV.cylinder_pos)
        action_space_vx = GazeboUAV.action_space_vx
        action_space_vy = GazeboUAV.action_space_vy
        att, rep, vx_world, vy_world = APF_Vel_ROS.vel_control(goal_pos=goal, curr_pos=curr_pos, obs_pos=obs_pos, mass=mass, obs_radius=2.0)
        yaw = GazeboUAV.self_state[2]
        # -------Convert the velocity in world frame to the body frame
        vx_uav, vy_uav = APF_Vel_ROS.convert_to_uav_frame(vx_world, vy_world, yaw)
        sum_vx_uav.append(vx_uav)
        sum_vy_uav.append(vy_uav)
        vx_uav_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vx_uav, action_space_vx, strategy='min')
        vy_uav_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vy_uav, action_space_vy, strategy='max')
        GazeboUAV.execute_linear_velocity(vx_uav_mapped, vy_uav_mapped)
        terminal, reward, success = GazeboUAV.get_reward_and_terminate(time_step=t)
        if terminal:
            if GazeboUAV.success:
                success_num += 1
                print("Step = ", t)
                avg_vx_uav = sum(sum_vx_uav) / len(sum_vx_uav)
                avg_vy_uav = sum(sum_vy_uav) / len(sum_vy_uav)
                print(f"Episode {i_episode}: Average vx_uav = {avg_vx_uav}, Average vy_uav = {avg_vy_uav}")
                print("Max Vx = ", max(sum_vx_uav))
                print("Max Vy = ", max(sum_vy_uav))
            break
print("Success Rate = {:.2f}%".format(success_num / total_episode * 100))




