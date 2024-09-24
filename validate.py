#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import env
import ddqn
import APF_Vel_ROS
import onnxruntime as ort


max_step_per_episode = 400
max_episode = 100
mass = 1.48
success_count = 0
step_count = 0


GazeboUAV = env.GazeboUAV(max_step=max_step_per_episode)
agent = ddqn.DQN(GazeboUAV, GazeboUAV.action_space_vx, GazeboUAV.action_space_vy, batch_size=64, memory_size=10000, target_update=4,
                 gamma=0.99, learning_rate=1e-4, eps=0.0, eps_min=0.0, eps_period=5000, network='Duel')
## ONNX Model for Drone Decision-Making ##
model = "Your_ONNX_File"
csv_path = 'Your_CSV_File'
tra_path = "Your_Tra_File"
sess = ort.InferenceSession(model)
obs_img = sess.get_inputs()[0].name
obs_pos_onnx = sess.get_inputs()[1].name


print('Action Space_vx = ', GazeboUAV.action_space_vx)
print('Action Space_vy = ', GazeboUAV.action_space_vy)
linear_x_values = []
linear_y_values = []
obs_data_gazebo = []
uav_pos_list = []
t = 1

for i in range(max_episode):
    state1, state2, dist_normalized = GazeboUAV.reset()
    print("dist_normalized = ", dist_normalized)
    time.sleep(0.1)
    for t in range(max_step_per_episode + 1):
        start_time = time.time()
        goal = np.array(GazeboUAV.goal)
        curr_pos = np.array(GazeboUAV.self_state[0:2])
        uav_pos_list.append(curr_pos)
        obs_pos = np.array(GazeboUAV.cylinder_pos)
        action_space_vx = GazeboUAV.action_space_vx
        action_space_vy = GazeboUAV.action_space_vy
        att, rep, vx_world, vy_world = APF_Vel_ROS.vel_control(goal_pos=goal, curr_pos=curr_pos, obs_pos=obs_pos,
                                                               mass=mass, obs_radius=1.0)
        yaw = GazeboUAV.self_state[2]
        # -------Convert the velocity in world frame to the body frame
        vx_uav, vy_uav = APF_Vel_ROS.convert_to_uav_frame(vx_world, vy_world, yaw)
        vx_uav_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vx_uav, action_space_vx, strategy='min')
        vy_uav_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vy_uav, action_space_vy, strategy='max')
        output_velocity = sess.run(None, {
            obs_img: np.array(np.expand_dims(state1, axis=0), dtype=np.float32),
            obs_pos_onnx: np.array(state2, dtype=np.float32).reshape(1, -1),
        })
        output_vx_index = np.argmax(output_velocity[0])  # linear_vx
        output_vy_index = np.argmax(output_velocity[1])  # linear_vy
        GazeboUAV.execute_linear_velocity(output_vx_index, output_vy_index)

        next_state1, next_state2, terminal, reward, success = GazeboUAV.step(time_step=t)
        if terminal:
            if success:
                np.savetxt(tra_path, uav_pos_list)
                df_velocity_pre = pd.DataFrame({
                    'linear_x': linear_x_values,
                    'linear_y': linear_y_values
                })
                success_count += 1
                step_count += t
                print('Time Step = ', t)
                linear_x_values.clear()
                linear_y_values.clear()
                uav_pos_list.clear()
            break
        state1 = next_state1
        state2 = next_state2

success_rate = success_count / max_episode
average_step = step_count / success_count
print("Success Rate: {:.2f}%".format(success_rate * 100))
print("Average Time Step: {:.0f}".format(average_step))

