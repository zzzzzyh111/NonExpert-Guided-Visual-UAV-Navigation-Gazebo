#!/usr/bin/python3

import numpy as np
import cv2
import onnxruntime as ort

import sys
import time
import traceback
import math
import pandas as pd

import av
import tellopy

import rospy
from geometry_msgs.msg import PoseStamped
import tf
import matplotlib.pyplot as plt

from helper_functions import *

## Global Variable ##
position_drone_1 = np.zeros((1, 3), dtype=np.float32)
orientation_drone_1 = np.zeros((1, 4), dtype=np.float32)
yaw = np.float32()
goal_position = np.zeros((1, 3), dtype=np.float32)

obs_pos_data = np.zeros((1, 2), dtype=np.float32)

action_space_vx = np.arange(-0.5, 0.75, 0.25).tolist()
action_space_vy = np.arange(-0.5, 0.75, 0.25).tolist()

telloID1 = "tello1"

frame_buffer = []
frame_count = 0
max_buffer_size = 4
obs_index = 0

## ONNX Model for Drone Decision Making ##
model = "Your_ONNX_File"
csv_path = 'YOUR_CSV_File'
sess = ort.InferenceSession(model)

obs_img = sess.get_inputs()[0].name
obs_pos = sess.get_inputs()[1].name
linear_x_values = []
linear_y_values = []

## Handlers and control functions ##

def current_positioning_drone_1(msg):
    global position_drone_1, orientation_drone_1, yaw

    position_drone_1[0, 0] = msg.pose.position.x
    position_drone_1[0, 1] = msg.pose.position.y
    position_drone_1[0, 2] = msg.pose.position.z

    orientation_drone_1[0, 0] = msg.pose.orientation.x
    orientation_drone_1[0, 1] = msg.pose.orientation.y
    orientation_drone_1[0, 2] = msg.pose.orientation.z
    orientation_drone_1[0, 3] = msg.pose.orientation.w
    quaternion = (msg.pose.orientation.x,
                  msg.pose.orientation.y,
                  msg.pose.orientation.z,
                  msg.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]

def goal_positioning(msg):
    global goal_position

    goal_position[0, 0] = msg.pose.position.x
    goal_position[0, 1] = msg.pose.position.y
    goal_position[0, 2] = msg.pose.position.z


def select_vel_cmd(vx_index, vy_index):
    global action_space_vx, action_space_vy
    linear_x = action_space_vx[vx_index] * 0.5
    linear_y = action_space_vy[vy_index] * 0.5
    return linear_x, linear_y

def cb_cmd_vel(drone, linear_x, linear_y):
    drone.set_roll(linear_x)
    drone.set_pitch(linear_y)
    print("Vx = ", linear_x)
    print("Vy = ", linear_y)



def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        pass


def main():
    global obs_pos_data
    global position_drone_1, orientation_drone_1, goal_position
    global frame_count, obs_index

    drone = tellopy.Tello()

    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

        drone.connect()
        drone.wait_for_connection(60.0)

        drone.takeoff()

        time.sleep(5)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')
                pass

        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                ####################

                if frame_count == 0:
                    start_time_all = time.time()

                frame_count += 1

                image = np.array(frame.to_image())[:, 120: 839, :]
                cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                # Get the latest four consecutive frames
                frame_buffer.append(image)
                if len(frame_buffer) > max_buffer_size:
                    frame_buffer.pop(0)  # Pop out the earliest frame
                if len(frame_buffer) < max_buffer_size:
                    continue
                processed_frame = image_processing(frame_buffer)
                print("Drone Position = ", position_drone_1)
                print("Goal Position = ", goal_position)
                dist_target, dist_normalized, alpha = goal2rob(goal_position, position_drone_1, yaw)
                obs_pos_data[0, 0] = dist_target / dist_normalized
                obs_pos_data[0, 1] = alpha
                print("obs_pos_data = ", obs_pos_data)

                output_velocity = sess.run(None, {
                    obs_img: processed_frame,
                    obs_pos: obs_pos_data,
                })
                output_vx_index = np.argmax(output_velocity[0])  # linear_vx
                output_vy_index = np.argmax(output_velocity[1])  # linear_vy
                linear_x, linear_y = select_vel_cmd(output_vx_index, output_vy_index)

                cb_cmd_vel(drone, -linear_y, linear_x)
                time.sleep(0.1)
                print("Time = ", time.time() - start_time)
                linear_x_values.append(-linear_y)
                linear_y_values.append(linear_x)


                if dist_target < 1.0:
                    cb_cmd_vel(drone, linear_x=0.0, linear_y=0.0)
                    drone.land()
                    print("End Frame:", frame_count)
                    print("End Time:", time.time() - start_time_all)
                    time.sleep(5)
                    drone.quit()
                    cv2.destroyAllWindows()

                ####################

                if frame.time_base < 1.0 / 60:
                    time_base = 1.0 / 60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time) / time_base)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

    df_velocity = pd.DataFrame({
        'linear_x': linear_x_values,
        'linear_y': linear_y_values
    })

    df_velocity.to_csv(csv_path + 'Your_CSV_Name', index=False)

    df_velocity = pd.read_csv(csv_path + 'Your_CSV_Name')

    # Plot vx and vy
    plt.figure()
    plt.plot(df_velocity['linear_x'], label='Linear X', color='blue')
    plt.plot(df_velocity['linear_y'], label='Linear Y', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Velocity')
    plt.title('Velocity Profile Over Time_Real')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    rospy.Subscriber(f'/vrpn_client_node/{telloID1}/pose', PoseStamped, current_positioning_drone_1)
    rospy.Subscriber(f'/vrpn_client_node/goal/pose', PoseStamped, goal_positioning)

    main()