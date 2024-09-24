#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import rospy
import numpy as np
import cv2
import time
import tf
import math as math
import random
import copy
import config
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image
from std_srvs.srv import Empty


class GazeboUAV():
    def __init__(self, max_step):
        rospy.init_node('GazeboUAV', anonymous=False)

        # -----------Params--------------------------------------------------
        self.image = None
        self.stacked_imgs = None
        self.bridge = CvBridge()
        self.position = []

        self.goal_space = config.goal_space
        self.start_space = config.start_space

        self.success = False
        self.dist_start = 0
        self.dist_init = 0
        self.dist = 0
        self.yaw = None
        self.reward = 0

        self.cylinder_pos = [[] for i in range(10)]
        self.uav_trajectory = [[], []]
        self.obstacle_state = []
        self.action_space_vx = np.arange(-1.0, 1.5, 0.5).tolist()
        self.action_space_vy = np.arange(-1.0, 1.5, 0.5).tolist()
        self.max_step_per_episode = max_step
        # ------------------------Publisher and Subscriber---------------------------
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
        self.image_sub = rospy.Subscriber("/front_cam/camera/image", Image, self.ImageCallBack)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.sleep(1.)

    def ModelStateCallBack(self, data):
        idx = data.name.index("quadrotor")
        quaternion = (data.pose[idx].orientation.x,
                      data.pose[idx].orientation.y,
                      data.pose[idx].orientation.z,
                      data.pose[idx].orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.self_state = [data.pose[idx].position.x,
                           data.pose[idx].position.y,
                           yaw,
                           data.twist[idx].linear.x,
                           data.twist[idx].linear.y,
                           data.twist[idx].angular.z]

        for i in range(10):
            idx = data.name.index("unit_cylinder" + str(i))
            self.cylinder_pos[i] = [data.pose[idx].position.x, data.pose[idx].position.y]

    def ImageCallBack(self, img):
        self.image = img

    def randomize_image_color_and_texture(self, cv_img):
        """
            Randomly change the color and texture of the image by applying
            random color transformations and adding noise.
            """
        # Randomly change the brightness and color
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        # hsv_img[:, :, 0] = (hsv_img[:, :, 0] + np.random.randint(-10, 10)) % 180  # Hue
        # hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * np.random.uniform(0.9, 1.1), 0, 255)  # Saturation
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)  # Brightness

        # Add noise to the V (Value/Brightness) channel
        noise = np.random.normal(0, 2, hsv_img[:, :, 2].shape).astype(np.int16)  # Use int16 to allow negative values
        hsv_img[:, :, 2] = hsv_img[:, :, 2].astype(np.int16)  # Convert to int16 before adding noise
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + noise, 0, 255)  # Add noise and clip values to [0, 255]
        hsv_img[:, :, 2] = hsv_img[:, :, 2].astype(np.uint8)  # Convert back to uint8
        cv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        cv_img = cv_img.astype(np.uint8)

        return cv_img


    def get_image_observation(self):
            # Ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
            cv_img = np.array(cv_img, dtype=np.uint8)
            cv_img[np.isnan(cv_img)] = 0
            # Apply domain randomization
            cv_img = self.randomize_image_color_and_texture(cv_img)

            return cv_img
        except Exception as err:
            print("Ros_to_Cv2 Failure: %s" % err)


    def goal2rob(self):
        # Calculate the distance between the goal and the agent
        theta = self.self_state[2]
        a_x = self.goal[0] - self.self_state[0]
        a_y = self.goal[1] - self.self_state[1]
        dist = math.sqrt(a_x ** 2 + a_y ** 2)
        alpha = math.atan2(a_y, a_x) - theta  # theta = yaw
        alpha = self.normalize_angle(alpha)
        return dist, alpha

    def obs2rob(self, o_x, o_y):
        theta = self.self_state[2]
        s_x = o_x - self.self_state[0]
        s_y = o_y - self.self_state[1]
        dist = math.sqrt(s_x ** 2 + s_y ** 2)
        beta = math.atan2(s_y, s_x) - theta
        beta = self.normalize_angle(beta)
        return dist, beta

    def detect_collision(self):
        collision = False
        for i in range(len(self.cylinder_pos)):
            e, _ = self.obs2rob(self.cylinder_pos[i][0], self.cylinder_pos[i][1])
            if e < 1.0:
                collision = True

        return collision

    def get_obs_state(self):
        self.obstacle_state.clear()
        for i in range(len(self.cylinder_pos)):
            dist, angle = self.obs2rob(self.cylinder_pos[i][0], self.cylinder_pos[i][1])
            self.obstacle_state.extend([dist, angle])


    def get_states(self):
        self.stacked_imgs = np.dstack([self.stacked_imgs[:, :, -9:], self.get_image_observation()])
        return self.stacked_imgs, self.position

    def set_uav_pose(self, x, y, theta):
        state = ModelState()
        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.linear.y = 0.0
        self.vel_pub.publish(move_cmd)
        rospy.sleep(0.3)
        state.model_name = 'quadrotor'
        state.reference_frame = 'world'  # ''ground_plane'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 1.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        # twist
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        self.set_state.publish(state)
        rospy.sleep(0.3)


    def set_obs_pose_random(self):
        state = ModelState()
        for i in range(10):
            state.model_name = 'unit_cylinder' + str(i)
            state.reference_frame = 'world'  # ''ground_plane'
            state.pose.position.x = config.obstacle_position[i][0] + random.uniform(-0.2, 0.2)
            state.pose.position.y = config.obstacle_position[i][1] + random.uniform(-0.2, 0.2)
            state.pose.position.z = 1.0
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0
            self.set_state.publish(state)

    def set_goal(self, x, y):
        self.goal = [x, y]

    def set_goal_pose(self, x, y):
        state = ModelState()
        state.model_name = 'unit_box_real'
        state.reference_frame = 'world'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.75 + random.uniform(-0.5, 0.5)
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        self.set_state.publish(state)

    def normalize_angle(self, alpha):
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        return alpha

    def reset(self):
        start_index = np.random.choice(len(self.start_space))
        goal_index = np.random.choice(len(self.goal_space))
        start = np.array(self.start_space[start_index]) + np.random.uniform(-0.3, 0.3)
        goal = np.array(self.goal_space[goal_index]) + np.random.uniform(-0.3, 0.3)
        # -------------------------------------
        theta = -math.pi / 2
        self.set_goal(goal[0], goal[1])
        self.set_uav_pose(start[0], start[1], theta)
        self.set_goal_pose(goal[0], goal[1])
        rospy.sleep(0.1)
        self.set_obs_pose_random()
        d0, alpha0 = self.goal2rob()
        self.position = [d0, alpha0]
        self.reward = 0
        self.dist_init = d0
        self.dist_start = abs(goal[1] - start[1])
        self.success = False
        rospy.sleep(0.1)
        self.stacked_imgs = np.dstack([self.get_image_observation()] * 4)
        img, pos = self.get_states()
        return img, pos, self.dist_start

    def execute(self, action_num):

        move_cmd = Twist()
        if action_num == 0:
            angular_z = 0.5
        elif action_num == 1:
            angular_z = 1.0
        elif action_num == 2:
            angular_z = -0.5
        elif action_num == 3:
            angular_z = -1.0
        elif action_num == 4:
            angular_z = 0
        else:
            raise Exception('Error discrete action')
        move_cmd.linear.x = 1.5
        move_cmd.angular.z = angular_z
        self.vel_pub.publish(move_cmd)
        rospy.sleep(0.2)  # execute time
    def execute_linear_velocity(self, vx_index, vy_index):

        move_cmd = Twist()
        move_cmd.linear.x = self.action_space_vx[vx_index]
        move_cmd.linear.y = self.action_space_vy[vy_index]
        self.vel_pub.publish(move_cmd)
        rospy.sleep(0.2)  # execute time
    def step(self, time_step):
        d1, alpha1 = self.goal2rob()
        self.position = [d1, alpha1]
        self.dist = d1
        terminal, reward, success = self.get_reward_and_terminate(time_step)
        self.reward = reward
        self.dist_init = self.dist
        img, pos = self.get_states()

        return img, pos, terminal, reward, success

    def get_reward_and_terminate(self, time_step):
        terminal = False
        dist2goal, _ = self.goal2rob()
        reward = (0.1 * (self.dist_init - self.dist) - 0.002)

        if (dist2goal < 0.5):
            reward = 10.0
            print("Arrival!")
            terminal = True
            self.success = True

        elif (self.self_state[0] >= 6.5) or (self.self_state[0] <= -6.5) or (self.self_state[1] >= 13.5) or (
                self.self_state[1] <= -13.5):
            reward = -1.0
            print("Out!")
            terminal = True

        elif self.detect_collision():
            reward = -1.0
            print("Collision!")
            terminal = True

        elif time_step == self.max_step_per_episode:
            # reward = -1.0
            print("Timeout!")
            terminal = True

        return terminal, reward, self.success
