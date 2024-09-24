import numpy as np
import math
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import Compose, Resize


UNIT_VECTOR_X = np.array([[1], [0], [0]], dtype=np.float32)

def image_processing(frame_buffer):
    target_size = (224, 224)
    resized_images = [TF.resize(Image.fromarray(frame), target_size) for frame in frame_buffer]
    combined_image = np.concatenate([np.array(img) for img in resized_images], axis=-1)
    tensor_image = torch.from_numpy(combined_image).float()
    tensor_image = tensor_image.unsqueeze(0)
    np_image = np.array(tensor_image, dtype=np.float32)
    return np_image

def quarternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]],
                          dtype=np.float32)

    return rot_matrix

def normalize_angle(theta):
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta

def convert_to_world_frame(vx_uav, vy_uav, yaw):
    # Conversion to body coordinate system
    vx_world = vx_uav * math.cos(yaw) - vy_uav * math.sin(yaw)
    vy_world = vx_uav * math.sin(yaw) + vy_uav * math.cos(yaw)
    return vx_world, vy_world

def goal2rob(goal_position, position_drone_1, yaw):
    dx = goal_position[0, 0] - position_drone_1[0, 0]
    dy = goal_position[0, 1] - position_drone_1[0, 1]
    dist_normalized = abs(position_drone_1[0, 1] - goal_position[0, 1])
    dist_target = np.sqrt((dx ** 2) + (dy ** 2))
    angle_goal = math.atan2(dy, dx)
    alpha = math.atan2(dy, dx) - yaw
    alpha = normalize_angle(alpha)
    return dist_target, dist_normalized, alpha
def obs2rob(o_x, o_y, yaw, position_drone_1):
    theta = yaw
    s_x = o_x - position_drone_1[0, 0]
    s_y = o_y - position_drone_1[0, 1]
    dist_obs = math.sqrt(s_x ** 2 + s_y ** 2)
    beta = math.atan2(s_y, s_x) - theta
    beta = normalize_angle(beta)
    return dist_obs, beta