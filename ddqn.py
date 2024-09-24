#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import math
import rospy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import env
import config
import random
import numpy as np
import torch.onnx
from collections import deque
import onnxruntime as ort


torch.autograd.set_detect_anomaly(True)
class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)


    # Add the replay memory
    def add(self, state1, state2, action, apf_index, reward, next_state1, next_state2, done):
        self.memory.append((state1, state2, action, apf_index, reward, next_state1, next_state2, done))

    # Sample the replay memory
    def sample_and_process(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        # Decompose each element
        states1, states2, actions, apf_indices, rewards, next_states1, next_states2, dones = zip(*batch)

        states1 = torch.FloatTensor(np.stack(states1)).to(self.device)
        states2 = torch.FloatTensor(np.stack(states2)).to(self.device)
        actions = torch.LongTensor(np.stack(actions)).to(self.device)
        actions_vx = actions[:, 0].long().view(-1, 1)
        actions_vy = actions[:, 1].long().view(-1, 1)
        apf_indices = torch.LongTensor(np.stack(apf_indices)).to(self.device)
        apf_index_vx = apf_indices[:, 0].long().view(-1, 1)
        apf_index_vy = apf_indices[:, 1].long().view(-1, 1)
        rewards = torch.FloatTensor(np.stack(rewards)).to(self.device)
        next_states1 = torch.FloatTensor(np.stack(next_states1)).to(self.device)
        next_states2 = torch.FloatTensor(np.stack(next_states2)).to(self.device)
        dones = torch.FloatTensor(np.stack(dones)).to(self.device)

        return states1, states2, actions_vx, actions_vy, apf_index_vx, apf_index_vy, rewards, next_states1, next_states2, dones

class DQNNet(nn.Module):
    def __init__(self, network, action_space_vx, action_space_vy):
        super(DQNNet, self).__init__()
        self.action_space_vx = action_space_vx
        self.action_space_vy = action_space_vy
        self.max_scene_dis = 30
        self.network = network
        # Modified Network Architecture
        self.cnn_1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(4, 3), stride=4)
        self.cnn_2 = nn.Conv2d(32, 64, kernel_size=(4, 3), stride=3)
        self.pool_1 = nn.AvgPool2d(2, stride=2)
        self.cnn_3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.fc_target = nn.Linear(2, 64)
        #  Modified the output size of the image convolution, previously it was 1 x 64 x 1 x 1, now it is 1 X 64 X 5 X 3,so the size of the fully connected layer needs to be changed as well
        # self.fc_1 = nn.Linear(64 * 1 * 1 + 64, 256)
        self.fc_1 = nn.Linear(64 * 18 * 18 + 64, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.output_vx = nn.Linear(256, len(self.action_space_vx))
        self.output_vy = nn.Linear(256, len(self.action_space_vy))

        self.advantage_vx = nn.Linear(128, len(self.action_space_vx))
        self.value_vx = nn.Linear(128, 1)

        self.advantage_vy = nn.Linear(128, len(self.action_space_vy))
        self.value_vy = nn.Linear(128, 1)

    def forward(self, state1, state2):
        batch_size = state1.size(0)
        img = state1 / 255
        x1 = F.relu(self.cnn_1(img.transpose(1, 3)))
        x2 = F.relu(self.cnn_2(x1))
        x3 = x2

        x_target = F.relu(self.fc_target(state2))
        x_merge = torch.cat((x3.view(batch_size, -1), x_target), axis=1)
        fc_1 = F.relu(self.fc_1(x_merge))
        fc_2 = F.relu(self.fc_2(fc_1))


        # Dueling DQN--Split the output
        if self.network == "Duel":
            advantage_vx, value_vx = torch.split(fc_2, 128, dim=1)
            advantage_vx = self.advantage_vx(advantage_vx)
            value_vx = self.value_vx(value_vx)
            vx_output = value_vx + advantage_vx - torch.mean(advantage_vx, dim=1, keepdim=True)

            advantage_vy, value_vy = torch.split(fc_2, 128, dim=1)
            advantage_vy = self.advantage_vy(advantage_vy)
            value_vy = self.value_vy(value_vy)
            vy_output = value_vy + advantage_vy - torch.mean(advantage_vy, dim=1, keepdim=True)
        else:
            vx_output = self.output_vx(fc_2)
            vy_output = self.output_vy(fc_2)

        return vx_output, vy_output


class DQN():
    def __init__(self, env, action_space_vx, action_space_vy, memory_size=50000, learning_rate=4e-5, batch_size=32, target_update=1000,
                 gamma=0.95, eps=0.95, eps_min=0.1, eps_period=2000, network='DQN'):
        super(DQN, self).__init__()
        self.env = env
        self.network = network
        self.action_space_vx = action_space_vx
        self.action_space_vy = action_space_vy

        # Torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        # Deep Q network
        self.predict_net = DQNNet(network=self.network, action_space_vx=self.action_space_vx, action_space_vy=self.action_space_vy).to(self.device)
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Target network
        self.target_net = DQNNet(network=self.network, action_space_vx=self.action_space_vx, action_space_vy=self.action_space_vy).to(self.device)
        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.target_update = target_update
        self.update_count = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        # Learning setting
        self.gamma = gamma

        # Exploration setting
        self.eps = eps
        self.eps_min = eps_min
        self.eps_period = eps_period
        self.alpha = 0.1
        self.decay_counter = 0
        # Select the algorithm
        if self.network == "DQN":
            print("DQN")
        elif self.network == "Double":
            print("DDQN")
        elif self.network == "Duel":
            print("Duel")

    # Get the action
    def get_action(self, state1, state2, dist_normalized):
        # Get the Q-values
        state1 = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
        state2 = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
        # If your existing environment trains well, use the following code, otherwise keep it commented out
        # state2[:, 0] = state2[:, 0] / dist_normalized
        # state2[:, 1] = state2[:, 1] / math.pi
        q_values_vx, q_values_vy = self.predict_net(state1, state2)

        # Use the action with the highest Q-value
        action_vx = np.argmax(q_values_vx.cpu().detach().numpy())
        action_vy = np.argmax(q_values_vy.cpu().detach().numpy())

        return action_vx, action_vy

    # Learn the policy
    def learn(self):
        # Replay buffer
        states1, states2, actions_vx, actions_vy, apf_index_vx, apf_index_vy, rewards, next_states1, next_states2, dones = self.replay_buffer.sample_and_process(self.batch_size)
        q_values_vx, q_values_vy = self.predict_net(states1, states2)
        loss_imitation_vx = F.cross_entropy(q_values_vx.squeeze(1), apf_index_vx.squeeze(1))
        loss_imitation_vy = F.cross_entropy(q_values_vy.squeeze(1), apf_index_vy.squeeze(1))
        loss_imitation = loss_imitation_vx + loss_imitation_vy
        # Calculate values and target values
        if self.network == 'Duel' or self.network == 'Double':
            q_values_vx_pred, q_values_vy_pred = self.predict_net(next_states1, next_states2)
            _, actions_prime_vx = torch.max(q_values_vx_pred, 1)
            _, actions_prime_vy = torch.max(q_values_vy_pred, 1)

            q_target_value_vx = self.target_net(next_states1, next_states2)[0].gather(1, actions_prime_vx.view(-1, 1))
            q_target_value_vy = self.target_net(next_states1, next_states2)[1].gather(1, actions_prime_vy.view(-1, 1))

            target_values_vx = (rewards.view(-1, 1) + self.gamma * q_target_value_vx * (1 - dones).view(-1, 1))
            target_values_vy = (rewards.view(-1, 1) + self.gamma * q_target_value_vy * (1 - dones).view(-1, 1))

            predict_values_vx = self.predict_net(states1, states2)[0].gather(1, actions_vx.view(-1, 1))
            predict_values_vy = self.predict_net(states1, states2)[1].gather(1, actions_vy.view(-1, 1))

        else:
            q_values_vx_pred, q_values_vy_pred = self.predict_net(states1, states2)
            q_values_target_vx, q_values_target_vy = self.target_net(next_states1, next_states2)

            target_values_vx = (rewards + self.gamma * torch.max(q_values_target_vx, 1)[0] * (1 - dones)).view(-1, 1)
            target_values_vy = (rewards + self.gamma * torch.max(q_values_target_vy, 1)[0] * (1 - dones)).view(-1, 1)
            predict_values_vx = q_values_vx_pred.gather(1, actions_vx.view(-1, 1))
            predict_values_vy = q_values_vy_pred.gather(1, actions_vy.view(-1, 1))

        # Calculate the loss and optimize the network
        loss_dqn_vx = self.loss_fn(predict_values_vx, target_values_vx)
        loss_dqn_vy = self.loss_fn(predict_values_vy, target_values_vy)
        loss_dqn = loss_dqn_vx + loss_dqn_vy
        loss = self.alpha * loss_dqn + (1 - self.alpha) * loss_imitation
        self.decay_counter += 1
        if self.decay_counter % 500 == 0:
            self.alpha += 0.05
            self.alpha = min(self.alpha, 0.9)
            print(f"Weight of the DQN Loss is set to {self.alpha}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update the target network
        self.update_count += 1
        if self.update_count == self.target_update:
            self.target_net.load_state_dict(self.predict_net.state_dict())
            self.update_count = 0
        return loss_imitation.item(), loss_dqn.item()
    def save_model(self, path):
        checkpoint = {
            'model_states': self.predict_net.state_dict(),
            'optimizer_states': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)
    def save_onnx_model(self, param_path_onnx):
        self.predict_net.eval()
        dummy_state1 = torch.randn(64, 224, 224, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(self.predict_net,
                    (dummy_state1, dummy_state2),
                          param_path_onnx,
                          input_names=['dummy_state1', 'dummy_state2'],
                          output_names=['output_velocity_x', 'output_velocity_y'],
                          dynamic_axes={'dummy_state1': {0: 'batch_size'},  # Dynamic batch size
                                        'dummy_state2': {0: 'batch_size'}}  # Dynamic batch size
                          )

    def load_model(self, filename, device):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.predict_net.load_state_dict(checkpoint['model_states'])
            self.optimizer.load_state_dict(checkpoint['optimizer_states'])
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")

    def load_onnx_model(onnx_file_path):
        """
        加载ONNX模型
        :param onnx_file_path: ONNX model file path
        :return: InferenceSession object for onnxruntime
        """
        sess = ort.InferenceSession(onnx_file_path)
        return sess