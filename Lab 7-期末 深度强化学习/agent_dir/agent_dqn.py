import collections
import os
import random
import copy
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = self.fc3(inputs)
        return inputs


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)

        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst).reshape(-1, 1)
        r_lst = np.array(r_lst).reshape(-1, 1)
        s_prime_lst = np.array(s_prime_lst)

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.long), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float)

    def clean(self):
        self.buffer = []


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)

        self.env = env
        self.args = args
        self.lr = args.lr
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.buffer = ReplayBuffer(50000)
        self.q_network = QNetwork(env.observation_space.shape[0], args.hidden_size, env.action_space.n)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.rewards = []

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.buffer.clean()

    def train(self):
        """
        Implement your training algorithm here
        """
        size = 32
        if len(self.buffer) < size:
            return
        for i in range(64):
            s, a, r, s_ = self.buffer.sample(size)

            q_out = self.q_network(s)
            q_a = q_out.gather(1, a)
            max_q = self.target_q_network(s_).max(1)[0].unsqueeze(1)
            target = r + self.args.gamma * max_q
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.q_network(torch.tensor(observation, dtype=torch.float)).argmax().item()

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        test_total_reward = 0
        for i_episode in range(self.args.n_frames):
            if i_episode == 8 or i_episode == 12 or i_episode == 20:
                self.lr /= 2
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
            total_reward = 0
            state = self.env.reset()[0]
            for t in range(200):
                action = self.make_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                used_reward = -abs(next_state[2]) - abs(next_state[0]) / 10
                self.buffer.push(state, action, used_reward, next_state)
                state = next_state
                if done:
                    break
            if i_episode > 1:
                self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
                self.train()
                if i_episode % 2 == 0:
                    self.target_q_network.load_state_dict(self.q_network.state_dict())
            tmp = self.epsilon
            self.epsilon = 0
            test_total_reward = 0
            observation = self.env.reset()[0]
            state = torch.tensor(observation, dtype=torch.float)
            for k in range(200):
                action = self.make_action(state)
                observation, reward, done, _, _ = self.env.step(action)
                test_total_reward += reward
                if done:
                    break
                state = torch.tensor(observation, dtype=torch.float)
            self.epsilon = tmp
            self.rewards.append(test_total_reward)
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
