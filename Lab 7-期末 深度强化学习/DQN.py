import time
import random
import matplotlib.pyplot as plt
import collections
from collections import deque, namedtuple
import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

env = gym.make("CartPole-v0")
o_dim = env.observation_space.shape[0]
a_dim = env.action_space.n


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(o_dim, args.hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(args.hidden, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.out(x)
        return action_value


class ReplayBuffer:
    def __init__(self):
        self.buffer = np.zeros((args.capacity, o_dim * 2 + 2))  # 定义大小
        self.buffer_counter = 0

    def push(self, obs, action, reward, next_obs):
        transition = np.hstack((obs, [action, reward], next_obs))
        index = self.buffer_counter % args.capacity  # 确定在buffer中的行数
        self.buffer[index, :] = transition  # 用新的数据覆盖之前的之前
        self.buffer_counter += 1

    def sample(self):
        if self.buffer_counter < args.batch_size:
            Sample = self.buffer
        else:
            sample_idex = np.random.choice(args.capacity, args.batch_size)
            Sample = self.buffer[sample_idex, :]  # 抽取选中的行数的数据
        return Sample


class DQN(object):
    def __init__(self):
        self.eval_net = QNet()
        self.target_net = QNet()
        self.optim = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)

        self.buffer = ReplayBuffer()
        self.loss_fn = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.learn_step = 0

    def choose_action(self, obs):

        action = np.random.randint(0, a_dim)
        return action

    def store_transition(self, obs, action, reward, next_obs):
        self.buffer.push(obs, action, reward, next_obs)

    def learn(self):
        # 目标网络更新，就是我们固定不动的网络
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1
        sample = self.buffer.sample()
        sample_obs = torch.FloatTensor(sample[:, :o_dim])
        sample_action = torch.LongTensor(sample[:, o_dim:o_dim + 1])
        sample_reward = torch.FloatTensor(sample[:, o_dim + 1:o_dim + 2])
        sample_next_obs = torch.FloatTensor(sample[:, -o_dim:])
        # 获得32个trasition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(sample_obs).gather(1, sample_action)  # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
        # q_next 不进行反向传播，故用detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_next = self.target_net(sample_next_obs).detach()
        # 先算出目标值q_target，max(1)[0]相当于计算出每一行中的最大值（注意不是上面的索引了,而是一个一维张量了），view()函数让其变成(32,1)
        q_target = sample_reward + args.gamma * q_next.max(1)[0].view(args.batch_size, 1)
        # 计算损失值
        loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差方向传播
        self.optim.step()  # 逐步的梯度优化


def main():
    dqn = DQN()
    Epsi = []
    num_500 = 0
    for i_episode in range(args.n_episodes):
        obs = env.reset()  # 重置环境
        episode_reward = 0  # 初始化每个周期的reward值
        if i_episode == 50:
            args.lr = 0.005
        while True:
            env.render()  # 开启画面
            action = dqn.choose_action(obs)  # 与环境互动选择action
            next_obs, reward, done, info= env.step(action)[0]
            x, x_dot, theta, theta_dot = next_obs
            # 小车的位置奖励归一化
            reward_1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.7
            # 小车的偏转角度奖励归一化
            reward_2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            nreward = reward_1 + reward_2
            dqn.store_transition(obs, action, nreward, next_obs)
            episode_reward += reward
            obs = next_obs
            if episode_reward == 500:
                num_500 += 1
            if dqn.buffer.buffer_counter > args.capacity:
                dqn.learn()
            if done:
                print(f"Episode:{i_episode}, Reward:{episode_reward}")
                Epsi.append(episode_reward)
                break
    print(num_500)
    plt.plot(Epsi)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward over Time')
    plt.show()

    window_size = 100
    avg_rewards = []
    for i in range(len(Epsi) - window_size + 1):
        avg_rewards.append(np.mean(Epsi[i:i + window_size]))
    # 绘制图表
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (Window Size={window_size})')
    plt.title('Average Reward ')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment name")
    parser.add_argument("--lr", default=0.014, type=float, help="learning rate")
    parser.add_argument("--hidden", default=50, type=int, help="dimension of args.hidden layer")
    parser.add_argument("--n_episodes", default=500, type=int, help="number of episodes")
    parser.add_argument("--gamma", default=0.9, type=float, help="discount factor")
    # parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity", default=100, type=int, help="capacity of replay buffer")
    parser.add_argument("--eps", default=0.8, type=float, help="args.eps of ε-greedy")
    # parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    # parser.add_argument("--eps_decay",      default=0.999,      type=float)
    parser.add_argument("--update_target", default=100, type=int, help="frequency to update target network")
    args = parser.parse_args()
    main()
