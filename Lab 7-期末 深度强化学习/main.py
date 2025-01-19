import argparse
import gym
from argument import dqn_arguments
from agent_dir.agent_dqn import AgentDQN
# 解析参数
def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')  # 是否训练DQN
    parser = dqn_arguments(parser)  # 添加DQN参数
    args = parser.parse_args()  # 解析参数
    return args

def run(args):
    if args.train_dqn:
        env_name = args.env_name  # 环境名称
        env = gym.make(env_name)  # 创建环境
        agent = AgentDQN(env, args)  # 创建DQN智能体
        agent.run()  # 运行智能体


if __name__ == '__main__':
    args = parse()
    run(args)