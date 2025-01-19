def dqn_arguments(parser):
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')  # 环境名称
    parser.add_argument("--seed", default=11037, type=int)  # 随机种子
    parser.add_argument("--hidden_size", default=16, type=int)  # 隐藏层大小
    parser.add_argument("--lr", default=0.02, type=float)  # 学习率
    parser.add_argument("--gamma", default=0.99, type=float)  # 折扣因子
    parser.add_argument("--grad_norm_clip", default=10, type=float)  # 梯度裁剪
    parser.add_argument("--test", default=False, type=bool)  # 是否测试
    parser.add_argument("--use_cuda", default=True, type=bool)  # 是否使用CUDA
    parser.add_argument("--n_frames", default=int(100), type=int)  # 训练帧数
    parser.add_argument("--batch_size", default=32, type=int)  # 批量大小
    return parser
