"""
DDPG (ASHRAE 专用版本)
仅支持 ASHRAE 表格数据训练与测试；已移除 Gym 环境兼容代码。
"""
import argparse
import numpy as np

from common.constants import default_seed
from common.utils import get_output_folder, train, test
from common.logger import logger
from env.build_ashrae_env import build_ashrae_envs
from model.ddpg import DDPG
from common.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description='DDPG for ASHRAE dataset (feature encoding + bandit option)')

    parser.add_argument('--mode', default='train', type=str, choices=['train','test'], help='运行模式: train 或 test')
    parser.add_argument('--env_name', default='ASHRAE', type=str, help='环境名称 (默认 ASHRAE)')
    # 网络与优化参数
    parser.add_argument('--hidden1', default=128, type=int, help='第一个隐藏层单元数')
    parser.add_argument('--hidden2', default=64, type=int, help='第二个隐藏层单元数')
    parser.add_argument('--rate', default=0.001, type=float, help='critic 网络学习率')
    parser.add_argument('--prate', default=0.0001, type=float, help='actor 网络学习率')
    parser.add_argument('--init_w', default=0.003, type=float, help='网络参数初始化范围')
    # 训练调度
    parser.add_argument('--train_iter', default=20000, type=int, help='训练总步数 (environment steps)')
    parser.add_argument('--warmup', default=200, type=int, help='经验回放预热步数 (仅随机动作)')
    parser.add_argument('--bsize', default=64, type=int, help='mini-batch 大小')
    parser.add_argument('--rmsize', default=200000, type=int, help='经验回放 buffer 容量')
    parser.add_argument('--window_length', default=1, type=int, help='状态帧堆叠长度 (ASHRAE 一般为 1)')
    parser.add_argument('--epsilon', default=3000, type=int, help='探索噪声线性衰减步数')
    parser.add_argument('--tau', default=0.001, type=float, help='target 网络软更新系数')
    parser.add_argument('--discount', default=0.0, type=float, help='折扣因子 (ASHRAE 推荐 0.0)')
    # 噪声
    parser.add_argument('--ou_theta', default=0.15, type=float, help='OU 噪声参数 theta')
    parser.add_argument('--ou_sigma', default=0.05, type=float, help='OU 噪声参数 sigma (表格回归可较小)')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='OU 噪声参数 mu')
    # 评估/日志
    parser.add_argument('--validate_steps', default=1000, type=int, help='每多少训练步进行一次验证')
    parser.add_argument('--validate_episodes', default=1, type=int, help='每次验证的 episode 数 (single_step 下=样本数)')
    parser.add_argument('--debug', action='store_true', help='打印调试日志')
    # 其它
    parser.add_argument('--output', default='output', type=str, help='输出主目录')
    parser.add_argument('--seed', default=default_seed, type=int, help='随机种子')
    parser.add_argument('--resume', default='latest', type=str, help='测试模式下的模型目录 (默认自动选择最新 run)')
    parser.add_argument('--corr_threshold', default=0.05, type=float, help='Pearson 相关系数绝对值阈值，低于该值的数值列将被删除')
    parser.add_argument('--single_step', action='store_true', help='单步 bandit 模式: 每条样本独立 episode')

    args = parser.parse_args()

    args.output = get_output_folder(args.output, args.env_name)

    logger.info(f"运行参数: {args}")

    # 构建 ASHRAE 环境
    # 兼容之前参数名称: single_step -> ashrae_single_step
    setattr(args, 'ashrae_single_step', args.single_step)
    train_env, test_env, nb_states, nb_actions = build_ashrae_envs(args)

    # 设定随机种子 (numpy)
    if args.seed is not None and args.seed >= 0:
        np.random.seed(args.seed)

    agent = DDPG(nb_states, nb_actions, args)
    evaluator = Evaluator(args.validate_episodes, args.validate_steps, args.output, max_episode_length=None)

    if args.mode == 'train':
        logger.info(f"开始训练 (ASHRAE only) 总步数={args.train_iter} single_step={args.single_step}")
        train(args.train_iter, agent, train_env, evaluator, args.validate_steps, args.output,
              args.warmup, max_episode_length=None, debug=args.debug, eval_env=test_env)
        logger.info('训练完成')
    else:  # test
        logger.info('进入测试模式')
        if args.resume == 'latest':
            args.resume = args.output
        # resume 目录设定：如果用户给的是上层 output/ASHRAE-runX，就使用它
        test(episodes:=args.validate_episodes, agent=agent, env=test_env, evaluate=evaluator,
             model_path=args.resume, visualize=False, debug=args.debug)
        logger.info('测试完成')

if __name__ == '__main__':
    main()
