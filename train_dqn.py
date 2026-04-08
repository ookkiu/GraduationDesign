import argparse
import time

import numpy as np

from ice_env import IcingEnv
from dqn_torch import DQNAgent, DQNConfig

"""
train_dqn.py 的作用就是在 ice_env.py（环境）里训练出一个 DQN 智能体，
并将训练结果（网络权重等）保存为模型文件「dqn_icing_model.pt」。
随后 evaluate_and_plot.py 等测试脚本会加载这个模型进行评估和可视化图表绘制。
"""


def parse_args():
    """
    使用 argparse 解析命令行参数。
    将超参数和环境配置提取到命令行中是一个极佳的工程习惯，
    这使得在服务器上批量运行不同配置的实验（如网格搜索）变得非常容易。
    """
    p = argparse.ArgumentParser(description="在输电线路覆冰环境中训练基于 PyTorch 的 DQN 智能体。")

    # 基础与系统配置
    p.add_argument("--data-path", type=str, default="cold_wave_data.csv")  # 气象数据路径
    p.add_argument("--total-steps", type=int, default=100_000)  # 训练的总环境交互步数
    p.add_argument("--seed", type=int, default=0)  # 随机种子，保证实验的可复现性
    p.add_argument("--device", type=str, default="cpu")  # 训练设备，如果 GPU 可用，建议在外部传入 "cuda"
    p.add_argument("--save-path", type=str, default="dqn_icing_model.pt")  # 模型最终的持久化保存路径

    # 环境的起始状态控制
    # 开启后，智能体将从数据序列的随机时间点开始，这有助于模型泛化，防止它死记硬背某一段固定的天气序列
    p.add_argument("--random-start", action="store_true")

    # 场景扰动参数（用于鲁棒性训练或消融实验）
    # 通常在基础训练时保持默认值，在评估模型面对极端异常天气时的表现才进行修改
    p.add_argument("--ice-growth-scale", type=float, default=1.0)
    p.add_argument("--temp-bias", type=float, default=0.0)
    p.add_argument("--humidity-bias", type=float, default=0.0)
    p.add_argument("--wind-speed-scale", type=float, default=1.0)

    # 多目标奖励函数的权重配置
    p.add_argument("--w-risk", type=float, default=100.0)  # 风险防范权重（最高优先级）
    p.add_argument("--w-reliability", type=float, default=10.0)  # 运行可靠性权重
    p.add_argument("--w-cost", type=float, default=1.0)  # 经济/能耗成本权重（最低优先级）

    # DQN 算法核心超参数
    p.add_argument("--lr", type=float, default=1e-3)  # 学习率
    p.add_argument("--gamma", type=float, default=0.99)  # 奖励折扣因子
    p.add_argument("--batch-size", type=int, default=64)  # 梯度更新时的批次大小
    p.add_argument("--buffer-size", type=int, default=50_000)  # 经验回放池容量
    p.add_argument("--learning-starts", type=int, default=1_000)  # 预热步数（在此之前只收集数据，不更新网络）
    p.add_argument("--train-freq", type=int, default=4)  # 训练频率（每走几步更新一次网络）
    p.add_argument("--target-update", type=int, default=1_000)  # 目标网络同步间隔

    # Epsilon-Greedy 探索策略的参数
    p.add_argument("--eps-start", type=float, default=1.0)  # 初始探索率 (100% 随机)
    p.add_argument("--eps-end", type=float, default=0.05)  # 最小探索率 (保留 5% 随机性)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)  # 探索率线性衰减完毕所需的步数

    return p.parse_args()


def main():
    # 1. 解析参数并设置全局随机种子
    args = parse_args()
    np.random.seed(args.seed)

    # 2. 实例化覆冰环境，并将命令行参数传入以构建特定配置的世界
    env = IcingEnv(
        data_path=args.data_path,
        random_start=args.random_start,
        temp_bias=args.temp_bias,
        humidity_bias=args.humidity_bias,
        wind_speed_scale=args.wind_speed_scale,
        ice_growth_scale=args.ice_growth_scale,
        w_risk=args.w_risk,
        w_reliability=args.w_reliability,
        w_cost=args.w_cost,
    )

    # 获取状态空间和动作空间的维度大小，用于动态构建神经网络
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    # 3. 将离散的 argparse 参数打包为面向对象配置，并实例化智能体
    cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
    )

    agent = DQNAgent(obs_dim, action_dim, cfg, device=args.device)

    # 4. 初始化环境，准备进入训练循环
    obs, _ = env.reset(seed=args.seed)

    # 过程指标记录变量
    episode_reward = 0.0  # 单集累计奖励
    episode_cost = 0.0  # 单集累计动作成本
    episode_steps = 0  # 单集存活步数
    episode = 0  # 训练的总集数

    t0 = time.time()  # 记录训练开始时间，用于计算 SPS (Steps Per Second)

    # 5. 核心强化学习主循环
    for step in range(args.total_steps):
        agent.global_step = step

        # 第一步：智能体观察当前环境状态，并根据策略（Epsilon-Greedy）选择一个动作
        action = agent.act(obs, deterministic=False)

        # 第二步：将动作作用于环境，环境推进到下一时刻，并返回新的状态和奖励等反馈信息
        next_obs, reward, terminated, truncated, info = env.step(action)
        # 只要发生终止（灾害断线）或截断（数据耗尽），均视为当前集结束
        done = bool(terminated or truncated)

        # 第三步：将这一步的交互经验打包存入经验回放池，供后续学习使用
        agent.remember(obs, action, reward, next_obs, done)

        # 累加指标
        episode_reward += float(reward)
        episode_cost += float(info.get("cost", 0.0))
        episode_steps += 1

        # 第四步：模型更新（训练）
        # 必须满足两个条件：1. 回放池里有足够数据（大于预热步数） 2. 达到指定的训练频率
        if step >= cfg.learning_starts and (step % cfg.train_freq == 0):
            agent.update()  # 进行一次梯度下降
            agent.maybe_update_target()  # 检查并执行可能的目标网络同步

        # 第五步：状态转移或环境重置
        if done:
            episode += 1
            # 每完成 5 集，打印一次训练日志以监控进度
            if episode % 5 == 0:
                elapsed = time.time() - t0
                sps = (step + 1) / max(1e-6, elapsed)
                print(
                    f"[episode {episode:4d}] steps={episode_steps:4d} "
                    f"ep_reward={episode_reward:8.2f} ep_cost={episode_cost:8.2f} "
                    f"epsilon={agent.epsilon():.3f} sps={sps:7.1f}"
                )

            # 重新开启新的一集，重置状态和指标
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
        else:
            # 状态滚动，进入下一步
            obs = next_obs

    # 6. 训练结束，将神经网络权重与超参数字典持久化到硬盘
    import torch

    torch.save(agent.state_dict(), args.save_path)
    print(f"训练完成，模型已保存到: {args.save_path}")


if __name__ == "__main__":
    main()