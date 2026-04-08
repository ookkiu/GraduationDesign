from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

"""
此文件是深度强化学习算法的实现文件，用于指导智能体在「ice_env」环境中进行决策。

核心组件包含：
1. QNetwork：基于多层感知机（MLP）的神经网络，输入状态特征，输出离散动作的 Q 值。
2. ReplayBuffer：经验回放池，用于打破数据间的时序相关性，稳定训练过程。
3. DQNAgent：智能体主类，封装了 Double DQN 的核心逻辑（动作选择、经验存储、梯度更新、目标网络软/硬同步）。
4. DQNConfig：数据类，集中管理训练过程中的所有超参数。
"""

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        """
        动作价值网络（Q-Network）。
        参数：
            obs_dim: 状态空间的维度（输入层大小）
            action_dim: 动作空间的维度（输出层大小）
            hidden_sizes: 隐藏层神经元数量，默认两层各 128 个节点
        """
        super().__init__()
        h1, h2 = hidden_sizes
        # 构建简单的全连接前馈神经网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：输入状态 x，输出该状态下所有可选动作的 Q 值预估
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        """
        经验回放池（Experience Replay Buffer）。
        采用预分配内存的 Numpy 数组实现，以提升存取效率并避免内存碎片化。
        """
        self.capacity = int(capacity)
        # 预分配连续内存
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.size = 0  # 当前池中已有的有效经验数量
        self.pos = 0   # 下一次写入时的指针位置

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        """
        向池中添加一条经验 (S, A, R, S', Done)。
        使用环形队列（Circular Buffer）逻辑，存满后会覆盖最旧的数据。
        """
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = 1.0 if done else 0.0

        # 指针向后移动，到达容量上限则折返到 0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        随机采样一批经验用于神经网络的梯度下降。
        随机采样可以打破马尔可夫决策过程（MDP）中相邻状态的时序强相关性。
        """
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


@dataclass
class DQNConfig:
    """
    集中化管理强化学习超参数，便于进行网格搜索或消融实验。
    """
    gamma: float = 0.99                # 折扣因子（Discount factor），决定对未来奖励的重视程度
    lr: float = 1e-3                   # 优化器的学习率
    batch_size: int = 64               # 每次从回放池中采样的批次大小
    buffer_size: int = 50_000          # 回放池的最大容量
    learning_starts: int = 1_000       # 训练前预填充回放池的随机探索步数，防止初期过拟合
    train_freq: int = 4                # 每隔多少个环境步执行一次网络更新（更新频率）
    target_update_interval: int = 1_000# 每隔多少个全局步硬同步一次目标网络
    max_grad_norm: float = 10.0        # 梯度裁剪阈值，防止梯度爆炸

    eps_start: float = 1.0             # 探索率（Epsilon）初始值（100% 随机探索）
    eps_end: float = 0.05              # 探索率最小值（保留 5% 的随机性避免陷入局部最优）
    eps_decay_steps: int = 50_000      # 探索率线性衰减的步数


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: DQNConfig, device: str = "cpu"):
        """
        Double DQN 智能体核心类。
        """
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.cfg = config
        self.device = torch.device(device)

        # 实例化在线网络（Online Network，负责当前决策并持续更新）
        self.q = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        # 实例化目标网络（Target Network，参数冻结，定期从在线网络拷贝，提供稳定的目标 Q 值）
        self.q_target = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval() # 目标网络只用于推断，不需要开启 Dropout/BatchNorm 等训练特性

        # 优化器
        self.optim = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        # 实例化经验回放池
        self.rb = ReplayBuffer(self.cfg.buffer_size, self.obs_dim)

        self.global_step = 0 # 记录智能体交互的总步数

    def epsilon(self) -> float:
        """
        计算当前步的 epsilon 值，实现探索与利用（Exploration vs Exploitation）的平衡。
        采用线性衰减策略。
        """
        t = min(self.global_step, self.cfg.eps_decay_steps)
        frac = t / float(self.cfg.eps_decay_steps)
        return float(self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        根据当前状态选择动作，使用 epsilon-greedy 策略。
        在评估或实际应用时，令 deterministic=True 以屏蔽随机探索。
        """
        # 探索阶段：以 epsilon 的概率随机选择动作
        if (not deterministic) and (random.random() < self.epsilon()):
            return random.randrange(self.action_dim)

        # 利用阶段：选择当前 Q 网络认为价值最大的动作
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)
        return int(torch.argmax(q, dim=1).item())

    def remember(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        """将交互数据存入经验池。"""
        self.rb.add(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        """
        核心学习逻辑：从经验池采样，计算时序差分（TD）误差，反向传播更新网络。
        这里实现的是 Double DQN (DDQN) 算法。
        """
        # 数据量不足时暂不更新
        if self.rb.size < max(self.cfg.batch_size, self.cfg.learning_starts):
            return None

        # 1. 经验采样并转换为计算图张量
        obs, actions, rewards, next_obs, dones = self.rb.sample(self.cfg.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 2. 计算当前状态动作的预测 Q 值：Q(s, a)
        # gather 的作用是提取 actions_t 对应的 Q 值
        q_values = self.q(obs_t).gather(1, actions_t)

        with torch.no_grad():
            # 3. Double DQN 核心逻辑：动作的选择与评估解耦
            # 3.1 使用「在线网络(self.q)」选择在下一状态 s' 的最优动作 a'
            next_actions = torch.argmax(self.q(next_obs_t), dim=1, keepdim=True)
            # 3.2 使用「目标网络(self.q_target)」评估该动作 a' 的价值
            next_q = self.q_target(next_obs_t).gather(1, next_actions)
            # 3.3 计算 TD 目标值：y = r + gamma * Q_target(s', a') * (1 - done)
            target = rewards_t + (1.0 - dones_t) * self.cfg.gamma * next_q

        # 4. 计算损失并反向传播
        # 使用 Smooth L1 Loss (Huber Loss) 代替 MSE，对异常值（Outliers）更鲁棒，防止梯度爆炸
        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optim.zero_grad(set_to_none=True) # 效率略高于 zero_grad()
        loss.backward()
        # 梯度裁剪：限制梯度的最大范数，进一步增强训练稳定性
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        return float(loss.item())

    def maybe_update_target(self) -> None:
        """
        目标网络硬同步（Hard Update）。
        每隔固定的步数，将在线网络的参数完整拷贝到目标网络。
        """
        if self.global_step % self.cfg.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def state_dict(self) -> dict:
        """序列化智能体状态，用于存档或断点续训。"""
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optim": self.optim.state_dict(),
            "global_step": int(self.global_step),
            "config": self.cfg.__dict__,
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
        }

    def load_state_dict(self, state: dict) -> None:
        """反序列化，加载已保存的模型状态。"""
        self.q.load_state_dict(state["q"])
        if "q_target" in state:
            self.q_target.load_state_dict(state["q_target"])
        if "optim" in state:
            self.optim.load_state_dict(state["optim"])
        if "global_step" in state:
            self.global_step = int(state["global_step"])