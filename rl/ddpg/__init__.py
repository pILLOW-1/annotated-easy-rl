"""
---
title: 深度确定性策略梯度 (DDPG) - Deep Deterministic Policy Gradient
summary: >
  基于蘑菇书EasyRL第十二章的深度确定性策略梯度(DDPG)算法的PyTorch实现，
  包含Actor-Critic架构、目标网络、经验回放等核心公式的逐行注释。
---

# 深度确定性策略梯度 (DDPG) - Deep Deterministic Policy Gradient

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第十二章的PyTorch实现，
参考论文[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)。

DDPG是DQN在连续动作空间的扩展，结合了DQN和确定性策略梯度的优点。

## 确定性策略梯度

在连续动作空间中，动作是连续值。
确定性策略直接输出动作，而不是动作的概率分布。

确定性策略梯度定理描述了梯度的计算方式。

## DDPG算法

DDPG使用四个神经网络：
1. **Actor**：确定性策略，直接输出动作
2. **Critic**：动作价值函数
3. **Target Actor**：目标策略网络
4. **Target Critic**：目标价值网络

### Critic更新

Critic通过最小化TD误差来学习。

目标值使用目标网络计算。

### Actor更新

Actor通过最大化Critic的输出来学习。

### 目标网络软更新

DDPG使用软更新来平滑地更新目标网络。

其中 tau 远小于1（通常取0.001~0.01）。

## 探索策略

DDPG使用Ornstein-Uhlenbeck (OU) 过程来添加探索噪声。

其中：
- theta 是均值回归速度
- mu 是均值
- sigma 是波动率
- W_t 是维纳过程（布朗运动）

OU过程产生的噪声具有时间相关性，适合连续控制任务。

## 蘑菇书中的DDPG

根据蘑菇书第十二章，DDPG的核心组件：
1. **Actor网络**：输入状态，输出连续动作
2. **Critic网络**：输入状态和动作，输出Q值
3. **经验回放缓冲区**：存储经验元组
4. **目标网络**：Actor和Critic各有对应的目标网络

训练流程：
1. Actor根据当前策略选择动作，加上OU噪声
2. 执行动作，观察奖励和下一状态
3. 存储经验到回放缓冲区
4. 从缓冲区采样小批量数据
5. 更新Critic：最小化TD误差
6. 更新Actor：最大化Q值
7. 软更新目标网络
"""

import torch
from torch import nn
import numpy as np


class Actor(nn.Module):
    """
    ## Actor网络（确定性策略）

    Actor 是一个确定性策略网络，
    输入状态，输出连续动作。

    输出通常使用tanh激活函数，将动作限制在 [-1, 1] 范围内，
    然后缩放到实际的动作空间。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出限制在 [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输出确定性动作。

        参数：
        - `state`: 状态

        返回值：
        - 缩放后的动作
        """
        action = self.network(state)
        return action * self.action_scale


class Critic(nn.Module):
    """
    ## Critic网络（动作价值函数）

    Critic 估计状态-动作对的价值。

    输入状态和动作，输出标量Q值。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # 状态和动作拼接后输入网络
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出标量Q值
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        输出动作价值。

        参数：
        - `state`: 状态
        - `action`: 动作

        返回值：
        - 动作价值
        """
        # 拼接状态和动作
        x = torch.cat([state, action], dim=-1)
        return self.network(x).squeeze(-1)


class DDPGLoss(nn.Module):
    """
    ## DDPG损失函数

    ### Critic损失

    使用MSE损失，目标值使用目标网络计算。

    ### Actor损失

    Actor通过最大化Critic的输出来学习。

    由于优化器执行最小化，Actor损失为负Q值的均值。
    """

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def compute_critic_loss(
        self,
        q_values: torch.Tensor,
        next_q_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算Critic损失。

        参数映射：
        - `q_values`: 当前状态-动作对的Q值
        - `next_q_values`: 目标网络对下一状态的Q值
        - `rewards`: 奖励
        - `dones`: 是否episode结束

        返回值：
        - Critic损失
        """
        # 目标值
        with torch.no_grad():
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # MSE损失
        critic_loss = nn.MSELoss()(q_values, target_q)

        return critic_loss

    def compute_actor_loss(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        计算Actor损失。

        参数：
        - `q_values`: Critic对Actor输出动作的Q值

        返回值：
        - Actor损失（取负号）
        """
        # 最大化Q值等价于最小化负Q值
        actor_loss = -q_values.mean()
        return actor_loss


class OUNoise:
    """
    ## Ornstein-Uhlenbeck噪声过程

    OU过程用于生成时间相关的探索噪声。

    OU过程的特性：
    1. 均值回归：噪声会趋向于均值
    2. 时间相关性：连续时间步的噪声相关
    3. 适合连续控制：噪声变化平滑
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        """重置噪声状态。"""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """
        采样OU噪声。

        返回值：
        - OU噪声样本
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    ## 目标网络软更新

    软更新使目标网络平滑地跟踪主网络，避免剧烈变化。

    参数：
    - `target`: 目标网络
    - `source`: 主网络
    - `tau`: 软更新系数（通常取0.001~0.01）
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target: nn.Module, source: nn.Module):
    """
    ## 目标网络硬更新

    直接将主网络的参数复制到目标网络。

    参数：
    - `target`: 目标网络
    - `source`: 主网络
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
