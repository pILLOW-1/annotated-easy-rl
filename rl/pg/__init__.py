"""
---
title: 策略梯度 (Policy Gradient) - REINFORCE算法
summary: >
  基于蘑菇书EasyRL第四章的策略梯度算法的PyTorch实现，
  包含REINFORCE算法、基线技巧、折扣回报等核心公式的逐行注释。
---

# 策略梯度 (Policy Gradient) - REINFORCE算法

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第四章的PyTorch实现。

策略梯度方法直接优化策略，通过梯度上升最大化期望累积奖励。

## 策略梯度定理

### 轨迹概率

一条轨迹的概率由初始状态概率、策略概率和状态转移概率的乘积决定。

### 期望奖励

期望奖励是轨迹总奖励的期望值。

### 策略梯度

对期望奖励求梯度，使用log-derivative技巧推导得到策略梯度公式。

## 基线技巧 (Baseline)

策略梯度的方差很大。我们可以减去一个基线来降低方差。

基线可以是状态价值函数，此时使用优势函数来加权梯度。

## 折扣回报 (Discounted Return)

使用折扣因子来权衡近期和远期奖励。

折扣回报满足递归关系：G_t = r_{t+1} + gamma * G_{t+1}

REINFORCE更新规则：使用折扣回报加权对数概率梯度。

参数更新方向沿梯度上升。
"""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """
    ## 策略网络

    策略网络输出在给定状态下每个动作的概率分布。

    使用softmax将网络输出转换为概率分布。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输出策略分布。

        参数：
        - `state`: 状态

        返回值：
        - 动作概率分布
        """
        logits = self.network(state)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs

    def select_action(self, state: torch.Tensor) -> tuple:
        """
        根据策略采样动作。

        返回值：
        - 采样的动作
        - 动作的对数概率
        """
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """
    ## 价值网络

    价值网络估计状态的期望累积奖励。

    用作策略梯度中的基线。
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输出状态价值。

        参数：
        - `state`: 状态

        返回值：
        - 状态价值
        """
        return self.network(state).squeeze(-1)


def compute_discounted_returns(rewards: list, gamma: float) -> list:
    """
    ## 计算折扣累积回报

    G_t = r_{t+1} + gamma * G_{t+1}

    从后往前递归计算，时间复杂度 O(T)。

    参数：
    - `rewards`: 轨迹的奖励序列
    - `gamma`: 折扣因子

    返回值：
    - `returns`: 每步的折扣累积回报
    """
    returns = []
    discounted_sum = 0

    # 从后往前计算
    for reward in reversed(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    return returns


def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    ## REINFORCE损失函数

    REINFORCE更新规则：使用折扣回报加权对数概率梯度。

    由于优化器执行最小化，我们取负号。

    参数：
    - `log_probs`: 每步动作的对数概率
    - `returns`: 每步的折扣累积回报

    返回值：
    - REINFORCE损失（取负号）
    """
    loss = -torch.sum(log_probs * returns)
    return loss


def actor_critic_loss(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor = None,
    entropy_coef: float = 0.01,
) -> tuple:
    """
    ## Actor-Critic损失函数

    Actor-Critic结合策略梯度和价值函数。

    优势函数 A(s_t, a_t) = G_t - V(s_t)。

    Critic学习价值函数。

    可选的熵正则化项鼓励探索。

    总损失 = Actor损失 + 0.5 * Critic损失 - 熵正则化项

    参数：
    - `log_probs`: 动作的对数概率
    - `values`: 状态价值估计
    - `returns`: 折扣累积回报
    - `entropy`: 策略熵（可选）
    - `entropy_coef`: 熵正则化系数

    返回值：
    - Actor损失
    - Critic损失
    - 总损失
    """
    # 优势函数
    advantages = returns - values.detach()

    # Actor损失
    actor_loss = -torch.mean(log_probs * advantages)

    # Critic损失
    critic_loss = torch.mean((returns - values) ** 2)

    # 总损失
    total_loss = actor_loss + 0.5 * critic_loss

    if entropy is not None:
        total_loss -= entropy_coef * entropy.mean()

    return actor_loss, critic_loss, total_loss
