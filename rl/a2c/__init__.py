"""
---
title: 演员-评论员算法 (Actor-Critic) - A2C
summary: >
  基于蘑菇书EasyRL第九章的演员-评论员(Actor-Critic)算法的PyTorch实现，
  包含A2C算法的核心公式逐行注释。
---

# 演员-评论员算法 (Actor-Critic) - A2C

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第九章的PyTorch实现。

Actor-Critic算法结合了策略梯度(Actor)和价值函数(Critic)的优点：
- **Actor** (演员)：负责学习策略，输出动作
- **Critic** (评论员)：负责学习价值函数，评估Actor的表现

## 优势函数

优势函数衡量在状态 s 下采取动作 a 相对于平均水平的优势。

优势函数可以通过TD残差估计。

## Actor-Critic目标函数

Actor的更新方向由优势函数加权。

Critic通过最小化TD误差来学习价值函数。

## A2C (Advantage Actor-Critic)

A2C是同步的Actor-Critic算法，多个并行环境同时收集经验，
然后同步更新所有环境的策略和价值函数。

### A2C算法流程

1. 每个并行环境运行 n 步，收集经验
2. 计算优势函数（使用GAE或TD残差）
3. 计算Actor损失和Critic损失
4. 同步更新所有环境的网络参数

### 损失函数

总损失由三部分组成：
- Actor损失：负的优势加权对数概率
- Critic损失：TD误差的MSE
- 策略熵（鼓励探索）

## 蘑菇书中的Actor-Critic

根据蘑菇书第九章，Actor-Critic的核心思想是：
- Actor根据策略选择动作
- Critic评估当前状态的价值
- 使用Critic的输出作为基线，降低策略梯度的方差

策略梯度更新沿优势加权对数概率梯度方向。

价值函数更新沿TD误差梯度方向。
"""

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """
    ## Actor-Critic共享网络

    共享特征提取器的Actor-Critic网络：
    - 共享层：提取状态特征
    - Actor头：输出策略分布
    - Critic头：输出状态价值
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # 共享特征提取器
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor头：输出动作分布
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic头：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        """
        前向传播。

        参数：
        - `state`: 状态

        返回值：
        - `action_probs`: 动作概率分布
        - `state_value`: 状态价值
        """
        features = self.feature(state)

        # Actor
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Critic
        state_value = self.critic(features).squeeze(-1)

        return action_probs, state_value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        评估给定状态-动作对。

        用于计算对数概率和策略熵。

        参数：
        - `state`: 状态
        - `action`: 动作

        返回值：
        - `log_prob`: 动作的对数概率
        - `state_value`: 状态价值
        - `entropy`: 策略熵
        """
        features = self.feature(state)
        action_logits = self.actor(features)
        state_value = self.critic(features).squeeze(-1)

        # 计算动作概率和对数概率
        action_probs = torch.softmax(action_logits, dim=-1)
        action_log_probs = torch.log(action_probs + 1e-8)

        # 选择给定动作的对数概率
        log_prob = action_log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        # 计算策略熵
        entropy = -(action_probs * action_log_probs).sum(dim=-1)

        return log_prob, state_value, entropy


class A2CLoss(nn.Module):
    """
    ## A2C损失函数

    A2C的总损失由三部分组成：
    - Actor损失
    - Critic损失
    - 策略熵（鼓励探索）

    优势函数通过TD残差或GAE估计。
    """

    def __init__(self, value_loss_coef: float = 0.5, entropy_coef: float = 0.01):
        super().__init__()
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def forward(
        self,
        log_probs: torch.Tensor,
        state_values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple:
        """
        计算A2C损失。

        参数映射：
        - `log_probs`: 动作的对数概率
        - `state_values`: 状态价值
        - `returns`: 折扣累积回报
        - `entropy`: 策略熵

        返回值：
        - Actor损失
        - Critic损失
        - 熵损失
        - 总损失
        """
        # 优势函数
        # detach防止梯度通过价值函数流向Actor
        advantages = returns - state_values.detach()

        # Actor损失
        # 负号因为优化器执行最小化
        actor_loss = -torch.mean(log_probs * advantages)

        # Critic损失
        critic_loss = torch.mean((returns - state_values) ** 2)

        # 熵损失：鼓励探索
        entropy_loss = -torch.mean(entropy)

        # 总损失
        total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        return actor_loss, critic_loss, entropy_loss, total_loss


def compute_td_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    ## 计算TD目标值

    如果episode结束（done=True），则目标值等于奖励。

    参数：
    - `rewards`: 每步奖励
    - `values`: 当前状态价值
    - `next_values`: 下一状态价值
    - `dones`: 是否episode结束
    - `gamma`: 折扣因子

    返回值：
    - TD目标值
    """
    td_targets = rewards + gamma * next_values * (1 - dones)
    return td_targets


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    ## 计算优势函数（单步TD）

    参数：
    - `rewards`: 每步奖励
    - `values`: 当前状态价值
    - `next_values`: 下一状态价值
    - `dones`: 是否episode结束
    - `gamma`: 折扣因子

    返回值：
    - 优势函数估计
    """
    td_targets = compute_td_targets(rewards, values, next_values, dones, gamma)
    advantages = td_targets - values
    return advantages
