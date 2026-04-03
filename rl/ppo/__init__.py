"""
---
title: 近端策略优化 (PPO) - Proximal Policy Optimization
summary: >
  基于蘑菇书EasyRL第五章的近端策略优化(PPO)算法的PyTorch实现，
  包含重要性采样、PPO-Penalty(PPO1)、PPO-Clip(PPO2)等核心公式的逐行注释。
---

# 近端策略优化 (PPO) - Proximal Policy Optimization

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第五章的PyTorch实现，
参考论文[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

PPO是一种基于策略梯度的强化学习算法。简单的策略梯度方法每个样本只做一次梯度更新，
如果对同一个样本做多次梯度更新会导致策略偏离过大，产生糟糕的策略。
PPO通过限制新旧策略之间的差异，使得我们可以对同一批样本做多次梯度更新。
它通过裁剪梯度流来实现这一目标，如果更新后的策略与采样时的策略差异过大，就会裁剪梯度。

## 策略梯度基础

在策略梯度方法中，我们希望最大化策略的期望累积奖励。

其中：
- r_t 是时刻 t 的奖励
- pi_theta 是参数为 theta 的策略
- tau 是从策略中采样的一条轨迹
- gamma 是折扣因子

根据log-derivative技巧，策略梯度可以写为带优势函数的形式。

## 重要性采样与异策略学习

我们可以使用重要性采样将期望从一个策略转移到另一个策略。

这让我们可以使用旧策略采集的数据来更新新策略。

## PPO目标函数

### PPO1 (带KL惩罚)

PPO1在目标函数中添加KL散度惩罚项。

beta 是自适应调整的参数。

### PPO2 (裁剪版本)

PPO2使用裁剪的目标函数。

其中 epsilon 是超参数（通常取0.1~0.2）。

裁剪操作确保新旧策略的比率不会偏离1太多，从而限制了策略更新的幅度。

## 优势函数估计

优势函数可以通过多种方式估计：

1. **蒙特卡洛估计**：A = G_t - V(s_t)
2. **TD残差**：A = r_t + gamma * V(s_{t+1}) - V(s_t)
3. **GAE (Generalized Advantage Estimation)**：加权TD残差的和

其中 delta_t 是TD残差。
"""

import torch
from torch import nn


class ClippedPPOLoss(nn.Module):
    """
    ## PPO裁剪损失

    PPO2的核心：裁剪策略比率以限制策略更新幅度。

    目标函数使用裁剪的策略比率和优势函数的乘积的最小值。

    推导过程使用重要性采样将期望从新策略转移到旧策略，
    得到CPI (Conservative Policy Iteration) 的目标函数。
    """

    def forward(
        self,
        log_pi: torch.Tensor,
        sampled_log_pi: torch.Tensor,
        advantage: torch.Tensor,
        clip: float,
    ) -> torch.Tensor:
        """
        计算PPO裁剪损失。

        参数映射：
        - `log_pi`: 当前策略的对数概率
        - `sampled_log_pi`: 旧策略的对数概率
        - `advantage`: 优势函数估计
        - `clip`: 裁剪参数

        返回值：
        - PPO损失（取负号，因为优化器执行最小化）
        """
        # 策略比率 = exp(log_pi - sampled_log_pi)
        # 注意：这里的 ratio 不同于奖励 r_t
        ratio = torch.exp(log_pi - sampled_log_pi)

        # ### 裁剪策略比率
        #
        # 策略比率被裁剪到 [1-epsilon, 1+epsilon] 范围内。
        # 取最小值确保：
        # - 当优势 > 0 时，ratio 不会超过 1+epsilon
        # - 当优势 < 0 时，ratio 不会低于 1-epsilon
        #
        # 这样限制了新旧策略之间的KL散度，防止策略更新过大导致性能崩溃。
        # 使用归一化的优势函数虽然引入了偏差，但大幅降低了方差。
        clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)

        # 记录裁剪比例，用于监控
        self.clip_fraction = (torch.abs(ratio - 1.0) > clip).to(torch.float).mean()

        # 取负号因为优化器执行最小化
        return -policy_reward.mean()


class ClippedValueFunctionLoss(nn.Module):
    """
    ## 裁剪的价值函数损失

    类似于策略的裁剪，价值函数的更新也需要裁剪以防止过大变化。

    裁剪的价值函数 = V_old + clip(V - V_old, -epsilon, epsilon)

    损失取未裁剪和裁剪的两个MSE损失的最大值。

    裁剪确保价值函数不会显著偏离旧价值函数。
    """

    def forward(
        self,
        value: torch.Tensor,
        sampled_value: torch.Tensor,
        sampled_return: torch.Tensor,
        clip: float,
    ) -> torch.Tensor:
        """
        计算裁剪的价值函数损失。

        参数映射：
        - `value`: 当前价值函数估计
        - `sampled_value`: 旧价值函数估计
        - `sampled_return`: 蒙特卡洛回报
        - `clip`: 裁剪参数
        """
        # 裁剪的价值函数
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)

        # 取两个损失的最大值：未裁剪的和裁剪的
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)

        # 乘以1/2是为了与MSE损失保持一致
        return 0.5 * vf_loss.mean()


class PPONetwork(nn.Module):
    """
    ## PPO网络

    共享特征提取器的Actor-Critic网络结构。
    Actor输出策略，Critic输出价值。

    PPO通常使用独立的Actor和Critic头：
    - Actor: 特征 -> 策略分布
    - Critic: 特征 -> 价值估计
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

        # Actor头：输出每个动作的对数概率（用于分类动作空间）
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic头：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        # 提取特征
        features = self.feature(state)

        # Actor: 输出动作 logits
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Critic: 输出状态价值
        state_value = self.critic(features).squeeze(-1)

        return action_probs, state_value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        评估给定状态-动作对的对数概率和价值。

        用于PPO更新时计算对数概率。
        """
        features = self.feature(state)
        action_logits = self.actor(features)
        state_value = self.critic(features).squeeze(-1)

        # 计算动作的对数概率
        action_probs = torch.softmax(action_logits, dim=-1)
        action_log_probs = torch.log(action_probs + 1e-8)

        # 选择给定动作的对数概率
        log_pi = action_log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return log_pi, state_value
