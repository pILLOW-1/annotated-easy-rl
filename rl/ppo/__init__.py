r"""
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

在策略梯度方法中，我们希望最大化策略的期望累积奖励：

$$\max_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

其中：
- $r_t$ 是时刻 $t$ 的奖励
- $\pi_\theta$ 是参数为 $\theta$ 的策略
- $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是从策略中采样的一条轨迹
- $\gamma \in [0, 1]$ 是折扣因子

轨迹的概率为：

$$p_\theta(\tau) = p(s_1) \prod_{t=1}^{T} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)$$

根据log-derivative技巧 $\nabla f(x) = f(x) \nabla \log f(x)$，策略梯度可以写为：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

其中 $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$ 是优势函数。

## 重要性采样与异策略学习

我们可以使用重要性采样将期望从 $\pi_\theta$ 转移到另一个策略 $\pi_{\theta'}$：

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$$

应用重要性采样，策略梯度可以改写为：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

这让我们可以使用旧策略 $\pi_{\theta'}$ 采集的数据来更新新策略 $\pi_\theta$。

## PPO目标函数

### PPO1 (带KL惩罚)

PPO1在目标函数中添加KL散度惩罚项：

$$J_{\text{PPO}}^{\theta^k}(\theta) = J^{\theta^k}(\theta) - \beta \text{KL}(\theta, \theta^k)$$

其中：

$$J^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta^k}(a_t | s_t)} A^{\theta^k}(s_t, a_t)$$

$\beta$ 是自适应调整的参数。

### PPO2 (裁剪版本)

PPO2使用裁剪的目标函数：

$$J_{\text{PPO2}}^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \min\left(\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta^k}(a_t | s_t)} A^{\theta^k}(s_t, a_t), \text{clip}\left(\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta^k}(a_t | s_t)}, 1-\varepsilon, 1+\varepsilon\right) A^{\theta^k}(s_t, a_t)\right)$$

其中 $\varepsilon$ 是超参数（通常取0.1~0.2）。

裁剪操作确保新旧策略的比率 $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta^k}(a_t | s_t)}$
不会偏离1太多，从而限制了策略更新的幅度。

## 优势函数估计

优势函数 $A(s_t, a_t)$ 可以通过多种方式估计：

1. **蒙特卡洛估计**：$A(s_t, a_t) = G_t - V(s_t)$，其中 $G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$
2. **TD残差**：$A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$
3. **GAE (Generalized Advantage Estimation)**：$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是TD残差。
"""

import torch
from torch import nn


class ClippedPPOLoss(nn.Module):
    r"""
    ## PPO裁剪损失

    PPO2的核心：裁剪策略比率以限制策略更新幅度。

    目标函数：
    $$J_{\text{PPO2}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

    其中：
    - $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是策略比率
    - $\hat{A}_t$ 是优势函数的估计
    - $\varepsilon$ 是裁剪参数

    推导过程：

    \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{old}})
    &= \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi_{old}}(s_t, a_t)\right] \\
    &= \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[A^{\pi_{old}}(s, a)\right] \\
    &= \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_{\theta_{old}}}, a \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{old}}(s, a)\right] \\
    &\approx \frac{1}{1-\gamma} \mathcal{L}^{CPI}
    \end{align}

    其中 $\mathcal{L}^{CPI}$ 是CPI (Conservative Policy Iteration) 的目标函数。
    """

    def forward(
        self,
        log_pi: torch.Tensor,
        sampled_log_pi: torch.Tensor,
        advantage: torch.Tensor,
        clip: float,
    ) -> torch.Tensor:
        r"""
        计算PPO裁剪损失。

        参数映射：
        - `log_pi`: $\log \pi_\theta(a_t|s_t)$ — 当前策略的对数概率
        - `sampled_log_pi`: $\log \pi_{\theta_{old}}(a_t|s_t)$ — 旧策略的对数概率
        - `advantage`: $\hat{A}_t$ — 优势函数估计
        - `clip`: $\varepsilon$ — 裁剪参数

        返回值：
        - PPO损失（取负号，因为优化器执行最小化）
        """
        # 策略比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$
        # 注意：这里的 ratio 不同于奖励 $r_t$
        ratio = torch.exp(log_pi - sampled_log_pi)

        # ### 裁剪策略比率
        #
        # $$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$
        #
        # 策略比率被裁剪到 $[1-\varepsilon, 1+\varepsilon]$ 范围内。
        # 取最小值确保：
        # - 当 $\hat{A}_t > 0$ 时，$r_t(\theta)$ 不会超过 $1+\varepsilon$
        # - 当 $\hat{A}_t < 0$ 时，$r_t(\theta)$ 不会低于 $1-\varepsilon$
        #
        # 这样限制了新旧策略之间的KL散度，防止策略更新过大导致性能崩溃。
        # 使用归一化的优势函数 $\bar{A}_t = \frac{\hat{A}_t - \mu(\hat{A}_t)}{\sigma(\hat{A}_t)}$
        # 虽然引入了偏差，但大幅降低了方差。
        clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)

        # 记录裁剪比例，用于监控
        self.clip_fraction = (torch.abs(ratio - 1.0) > clip).to(torch.float).mean()

        # 取负号因为优化器执行最小化
        return -policy_reward.mean()


class ClippedValueFunctionLoss(nn.Module):
    r"""
    ## 裁剪的价值函数损失

    类似于策略的裁剪，价值函数的更新也需要裁剪以防止过大变化。

    $$V_{CLIP}^{\pi_\theta}(s_t) = V^{\pi_{\theta_{old}}}(s_t) + \text{clip}\left(V^{\pi_\theta}(s_t) - V^{\pi_{\theta_{old}}}(s_t), -\varepsilon, +\varepsilon\right)$$

    $$\mathcal{L}^{VF}(\theta) = \frac{1}{2} \mathbb{E}_t\left[\max\left((V^{\pi_\theta}(s_t) - R_t)^2, (V_{CLIP}^{\pi_\theta}(s_t) - R_t)^2\right)\right]$$

    裁剪确保价值函数 $V_\theta$ 不会显著偏离 $V_{\theta_{old}}$。
    """

    def forward(
        self,
        value: torch.Tensor,
        sampled_value: torch.Tensor,
        sampled_return: torch.Tensor,
        clip: float,
    ) -> torch.Tensor:
        r"""
        计算裁剪的价值函数损失。

        参数映射：
        - `value`: $V^{\pi_\theta}(s_t)$ — 当前价值函数估计
        - `sampled_value`: $V^{\pi_{\theta_{old}}}(s_t)$ — 旧价值函数估计
        - `sampled_return`: $R_t$ — 蒙特卡洛回报
        - `clip`: $\varepsilon$ — 裁剪参数
        """
        # 裁剪的价值函数：$V_{CLIP} = V_{old} + \text{clip}(V - V_{old}, -\varepsilon, \varepsilon)$
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)

        # 取两个损失的最大值：未裁剪的和裁剪的
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)

        # 乘以1/2是为了与MSE损失保持一致
        return 0.5 * vf_loss.mean()


class PPONetwork(nn.Module):
    r"""
    ## PPO网络

    共享特征提取器的Actor-Critic网络结构。
    Actor输出策略 $\pi_\theta(a|s)$，Critic输出价值 $V_\theta(s)$。

    网络结构：
    $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

    但PPO通常使用独立的Actor和Critic头：
    - Actor: 特征 → 策略分布 $\pi_\theta(\cdot|s)$
    - Critic: 特征 → 价值估计 $V_\theta(s)$
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

        # Critic头：输出状态价值 $V(s)$
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
        r"""
        评估给定状态-动作对的对数概率和价值。

        用于PPO更新时计算 $\log \pi_\theta(a_t|s_t)$。
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
