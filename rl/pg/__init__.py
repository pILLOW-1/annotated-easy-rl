r"""
---
title: 策略梯度 (Policy Gradient) - REINFORCE算法
summary: >
  基于蘑菇书EasyRL第四章的策略梯度算法的PyTorch实现，
  包含REINFORCE算法、基线技巧、折扣回报等核心公式的逐行注释。
---

# 策略梯度 (Policy Gradient) - REINFORCE算法

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第四章的PyTorch实现。

策略梯度方法直接优化策略 $\pi_\theta(a|s)$，通过梯度上升最大化期望累积奖励。

## 策略梯度定理

### 轨迹概率

一条轨迹 $\tau = (s_1, a_1, s_2, a_2, \ldots, s_T, a_T)$ 的概率为：

$$p_\theta(\tau) = p(s_1) \prod_{t=1}^{T} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)$$

### 期望奖励

$$\bar{R}_\theta = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]$$

其中 $R(\tau) = \sum_{t=1}^{T} r_t$ 是轨迹的总奖励。

### 策略梯度

对期望奖励求梯度：

$$\nabla_\theta \bar{R}_\theta = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[R(\tau) \nabla_\theta \log p_\theta(\tau)\right]$$

使用log-derivative技巧 $\nabla f(x) = f(x) \nabla \log f(x)$：

\begin{align}
\nabla_\theta \bar{R}_\theta &= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[R(\tau) \nabla_\theta \log p_\theta(\tau)\right] \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[R(\tau) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)\right] \\
&\approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla_\theta \log \pi_\theta(a_t^n | s_t^n)
\end{align}

## 基线技巧 (Baseline)

策略梯度的方差很大。我们可以减去一个基线 $b$ 来降低方差：

$$\nabla_\theta \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R(\tau^n) - b) \nabla_\theta \log \pi_\theta(a_t^n | s_t^n)$$

基线 $b$ 可以是状态价值函数 $V(s_t)$，此时：

$$\nabla_\theta \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A(s_t^n, a_t^n) \nabla_\theta \log \pi_\theta(a_t^n | s_t^n)$$

其中 $A(s_t, a_t) = R(\tau^n) - V(s_t)$ 是优势函数。

## 折扣回报 (Discounted Return)

使用折扣因子 $\gamma$ 来权衡近期和远期奖励：

$$G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$$

折扣回报满足递归关系：
$$G_t = r_{t+1} + \gamma G_{t+1}$$

REINFORCE更新规则：
$$\nabla_\theta \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} G_t^n \nabla_\theta \log \pi_\theta(a_t^n | s_t^n)$$

参数更新：
$$\theta \leftarrow \theta + \eta \nabla_\theta \bar{R}_\theta$$

其中 $\eta$ 是学习率。
"""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    r"""
 ## 策略网络

    策略网络 $\pi_\theta(a|s)$ 输出在给定状态下每个动作的概率分布。

    $$\pi_\theta(a|s) = \frac{e^{f_\theta(s, a)}}{\sum_{a'} e^{f_\theta(s, a')}}$$

    其中 $f_\theta(s, a)$ 是网络的输出（logits）。
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
        r"""
        输出策略分布 $\pi_\theta(\cdot|s)$。

        参数：
        - `state`: $s$ — 状态

        返回值：
        - 动作概率分布 $\pi_\theta(a|s)$
        """
        logits = self.network(state)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs

    def select_action(self, state: torch.Tensor) -> tuple:
        r"""
        根据策略 $\pi_\theta(a|s)$ 采样动作。

        返回值：
        - 采样的动作 $a \sim \pi_\theta(\cdot|s)$
        - 动作的对数概率 $\log \pi_\theta(a|s)$
        """
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class ValueNetwork(nn.Module):
    r"""
 ## 价值网络

    价值网络 $V_\phi(s)$ 估计状态的期望累积奖励。

    $$V_\phi(s) \approx \mathbb{E}_\pi\left[G_t \mid s_t = s\right]$$

    用作策略梯度中的基线 $b = V_\phi(s_t)$。
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
        r"""
        输出状态价值 $V_\phi(s)$。

        参数：
        - `state`: $s$ — 状态

        返回值：
        - 状态价值 $V_\phi(s)$
        """
        return self.network(state).squeeze(-1)


def compute_discounted_returns(rewards: list, gamma: float) -> list:
    r"""
 ## 计算折扣累积回报

    $$G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k = r_{t+1} + \gamma G_{t+1}$$

    从后往前递归计算，时间复杂度 $O(T)$。

    参数：
    - `rewards`: $[r_1, r_2, \ldots, r_T]$ — 轨迹的奖励序列
    - `gamma`: $\gamma$ — 折扣因子

    返回值：
    - `returns`: $[G_0, G_1, \ldots, G_{T-1}]$ — 每步的折扣累积回报
    """
    returns = []
    discounted_sum = 0

 # 从后往前计算
    for reward in reversed(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    return returns


def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    r"""
 ## REINFORCE损失函数

    REINFORCE更新规则：
    $$\nabla_\theta \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} G_t^n \nabla_\theta \log \pi_\theta(a_t^n | s_t^n)$$

    由于优化器执行最小化，我们取负号：
    $$L(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} G_t^n \log \pi_\theta(a_t^n | s_t^n)$$

    参数：
    - `log_probs`: $[\log \pi_\theta(a_1|s_1), \ldots, \log \pi_\theta(a_T|s_T)]$ — 每步动作的对数概率
    - `returns`: $[G_0, G_1, \ldots, G_{T-1}]$ — 每步的折扣累积回报

    返回值：
    - REINFORCE损失（取负号）
    """
 #
    loss = -torch.sum(log_probs * returns)
    return loss


def actor_critic_loss(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor = None,
    entropy_coef: float = 0.01,
) -> tuple:
    r"""
 ## Actor-Critic损失函数

    Actor-Critic结合策略梯度和价值函数：
    $$\nabla_\theta J \approx \mathbb{E}_t\left[A(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

    其中优势函数 $A(s_t, a_t) = G_t - V(s_t)$。

    Critic学习价值函数：
    $$L_{critic} = \mathbb{E}_t\left[(G_t - V(s_t))^2\right]$$

    可选的熵正则化项鼓励探索：
    $$L_{entropy} = -\mathbb{E}_t\left[\mathcal{H}(\pi_\theta(\cdot|s_t))\right]$$

    总损失：
    $$L = L_{actor} + c_1 L_{critic} + c_2 L_{entropy}$$

    参数：
    - `log_probs`: $\log \pi_\theta(a_t|s_t)$ — 动作的对数概率
    - `values`: $V(s_t)$ — 状态价值估计
    - `returns`: $G_t$ — 折扣累积回报
    - `entropy`: $\mathcal{H}(\pi_\theta(\cdot|s_t))$ — 策略熵（可选）
    - `entropy_coef`: 熵正则化系数

    返回值：
    - Actor损失
    - Critic损失
    - 总损失
    """
 # 优势函数：
    advantages = returns - values.detach()

 # Actor损失：
    actor_loss = -torch.mean(log_probs * advantages)

 # Critic损失：
    critic_loss = torch.mean((returns - values) ** 2)

 # 总损失
    total_loss = actor_loss + 0.5 * critic_loss

    if entropy is not None:
        total_loss -= entropy_coef * entropy.mean()

    return actor_loss, critic_loss, total_loss
