"""
---
title: 深度Q网络 (DQN) - Deep Q Networks
summary: >
  基于蘑菇书EasyRL第六章的深度Q网络(DQN)算法的PyTorch实现，
  包含目标网络、经验回放、Dueling网络、Double DQN等核心公式的逐行注释。
---

# 深度Q网络 (DQN) - Deep Q Networks

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第六章的PyTorch实现，
参考论文[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，
以及[Dueling Network](https://arxiv.org/abs/1511.06581)和
[Double Q-Learning](https://arxiv.org/abs/1509.06461)。

## Q函数与贝尔曼方程

在强化学习中，Q函数定义为：

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t=s, a_t=a\right]$$

最优Q函数满足贝尔曼最优方程：

$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{E}}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

## DQN的核心思想

DQN使用神经网络来近似Q函数：$Q(s, a; \theta) \approx Q^*(s, a)$

训练目标是使Q函数的预测满足贝尔曼方程。定义目标值：

$$y_i = r_i + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^-)$$

其中 $\theta^-$ 是目标网络的参数，定期从主网络 $\theta$ 复制。

损失函数为：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)}\left[(y_i - Q(s, a; \theta))^2\right]$$

## 目标网络 🎯

为了提高训练稳定性，DQN使用两个网络：
- **主网络** (policy network)：参数 $\theta$，用于选择动作和计算Q值
- **目标网络** (target network)：参数 $\theta^-$，定期从主网络复制，用于计算目标值

目标值：
$$y_i = r_i + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^-)$$

目标网络的参数 $\theta^-$ 每隔一定步数更新为 $\theta$，这打破了样本间的相关性，提高了训练稳定性。

## Double DQN

标准DQN中，max操作符使用同一个网络来选择和评估动作：

$$\max_{a'} Q(s', a'; \theta) = Q\left(s', \underset{a'}{\arg\max} Q(s', a'; \theta); \theta\right)$$

这会导致Q值的高估。Double DQN解耦了动作选择和价值评估：

$$\max_{a'} Q(s', a'; \theta^-) = Q\left(s', \underset{a'}{\arg\max} Q(s', a'; \theta); \theta^-\right)$$

- 使用主网络 $\theta$ 选择动作：$\underset{a'}{\arg\max} Q(s', a'; \theta)$
- 使用目标网络 $\theta^-$ 评估价值：$Q(s', \text{best\_action}; \theta^-)$

Double DQN的损失函数：

$$L(\theta) = \mathbb{E}\left[\left(r + \gamma Q\left(s', \underset{a'}{\arg\max} Q(s', a'; \theta); \theta^-\right) - Q(s, a; \theta)\right)^2\right]$$

## Dueling Network ⚔️

Dueling网络将Q值分解为状态价值和优势函数：

\begin{align}
Q^\pi(s, a) &= V^\pi(s) + A^\pi(s, a) \\
\mathbb{E}_{a \sim \pi(s)}\left[A^\pi(s, a)\right] &= 0
\end{align}

实际计算中：

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a' \in \mathcal{A}} A(s, a')\right)$$

Dueling网络的优势在于：
- 在很多状态下，动作并不重要（优势函数接近0）
- 在少数关键状态下，动作非常重要
- 这种结构让网络可以更好地学习这种差异

## 经验回放

DQN使用经验回放缓冲区来存储和采样经验：

$$D = \{(s_t, a_t, r_t, s_{t+1})\}$$

训练时从缓冲区中随机采样：
$$(s, a, r, s') \sim U(D)$$

经验回放的好处：
1. 打破样本间的时间相关性
2. 提高数据利用率
3. 使训练更加稳定

## 探索策略

### $\varepsilon$-贪心策略

$$a = \begin{cases}
\underset{a}{\arg\max} Q(s, a) & \text{概率 } 1-\varepsilon \\
\text{随机动作} & \text{概率 } \varepsilon
\end{cases}$$

$\varepsilon$ 通常从大到小衰减，初期多探索，后期多利用。

### Boltzmann探索

$$\pi(a|s) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}$$

其中 $T$ 是温度参数，控制探索程度。
"""

from typing import Tuple

import torch
from torch import nn


class QFuncLoss(nn.Module):
    """
    ## DQN损失函数

    我们想要找到最优的动作价值函数：

    \begin{align}
    Q^*(s,a) &= \max_\pi \mathbb{E}\left[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t=s, a_t=a, \pi\right] \\
    Q^*(s,a) &= \mathbb{E}_{s' \sim \mathcal{E}}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]
    \end{align}

    ### 目标网络 🎯

    使用经验回放 $U(D)$ 随机采样历史经验，并使用独立的Q网络
    参数 $\textcolor{orange}{\theta^-}$ 来计算目标值。
    $\textcolor{orange}{\theta^-}$ 定期更新。

    损失函数：
    $$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)}\left[\left(r + \gamma \max_{a'} Q(s', a'; \textcolor{orange}{\theta^-}) - Q(s,a;\theta)\right)^2\right]$$

    ### Double Q-Learning

    使用Double Q-Learning，其中：
    - $\textcolor{cyan}{\theta}$ 用于选择动作
    - $\textcolor{orange}{\theta^-}$ 用于评估价值

    \begin{align}
    \mathcal{L}(\theta) = \mathbb{E}\Bigg[\bigg(&r + \gamma \textcolor{orange}{Q}\Big(s', \underset{a'}{\arg\max} \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta}); \textcolor{orange}{\theta^-}\Big) \\
    &- Q(s,a;\theta)\bigg)^2\Bigg]
    \end{align}
    """

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.huber_loss = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        q: torch.Tensor,
        action: torch.Tensor,
        double_q: torch.Tensor,
        target_q: torch.Tensor,
        done: torch.Tensor,
        reward: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算DQN损失。

        参数映射：
        - `q`: $Q(s,a;\theta)$ — 主网络输出的Q值
        - `action`: $a$ — 采取的动作
        - `double_q`: $\textcolor{cyan}{Q}(s';\textcolor{cyan}{\theta})$ — 主网络对下一状态的Q值
        - `target_q`: $\textcolor{orange}{Q}(s';\textcolor{orange}{\theta^-})$ — 目标网络对下一状态的Q值
        - `done`: 是否episode结束
        - `reward`: $r$ — 奖励
        - `weights`: 优先级经验回放的权重

        返回值：
        - TD误差（用于优先级回放）
        - 损失值
        """
        # $Q(s,a;\theta)$ — 选择当前状态-动作对的Q值
        q_sampled_action = q.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)

        # 不传播梯度到目标网络
        # $$r + \gamma \textcolor{orange}{Q}\Big(s', \underset{a'}{\arg\max} \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta}); \textcolor{orange}{\theta^-}\Big)$$
        with torch.no_grad():
            # 使用主网络选择最佳动作
            # $$\underset{a'}{\arg\max} \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta})$$
            best_next_action = torch.argmax(double_q, -1)

            # 使用目标网络评估最佳动作的价值
            # $$\textcolor{orange}{Q}\Big(s', \underset{a'}{\arg\max} \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta}); \textcolor{orange}{\theta^-}\Big)$$
            best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)

            # 计算目标Q值
            # 乘以 `(1 - done)` 确保episode结束时不加上下一状态的价值
            # $$r + \gamma \textcolor{orange}{Q}\Big(s', \text{best\_action}; \textcolor{orange}{\theta^-}\Big)$$
            q_update = reward + self.gamma * best_next_q_value * (1 - done)

            # TD误差 $\delta = Q(s,a;\theta) - y$
            td_error = q_sampled_action - q_update

        # 使用Huber损失代替MSE，对异常值更不敏感
        losses = self.huber_loss(q_sampled_action, q_update)

        if weights is not None:
            # 优先级经验回放的加权损失
            loss = torch.mean(weights * losses)
        else:
            loss = losses.mean()

        return td_error, loss


class DuelingQNetwork(nn.Module):
    """
    ## Dueling网络 ⚔️ 结构

    使用[Dueling网络](https://arxiv.org/abs/1511.06581)来计算Q值。

    Dueling网络的直觉：
    - 在大多数状态下，动作并不重要
    - 在少数状态下，动作非常关键
    - Dueling网络可以很好地表示这种差异

    \begin{align}
    Q^\pi(s,a) &= V^\pi(s) + A^\pi(s, a) \\
    \mathbb{E}_{a \sim \pi(s)}\left[A^\pi(s, a)\right] &= 0
    \end{align}

    实际计算：
    $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a' \in \mathcal{A}} A(s, a')\right)$$

    $V$ 和 $A$ 网络共享前面的卷积层。
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

        # 状态价值头 $V(s)$
        self.state_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 优势函数头 $A(s, a)$
        self.action_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # 共享特征提取
        features = self.feature(state)

        # $V(s)$ — 状态价值
        state_value = self.state_value(features)

        # $A(s, a)$ — 动作优势
        action_value = self.action_value(features)

        # $A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$ — 中心化的优势函数
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)

        # $Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$
        q = state_value + action_score_centered

        return q


class EpsilonGreedy:
    """
    ## $\varepsilon$-贪心探索策略

    $$a = \begin{cases}
    \underset{a}{\arg\max} Q(s, a) & \text{概率 } 1-\varepsilon \\
    \text{随机动作} & \text{概率 } \varepsilon
    \end{cases}$$

    $\varepsilon$ 随时间衰减：
    $$\varepsilon_t = \varepsilon_{end} + (\varepsilon_{start} - \varepsilon_{end}) \times e^{-t/\text{decay}}$$
    """

    def __init__(
        self,
        action_dim: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay: int = 1000,
    ):
        self.action_dim = action_dim
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = decay
        self.epsilon = epsilon_start
        self.count = 0

    def select_action(self, q_values: torch.Tensor) -> int:
        """
        根据$\varepsilon$-贪心策略选择动作。

        参数：
        - `q_values`: $Q(s, a;\theta)$ — 当前状态的Q值

        返回值：
        - 选择的动作
        """
        self.count += 1
        # 衰减$\varepsilon$
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
            -self.count / self.decay
        )
        self.epsilon = max(self.epsilon, self.epsilon_end)

        if torch.rand(1).item() < self.epsilon:
            # 探索：随机动作
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            # 利用：选择Q值最大的动作
            return torch.argmax(q_values).item()
