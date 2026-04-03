"""
---
title: 广义优势估计 (GAE) - Generalized Advantage Estimation
summary: 蘑菇书EasyRL中PPO算法使用的广义优势估计(GAE)的PyTorch实现。
---

# 广义优势估计 (GAE) - Generalized Advantage Estimation

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)中PPO算法使用的
广义优势估计(Generalized Advantage Estimation, GAE)的PyTorch实现。

参考论文：[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## 优势函数估计的权衡

在策略梯度方法中，我们需要估计优势函数 $A(s_t, a_t)$。有多种估计方式：

### k步估计

\begin{align}
\hat{A}_t^{(1)} &= r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t^{(2)} &= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t) \\
&\vdots \\
\hat{A}_t^{(\infty)} &= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t)
\end{align}

- $\hat{A}_t^{(1)}$：**高偏差，低方差** — 只使用一步奖励，估计不准确但稳定
- $\hat{A}_t^{(\infty)}$：**无偏差，高方差** — 使用完整回报，准确但不稳定

### GAE：加权平均

GAE通过对不同k步估计进行加权平均来平衡偏差和方差：

$$\hat{A}_t^{GAE} = \frac{\sum_{k=1}^{\infty} w_k \hat{A}_t^{(k)}}{\sum_{k=1}^{\infty} w_k}$$

取权重 $w_k = \lambda^{k-1}$，可以得到简洁的递归计算形式。

### GAE的递归形式

定义TD残差：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

则GAE可以递归计算：
$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

其中：
- $\gamma$：折扣因子，控制未来奖励的衰减
- $\lambda$：GAE参数，控制偏差-方差权衡
  - $\lambda = 0$：退化为单步TD估计（高偏差，低方差）
  - $\lambda = 1$：退化为蒙特卡洛估计（无偏差，高方差）

## 蘑菇书中的公式

根据蘑菇书第五章，PPO的优势函数估计为：

$$A_t = G_t - V(s_t)$$

其中 $G_t$ 是折扣累积回报：
$$G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k = r_{t+1} + \gamma G_{t+1}$$

在GAE中，我们使用更一般的优势估计：
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
"""

import numpy as np


class GAE:
    """
    ## 广义优势估计计算器

    实现GAE公式：
    $$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

    其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是TD残差。
    """

    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        """
        初始化GAE计算器。

        参数：
        - `n_workers`: 并行环境数量
        - `worker_steps`: 每个环境的步数
        - `gamma`: $\gamma$ — 折扣因子
        - `lambda_`: $\lambda$ — GAE参数
        """
        self.lambda_ = lambda_
        self.gamma = gamma
        self.worker_steps = worker_steps
        self.n_workers = n_workers

    def __call__(
        self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        计算优势函数。

        参数映射：
        - `done`: 是否 episode 结束
        - `rewards`: $r_t$ — 每步奖励
        - `values`: $V(s_t)$ — 每步价值估计

        返回值：
        - `advantages`: $\hat{A}_t$ — 优势函数估计

        计算过程：
        $$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$
        其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
        """
        # 优势函数表
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        # $V(s_{T+1})$ — 最后一步的下一个状态价值
        last_value = values[:, -1]

        # 从后往前递归计算
        for t in reversed(range(self.worker_steps)):
            # 如果episode已结束，mask为0，否则为1
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            # TD残差：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            # GAE递归：$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage

            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages


def compute_returns(rewards: np.ndarray, gamma: float, dones: np.ndarray) -> np.ndarray:
    """
    ## 计算折扣累积回报

    $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = r_t + \gamma G_{t+1}$$

    这是蘑菇书中使用的简单回报计算方式，不使用GAE时的替代方案。

    参数：
    - `rewards`: $r_t$ — 每步奖励
    - `gamma`: $\gamma$ — 折扣因子
    - `dones`: 是否episode结束

    返回值：
    - `returns`: $G_t$ — 折扣累积回报
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    last_return = 0

    for t in reversed(range(rewards.shape[1])):
        mask = 1.0 - dones[:, t]
        last_return = last_return * mask
        # $G_t = r_t + \gamma G_{t+1}$
        last_return = rewards[:, t] + gamma * last_return
        returns[:, t] = last_return

    return returns
