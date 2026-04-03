r"""
---
title: 表格型方法 (Tabular Methods) - Q-Learning & Sarsa
summary: >
  基于蘑菇书EasyRL第三章的表格型强化学习方法的Python实现，
  包含Q-Learning、Sarsa、Value Iteration等核心算法的逐行注释。
---

# 表格型方法 (Tabular Methods) - Q-Learning & Sarsa

本文件是[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)第三章的Python实现。

表格型方法使用表格来存储状态-动作价值函数 $Q(s, a)$ 或状态价值函数 $V(s)$。
适用于状态空间和动作空间都是离散且较小的场景。

## 贝尔曼方程

### 状态价值函数

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s\right]$$

贝尔曼期望方程：
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) \left[r + \gamma V^\pi(s')\right]$$

### 动作价值函数

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s, a_t = a\right]$$

贝尔曼最优方程：
$$Q^*(s, a) = \sum_{s', r} p(s', r|s, a) \left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

## Q-Learning (异策略)

Q-Learning是异策略(off-policy)的时序差分(TD)控制算法。

### Q值更新规则

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)\right]$$

其中：
- $\alpha$ 是学习率
- $\gamma$ 是折扣因子
- $r_{t+1} + \gamma \max_a Q(s_{t+1}, a)$ 是TD目标
- $r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)$ 是TD误差

Q-Learning使用 $\max_a Q(s_{t+1}, a)$ 来估计下一状态的价值，
这对应于贪婪策略 $\pi(s) = \arg\max_a Q(s, a)$，
但行为策略可以是 $\varepsilon$-贪心策略，因此是异策略的。

## Sarsa (同策略)

Sarsa是同策略(on-policy)的时序差分(TD)控制算法。

### Q值更新规则

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]$$

Sarsa使用实际采取的下一个动作 $a_{t+1}$ 的Q值来更新，
因此学习的就是行为策略本身的价值，是同策略的。

## Q-Learning vs Sarsa

| 特性 | Q-Learning | Sarsa |
|------|-----------|-------|
| 策略类型 | 异策略(off-policy) | 同策略(on-policy) |
| TD目标 | $r + \gamma \max_a Q(s', a)$ | $r + \gamma Q(s', a')$ |
| 行为策略 | $\varepsilon$-贪心 | $\varepsilon$-贪心 |
| 目标策略 | 贪婪 | $\varepsilon$-贪心 |
| 收敛性 | 收敛到最优Q值 | 收敛到行为策略的Q值 |

## 价值迭代 (Value Iteration)

价值迭代是一种动态规划算法，用于求解已知环境模型的MDP。

### 更新规则

$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r|s, a) \left[r + \gamma V_k(s')\right]$$

价值迭代直接更新状态价值函数，直到收敛。
收敛后，最优策略可以通过贪婪选择得到：

$$\pi^*(s) = \arg\max_a \sum_{s', r} p(s', r|s, a) \left[r + \gamma V^*(s')\right]$$

## $\varepsilon$-贪心策略

$$\pi(a|s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}$$

或者简化为：
$$a = \begin{cases}
\arg\max_a Q(s, a) & \text{概率 } 1-\varepsilon \\
\text{随机动作} & \text{概率 } \varepsilon
\end{cases}$$
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    r"""
    ## Q-Learning智能体

    Q-Learning更新规则：
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

    其中：
    - $\alpha$：学习率
    - $\gamma$：折扣因子
    - $r + \gamma \max_{a'} Q(s', a')$：TD目标
    - $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$：TD误差 $\delta$
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate  # $\alpha$
        self.gamma = gamma  # $\gamma$
        self.epsilon = epsilon  # $\varepsilon$
        # Q表：$Q(s, a)$
        self.q_table = np.zeros((state_dim, action_dim), dtype=np.float32)

    def select_action(self, state: int) -> int:
        r"""
        使用$\varepsilon$-贪心策略选择动作。

        $$a = \begin{cases}
        \arg\max_a Q(s, a) & \text{概率 } 1-\varepsilon \\
        \text{随机动作} & \text{概率 } \varepsilon
        \end{cases}$$
        """
        if np.random.random() < self.epsilon:
            # 探索：随机动作
            return np.random.randint(self.action_dim)
        else:
            # 利用：贪婪动作
            return np.argmax(self.q_table[state])

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ):
        r"""
        更新Q值。

        $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') \times (1 - \text{done}) - Q(s, a)\right]$$

        参数：
        - `state`: $s_t$ — 当前状态
        - `action`: $a_t$ — 当前动作
        - `reward`: $r_{t+1}$ — 奖励
        - `next_state`: $s_{t+1}$ — 下一状态
        - `done`: 是否episode结束
        """
        # TD目标：$r + \gamma \max_{a'} Q(s', a') \times (1 - \text{done})$
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD误差：$\delta = \text{TD目标} - Q(s, a)$
        td_error = td_target - self.q_table[state, action]

        # 更新Q值：$Q(s, a) \leftarrow Q(s, a) + \alpha \delta$
        self.q_table[state, action] += self.lr * td_error


class SarsaAgent:
    r"""
    ## Sarsa智能体

    Sarsa更新规则：
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma Q(s', a') - Q(s, a)\right]$$

    与Q-Learning不同，Sarsa使用实际采取的下一个动作 $a'$ 的Q值，
    因此是同策略(on-policy)算法。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_dim, action_dim), dtype=np.float32)

    def select_action(self, state: int) -> int:
        r"""$\varepsilon$-贪心策略选择动作。"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state])

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ):
        r"""
        更新Q值。

        $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma Q(s', a') \times (1 - \text{done}) - Q(s, a)\right]$$

        注意：Sarsa需要下一个动作 $a'$ 来更新，这是同策略的关键。
        """
        # TD目标：$r + \gamma Q(s', a') \times (1 - \text{done})$
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state, next_action]

        # TD误差
        td_error = td_target - self.q_table[state, action]

        # 更新Q值
        self.q_table[state, action] += self.lr * td_error


class ValueIterationAgent:
    r"""
    ## 价值迭代智能体

    价值迭代是一种动态规划算法，适用于已知环境模型的场景。

    更新规则：
    $$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r|s, a) \left[r + \gamma V_k(s')\right]$$

    收敛后，最优策略：
    $$\pi^*(s) = \arg\max_a \sum_{s', r} p(s', r|s, a) \left[r + \gamma V^*(s')\right]$$
    """

    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99, theta: float = 1e-6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.theta = theta  # 收敛阈值
        self.value_table = np.zeros(state_dim, dtype=np.float32)
        self.policy = np.zeros(state_dim, dtype=np.int32)
        # 环境模型：P[s][a] = [(prob, next_state, reward, done), ...]
        self.P = None

    def set_model(self, P: dict):
        """
        设置环境模型。

        参数：
        - `P`: 状态转移模型
          P[s][a] = [(prob, next_state, reward, done), ...]
          其中prob是转移到next_state的概率
        """
        self.P = P

    def iterate(self):
        r"""
        执行一次价值迭代。

        $$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r|s, a) \left[r + \gamma V_k(s')\right]$$

        返回值：
        - 价值函数的最大变化量
        """
        delta = 0
        for s in range(self.state_dim):
            v = self.value_table[s]
            # 计算每个动作的期望价值
            action_values = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                for prob, next_state, reward, done in self.P[s][a]:
                    # 期望价值：$\sum_{s', r} p(s', r|s, a) [r + \gamma V(s')]$
                    action_values[a] += prob * (reward + self.gamma * self.value_table[next_state] * (1 - done))

            # 选择最大价值：$V_{k+1}(s) = \max_a Q(s, a)$
            self.value_table[s] = np.max(action_values)
            delta = max(delta, abs(v - self.value_table[s]))

        return delta

    def solve(self, max_iterations: int = 1000):
        """
        求解最优价值函数和策略。

        迭代直到价值函数收敛（变化小于theta）或达到最大迭代次数。
        """
        for _ in range(max_iterations):
            delta = self.iterate()
            if delta < self.theta:
                break

        # 提取最优策略
        self.extract_policy()

    def extract_policy(self):
        r"""
        从最优价值函数提取最优策略。

        $$\pi^*(s) = \arg\max_a \sum_{s', r} p(s', r|s, a) \left[r + \gamma V^*(s')\right]$$
        """
        for s in range(self.state_dim):
            action_values = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                for prob, next_state, reward, done in self.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.value_table[next_state] * (1 - done))
            self.policy[s] = np.argmax(action_values)
