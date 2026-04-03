r"""
---
title: 强化学习算法实现 (Reinforcement Learning)
summary: >
  基于蘑菇书EasyRL理论公式与代码的强化学习算法注释实现，
  采用labmlai风格的侧边对照注释格式。
---

# 强化学习算法实现

本目录包含[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)中各章节
理论公式与代码的[labmlai风格](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
注释实现。

每个算法文件都将理论公式与代码实现逐行对照，帮助读者更好地理解强化学习算法的数学原理与实现细节。

## 算法列表

### 表格型方法 (Chapter 3)

- [Q-Learning](tabular/__init__.py) — 异策略时序差分控制算法
- [Sarsa](tabular/__init__.py) — 同策略时序差分控制算法
- [Value Iteration](tabular/__init__.py) — 价值迭代动态规划算法

### 策略梯度 (Chapter 4)

- [REINFORCE](pg/__init__.py) — 蒙特卡洛策略梯度算法
- [基线技巧](pg/__init__.py) — 降低策略梯度方差的基线方法
- [折扣回报](pg/__init__.py) — 折扣累积回报的计算

### 近端策略优化 (Chapter 5)

- [PPO-Clip](ppo/__init__.py) — 裁剪版本的PPO算法(PPO2)
- [PPO-Penalty](ppo/__init__.py) — 带KL惩罚的PPO算法(PPO1)
- [GAE](ppo/gae.py) — 广义优势估计

### 深度Q网络 (Chapter 6-8)

- [DQN](dqn/__init__.py) — 深度Q网络
- [Double DQN](dqn/__init__.py) — 双Q网络
- [Dueling DQN](dqn/__init__.py) — Dueling网络结构
- [经验回放](dqn/__init__.py) — 优先级经验回放

### 演员-评论员 (Chapter 9)

- [A2C](a2c/__init__.py) — 优势演员-评论员算法

### 深度确定性策略梯度 (Chapter 12)

- [DDPG](ddpg/__init__.py) — 深度确定性策略梯度
- [OU噪声](ddpg/__init__.py) — Ornstein-Uhlenbeck探索噪声
- [软更新](ddpg/__init__.py) — 目标网络软更新

## 注释风格

本实现采用labmlai的侧边对照注释风格：

1. **模块级文档字符串**：每个文件开头包含完整的理论推导，使用LaTeX公式
2. **类/方法级文档字符串**：每个组件都有详细的公式说明
3. **代码行注释**：每行关键代码前都有对应的数学公式
4. **参数映射**：函数参数与公式变量一一对应

### 公式与代码对照示例

```python
# TD目标：$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
with torch.no_grad():
    target_q = reward + self.gamma * target_q.max(dim=-1)[0] * (1 - done)

# TD误差：$\delta = Q(s, a; \theta) - y$
td_error = q_value - target_q

# 损失：$L(\theta) = \mathbb{E}[\delta^2]$
loss = td_error.pow(2).mean()
```

## 使用方式

### 安装依赖

```bash
pip install torch numpy gym
```

### 导入算法

```python
# Q-Learning
from rl.tabular import QLearningAgent

# PPO
from rl.ppo import ClippedPPOLoss, PPONetwork
from rl.ppo.gae import GAE

# DQN
from rl.dqn import QFuncLoss, DuelingQNetwork, EpsilonGreedy

# A2C
from rl.a2c import ActorCriticNetwork, A2CLoss

# DDPG
from rl.ddpg import Actor, Critic, DDPGLoss, OUNoise, soft_update
```

## 参考

- [蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/) — 强化学习中文教程
- [labml.ai Annotated Deep Learning](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — 注释风格参考
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO论文
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — DQN论文
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) — DDPG论文
"""
