# 蘑菇书EasyRL — 理论公式与代码注释实现

基于[蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/)理论公式与代码，
采用[labmlai annotated deep learning paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
风格的侧边对照注释排版。

## 概述

本项目将蘑菇书中的强化学习理论公式与Python/PyTorch代码实现进行逐行对照注释，
帮助读者更好地理解每个算法的数学原理与实现细节。

## 目录结构

```
annotated-rl/
├── generate_html.py               # HTML生成脚本
├── README.md                      # 本文件
├── docs/                          # GitHub Pages 部署目录
│   ├── index.html                 # 算法导航首页
│   ├── tabular/index.html         # 表格型方法
│   ├── pg/index.html              # 策略梯度
│   ├── ppo/index.html             # PPO
│   ├── ppo/gae.html               # GAE
│   ├── dqn/index.html             # DQN
│   ├── a2c/index.html             # A2C
│   └── ddpg/index.html            # DDPG
├── html/                          # 本地HTML页面（左右分栏）
│   └── ...                        # 与 docs/ 内容相同
└── rl/                            # 带注释的Python源码
    ├── __init__.py                # 强化学习算法索引
    ├── tabular/__init__.py        # 表格型方法 (Q-Learning, Sarsa, Value Iteration)
    ├── pg/__init__.py             # 策略梯度 (REINFORCE, 基线技巧, 折扣回报)
    ├── ppo/
    │   ├── __init__.py            # PPO算法 (PPO-Clip, PPO-Penalty)
    │   └── gae.py                 # 广义优势估计 (GAE)
    ├── dqn/__init__.py            # DQN算法 (DQN, Double DQN, Dueling DQN)
    ├── a2c/__init__.py            # 优势演员-评论员 (A2C)
    └── ddpg/__init__.py           # 深度确定性策略梯度 (DDPG)
```

## 算法覆盖

| 章节 | 算法 | Python源码 |
|------|------|-----------|
| 第3章 | Q-Learning | `rl/tabular/__init__.py` |
| 第3章 | Sarsa | `rl/tabular/__init__.py` |
| 第3章 | Value Iteration | `rl/tabular/__init__.py` |
| 第4章 | REINFORCE (策略梯度) | `rl/pg/__init__.py` |
| 第5章 | PPO-Clip (PPO2) | `rl/ppo/__init__.py` |
| 第5章 | PPO-Penalty (PPO1) | `rl/ppo/__init__.py` |
| 第5章 | GAE (广义优势估计) | `rl/ppo/gae.py` |
| 第6章 | DQN | `rl/dqn/__init__.py` |
| 第6章 | Double DQN | `rl/dqn/__init__.py` |
| 第6章 | Dueling DQN | `rl/dqn/__init__.py` |
| 第9章 | A2C (优势演员-评论员) | `rl/a2c/__init__.py` |
| 第12章 | DDPG | `rl/ddpg/__init__.py` |

## 注释风格

采用labmlai的**侧边对照注释**风格：

### 1. 模块级文档字符串

每个Python文件以完整的理论推导开始，包含LaTeX公式：

```python
"""
---
title: 近端策略优化 (PPO)
summary: 基于蘑菇书EasyRL第五章的PPO算法实现。
---

# 近端策略优化 (PPO)

PPO目标函数：
$$J_{\text{PPO2}}(\theta) \approx \sum \min\left(\frac{\pi_\theta}{\pi_{\theta_{old}}} A, \text{clip}(\cdots) A\right)$$
"""
```

### 2. 公式与代码逐行对照

每行关键代码前都有对应的数学公式注释：

```python
# 策略比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
ratio = torch.exp(log_pi - sampled_log_pi)

# TD目标：$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
target_q = reward + gamma * next_q.max(dim=-1)[0] * (1 - done)
```

### 3. 参数映射

函数参数与公式变量一一对应：

```python
def forward(self, q: torch.Tensor, action: torch.Tensor, ...):
    """
    * `q` - $Q(s,a;\theta)$
    * `action` - $a$
    * `target_q` - $Q(s';\theta^-)$
    """
```

### 4. HTML左右分栏布局

- **左侧 40%**：理论公式推导（KaTeX渲染）
- **右侧 60%**：带语法高亮和行号的代码实现
- 每个class/function的docstring提取为左侧文档，代码体保留在右侧

## 快速开始

### 生成HTML页面

```bash
cd annotated-rl
python generate_html.py
```

生成的HTML文件保存在 `html/` 目录下，直接用浏览器打开 `html/index.html` 即可查看。

### 安装依赖

```bash
pip install torch numpy
```

### 使用示例

#### Q-Learning

```python
from rl.tabular import QLearningAgent

agent = QLearningAgent(state_dim=16, action_dim=4, learning_rate=0.1, gamma=0.99)

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

#### PPO

```python
from rl.ppo import ClippedPPOLoss, PPONetwork
from rl.ppo.gae import GAE

network = PPONetwork(state_dim=4, action_dim=2)
ppo_loss = ClippedPPOLoss()
gae = GAE(n_workers=4, worker_steps=128, gamma=0.99, lambda_=0.95)

log_pi, value = network.evaluate(states, actions)
advantages = gae(dones, rewards, values)
loss = ppo_loss(log_pi, old_log_pi, advantages, clip=0.2)
```

#### DQN

```python
from rl.dqn import QFuncLoss, DuelingQNetwork, EpsilonGreedy

policy_net = DuelingQNetwork(state_dim=84*84*4, action_dim=4)
target_net = DuelingQNetwork(state_dim=84*84*4, action_dim=4)
dqn_loss = QFuncLoss(gamma=0.99)
exploration = EpsilonGreedy(action_dim=4)

q_values = policy_net(states)
target_q_values = target_net(next_states)
td_error, loss = dqn_loss(q_values, actions, double_q_values, target_q_values, dones, rewards)
```

## 参考资源

- [蘑菇书EasyRL](https://github.com/datawhalechina/easy-rl/) — 强化学习中文教程
- [蘑菇书在线阅读](https://datawhalechina.github.io/easy-rl/) — 实时更新版本
- [labml.ai Annotated Deep Learning](https://nn.labml.ai/) — 注释风格参考
- [OpenAI Spinning Up](https://spinningup.openai.com/) — 强化学习学习资源

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{annotated-easy-rl,
  title={Annotated Easy-RL: Reinforcement Learning Theory and Code},
  author={Based on Easy-RL by Wang, Yang, and Jiang},
  year={2026},
  url={https://github.com/datawhalechina/easy-rl/}
}
```

## 许可证

本项目的注释实现遵循蘑菇书EasyRL的[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)许可证。
