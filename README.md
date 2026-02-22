# SPARC: Spatial-Aware Path Planning via Attentive Robot Communication

> Submitted to IROS 2026
>
> Sayang Mu\* · Xiangyu Wu\* · (\*Equal contribution) · Nanyang Technological University, Singapore

[English](#english) | [中文](#中文)

## English

### Overview

**SPARC** addresses two fundamental bottlenecks in Multi-Robot Path Planning (MRPP): *low communication efficiency* and *insufficient cooperative optimization*. We propose **RMHA** (Relation-enhanced Multi-Head Attention), a communication mechanism that explicitly encodes spatial relationships between robots into the attention weight computation.

By embedding inter-robot Manhattan distances as edge features alongside observation content, RMHA enables communication weights to adapt dynamically to topological relationships. This reduces communication overhead while significantly improving path planning success rates — especially in high-density, obstacle-rich environments.

### Key Contributions

1. **Spatial Relation-Enhanced Attention** — Inter-robot relative distance is incorporated as explicit edge information into multi-head attention, enabling distance-aware communication weight allocation that jointly considers spatial proximity and message content.

2. **Communication Architecture for Stable Online RL** — Distance-constrained attention masking limits message passing to local neighbors, while GRU-gated message fusion adaptively balances new communications against prior beliefs. Together these address the training instabilities typical of Transformer-based communication in online multi-agent RL.

3. **Zero-Shot Scalability** — Attention-based communication with parameter sharing enables zero-shot generalization from 8-robot training to 128-robot deployment, with no performance collapse across varying obstacle densities.

### Method

RMHA replaces standard dot-product attention with a spatially-aware variant.

#### Standard attention

`s_ij = o_i · Wq^T · Wk · o_j`

#### RMHA (ours)

`s_ij = (o_i + d_{i→j}) · Wq^T · Wk · (o_j + d_{j→i})`

where `d_{*→*}` is the learned embedding of the Manhattan distance between robots. A communication radius mask further restricts message passing to local neighbors, modeling realistic bandwidth constraints.

The full system integrates:

- **Encoder** — CNN extracts spatial features from each robot's 8-channel 3×3 local observation (obstacle map, teammate positions, goal heuristics); a separate FC branch encodes a 7-dim vector input (goal direction, prior rewards, last action); both are fused and passed through LSTM for temporal memory across steps under partial observability (Dec-POMDP)
- **RMHA communication module** — distance-aware multi-head attention with GRU-gated fusion
- **Policy & Value networks** — shared post-communication features feed into an action distribution head and dual value heads (extrinsic + intrinsic)
- **MAPPO** (Centralized Training, Decentralized Execution)

### Results

All evaluations use **128 robots** on a **40×40 grid** at obstacle densities of 0%, 15%, and 30%. **SR** = Success Rate (percentage of robots reaching their goal within the episode time limit).

#### Ablation: Communication Mechanism

| Method | SR @ 0% | SR @ 15% | SR @ 30% |
| --- | --- | --- | --- |
| MAPPO (no comm) | ~95% | ~60% | ~40% |
| MAPPO + Graph Comm | ~98% | ~75% | ~50% |
| **SPARC / RMHA (ours)** | **~100%** | **~90%** | **~75%** |

At 30% obstacle density, SPARC outperforms graph communication (no distance encoding) by **+25% SR**, and no-communication baseline by **+53% SR**.

#### Comparison with State-of-the-Art

All methods evaluated under identical conditions: 128 robots, 40×40 random obstacle map, 256-step episode limit.

| Method | SR @ 0% | SR @ 15% | SR @ 30% |
| --- | --- | --- | --- |
| ODrM* | ~100% | ~60% | ~20% |
| SCRIMP | ~98% | ~70% | ~50% |
| DHC † | ~95% | ~30% | ~0% |
| PICO ‡ | ~95% | ~30% | ~0% |
| **SPARC (ours)** | **~100%** | **~90%** | **~75%** |

† DHC uses a 9×9 FOV. ‡ PICO uses an 11×11 FOV. SPARC uses a 3×3 FOV.

### Installation

Requirements: Python 3.7+, PyTorch 2.1+, CUDA 11.8+

```bash
git clone https://github.com/FirmamentWu/sparc.git
cd sparc
conda create -n sparc python=3.7
conda activate sparc
pip install -r requirements.txt
```

### Quick Start

```bash
# Training
python train.py --config config/train_config.yaml --algo rmha

# Testing
python test.py --checkpoint results/checkpoints/rmha_best.pth \
               --num_robots 128 --grid_size 40 --obstacle_density 0.3

# Visualization
python visualize.py --checkpoint results/checkpoints/rmha_best.pth \
                    --save_path results/figures/trajectory.gif

# Ablation and baselines
python experiments/ablation_study.py
python experiments/compare_baselines.py
```

### Project Structure

```text
sparc/
├── algorithms/          # MAPPO trainer and RMHA agent
├── config/              # Training / evaluation configs
├── envs/                # Grid-world MRPP environment
├── experiments/         # Ablation and baseline comparison scripts
├── models/              # Encoder, RMHA module, policy, value networks
├── paper/               # LaTeX source (IROS 2026)
├── results/
│   ├── checkpoints/     # Saved model weights (not tracked)
│   └── logs/            # Training logs (not tracked)
├── utils/               # Replay buffer, logger, metrics
├── train.py
├── test.py
└── visualize.py
```

### Citation

```bibtex
@inproceedings{mu2026sparc,
  title     = {{SPARC}: Spatial-Aware Path Planning via Attentive Robot Communication},
  author    = {Mu, Sayang and Wu, Xiangyu},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on
               Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

### Contact

For questions, open an Issue or email: `xiangyu015@e.ntu.edu.sg`

---

## 中文

### 项目简介

**SPARC** 针对多机器人路径规划（MRPP）中通信效率低下与协同优化不足两大核心问题，提出了空间关系增强多头注意力通信方法 **RMHA**（Relation-enhanced Multi-Head Attention）。

RMHA 将机器人间的相对曼哈顿距离作为边特征显式嵌入注意力权重计算，使通信权重能够动态适应机器人的拓扑关系，在有效降低通信负载的同时显著提升路径规划成功率——尤其在高密度复杂障碍环境下优势突出。

### 核心贡献

1. **空间关系增强注意力机制** — 将机器人间相对距离作为显式边信息融入多头注意力，实现同时感知空间邻近性与消息内容的动态通信权重分配。

2. **面向在线强化学习的通信架构** — 距离约束的注意力掩码将消息传递限制在局部邻居范围内；GRU 门控消息融合自适应平衡新通信与历史信念。两者共同解决了 Transformer 通信模块在在线多智能体 RL 训练中的不稳定问题。

3. **零样本规模泛化** — 基于注意力机制的通信与参数共享，实现从 8 机器人训练直接零样本扩展至 128 机器人部署，在不同障碍密度下均无性能崩塌。

### 方法

RMHA 将标准点积注意力替换为空间感知变体。

#### 标准注意力

`s_ij = o_i · Wq^T · Wk · o_j`

#### RMHA（本文）

`s_ij = (o_i + d_{i→j}) · Wq^T · Wk · (o_j + d_{j→i})`

其中 `d_{*→*}` 为机器人间曼哈顿距离的可学习嵌入，通信半径掩码进一步将消息传递限制在局部邻居范围内。

完整系统由以下模块组成：

- **编码器** — CNN 从每个机器人的 8 通道 3×3 局部观测中提取空间特征；独立 FC 分支对 7 维向量输入（目标方向、历史奖励、上步动作）进行编码；两路特征融合后经 LSTM 保持跨时间步的记忆，适应部分可观测决策场景（Dec-POMDP）
- **RMHA 通信模块** — 距离感知多头注意力 + GRU 门控消息融合
- **策略网络与价值网络** — 共享通信后特征，分别输出动作概率分布与双路价值估计（外在 + 内在）
- **MAPPO**（集中训练、分散执行）

### 实验结果

所有测试在 **128 机器人 · 40×40 网格**下进行，障碍密度分别为 0%、15%、30%。**SR**（成功率）= 在回合时间限制内到达目标的机器人比例。

#### 通信消融实验

| 方法 | SR @ 0% | SR @ 15% | SR @ 30% |
| --- | --- | --- | --- |
| MAPPO（无通信） | ~95% | ~60% | ~40% |
| MAPPO + 图通信 | ~98% | ~75% | ~50% |
| **SPARC / RMHA（本文）** | **~100%** | **~90%** | **~75%** |

在 30% 障碍密度下，SPARC 成功率比无距离编码的图通信高 **+25%**，比无通信基线高 **+53%**。

#### 与 SOTA 方法对比

所有方法在相同条件下评估：128 机器人，40×40 随机障碍地图，最大 256 步。

| 方法 | SR @ 0% | SR @ 15% | SR @ 30% |
| --- | --- | --- | --- |
| ODrM* | ~100% | ~60% | ~20% |
| SCRIMP | ~98% | ~70% | ~50% |
| DHC † | ~95% | ~30% | ~0% |
| PICO ‡ | ~95% | ~30% | ~0% |
| **SPARC（本文）** | **~100%** | **~90%** | **~75%** |

† DHC 使用 9×9 视野。‡ PICO 使用 11×11 视野。SPARC 仅使用 3×3 视野。

### 安装与使用

```bash
git clone https://github.com/FirmamentWu/sparc.git
cd sparc
conda create -n sparc python=3.7
conda activate sparc
pip install -r requirements.txt
```

```bash
# 训练
python train.py --config config/train_config.yaml --algo rmha

# 测试
python test.py --checkpoint results/checkpoints/rmha_best.pth \
               --num_robots 128 --grid_size 40 --obstacle_density 0.3

# 可视化
python visualize.py --checkpoint results/checkpoints/rmha_best.pth \
                    --save_path results/figures/trajectory.gif
```

### 项目结构

```text
sparc/
├── algorithms/          # MAPPO 训练框架 & RMHA 智能体
├── config/              # 训练 / 测试配置文件
├── envs/                # 网格世界 MRPP 环境
├── experiments/         # 消融实验 & 基线对比脚本
├── models/              # 编码器、RMHA 模块、策略网络、价值网络
├── paper/               # LaTeX 论文源码（IROS 2026）
├── results/
│   ├── checkpoints/     # 模型权重（不纳入版本控制）
│   └── logs/            # 训练日志（不纳入版本控制）
├── utils/               # 经验回放、日志、评估指标
├── train.py
├── test.py
└── visualize.py
```

### 联系方式

如有问题请提交 Issue 或发送邮件至：`xiangyu015@e.ntu.edu.sg`

---

*本项目仅用于学术研究。*
