# SPARC: Spatial-Aware Path Planning via Attentive Robot Communication

> Submitted to IROS 2026
>
> Sayang Mu\* · Xiangyu Wu\* · (\*Equal contribution) · Nanyang Technological University, Singapore

[English](#english) | [中文](#chinese)

## English

### Overview

**SPARC** addresses two fundamental bottlenecks in Multi-Robot Path Planning (MRPP): *low communication efficiency* and *insufficient cooperative optimization*. We propose **RMHA** (Relation-enhanced Multi-Head Attention), a communication mechanism built on graph attention that explicitly encodes spatial relationships between robots into the attention weight computation.

By embedding inter-robot Manhattan distances as edge features alongside observation content, RMHA enables communication weights to adapt dynamically to topological relationships. This reduces communication overhead while significantly improving path planning success rates — especially in high-density, obstacle-rich environments.

### Key Contributions

1. **Spatial Relation-Enhanced Attention** — Inter-robot relative distance is incorporated as explicit edge information into multi-head attention, enabling distance-aware communication weight allocation.

2. **End-to-End MAPPO Framework** — Distance-constrained local communication with attention masking reduces overhead while maintaining effective information exchange and training stability.

3. **Scalability from 8 to 128 robots** — Trained on 8 robots, SPARC generalizes to 128 robots at test time, maintaining high success rates across varying obstacle densities.

### Method

RMHA replaces standard dot-product attention with a spatially-aware variant.

#### Standard attention

`s_ij = o_i · Wq^T · Wk · o_j`

#### RMHA (ours)

`s_ij = (o_i + d_{i→j}) · Wq^T · Wk · (o_j + d_{j→i})`

where `d_{*→*}` is the learned embedding of the Manhattan distance between robots. A communication radius mask further restricts message passing to local neighbors, modeling realistic bandwidth constraints.

The full system integrates:

- **CNN + LSTM encoder** for multi-modal local observations
- **RMHA communication module** (Transformer encoder with spatial edge features)
- **MAPPO** (Centralized Training, Decentralized Execution)

### Results

All evaluations use **128 robots** on a **40×40 grid** at obstacle densities of 0%, 15%, and 30%.

#### Ablation: Communication Mechanism

| Method | SR @ 0% | SR @ 15% | SR @ 30% |
| --- | --- | --- | --- |
| MAPPO (no comm) | ~95% | ~60% | ~40% |
| MAPPO + Graph Comm | ~98% | ~75% | ~50% |
| **SPARC / RMHA (ours)** | **~100%** | **~90%** | **~75%** |

At 30% obstacle density, SPARC outperforms non-enhanced communication by **+53% success rate**.

#### Comparison with State-of-the-Art

| Method | SR @ 0% | SR @ 15% | SR @ 30% |
| --- | --- | --- | --- |
| ODrM* | ~100% | ~60% | ~20% |
| SCRIMP | ~98% | ~70% | ~50% |
| DHC | ~95% | ~30% | ~0% |
| PICO | ~95% | ~30% | ~0% |
| **SPARC (ours)** | **~100%** | **~90%** | **~75%** |

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

**SPARC** 针对多机器人路径规划（MRPP）中通信效率低下与协同优化不足两大核心问题，提出了基于图注意力机制的高效通信方法 **RMHA**（空间关系增强多头注意力）。

RMHA 将机器人间的相对曼哈顿距离作为边特征显式嵌入注意力权重计算，使通信权重能够动态适应机器人的拓扑关系，在有效降低通信负载的同时显著提升路径规划成功率——尤其在高密度复杂障碍环境下优势突出。

### 核心贡献

1. **空间关系增强注意力机制** — 将机器人间相对距离作为显式边信息融入多头注意力，实现距离感知的通信权重动态分配。

2. **端到端 MAPPO 训练框架** — 引入距离约束的局部通信与注意力掩码，在保证有效信息交互的同时降低通信开销，提升训练稳定性。

3. **8 至 128 机器人的强泛化能力** — 仅用 8 个机器人训练，测试时直接扩展至 128 个机器人，在不同障碍密度下均保持高成功率。

### 实验结果

所有测试在 **128 机器人 · 40×40 网格** 下进行，障碍密度分别为 0%、15%、30%。

#### 通信消融实验

| 方法 | 成功率 0% | 成功率 15% | 成功率 30% |
| --- | --- | --- | --- |
| MAPPO（无通信） | ~95% | ~60% | ~40% |
| MAPPO + 图通信 | ~98% | ~75% | ~50% |
| **SPARC / RMHA（本文）** | **~100%** | **~90%** | **~75%** |

在 30% 障碍密度下，SPARC 成功率比无距离增强通信方法高出 **53%**。

#### 与 SOTA 方法对比

| 方法 | 成功率 0% | 成功率 15% | 成功率 30% |
| --- | --- | --- | --- |
| ODrM* | ~100% | ~60% | ~20% |
| SCRIMP | ~98% | ~70% | ~50% |
| DHC | ~95% | ~30% | ~0% |
| PICO | ~95% | ~30% | ~0% |
| **SPARC（本文）** | **~100%** | **~90%** | **~75%** |

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
