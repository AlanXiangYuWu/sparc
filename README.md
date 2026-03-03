# SPARC: Spatial-Aware Path Planning via Attentive Robot Communication

> Submitted to IROS 2026
>
> Sayang Mu · Xiangyu Wu · Bo An† · Nanyang Technological University, Singapore
>
> †Corresponding author

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

#### Model Architecture

```text
═══ Stage 1: Observation Encoder ═══

image (N, 8, 3, 3)                         vector (N, 7)
       │                                        │
  CNN (4×Conv2d + AdaptiveAvgPool)          FC: Linear(7→128) + ReLU
       → (N, 64)                                → (N, 128)
       │                                        │
       └──────────→ Concat (N, 192) ←───────────┘
                          │
                 3-layer FC (192→256→256→256)
                          │
                    LSTM (256, 256)
                          │
                    h_i^t (N, 256)     ← per-robot encoded features

═══ Stage 2: RMHA Communication (×3 layers) ═══

distance_matrix (N, N)          messages M = h_i^t
         │                              │
  Embedding(201, 64)                    │
  W₁(64→256) → Q branch                │
  W₂(64→256) → K branch                │
         │                              │
         ▼                              ▼
  Q = W_q(M + W₁·D)    K = W_k(M + W₂·D)    V = W_v(M)
                        │
         Masked Multi-Head Attention (4 heads, d_k=64)
           comm_radius mask: dist > R → −∞
                        │
                  GRU-gated residual
                        │
              FFN(256→1024→256) + LayerNorm
                        │
            updated messages (N, 256)

═══ Stage 3: Output Heads ═══

            Concat[h_i^t, messages] (N, 512)
                        │
               ┌────────┴────────┐
               │                 │
          Policy Network    Value Network
          (512→256→128→5)   (512→256→128→1)
               │                 │
           π(a|o)             V(s)
```

#### Feature Input Design

**Image (N, 8, 3, 3)** — 8-channel local observation within 3×3 FOV (not a rendered image, but stacked binary/feature maps):

| Channel | Content | Purpose |
| --- | --- | --- |
| 1-4 | Heuristic maps (up/down/left/right) | Whether moving in each direction gets closer to goal (binary) |
| 5 | Obstacle map | Walls and boundaries within FOV |
| 6 | Other robots map | Teammate positions within FOV |
| 7 | Own goal map | Own goal position (when visible in FOV) |
| 8 | Other goals map | Teammates' goal positions within FOV |

**Vector (N, 7)** — complements the image with information beyond the 3×3 FOV:

| Dims | Content | Purpose |
| --- | --- | --- |
| dx, dy | Normalized offset to goal | Global navigation direction (image only covers 3×3) |
| d | Normalized Euclidean distance to goal | How far the goal is |
| re_prev | Previous extrinsic reward | Immediate feedback (e.g., -2.0 = collision last step) |
| ri_prev | Previous intrinsic reward | Exploration signal |
| dmin_prev | Min distance to historical positions | Novelty indicator for exploration |
| a_prev | Previous action (0-4) | Action continuity (avoid oscillation) |

**Distance Matrix (N, N)** — pairwise Manhattan distances between all robots, recomputed every step; used by RMHA to generate spatially-aware attention weights.

#### Collision Handling

At each timestep, the environment checks for two collision types and resolves them before updating positions:

- **Vertex collision**: two robots move to the same cell → both stay in place, each receives -2.0 penalty
- **Edge collision (swap)**: two robots exchange positions → both stay in place, each receives -2.0 penalty

#### Reward Design

The reward signal combines rule-based extrinsic and intrinsic components (no learned reward networks):

**Extrinsic reward** (per robot per step):

| Component | Condition | Value | Purpose |
| --- | --- | --- | --- |
| Move cost | Any movement action | -0.3 | Encourage efficiency |
| Idle cost | STAY but not on goal | -0.3 | Discourage loitering |
| Distance shaping | Moved closer to goal | +0.1 per step closer | Continuous gradient signal |
| Collision penalty | Vertex or edge collision | -2.0 | Avoid conflicts |
| Reach goal | First arrival at goal | configurable | Goal incentive |

**Intrinsic reward** (episodic count-based exploration):

Each robot maintains a sparse buffer of visited milestone positions. When the current position has Manhattan distance >= 2 to all buffer entries, a +0.1 exploration bonus is awarded and the position is added to the buffer. This prevents robots from circling in familiar areas.

#### Multi-Hop Information Propagation

RMHA uses 3 stacked Transformer layers. Each layer enables one hop of message passing, progressively expanding each robot's information horizon:

```text
Layer 1: Robot A directly receives Robot B's message          (1-hop)
Layer 2: Robot A receives B's message which already contains C (2-hop)
Layer 3: Robot A indirectly perceives Robot D via B→C→D chain  (3-hop)
```

This multi-hop design allows robots to coordinate beyond their local communication radius without explicit global broadcasting.

#### Zero-Shot Scalability Setup

The model is trained at small scale and tested at large scale without any fine-tuning:

| | Training | Testing |
| --- | --- | --- |
| Grid size | Random from {10, 25, 40} | Fixed 40×40 |
| Obstacle density | Triangular distribution [0%, 50%], peak 33% | Fixed 0% / 15% / 30% |
| Robot count | **8** | **16 / 32 / 64 / 128** |

This train-test discrepancy is intentional: the attention mechanism naturally handles variable-length sequences (N changes only the attention matrix size), and parameter sharing across robots means individual robot behavior is N-invariant.

#### PPO Loss Computation

Training uses MAPPO (Multi-Agent PPO) with GAE advantage estimation. Each update collects 256 steps × 8 robots = 2048 samples, then performs 4 epochs × 8 mini-batches = 32 gradient updates per rollout.

**Step 1: GAE (Generalized Advantage Estimation)** — computed via backward recursion over the rollout (`buffer.py`):

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_{t+1}) − V(s_t)
A_t = δ_t + γ · λ · (1 - done_{t+1}) · A_{t+1}
```

where γ = 0.99 (discount factor), λ = 0.95 (GAE smoothing). Advantages are normalized to zero mean and unit variance before use.

**Step 2: Returns** — derived directly from GAE:

```
R_t = A_t + V(s_t)
```

**Step 3: Policy loss** — clipped surrogate objective (`mappo.py`):

```
ratio = exp(log π_new(a|s) − log π_old(a|s))
L_policy = −min(ratio · A, clip(ratio, 1−ε, 1+ε) · A).mean()
```

where ε = 0.2. The clipping prevents destructively large policy updates.

**Step 4: Value loss** — clipped MSE against returns:

```
V_clipped = V_old + clip(V_new − V_old, −ε, +ε)
L_value = 0.5 · max((V_new − R)², (V_clipped − R)²).mean()
```

**Step 5: Entropy loss** — encourages exploration:

```
L_entropy = −H(π).mean()
```

**Total loss:**

```
L = L_policy + 0.5 · L_value + 0.2 · L_entropy
```

| Hyperparameter | Value |
| --- | --- |
| Learning rate | 3e-4 → 3e-5 (cosine annealing) |
| Optimizer | Adam (eps=1e-5) |
| Gradient clipping | max_norm = 0.5 |
| PPO epochs | 4 |
| Mini-batches | 8 |
| Clip parameter ε | 0.2 |
| Entropy coefficient | 0.2 |

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
git clone https://github.com/AlanXiangYuWu/sparc.git
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
@inproceedings{wu2026sparc,
  title     = {{SPARC}: Spatial-Aware Path Planning via Attentive Robot Communication},
  author    = {Mu, Sayang and Wu, Xiangyu and An, Bo},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on
               Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

### Contact

For questions, open an Issue or email: `xiangyu015@e.ntu.edu.sg`

---

## 中文

> 慕飒扬 · 武翔宇 · 安波† · 南洋理工大学（新加坡）
>
> †通讯作者

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

#### 模型架构

```text
═══ 阶段一：观测编码器 ═══

image (N, 8, 3, 3)                         vector (N, 7)
       │                                        │
  CNN (4×Conv2d + AdaptiveAvgPool)          FC: Linear(7→128) + ReLU
       → (N, 64)                                → (N, 128)
       │                                        │
       └──────────→ Concat (N, 192) ←───────────┘
                          │
                 3层FC (192→256→256→256)
                          │
                    LSTM (256, 256)
                          │
                    h_i^t (N, 256)     ← 每个机器人的编码特征

═══ 阶段二：RMHA 通信模块（×3 层）═══

distance_matrix (N, N)          消息 M = h_i^t
         │                              │
  Embedding(201, 64)                    │
  W₁(64→256) → Q 分支                   │
  W₂(64→256) → K 分支                   │
         │                              │
         ▼                              ▼
  Q = W_q(M + W₁·D)    K = W_k(M + W₂·D)    V = W_v(M)
                        │
         掩码多头注意力（4头, d_k=64）
           通信半径掩码: dist > R → −∞
                        │
                  GRU 门控残差
                        │
              FFN(256→1024→256) + LayerNorm
                        │
            更新后消息 (N, 256)

═══ 阶段三：输出头 ═══

            Concat[h_i^t, messages] (N, 512)
                        │
               ┌────────┴────────┐
               │                 │
            策略网络          价值网络
          (512→256→128→5)   (512→256→128→1)
               │                 │
           π(a|o)             V(s)
```

#### 特征输入设计

**Image (N, 8, 3, 3)** — 8 通道局部观测（非渲染图片，而是堆叠的二值/特征矩阵）：

| 通道 | 内容 | 用途 |
| --- | --- | --- |
| 1-4 | 启发式地图（上/下/左/右） | 向该方向移动是否更接近目标（二值） |
| 5 | 障碍物地图 | FOV 内的墙壁和边界 |
| 6 | 其他机器人地图 | FOV 内队友位置 |
| 7 | 自身目标地图 | 自身目标位置（在 FOV 内时标记） |
| 8 | 他人目标地图 | FOV 内队友的目标位置 |

**Vector (N, 7)** — 补充 image 无法覆盖的 3×3 FOV 以外的信息：

| 维度 | 内容 | 用途 |
| --- | --- | --- |
| dx, dy | 到目标的归一化偏移 | 全局导航方向 |
| d | 到目标的归一化欧氏距离 | 目标远近 |
| re_prev | 上一步外在奖励 | 即时反馈（如 -2.0 = 上步碰撞） |
| ri_prev | 上一步内在奖励 | 探索信号 |
| dmin_prev | 与历史位置最小距离 | 新颖度指标 |
| a_prev | 上一步动作 (0-4) | 动作连贯性（避免来回震荡） |

**Distance Matrix (N, N)** — 所有机器人之间的曼哈顿距离矩阵，每步重新计算；RMHA 据此生成空间感知的注意力权重。

#### 碰撞处理

每个时间步中，环境检测两种碰撞类型并在更新位置前解决：

- **顶点碰撞**：两个机器人移动到同一格子 → 双方退回原位，各扣 -2.0 惩罚
- **边碰撞（交换碰撞）**：两个机器人互换位置 → 双方退回原位，各扣 -2.0 惩罚

#### 奖励设计

奖励信号由规则计算的外在奖励和内在奖励组成（不使用可学习的奖励网络）：

**外在奖励**（每步每个机器人）：

| 分量 | 条件 | 值 | 作用 |
| --- | --- | --- | --- |
| 移动代价 | 执行移动动作 | -0.3 | 鼓励高效到达 |
| 怠惰惩罚 | 未到目标但原地不动 | -0.3 | 防止摸鱼 |
| 距离塑形 | 向目标靠近 | +0.1/步 | 持续梯度信号 |
| 碰撞惩罚 | 顶点或边碰撞 | -2.0 | 避免冲突 |
| 到达奖励 | 首次到达目标 | 可配置 | 目标激励 |

**内在奖励**（基于 episode 内计数的探索机制）：

每个机器人维护一个稀疏的已访问里程碑缓冲区。当前位置与缓冲区中所有位置的曼哈顿距离均 >= 2 时，给予 +0.1 探索奖励并将该位置加入缓冲区。此机制防止机器人在已访问区域原地打转。

#### 多跳信息传播

RMHA 使用 3 层堆叠的 Transformer 层，每层实现一跳消息传递，逐步扩大每个机器人的信息感知范围：

```text
第1层: 机器人 A 直接获取机器人 B 的信息          (1-hop)
第2层: 机器人 A 获取"B 眼中的 C"的信息            (2-hop)
第3层: 机器人 A 间接感知机器人 D (经 B→C→D 链路)  (3-hop)
```

多跳设计使机器人无需全局广播即可实现超出局部通信半径的协调。

#### 零样本规模泛化设置

模型在小规模训练，在大规模测试，无需任何微调：

| | 训练时 | 测试时 |
| --- | --- | --- |
| 网格大小 | 从 {10, 25, 40} 随机采样 | 固定 40×40 |
| 障碍物密度 | 三角分布 [0%, 50%]，峰值 33% | 固定 0% / 15% / 30% |
| 机器人数量 | **8** | **16 / 32 / 64 / 128** |

训练与测试的规模差异是有意设计的：注意力机制天然支持变长序列（N 变化仅影响注意力矩阵大小），参数共享使单个机器人的行为与 N 无关。

#### PPO 损失函数计算

训练使用 MAPPO（Multi-Agent PPO）结合 GAE 优势估计。每次更新收集 256 步 × 8 机器人 = 2048 条样本，随后进行 4 轮 epoch × 8 个 mini-batch = 32 次梯度更新。

**第一步：GAE（广义优势估计）** — 在 rollout 上反向递归计算（`buffer.py`）：

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_{t+1}) − V(s_t)
A_t = δ_t + γ · λ · (1 - done_{t+1}) · A_{t+1}
```

其中 γ = 0.99（折扣因子），λ = 0.95（GAE 平滑参数）。优势值在使用前归一化为零均值单位方差。

**第二步：回报（Returns）** — 由 GAE 直接推导：

```
R_t = A_t + V(s_t)
```

**第三步：策略损失** — 裁剪替代目标函数（`mappo.py`）：

```
ratio = exp(log π_new(a|s) − log π_old(a|s))
L_policy = −min(ratio · A, clip(ratio, 1−ε, 1+ε) · A).mean()
```

ε = 0.2，裁剪防止策略更新过大导致训练崩溃。

**第四步：价值损失** — 对回报的裁剪 MSE：

```
V_clipped = V_old + clip(V_new − V_old, −ε, +ε)
L_value = 0.5 · max((V_new − R)², (V_clipped − R)²).mean()
```

**第五步：熵损失** — 鼓励探索：

```
L_entropy = −H(π).mean()
```

**总损失：**

```
L = L_policy + 0.5 · L_value + 0.2 · L_entropy
```

| 超参数 | 值 |
| --- | --- |
| 学习率 | 3e-4 → 3e-5（余弦退火） |
| 优化器 | Adam (eps=1e-5) |
| 梯度裁剪 | max_norm = 0.5 |
| PPO 轮数 | 4 |
| Mini-batch 数 | 8 |
| 裁剪参数 ε | 0.2 |
| 熵系数 | 0.2 |
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
git clone https://github.com/AlanXiangYuWu/sparc.git
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
