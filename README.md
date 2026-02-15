# RMHA: 空间关系增强的多头注意力通信多机器人路径规划

## 项目简介

本项目实现了论文《面向多机器人路径规划的空间关系增强多头注意力通信方法》中提出的RMHA（Relation-enhanced Multi-head Attention）算法。

RMHA通过引入空间关系增强的多头注意力机制，将机器人间的相对距离信息与通信内容有机结合，实现通信权重的动态分配与优化，从而有效降低通信负载并提升多机器人路径规划性能。

## 核心特性

- ✅ 基于Transformer的空间关系增强多头注意力通信
- ✅ MAPPO（Multi-Agent PPO）训练框架
- ✅ 支持大规模多机器人场景（128+机器人）
- ✅ 完整的消融实验和基线对比
- ✅ 详细的可视化和评估工具

## 环境要求

### 硬件要求
- GPU: NVIDIA GPU（建议RTX 3080或更高）
- RAM: 16GB+
- 存储: 50GB+

### 软件要求
- Python 3.7+
- PyTorch 2.1+
- CUDA 11.8+（如使用GPU）

## 安装指南

### 1. 克隆项目
```bash
git clone <repository_url>
cd RMHA_MRPP
```

### 2. 创建虚拟环境
```bash
conda create -n rmha python=3.7
conda activate rmha
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 项目结构

```
RMHA_MRPP/
├── config/                      # 配置文件
├── envs/                        # MRPP环境实现
├── models/                      # 神经网络模型
│   ├── encoder.py              # 观测编码器
│   ├── rmha.py                 # RMHA通信模块
│   ├── policy.py               # 策略网络
│   └── value.py                # 价值网络
├── algorithms/                  # 算法实现
│   ├── mappo.py                # MAPPO训练框架
│   ├── rmha_agent.py           # RMHA智能体
│   └── baselines/              # 基线算法
├── utils/                       # 工具函数
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
├── visualize.py                 # 可视化脚本
└── experiments/                 # 实验脚本
```

## 快速开始

### 训练RMHA模型

```bash
python train.py --config config/train_config.yaml --algo rmha
```

### 训练基线模型

```bash
# MAPPO (无通信)
python train.py --config config/train_config.yaml --algo mappo

# MAPPO + 图通信
python train.py --config config/train_config.yaml --algo mappo_gnn
```

### 测试模型

```bash
python test.py --checkpoint results/checkpoints/rmha_best.pth \
               --num_robots 128 \
               --grid_size 40 \
               --obstacle_density 0.3 \
               --num_episodes 100
```

### 可视化结果

```bash
python visualize.py --checkpoint results/checkpoints/rmha_best.pth \
                    --save_path results/figures/trajectory.gif
```

## 实验复现

### 实验1：通信消融实验

```bash
python experiments/ablation_study.py
```

这将运行：
- MAPPO（无通信）
- MAPPO + 图通信
- RMHA（完整方法）

在不同障碍物密度（0%, 15%, 30%）下的对比实验。

### 实验2：与SOTA算法对比

```bash
python experiments/compare_baselines.py
```

对比算法：
- RMHA（本文方法）
- SCRIMP
- DHC
- PICO
- ODrM*

### 生成所有结果图表

```bash
python experiments/generate_results.py --output_dir results/figures
```

## 评估指标

本项目实现了论文中的三个核心评估指标：

1. **成功率（Success Rate, SR）**：所有机器人无碰撞到达目标的比例
2. **最大目标达成数（Max Reached, MR）**：最多到达目标的机器人数量
3. **碰撞率（Collision Rate, CO）**：碰撞次数的归一化比例

## 配置说明

主要配置文件：`config/train_config.yaml`

```yaml
# 环境配置
env:
  num_robots: 8                # 训练时机器人数量
  grid_size: [10, 40]          # 网格大小范围
  obstacle_density: [0.0, 0.5] # 障碍物密度范围
  fov_size: 3                  # 视野范围
  max_episode_steps: 256       # 最大回合步数

# 模型配置
model:
  hidden_dim: 256              # 隐藏层维度
  num_heads: 4                 # 注意力头数
  num_layers: 3                # Transformer层数
  comm_radius: 40              # 通信半径

# 训练配置
training:
  total_steps: 10000000        # 总训练步数
  batch_size: 32               # 批次大小
  lr: 3e-4                     # 学习率
  gamma: 0.99                  # 折扣因子
  gae_lambda: 0.95             # GAE参数
  clip_param: 0.2              # PPO裁剪参数
```

## 实验结果

根据论文实验结果：

### 成功率对比（128机器人，40×40网格）

| 算法 | 0%障碍物 | 15%障碍物 | 30%障碍物 |
|------|----------|-----------|-----------|
| RMHA | ~100% | ~90% | ~75% |
| SCRIMP | ~98% | ~70% | ~50% |
| DHC | ~95% | ~30% | ~0% |
| PICO | ~95% | ~30% | ~0% |
| MAPPO | ~95% | ~60% | ~40% |

### 主要发现

1. **空间关系编码的重要性**：RMHA通过融合机器人间距离信息，在高密度障碍物环境下成功率比无距离编码的方法提高53%

2. **通信机制的有效性**：相比无通信的MAPPO，图通信方法在复杂环境下显著提升性能

3. **可扩展性**：从训练时的8个机器人扩展到测试时的128个机器人，验证了良好的泛化能力

## 常见问题

### Q1: 训练需要多长时间？
A: 使用双NVIDIA 8000 GPU，完整训练约需5-7天。使用单GPU可能需要10-14天。

### Q2: 如何减少训练时间？
A: 
- 减少总训练步数（可能影响性能）
- 使用更大的批次大小（需要更多GPU内存）
- 使用多进程并行环境

### Q3: 内存不足怎么办？
A:
- 减少批次大小
- 减少并行环境数量
- 使用梯度累积

### Q4: 如何调整超参数？
A: 参考`config/train_config.yaml`中的注释，主要调整学习率、批次大小、模型维度等。

## 引用

如果您使用本代码，请引用原论文：

```bibtex
@article{rmha2025,
  title={面向多机器人路径规划的空间关系增强多头注意力通信方法},
  author={...},
  journal={...},
  year={2025}
}
```

## 许可证

本项目仅用于学术研究目的。

## 联系方式

如有问题或建议，请提交Issue或联系：[xiangyu015@e.ntu.edu.sg]

---

**更新日期**：2025-11-26  
**版本**：v1.0

