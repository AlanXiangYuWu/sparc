"""
可视化脚本
用于生成轨迹动画和对比图表
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from typing import List, Dict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="可视化RMHA结果")
    parser.add_argument("--data_file", type=str, default=None,
                        help="测试数据文件")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                        help="输出目录")
    parser.add_argument("--plot_type", type=str, default="all",
                        choices=["trajectory", "comparison", "all"],
                        help="绘图类型")
    return parser.parse_args()


def plot_trajectory(trajectory_data: Dict, save_path: str):
    """
    绘制单个轨迹动画
    
    Args:
        trajectory_data: 轨迹数据
        save_path: 保存路径
    """
    trajectory = trajectory_data["trajectory"]
    grid_size = 40  # 假设
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 颜色映射
    num_robots = len(trajectory[0]["positions"])
    colors = plt.cm.tab20(np.linspace(0, 1, num_robots))
    
    def init():
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("多机器人路径规划轨迹", fontsize=16, pad=20)
        return []
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"时间步: {frame}/{len(trajectory)}", fontsize=16, pad=20)
        
        positions = trajectory[frame]["positions"]
        
        # 绘制机器人
        for i, pos in enumerate(positions):
            ax.plot(pos[1], pos[0], 'o', color=colors[i], markersize=10)
            ax.text(pos[1], pos[0], str(i), ha='center', va='center',
                   fontsize=8, color='white', weight='bold')
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(trajectory), interval=200, blit=True)
    
    # 保存动画
    writer = PillowWriter(fps=5)
    anim.save(save_path, writer=writer)
    plt.close()
    
    print(f"轨迹动画已保存到: {save_path}")


def plot_comparison_curves(metrics_files: List[str], save_dir: str):
    """
    绘制对比曲线
    
    Args:
        metrics_files: 指标文件列表
        save_dir: 保存目录
    """
    # 读取所有指标
    all_metrics = {}
    for file_path in metrics_files:
        algo_name = os.path.basename(file_path).split("_")[0]
        with open(file_path, "r") as f:
            all_metrics[algo_name] = json.load(f)
    
    # 绘制成功率对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    obstacle_densities = [0.0, 0.15, 0.30]
    
    for algo_name, metrics in all_metrics.items():
        success_rates = []
        for density in obstacle_densities:
            key = f"success_rate_obs{int(density*100)}"
            success_rates.append(metrics.get(key, 0.0))
        
        ax.plot(obstacle_densities, success_rates, marker='o', 
               label=algo_name, linewidth=2, markersize=8)
    
    ax.set_xlabel("障碍物密度", fontsize=14)
    ax.set_ylabel("成功率", fontsize=14)
    ax.set_title("不同算法在不同障碍物密度下的成功率对比", fontsize=16, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "success_rate_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"成功率对比图已保存到: {save_path}")


def plot_metrics_heatmap(metrics_dict: Dict, save_path: str):
    """
    绘制指标热力图
    
    Args:
        metrics_dict: 指标字典
        save_path: 保存路径
    """
    # 提取关键指标
    metrics_to_plot = {
        "成功率": "success_mean",
        "最大目标达成数": "max_reached_mean",
        "碰撞率": "collision_rate_mean",
        "平均回合长度": "episode_length_mean"
    }
    
    data = []
    labels = []
    
    for label, key in metrics_to_plot.items():
        if key in metrics_dict:
            data.append(metrics_dict[key])
            labels.append(label)
    
    # 归一化
    data_normalized = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data_normalized.reshape(-1, 1), cmap='RdYlGn', aspect='auto')
    
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xticks([])
    
    # 添加数值标签
    for i, (d, d_norm) in enumerate(zip(data, data_normalized)):
        text_color = 'white' if d_norm < 0.5 else 'black'
        ax.text(0, i, f'{d:.3f}', ha='center', va='center',
               color=text_color, fontsize=14, weight='bold')
    
    ax.set_title("评估指标概览", fontsize=16, pad=20)
    plt.colorbar(im, ax=ax, label="归一化值")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"指标热力图已保存到: {save_path}")


def plot_ablation_results(save_dir: str):
    """
    绘制消融实验结果
    
    Args:
        save_dir: 保存目录
    """
    # 模拟数据（实际应从文件读取）
    algorithms = ["MAPPO", "MAPPO+GNN", "RMHA"]
    obstacle_densities = [0.0, 0.15, 0.30]
    
    # 成功率数据（示例）
    success_rates = {
        "MAPPO": [0.95, 0.60, 0.40],
        "MAPPO+GNN": [0.98, 0.70, 0.50],
        "RMHA": [1.00, 0.90, 0.75]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(obstacle_densities))
    width = 0.25
    
    for i, algo in enumerate(algorithms):
        offset = (i - 1) * width
        ax.bar(x + offset, success_rates[algo], width, label=algo, alpha=0.8)
    
    ax.set_xlabel("障碍物密度", fontsize=14)
    ax.set_ylabel("成功率", fontsize=14)
    ax.set_title("消融实验：通信机制的影响", fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d*100)}%" for d in obstacle_densities])
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "ablation_study.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"消融实验图已保存到: {save_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("生成可视化...")
    
    if args.plot_type in ["all", "comparison"]:
        # 绘制消融实验结果
        plot_ablation_results(args.output_dir)
    
    if args.plot_type in ["all", "trajectory"] and args.data_file:
        # 加载数据
        with open(args.data_file, "r") as f:
            data = json.load(f)
        
        # 绘制第一个episode的轨迹
        if len(data) > 0 and "trajectory" in data[0]:
            save_path = os.path.join(args.output_dir, "trajectory.gif")
            plot_trajectory(data[0], save_path)
    
    print("\n可视化完成！")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

