"""
消融实验脚本
对比 MAPPO、MAPPO+GNN、RMHA
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--train", action="store_true",
                        help="是否训练模型（否则仅测试）")
    parser.add_argument("--test", action="store_true",
                        help="是否测试模型")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成可视化")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="配置文件")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                        help="输出目录")
    return parser.parse_args()


def train_models(config_path, algorithms):
    """训练所有模型"""
    print("="*60)
    print("开始训练模型")
    print("="*60)
    
    for algo in algorithms:
        print(f"\n训练 {algo}...")
        cmd = [
            "python", "train.py",
            "--config", config_path,
            "--algo", algo,
            "--seed", "42"
        ]
        
        subprocess.run(cmd, check=True)
        print(f"{algo} 训练完成！")


def test_models(algorithms, test_scenarios, output_dir):
    """测试所有模型"""
    print("\n" + "="*60)
    print("开始测试模型")
    print("="*60)
    
    results = {}
    
    for algo in algorithms:
        print(f"\n测试 {algo}...")
        algo_results = {}
        
        checkpoint_path = f"results/checkpoints/{algo}_seed42/{algo}_best.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"警告: 未找到检查点 {checkpoint_path}，跳过")
            continue
        
        for scenario in test_scenarios:
            scenario_name = scenario["name"]
            obstacle_density = scenario["obstacle_density"]
            num_robots = scenario.get("num_robots", 128)
            grid_size = scenario.get("grid_size", 40)
            num_episodes = scenario.get("num_episodes", 100)
            
            print(f"  场景: {scenario_name} (障碍物密度: {obstacle_density:.0%})")
            
            cmd = [
                "python", "test.py",
                "--checkpoint", checkpoint_path,
                "--num_robots", str(num_robots),
                "--grid_size", str(grid_size),
                "--obstacle_density", str(obstacle_density),
                "--num_episodes", str(num_episodes),
                "--output_dir", output_dir,
                "--seed", "123"
            ]
            
            # 运行测试
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 解析结果（从输出文件读取）
            metrics_file = os.path.join(
                output_dir,
                f"test_n{num_robots}_grid{grid_size}_obs{int(obstacle_density*100)}_metrics.json"
            )
            
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                algo_results[scenario_name] = metrics
            else:
                print(f"  警告: 未找到指标文件 {metrics_file}")
        
        results[algo] = algo_results
    
    # 保存所有结果
    results_file = os.path.join(output_dir, "ablation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n所有结果已保存到: {results_file}")
    
    return results


def visualize_results(results, output_dir):
    """可视化结果"""
    print("\n" + "="*60)
    print("生成可视化")
    print("="*60)
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    algorithms = list(results.keys())
    scenarios = ["low_density", "medium_density", "high_density"]
    densities = [0.0, 0.15, 0.30]
    
    # 绘制成功率对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algorithms:
        success_rates = []
        for scenario in scenarios:
            if scenario in results[algo]:
                sr = results[algo][scenario].get("success_mean", 0.0)
                success_rates.append(sr * 100)  # 转换为百分比
            else:
                success_rates.append(0.0)
        
        ax.plot(densities, success_rates, marker='o', label=algo.upper(),
               linewidth=2, markersize=8)
    
    ax.set_xlabel("障碍物密度", fontsize=14)
    ax.set_ylabel("成功率 (%)", fontsize=14)
    ax.set_title("消融实验：成功率对比 (128机器人, 40×40网格)", fontsize=16, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    ax.set_xticks(densities)
    ax.set_xticklabels([f"{int(d*100)}%" for d in densities])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ablation_success_rate.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"成功率对比图已保存到: {save_path}")
    
    # 绘制最大目标达成数对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algorithms:
        max_reached = []
        for scenario in scenarios:
            if scenario in results[algo]:
                mr = results[algo][scenario].get("max_reached_mean", 0.0)
                max_reached.append(mr)
            else:
                max_reached.append(0.0)
        
        ax.plot(densities, max_reached, marker='s', label=algo.upper(),
               linewidth=2, markersize=8)
    
    ax.set_xlabel("障碍物密度", fontsize=14)
    ax.set_ylabel("最大目标达成数", fontsize=14)
    ax.set_title("消融实验：最大目标达成数对比", fontsize=16, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(densities)
    ax.set_xticklabels([f"{int(d*100)}%" for d in densities])
    ax.set_ylim([0, 130])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ablation_max_reached.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"最大目标达成数对比图已保存到: {save_path}")
    
    # 打印数值结果表格
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'算法':<15} {'0%障碍物':<15} {'15%障碍物':<15} {'30%障碍物':<15}")
    print("-"*60)
    
    for algo in algorithms:
        row = [algo.upper()]
        for scenario in scenarios:
            if scenario in results[algo]:
                sr = results[algo][scenario].get("success_mean", 0.0) * 100
                row.append(f"{sr:.1f}%")
            else:
                row.append("N/A")
        print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    
    print("="*60)


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义要对比的算法
    algorithms = ["mappo", "mappo_gnn", "rmha"]
    
    # 定义测试场景
    test_scenarios = [
        {"name": "low_density", "obstacle_density": 0.0, "num_robots": 128,
         "grid_size": 40, "num_episodes": 100},
        {"name": "medium_density", "obstacle_density": 0.15, "num_robots": 128,
         "grid_size": 40, "num_episodes": 100},
        {"name": "high_density", "obstacle_density": 0.30, "num_robots": 128,
         "grid_size": 40, "num_episodes": 100}
    ]
    
    # 训练
    if args.train:
        train_models(args.config, algorithms)
    
    # 测试
    if args.test:
        results = test_models(algorithms, test_scenarios, args.output_dir)
    else:
        # 尝试加载已有结果
        results_file = os.path.join(args.output_dir, "ablation_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
        else:
            print("错误: 未找到结果文件，请先运行测试")
            return
    
    # 可视化
    if args.visualize:
        visualize_results(results, args.output_dir)
    
    print("\n消融实验完成！")


if __name__ == "__main__":
    main()

