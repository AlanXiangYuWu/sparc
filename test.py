"""
RMHA测试脚本
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import json

from envs.mrpp_env import MRPPEnv
from algorithms.rmha_agent import RMHAAgent
from utils.metrics import compute_metrics, print_metrics, save_metrics_to_file


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试RMHA模型")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--config", type=str, default="config/test_config.yaml",
                        help="测试配置文件")
    parser.add_argument("--num_robots", type=int, default=128,
                        help="机器人数量")
    parser.add_argument("--grid_size", type=int, default=40,
                        help="网格大小")
    parser.add_argument("--obstacle_density", type=float, default=0.3,
                        help="障碍物密度")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="测试episode数量")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（如果不指定，将从检查点文件名或配置文件读取）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--render", action="store_true",
                        help="是否渲染")
    parser.add_argument("--save_trajectories", action="store_true",
                        help="是否保存轨迹")
    parser.add_argument("--output_dir", type=str, default="results/data",
                        help="输出目录")
    return parser.parse_args()


def create_test_env(num_robots, grid_size, obstacle_density, fov_size=3):
    """创建测试环境"""
    env_config = {
        "num_robots": num_robots,
        "grid_size_range": [grid_size, grid_size],  # 固定大小
        "obstacle_density_range": [obstacle_density, obstacle_density],  # 固定密度
        "obstacle_density_peak": obstacle_density,
        "fov_size": fov_size,
        "max_episode_steps": 256,
        "rewards": {
            "move": -0.3,
            "stay_on_goal": 0.0,
            "stay_off_goal": -0.3,
            "collision": -2.0,
            "blocking": -1.0,
            "reach_goal": 0.0
        },
        "use_intrinsic_reward": False  # 测试时不使用内在奖励
    }
    return MRPPEnv(env_config)


def load_agent(checkpoint_path, device, config_path=None):
    """加载智能体"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 尝试从检查点恢复配置
    checkpoint_config = checkpoint.get("config", {})  # 算法配置
    full_config = checkpoint.get("full_config", {})  # 完整配置（包括模型配置）
    
    # 默认配置（与train_config_stable.yaml一致）
    default_config = {
        "fov_size": 3,
        "encoder": {
            "conv_channels": [8, 16],
            "conv_kernels": [3, 3],
            "pool_layers": [],
            "fc_dims": [128],
            "lstm_hidden_dim": 128
        },
        "communication": {
            "hidden_dim": 128,
            "num_heads": 2,
            "num_layers": 2,
            "dropout": 0.0,
            "use_gru": True,
            "distance_embedding_dim": 32,
            "comm_radius": 15,
            "use_distance_encoding": True,
            "use_communication": True
        },
        "policy": {
            "hidden_dims": [128, 64],
            "activation": "relu"
        },
        "value": {
            "hidden_dims": [128, 64],
            "activation": "relu"
        }
    }
    
    # 如果提供了配置文件，优先使用配置文件
    if config_path and os.path.exists(config_path):
        print(f"从配置文件加载模型配置: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)
        
        # 从配置文件获取模型配置
        if "model" in file_config:
            model_config = file_config["model"]
            agent_config = {
                "fov_size": file_config.get("env", {}).get("fov_size", 3),
                "encoder": model_config.get("encoder", default_config["encoder"]),
                "communication": model_config.get("communication", default_config["communication"]),
                "policy": model_config.get("policy", default_config["policy"]),
                "value": model_config.get("value", default_config["value"])
            }
        else:
            print("警告: 配置文件中没有model配置，使用默认配置")
            agent_config = default_config
    else:
        # 尝试从检查点获取配置（优先使用full_config）
        if full_config.get("model"):
            print("从检查点的完整配置加载模型配置")
            agent_config = {
                "fov_size": full_config.get("env", {}).get("fov_size", 3),
                "encoder": full_config.get("model", {}).get("encoder", default_config["encoder"]),
                "communication": full_config.get("model", {}).get("communication", default_config["communication"]),
                "policy": full_config.get("model", {}).get("policy", default_config["policy"]),
                "value": full_config.get("model", {}).get("value", default_config["value"])
            }
        elif checkpoint_config.get("model"):
            print("从检查点的算法配置加载模型配置")
            agent_config = {
                "fov_size": checkpoint_config.get("env", {}).get("fov_size", 3),
                "encoder": checkpoint_config.get("model", {}).get("encoder", default_config["encoder"]),
                "communication": checkpoint_config.get("model", {}).get("communication", default_config["communication"]),
                "policy": checkpoint_config.get("model", {}).get("policy", default_config["policy"]),
                "value": checkpoint_config.get("model", {}).get("value", default_config["value"])
            }
        else:
            print("警告: 检查点中没有模型配置，使用默认配置（train_config_stable.yaml）")
            agent_config = default_config
    
    # 确保communication配置包含必要的字段
    if "use_communication" not in agent_config["communication"]:
        agent_config["communication"]["use_communication"] = True
    if "use_distance_encoding" not in agent_config["communication"]:
        agent_config["communication"]["use_distance_encoding"] = True
    
    # 创建智能体
    print(f"创建智能体: encoder={agent_config['encoder'].get('lstm_hidden_dim')}, "
          f"comm={agent_config['communication'].get('hidden_dim')}, "
          f"policy={agent_config['policy'].get('hidden_dims')}")
    agent = RMHAAgent(agent_config, device=device)
    
    # 尝试加载状态字典
    try:
        agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
        print("模型加载成功（strict=True）")
    except RuntimeError as e:
        print(f"警告: 严格模式加载失败，尝试使用strict=False: {e}")
        try:
            agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
            print("模型加载成功（strict=False，部分参数可能未加载）")
        except RuntimeError as e2:
            print(f"错误: 模型加载失败: {e2}")
            raise
    
    agent.eval()
    
    return agent


def test_episode(env, agent, device, render=False, seed=None):
    """测试单个episode"""
    # 如果提供了seed，使用它；否则使用环境的默认行为
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    collision_count = 0
    max_robots_reached = 0
    hidden_states = None
    
    robots_on_goal_history = []
    trajectory = []  # 用于保存轨迹
    
    # 获取最大步数（从环境配置或默认值）
    max_steps = env.max_episode_steps if hasattr(env, 'max_episode_steps') else 400
    
    while not done and episode_length < max_steps:
        # 准备输入
        obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
        obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
        distance_matrix = torch.from_numpy(
            env.get_distance_matrix()
        ).float().unsqueeze(0).to(device)
        
        # 推理（测试时使用确定性策略，选择最优动作）
        # 注意：训练时使用deterministic=False（探索），测试时使用deterministic=True（利用）
        # 这是标准的训练-测试差异：训练时探索，测试时利用
        with torch.no_grad():
            output = agent.forward(
                obs_image,
                obs_vector,
                distance_matrix,
                hidden_states,
                deterministic=True  # 测试时使用确定性策略，选择最优动作
            )
        
        actions = output["actions"].cpu().numpy()[0]
        hidden_states = output["new_hidden_states"]
        
        # 保存轨迹
        trajectory.append({
            "positions": env.grid_world.robot_positions.copy(),
            "actions": actions.copy()
        })
        
        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions.tolist())
        done = terminated or truncated
        
        episode_reward += np.sum(rewards)
        episode_length += 1
        collision_count = info.get("collision_count", 0)
        robots_on_goal = info.get("robots_on_goal", 0)
        robots_on_goal_history.append(robots_on_goal)
        max_robots_reached = max(max_robots_reached, robots_on_goal)
        
        if render:
            env.render()
    
    # 检查是否所有机器人都到达目标
    # 成功条件：episode正常终止（terminated=True）
    # 注意：根据env.step()的实现，terminated=True 表示所有机器人到达目标
    # truncated=True 表示达到最大步数（超时）
    all_robots_on_goal = terminated  # terminated=True 意味着所有机器人都在目标上
    
    # 调试信息：记录最后一步的状态
    final_robots_on_goal = robots_on_goal
    final_terminated = terminated
    final_truncated = truncated
    
    episode_data = {
        "done": done,
        "all_robots_on_goal": all_robots_on_goal,
        "max_robots_reached": max_robots_reached,
        "collision_count": collision_count,
        "episode_length": episode_length,
        "num_robots": env.num_robots,
        "episode_return": episode_reward,
        "robots_on_goal_history": robots_on_goal_history,
        "trajectory": trajectory,
        "final_robots_on_goal": final_robots_on_goal,
        "final_terminated": final_terminated,
        "final_truncated": final_truncated
    }
    
    return episode_data


def test():
    """主测试函数"""
    args = parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 确定随机种子
    # 优先级：命令行参数 > 检查点文件名 > 配置文件 > 默认值42
    seed = args.seed
    
    # 如果未指定，尝试从检查点文件名提取（格式：rmha_seed42/xxx.pth）
    if seed is None:
        import re
        checkpoint_path = args.checkpoint
        match = re.search(r'seed(\d+)', checkpoint_path)
        if match:
            seed = int(match.group(1))
            print(f"从检查点文件名提取seed: {seed}")
    
    # 如果仍未找到，尝试从配置文件读取
    if seed is None and args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)
            if "training" in file_config and "seed" in file_config["training"]:
                seed = file_config["training"]["seed"]
                print(f"从配置文件读取seed: {seed}")
    
    # 如果仍未找到，使用默认值42（与训练时一致）
    if seed is None:
        seed = 42
        print(f"使用默认seed: {seed}")
    
    print(f"最终使用的随机种子: {seed}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    agent = load_agent(args.checkpoint, device, config_path=args.config)
    print("模型加载完成")
    
    # 创建测试环境
    # 如果提供了配置文件，尝试从配置文件读取环境配置
    env_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)
            if "env" in file_config:
                env_config = file_config["env"]
                print(f"\n从配置文件读取环境配置: {args.config}")
                print(f"  机器人数量: {env_config.get('num_robots', args.num_robots)}")
                print(f"  网格大小范围: {env_config.get('grid_size_range', [args.grid_size, args.grid_size])}")
                print(f"  障碍物密度范围: {env_config.get('obstacle_density_range', [args.obstacle_density, args.obstacle_density])}")
    
    # 如果配置文件中有环境配置，使用配置文件；否则使用命令行参数
    if env_config:
        # 确保使用配置文件中的完整配置（包括奖励配置）
        # 如果配置文件中没有某些字段，使用默认值
        if "rewards" not in env_config:
            print("警告: 配置文件中没有rewards配置，使用默认值")
            env_config["rewards"] = {
                "move": 0.0,
                "stay_on_goal": 5.0,
                "stay_off_goal": 0.0,
                "collision": -0.1,
                "blocking": -0.05,
                "reach_goal": 10.0,
                "distance_reward": 0.2,
                "proximity_reward": 0.1,
                "success_bonus": 50.0
            }
        
        # 使用配置文件创建环境
        env = MRPPEnv(env_config)
        print(f"\n创建测试环境（使用配置文件）:")
        print(f"  机器人数量: {env_config.get('num_robots')}")
        print(f"  网格大小范围: {env_config.get('grid_size_range')}")
        print(f"  障碍物密度范围: {env_config.get('obstacle_density_range')}")
        print(f"  最大步数: {env_config.get('max_episode_steps', 'N/A')}")
        print(f"  奖励配置: {list(env_config.get('rewards', {}).keys())}")
    else:
        # 使用命令行参数创建环境
        print(f"\n创建测试环境（使用命令行参数）:")
        print(f"  机器人数量: {args.num_robots}")
        print(f"  网格大小: {args.grid_size}x{args.grid_size}")
        print(f"  障碍物密度: {args.obstacle_density:.1%}")
        
        env = create_test_env(
            num_robots=args.num_robots,
            grid_size=args.grid_size,
            obstacle_density=args.obstacle_density
        )
    
    # 显示环境配置信息
    print(f"\n环境配置信息:")
    print(f"  最大步数: {env.max_episode_steps}")
    print(f"  机器人数量: {env.num_robots}")
    print(f"  网格大小范围: {env.grid_size_range}")
    print(f"  障碍物密度范围: {env.obstacle_density_range}")
    
    # 先重置一次环境，获取实际的网格大小（用于显示）
    # 注意：这里重置后，第一个episode会再次重置，但seed不同，所以网格大小可能不同
    # 这只是为了显示，不影响测试
    try:
        test_obs, test_info = env.reset(seed=seed)
        print(f"  当前网格大小: {env.current_grid_size}")
        print(f"  当前障碍物密度: {env.current_obstacle_density:.2%}")
        print(f"  网格世界已创建: {hasattr(env, 'grid_world') and env.grid_world is not None}")
    except Exception as e:
        print(f"  警告: 环境重置失败: {e}")
        print(f"  当前网格大小: 未初始化（将在第一次测试时设置）")
    
    # 测试
    print(f"\n开始测试 ({args.num_episodes} episodes)...")
    episode_data_list = []
    
    # 统计信息
    success_count = 0
    terminated_count = 0
    truncated_count = 0
    
    for episode in tqdm(range(args.num_episodes), desc="测试进度"):
        # 设置种子以确保可重复性
        # 使用基础seed + episode编号，确保每个episode可重复但不同
        # 注意：与训练时保持一致，训练时使用seed=42，这里也使用相同的seed
        episode_seed = seed + episode
        
        episode_data = test_episode(
            env, 
            agent, 
            device, 
            render=args.render,
            seed=episode_seed  # 传递seed，test_episode会调用env.reset(seed=seed)
        )
        
        episode_data_list.append(episode_data)
        
        # 统计
        if episode_data.get("all_robots_on_goal", False):
            success_count += 1
        if episode_data.get("final_terminated", False):
            terminated_count += 1
        if episode_data.get("final_truncated", False):
            truncated_count += 1
        
        # 前10个episode打印详细信息
        if episode < 10:
            print(f"\n  Episode {episode+1} (seed={episode_seed}):")
            print(f"    长度: {episode_data['episode_length']}")
            print(f"    终止: terminated={episode_data.get('final_terminated', False)}, "
                  f"truncated={episode_data.get('final_truncated', False)}")
            print(f"    到达目标: {episode_data.get('final_robots_on_goal', 0)}/{env.num_robots}")
            print(f"    最大到达: {episode_data['max_robots_reached']}")
            print(f"    成功: {episode_data.get('all_robots_on_goal', False)}")
            print(f"    奖励: {episode_data['episode_return']:.2f}")
            
            # 检查环境状态
            if hasattr(env, 'current_grid_size'):
                print(f"    网格大小: {env.current_grid_size}")
            if hasattr(env, 'current_obstacle_density'):
                print(f"    障碍物密度: {env.current_obstacle_density:.2%}")
            if hasattr(env, 'grid_world') and env.grid_world is not None:
                print(f"    网格世界: {env.grid_world.grid_size}x{env.grid_world.grid_size}")
                print(f"    障碍物数量: {len(env.grid_world.obstacles)}")
            
            # 检查机器人位置历史
            if episode_data.get('trajectory') and len(episode_data['trajectory']) > 0:
                start_pos = episode_data['trajectory'][0]['positions']
                end_pos = episode_data['trajectory'][-1]['positions']
                print(f"    起始位置: {start_pos}")
                print(f"    结束位置: {end_pos}")
                
                # 检查目标位置
                if hasattr(env, 'grid_world') and env.grid_world is not None:
                    goals = env.grid_world.goal_positions
                    print(f"    目标位置: {goals}")
                    
                    # 计算距离
                    for i, (end, goal) in enumerate(zip(end_pos, goals)):
                        dist = env.grid_world.manhattan_distance(end, goal)
                        on_goal = (end == goal)
                        print(f"      机器人{i}: 位置={end}, 目标={goal}, 距离={dist}, 到达={'✓' if on_goal else '✗'}")
    
    print(f"\n测试统计:")
    print(f"  成功episode: {success_count}/{args.num_episodes} ({success_count/args.num_episodes*100:.1f}%)")
    print(f"  正常终止: {terminated_count}/{args.num_episodes}")
    print(f"  超时终止: {truncated_count}/{args.num_episodes}")
    
    # 计算指标
    print("\n计算评估指标...")
    metrics = compute_metrics(episode_data_list, return_individual=False)
    
    # 打印结果
    print_metrics(metrics, title=f"测试结果 ({args.num_episodes} episodes)")
    
    # 保存结果
    output_prefix = f"test_n{args.num_robots}_grid{args.grid_size}_obs{int(args.obstacle_density*100)}"
    
    # 保存指标
    metrics_path = os.path.join(args.output_dir, f"{output_prefix}_metrics.json")
    save_metrics_to_file(metrics, metrics_path)
    
    # 保存详细数据
    if args.save_trajectories:
        data_path = os.path.join(args.output_dir, f"{output_prefix}_data.json")
        # 移除轨迹以减小文件大小（可选）
        episode_data_list_to_save = [
            {k: v for k, v in data.items() if k != "trajectory"}
            for data in episode_data_list
        ]
        with open(data_path, "w") as f:
            json.dump(episode_data_list_to_save, f, indent=2)
        print(f"详细数据已保存到: {data_path}")
    
    print("\n测试完成！")
    
    # 返回关键指标
    return {
        "success_rate": metrics.get("success_mean", 0.0),
        "max_reached": metrics.get("max_reached_mean", 0.0),
        "collision_rate": metrics.get("collision_rate_mean", 0.0)
    }


if __name__ == "__main__":
    test()

