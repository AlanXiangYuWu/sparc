"""
课程学习训练脚本
从简单环境开始，逐步增加难度
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from envs.mrpp_env import MRPPEnv
from algorithms.rmha_agent import RMHAAgent
from algorithms.mappo import MAPPO
from utils.buffer import RolloutBuffer
from utils.logger import Logger


def create_curriculum_configs():
    """创建课程学习配置"""
    configs = [
        {
            "name": "stage1_easy",
            "num_robots": 2,
            "grid_size_range": [6, 8],
            "obstacle_density_range": [0.0, 0.05],
            "obstacle_density_peak": 0.02,
            "max_episode_steps": 200,
            "rewards": {
                "move": 0.0,
                "stay_on_goal": 50.0,
                "stay_off_goal": 0.0,
                "collision": -0.5,
                "blocking": -0.1,
                "reach_goal": 100.0,
                "distance_reward": 2.0,
                "proximity_reward": 1.0
            },
            "total_steps": 500000,  # 50万步
            "success_threshold": 0.3  # 30%成功率进入下一阶段
        },
        {
            "name": "stage2_medium",
            "num_robots": 3,
            "grid_size_range": [8, 12],
            "obstacle_density_range": [0.0, 0.1],
            "obstacle_density_peak": 0.05,
            "max_episode_steps": 250,
            "rewards": {
                "move": -0.1,
                "stay_on_goal": 30.0,
                "stay_off_goal": -0.1,
                "collision": -1.0,
                "blocking": -0.5,
                "reach_goal": 50.0,
                "distance_reward": 1.0,
                "proximity_reward": 0.5
            },
            "total_steps": 1000000,  # 100万步
            "success_threshold": 0.2  # 20%成功率进入下一阶段
        },
        {
            "name": "stage3_hard",
            "num_robots": 4,
            "grid_size_range": [10, 20],
            "obstacle_density_range": [0.0, 0.2],
            "obstacle_density_peak": 0.1,
            "max_episode_steps": 256,
            "rewards": {
                "move": -0.2,
                "stay_on_goal": 10.0,
                "stay_off_goal": -0.2,
                "collision": -1.5,
                "blocking": -0.8,
                "reach_goal": 20.0,
                "distance_reward": 0.5,
                "proximity_reward": 0.2
            },
            "total_steps": 2000000,  # 200万步
            "success_threshold": 0.1  # 10%成功率进入下一阶段
        }
    ]
    return configs


def train_stage(stage_config, agent, mappo, device, resume_checkpoint=None):
    """训练一个阶段"""
    print(f"\n{'='*60}")
    print(f"开始训练阶段: {stage_config['name']}")
    print(f"{'='*60}")
    print(f"机器人数量: {stage_config['num_robots']}")
    print(f"网格大小: {stage_config['grid_size_range']}")
    print(f"障碍物密度: {stage_config['obstacle_density_range']}")
    print(f"总步数: {stage_config['total_steps']:,}")
    print(f"成功阈值: {stage_config['success_threshold']:.0%}")
    
    # 创建环境
    env_config = {
        "num_robots": stage_config["num_robots"],
        "grid_size_range": stage_config["grid_size_range"],
        "obstacle_density_range": stage_config["obstacle_density_range"],
        "obstacle_density_peak": stage_config["obstacle_density_peak"],
        "fov_size": 3,
        "max_episode_steps": stage_config["max_episode_steps"],
        "rewards": stage_config["rewards"]
    }
    
    env = MRPPEnv(env_config)
    
    # 创建回放缓冲区
    obs_shape = {
        "image": (8, 3, 3),
        "vector": (7,)
    }
    rollout_buffer = RolloutBuffer(
        num_robots=stage_config["num_robots"],
        num_steps=256,
        obs_shape=obs_shape,
        device=device
    )
    
    # 训练参数
    total_steps = stage_config["total_steps"]
    num_steps = 256
    log_interval = 10
    save_interval = 50
    
    # 训练循环
    obs, info = env.reset(seed=42)
    episode = 0
    episode_rewards = []
    episode_successes = []
    best_success_rate = 0.0
    
    global_step = 0
    pbar = tqdm(total=total_steps, desc=f"阶段: {stage_config['name']}")
    
    while global_step < total_steps:
        episode += 1
        rollout_buffer.reset()
        
        episode_reward = 0
        episode_length = 0
        hidden_states = None
        episode_terminated = False
        episode_info = {}
        
        # 收集一个rollout
        for step in range(num_steps):
            obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
            obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
            distance_matrix = torch.from_numpy(env.get_distance_matrix()).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = agent.forward(obs_image, obs_vector, distance_matrix, hidden_states, deterministic=False)
            
            actions = output["actions"].cpu().numpy()[0]
            log_probs = output["log_probs"].cpu().numpy()[0]
            values_ext = output["values_ext"].cpu().numpy()[0]
            hidden_states = output["new_hidden_states"]
            
            next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
            done = terminated or truncated
            
            rollout_buffer.add(
                obs=obs,
                actions=actions,
                rewards=np.array(rewards, dtype=np.float32),
                values=values_ext,
                log_probs=log_probs,
                dones=np.array([done] * len(rewards), dtype=bool)
            )
            
            obs = next_obs
            episode_reward += np.sum(rewards)
            episode_length += 1
            global_step += 1
            pbar.update(1)
            
            episode_terminated = terminated
            episode_info = info
            
            if done:
                break
        
        # 计算最后一步的价值
        with torch.no_grad():
            obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
            obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
            distance_matrix = torch.from_numpy(env.get_distance_matrix()).float().unsqueeze(0).to(device)
            last_values_ext, _ = agent.get_value(obs_image, obs_vector, distance_matrix, hidden_states)
            last_values = last_values_ext.cpu().numpy()[0]
        
        # 计算回报和优势
        rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=np.array([done] * len(last_values), dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95
        )
        
        # 更新策略
        train_stats = mappo.update(rollout_buffer)
        
        # 记录统计
        episode_rewards.append(episode_reward)
        success = episode_terminated
        episode_successes.append(1.0 if success else 0.0)
        
        # 日志
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_success = np.mean(episode_successes[-log_interval:])
            
            print(f"\nEpisode {episode}:")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  成功率: {avg_success:.2%}")
            print(f"  到达目标: {episode_info.get('robots_on_goal', 0)}/{stage_config['num_robots']}")
        
        # 保存模型
        if episode % save_interval == 0:
            checkpoint_path = f"results/checkpoints/{stage_config['name']}_ep{episode}.pth"
            mappo.save(checkpoint_path)
            
            if len(episode_successes) >= 100:
                recent_success_rate = np.mean(episode_successes[-100:])
                if recent_success_rate > best_success_rate:
                    best_success_rate = recent_success_rate
                    best_path = f"results/checkpoints/{stage_config['name']}_best.pth"
                    mappo.save(best_path)
                    print(f"  新的最佳模型！成功率: {best_success_rate:.2%}")
                
                # 检查是否达到成功阈值
                if recent_success_rate >= stage_config["success_threshold"]:
                    print(f"\n✓ 达到成功阈值 {stage_config['success_threshold']:.0%}！可以进入下一阶段")
                    break
        
        # 重置环境
        obs, info = env.reset()
        hidden_states = None
    
    pbar.close()
    
    return best_success_rate, mappo


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # 创建智能体和训练器
    agent_config = {
        "fov_size": 3,
        "encoder": {"lstm_hidden_dim": 128},
        "communication": {
            "hidden_dim": 128,
            "use_communication": True,
            "use_distance_encoding": True
        },
        "policy": {"hidden_dims": [128, 64]},
        "value": {"hidden_dims": [128, 64]}
    }
    
    agent = RMHAAgent(agent_config, device=device)
    
    algorithm_config = {
        "lr": 2.0e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.2,
        "max_grad_norm": 1.0,
        "num_steps": 256,
        "num_mini_batch": 4,
        "ppo_epoch": 8,
        "use_intrinsic_reward": True,
        "intrinsic_reward": {
            "tau": 0.5,
            "phi": 1.0,
            "beta": 0.01,
            "buffer_size": 3
        }
    }
    
    mappo = MAPPO(agent, algorithm_config, device=device)
    
    # 课程学习
    configs = create_curriculum_configs()
    checkpoint_path = None
    
    for stage_idx, stage_config in enumerate(configs):
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n从检查点恢复: {checkpoint_path}")
            mappo.load(checkpoint_path)
        
        success_rate, mappo = train_stage(stage_config, agent, mappo, device, checkpoint_path)
        
        checkpoint_path = f"results/checkpoints/{stage_config['name']}_best.pth"
        
        print(f"\n阶段 {stage_idx + 1} 完成！最终成功率: {success_rate:.2%}")
        
        if success_rate < stage_config["success_threshold"]:
            print(f"警告: 未达到成功阈值 {stage_config['success_threshold']:.0%}")
            print("建议: 继续训练当前阶段或降低阈值")
    
    print("\n课程学习完成！")


if __name__ == "__main__":
    main()


