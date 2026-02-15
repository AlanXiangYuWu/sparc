"""
RMHA训练脚本
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
from utils.metrics import compute_episode_metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练RMHA模型")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--algo", type=str, default="rmha",
                        choices=["rmha", "mappo", "mappo_gnn"],
                        help="算法类型")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_model(env, agent, device, num_episodes=20):
    """
    验证模型性能
    
    Args:
        env: 环境
        agent: 智能体
        device: 设备
        num_episodes: 验证episode数量
        
    Returns:
        验证集成功率
    """
    agent.eval()
    successes = []
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # 使用不同的seed（9999+episode），确保与训练数据不同
            obs, info = env.reset(seed=9999 + episode)
            done = False
            hidden_states = None
            
            while not done:
                obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
                obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
                distance_matrix = torch.from_numpy(
                    env.get_distance_matrix()
                ).float().unsqueeze(0).to(device)
                
                # 验证时使用确定性策略（选择最优动作）
                output = agent.forward(
                    obs_image, obs_vector, distance_matrix,
                    hidden_states, deterministic=True
                )
                
                actions = output["actions"].cpu().numpy()[0]
                hidden_states = output["new_hidden_states"]
                
                obs, rewards, terminated, truncated, info = env.step(actions.tolist())
                done = terminated or truncated
                
                if done:
                    successes.append(1.0 if terminated else 0.0)
                    break
    
    agent.train()
    return np.mean(successes) if successes else 0.0


def create_env(config: dict):
    """创建环境"""
    env_config = config["env"]
    return MRPPEnv(env_config)


def create_agent(config: dict, algo: str, device: str):
    """创建智能体"""
    agent_config = {
        "fov_size": config["env"]["fov_size"],
        "encoder": config["model"]["encoder"],
        "communication": config["model"]["communication"],
        "policy": config["model"]["policy"],
        "value": config["model"]["value"]
    }
    
    # 根据算法类型调整配置
    if algo == "mappo":
        # 无通信
        agent_config["communication"]["use_communication"] = False
        agent_config["communication"]["use_distance_encoding"] = False
    elif algo == "mappo_gnn":
        # 有通信但无距离编码
        agent_config["communication"]["use_communication"] = True
        agent_config["communication"]["use_distance_encoding"] = False
    elif algo == "rmha":
        # 完整RMHA
        agent_config["communication"]["use_communication"] = True
        agent_config["communication"]["use_distance_encoding"] = True
    
    return RMHAAgent(agent_config, device=device)


def train():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置随机种子
    seed = args.seed if args.seed is not None else config["training"].get("seed", 42)
    set_seed(seed)
    print(f"随机种子: {seed}")
    
    # 创建日志目录
    log_dir = os.path.join(config["training"]["log_dir"], f"{args.algo}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(config["training"]["checkpoint_dir"], f"{args.algo}_seed{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化日志记录器
    logger = Logger(
        log_dir=log_dir,
        use_tensorboard=config["training"].get("use_tensorboard", True),
        use_wandb=config["training"].get("use_wandb", False),
        wandb_config={
            "project": config["training"].get("wandb_project", "RMHA_MRPP"),
            "name": f"{args.algo}_seed{seed}",
            "config": config
        } if config["training"].get("use_wandb", False) else None
    )
    
    # 创建环境
    print("创建环境...")
    env = create_env(config)
    
    # 创建智能体
    print(f"创建智能体: {args.algo}")
    agent = create_agent(config, args.algo, device)
    print(f"模型参数数量: {sum(p.numel() for p in agent.parameters()):,}")
    
    # 创建MAPPO训练器
    mappo = MAPPO(agent, config["algorithm"], device=device)
    
    # 恢复训练
    start_episode = 0
    if args.resume is not None:
        print(f"从检查点恢复: {args.resume}")
        mappo.load(args.resume)
        # 从文件名提取episode数
        start_episode = int(args.resume.split("_ep")[-1].split(".")[0])
    
    # 创建回放缓冲区
    obs_shape = {
        "image": (8, config["env"]["fov_size"], config["env"]["fov_size"]),
        "vector": (7,)
    }
    rollout_buffer = RolloutBuffer(
        num_robots=config["env"]["num_robots"],
        num_steps=config["algorithm"]["num_steps"],
        obs_shape=obs_shape,
        device=device
    )
    
    # 训练参数
    total_timesteps = config["training"]["total_timesteps"]
    num_steps = config["algorithm"]["num_steps"]
    log_interval = config["training"].get("log_interval", 10)
    save_interval = config["training"].get("save_interval", 100)
    eval_interval = config["training"].get("eval_interval", 50)
    
    # 训练循环
    print("\n开始训练...")
    print(f"总时间步: {total_timesteps:,}")
    print(f"每回合步数: {num_steps}")
    print(f"预计总回合数: {total_timesteps // num_steps:,}\n")
    
    obs, info = env.reset(seed=seed)
    episode = start_episode
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    best_success_rate = 0.0
    
    global_step = start_episode * num_steps
    pbar = tqdm(total=total_timesteps, initial=global_step, desc="训练进度")
    
    while global_step < total_timesteps:
        episode += 1
        rollout_buffer.reset()
        
        episode_reward = 0
        episode_length = 0
        hidden_states = None
        episode_terminated = False  # 初始化终止标志
        episode_info = {}  # 初始化info
        
        # 收集一个rollout
        for step in range(num_steps):
            # 准备输入
            obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
            obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
            
            # 获取距离矩阵
            distance_matrix = torch.from_numpy(
                env.get_distance_matrix()
            ).float().unsqueeze(0).to(device)
            
            # 前向传播
            with torch.no_grad():
                output = agent.forward(
                    obs_image,
                    obs_vector,
                    distance_matrix,
                    hidden_states,
                    deterministic=False
                )
            
            actions = output["actions"].cpu().numpy()[0]
            log_probs = output["log_probs"].cpu().numpy()[0]
            values_ext = output["values_ext"].cpu().numpy()[0]
            hidden_states = output["new_hidden_states"]
            
            # 环境交互
            next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
            done = terminated or truncated
            
            # 应用奖励缩放（如果配置了）
            reward_scale = config.get("regularization", {}).get("reward_scale", 1.0)
            scaled_rewards = np.array(rewards, dtype=np.float32) * reward_scale
            
            # 存储经验
            rollout_buffer.add(
                obs=obs,
                actions=actions,
                rewards=scaled_rewards,
                values=values_ext,
                log_probs=log_probs,
                dones=np.array([done] * len(rewards), dtype=bool)
            )
            
            # 更新
            obs = next_obs
            episode_reward += np.sum(rewards)  # 日志使用原始奖励
            episode_length += 1
            global_step += 1
            pbar.update(1)
            
            # 保存终止信息（用于计算成功率）
            episode_terminated = terminated
            episode_info = info
            
            if done:
                break
        
        # 计算最后一步的价值
        with torch.no_grad():
            obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(device)
            obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(device)
            distance_matrix = torch.from_numpy(
                env.get_distance_matrix()
            ).float().unsqueeze(0).to(device)
            
            last_values_ext, _ = agent.get_value(
                obs_image,
                obs_vector,
                distance_matrix,
                hidden_states
            )
            last_values = last_values_ext.cpu().numpy()[0]
        
        # 计算回报和优势
        rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=np.array([done] * len(last_values), dtype=np.float32),
            gamma=config["algorithm"]["gamma"],
            gae_lambda=config["algorithm"]["gae_lambda"]
        )
        
        # 更新策略
        train_stats = mappo.update(rollout_buffer)
        
        # 记录episode统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 计算成功率
        # terminated=True表示所有机器人都到达目标（成功）
        # truncated=True表示达到最大步数（失败）
        success = episode_terminated  # 如果terminated=True，说明所有机器人都到达目标
        episode_successes.append(1.0 if success else 0.0)
        
        # 日志记录
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            avg_success = np.mean(episode_successes[-log_interval:])
            
            # 计算最近episode的详细信息
            recent_robots_on_goal = episode_info.get("robots_on_goal", 0)
            recent_max_reached = episode_info.get("max_robots_reached", 0)
            recent_collisions = episode_info.get("collision_count", 0)
            
            logger.log_episode(episode, {
                "reward": avg_reward,
                "length": avg_length,
                "success_rate": avg_success,
                "robots_on_goal": recent_robots_on_goal,
                "max_robots_reached": recent_max_reached,
                "collision_count": recent_collisions,
                "terminated": episode_terminated,
                **train_stats
            })
            
            logger.print_stats(episode, {
                "avg_reward": avg_reward,
                "avg_length": avg_length,
                "success_rate": avg_success,
                "last_robots_on_goal": recent_robots_on_goal,
                "last_max_reached": recent_max_reached,
                "last_terminated": episode_terminated,
                **train_stats
            })
        
        # 验证集评估（每10个episode评估一次）
        val_success_rate = None
        if episode % 10 == 0 and episode > 0:
            val_success_rate = validate_model(env, agent, device, num_episodes=20)
            logger.log_scalar("validation/success_rate", val_success_rate, episode)
            print(f"验证集成功率: {val_success_rate:.2%}")
        
        # 保存模型
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pth")
            mappo.save(checkpoint_path, full_config=config)
            
            # 保存最佳模型（基于验证集性能）
            if val_success_rate is not None:
                if val_success_rate > best_success_rate:
                    best_success_rate = val_success_rate
                    best_path = os.path.join(checkpoint_dir, f"{args.algo}_best.pth")
                    mappo.save(best_path, full_config=config)
                    print(f"新的最佳模型！验证集成功率: {best_success_rate:.2%}")
            elif len(episode_successes) >= 100:
                # 如果没有验证集，使用训练集成功率
                recent_success_rate = np.mean(episode_successes[-100:])
                if recent_success_rate > best_success_rate:
                    best_success_rate = recent_success_rate
                    best_path = os.path.join(checkpoint_dir, f"{args.algo}_best.pth")
                    mappo.save(best_path, full_config=config)
                    print(f"新的最佳模型！训练集成功率: {best_success_rate:.2%}")
        
        # 重置环境
        # 使用seed + episode确保环境可复现，同时每个episode不同
        # 注意：网格大小和障碍物密度仍然会在配置的范围内随机变化（这是设计如此）
        obs, info = env.reset(seed=seed + episode)
        hidden_states = None
    
    pbar.close()
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, f"{args.algo}_final.pth")
    mappo.save(final_path, full_config=config)
    
    # 关闭日志
    logger.close()
    
    print("\n训练完成！")
    print(f"最终模型保存在: {final_path}")
    print(f"最佳模型成功率: {best_success_rate:.2%}")


if __name__ == "__main__":
    train()

