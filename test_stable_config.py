"""
测试稳定配置
"""

import numpy as np
import torch
import yaml
from envs.mrpp_env import MRPPEnv

print("="*60)
print("测试稳定配置")
print("="*60)

# 加载稳定配置
with open("config/train_config_stable.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

env_config = config["env"]
print(f"\n环境配置:")
print(f"  机器人数量: {env_config['num_robots']}")
print(f"  网格大小: {env_config['grid_size_range']}")
print(f"  障碍物密度: {env_config['obstacle_density_range']}")
print(f"  最大步数: {env_config['max_episode_steps']}")

print(f"\n奖励配置:")
for key, value in env_config['rewards'].items():
    print(f"  {key}: {value}")

# 创建环境
env = MRPPEnv(env_config)
obs, info = env.reset(seed=42)

print(f"\n环境创建成功:")
print(f"  网格大小: {env.current_grid_size}x{env.current_grid_size}")
print(f"  障碍物密度: {env.current_obstacle_density:.2%}")

# 测试奖励计算
print(f"\n测试奖励计算（使用启发式策略）...")
total_rewards = []
success_count = 0
success_bonus_count = 0

for episode in range(10):
    obs, info = env.reset(seed=42 + episode)
    episode_reward = 0
    episode_terminated = False
    
    for step in range(env_config['max_episode_steps']):
        # 使用启发式策略：向目标移动
        actions = []
        for robot_id in range(env.num_robots):
            pos = env.grid_world.robot_positions[robot_id]
            goal = env.grid_world.goal_positions[robot_id]
            
            dx = goal[1] - pos[1]
            dy = goal[0] - pos[0]
            
            if dx > 0:
                action = 3  # RIGHT
            elif dx < 0:
                action = 2  # LEFT
            elif dy > 0:
                action = 1  # DOWN
            elif dy < 0:
                action = 0  # UP
            else:
                action = 4  # STAY
            
            actions.append(action)
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_reward += sum(rewards)
        episode_terminated = terminated
        
        if terminated or truncated:
            break
    
    total_rewards.append(episode_reward)
    if episode_terminated:
        success_count += 1
        # 检查是否有成功奖励
        if episode_reward > 50:  # 成功奖励应该是50+
            success_bonus_count += 1
    
    if episode < 3:
        print(f"  Episode {episode+1}: 奖励={episode_reward:.2f}, 成功={episode_terminated}, 到达目标={info.get('robots_on_goal', 0)}/{env.num_robots}")

avg_reward = np.mean(total_rewards)
success_rate = success_count / 10

print(f"\n测试结果:")
print(f"  平均奖励: {avg_reward:.2f}")
print(f"  成功率: {success_rate:.0%}")
print(f"  成功奖励次数: {success_bonus_count}/{success_count}")

# 检查奖励缩放
reward_scale = config.get("regularization", {}).get("reward_scale", 1.0)
print(f"\n奖励缩放配置:")
print(f"  reward_scale: {reward_scale}")
print(f"  实际奖励值范围: {min(total_rewards):.2f} - {max(total_rewards):.2f}")
print(f"  缩放后奖励值范围: {min(total_rewards)*reward_scale:.2f} - {max(total_rewards)*reward_scale:.2f}")

print("\n" + "="*60)
print("测试完成！")
print("="*60)

if success_rate > 0 and avg_reward > 0:
    print("\n[OK] 配置正常，可以开始训练")
    print("使用: python train.py --config config/train_config_stable.yaml --algo rmha")
else:
    print("\n[WARN] 可能需要进一步调整配置")


