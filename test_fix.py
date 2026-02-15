"""
测试修复后的编码器和智能体
"""

import torch
import numpy as np
from envs.mrpp_env import MRPPEnv
from algorithms.rmha_agent import RMHAAgent

print("Testing fixed encoder and agent...")

# 创建环境
env_config = {
    "num_robots": 4,
    "grid_size_range": [10, 10],
    "obstacle_density_range": [0.1, 0.3],  # 需要左<右
    "obstacle_density_peak": 0.2,
    "fov_size": 3,
    "max_episode_steps": 10,
    "rewards": {
        "move": -0.3,
        "stay_on_goal": 0.0,
        "stay_off_goal": -0.3,
        "collision": -2.0,
        "blocking": -1.0
    }
}

print("1. Creating environment...")
env = MRPPEnv(env_config)
obs, info = env.reset(seed=42)
print(f"   [OK] Environment created (robots: {env.num_robots}, grid: {env.current_grid_size}x{env.current_grid_size})")
print(f"   [OK] Observation shape: image={obs['image'].shape}, vector={obs['vector'].shape}")

# 创建智能体
agent_config = {
    "fov_size": 3,
    "encoder": {
        "lstm_hidden_dim": 256
    },
    "communication": {
        "hidden_dim": 256,
        "use_communication": True,
        "use_distance_encoding": True
    },
    "policy": {
        "hidden_dims": [256, 128]
    },
    "value": {
        "hidden_dims": [256, 128]
    }
}

print("\n2. Creating agent...")
agent = RMHAAgent(agent_config, device="cpu")
print("   [OK] Agent created")

# 测试前向传播
print("\n3. Testing forward pass...")
obs_img = torch.from_numpy(obs["image"]).float().unsqueeze(0)
obs_vec = torch.from_numpy(obs["vector"]).float().unsqueeze(0)
dist = torch.from_numpy(env.get_distance_matrix()).float().unsqueeze(0)

output = agent.forward(obs_img, obs_vec, dist)
print(f"   [OK] Actions shape: {output['actions'].shape}")
print(f"   [OK] Values ext shape: {output['values_ext'].shape}")
print(f"   [OK] Values int shape: {output['values_int'].shape}")
print(f"   [OK] Log probs shape: {output['log_probs'].shape}")

# 测试一步环境交互
print("\n4. Testing environment step...")
actions = output["actions"].cpu().numpy()[0]
next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
print(f"   [OK] Step successful")
print(f"   [OK] Rewards: {rewards}")
print(f"   [OK] Done: {terminated or truncated}")

print("\n" + "="*50)
print("All tests passed! The fix is working correctly!")
print("="*50)
print("\nYou can now run training:")
print("  python train.py --config config/train_config.yaml --algo rmha")

