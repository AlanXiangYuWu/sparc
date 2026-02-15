"""
Quick test for optimized configuration
"""

import numpy as np
import torch
from envs.mrpp_env import MRPPEnv
from algorithms.rmha_agent import RMHAAgent

print("="*60)
print("Quick Test: Optimized Configuration")
print("="*60)

# Use aggressive configuration
env_config = {
    "num_robots": 2,
    "grid_size_range": [6, 8],
    "obstacle_density_range": [0.0, 0.05],
    "obstacle_density_peak": 0.02,
    "fov_size": 3,
    "max_episode_steps": 300,
    "rewards": {
        "move": 0.0,
        "stay_on_goal": 50.0,
        "stay_off_goal": 0.0,
        "collision": -0.5,
        "blocking": -0.1,
        "reach_goal": 100.0,
        "distance_reward": 2.0,
        "proximity_reward": 1.0
    }
}

print("\n1. Creating environment...")
env = MRPPEnv(env_config)
obs, info = env.reset(seed=42)

print(f"   [OK] Environment created")
print(f"   Grid size: {env.current_grid_size}x{env.current_grid_size}")
print(f"   Obstacle density: {env.current_obstacle_density:.2%}")
print(f"   Number of robots: {env.num_robots}")

# Check reward configuration
print(f"\n2. Checking reward configuration...")
print(f"   Move reward: {env.rewards.get('move', 0)}")
print(f"   Reach goal reward: {env.rewards.get('reach_goal', 0)}")
print(f"   Stay on goal: {env.rewards.get('stay_on_goal', 0)}")
print(f"   Distance reward: {env.rewards.get('distance_reward', 0)}")
print(f"   Proximity reward: {env.rewards.get('proximity_reward', 0)}")

# Test reward calculation
print(f"\n3. Testing reward calculation...")
total_rewards = []
success_count = 0

for episode in range(10):
    obs, info = env.reset(seed=42 + episode)
    episode_reward = 0
    episode_terminated = False
    
    for step in range(100):
        # Use heuristic: move towards goal
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
    
    if episode < 3:
        print(f"   Episode {episode+1}: reward={episode_reward:.2f}, success={episode_terminated}, robots_on_goal={info.get('robots_on_goal', 0)}/{env.num_robots}")

avg_reward = np.mean(total_rewards)
success_rate = success_count / 10

print(f"\n4. Test results:")
print(f"   Average reward: {avg_reward:.2f}")
print(f"   Success rate: {success_rate:.0%}")
print(f"   Max reward: {max(total_rewards):.2f}")

if success_rate > 0:
    print(f"\n   [OK] Environment is reasonable, heuristic can reach goals")
else:
    print(f"\n   [WARN] Even heuristic cannot reach goals, may need simpler environment")

# Test agent
print(f"\n5. Testing agent...")
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

agent = RMHAAgent(agent_config, device="cpu")
agent.eval()

obs, info = env.reset(seed=42)
obs_image = torch.from_numpy(obs["image"]).float().unsqueeze(0)
obs_vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0)
distance_matrix = torch.from_numpy(env.get_distance_matrix()).float().unsqueeze(0)

with torch.no_grad():
    output = agent.forward(obs_image, obs_vector, distance_matrix)

print(f"   [OK] Agent forward pass successful")
print(f"   Action shape: {output['actions'].shape}")
print(f"   Actions: {output['actions'].cpu().numpy()[0]}")

print("\n" + "="*60)
print("Test completed!")
print("="*60)
print("\nRecommendations:")
if success_rate > 0:
    print("[OK] Environment is reasonable, you can start training:")
    print("  python train.py --config config/train_config_aggressive.yaml --algo rmha")
else:
    print("[WARN] Environment may still be too difficult:")
    print("  1. Further reduce obstacle density")
    print("  2. Use smaller grid")
    print("  3. Check if start/goal positions are reasonable")
