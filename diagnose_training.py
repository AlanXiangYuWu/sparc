"""
训练诊断脚本
用于分析为什么成功率是0
"""

import numpy as np
import torch
from envs.mrpp_env import MRPPEnv
from algorithms.rmha_agent import RMHAAgent
import yaml

def diagnose_environment():
    """诊断环境设置"""
    print("="*60)
    print("环境诊断")
    print("="*60)
    
    # 创建简单环境
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
    
    env = MRPPEnv(env_config)
    
    # 测试几个episode
    print("\n测试环境设置...")
    for episode in range(5):
        obs, info = env.reset(seed=42 + episode)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  网格大小: {env.current_grid_size}x{env.current_grid_size}")
        print(f"  障碍物密度: {env.current_obstacle_density:.2%}")
        print(f"  机器人数量: {env.num_robots}")
        
        # 检查起始和目标位置
        for i in range(env.num_robots):
            start = env.grid_world.start_positions[i]
            goal = env.grid_world.goal_positions[i]
            dist = env.grid_world.manhattan_distance(start, goal)
            print(f"  机器人{i}: 起始{start} -> 目标{goal}, 距离={dist}")
        
        # 测试随机动作
        total_reward = 0
        for step in range(50):
            actions = np.random.randint(0, 5, size=env.num_robots).tolist()
            obs, rewards, terminated, truncated, info = env.step(actions)
            total_reward += sum(rewards)
            
            if terminated:
                print(f"  成功！步数={step+1}, 总奖励={total_reward:.2f}")
                break
            if truncated:
                print(f"  超时，步数={step+1}, 总奖励={total_reward:.2f}")
                break
        
        # 检查奖励
        print(f"  最终奖励: {total_reward:.2f}")
        print(f"  到达目标数: {info.get('robots_on_goal', 0)}/{env.num_robots}")

def diagnose_rewards():
    """诊断奖励计算"""
    print("\n" + "="*60)
    print("奖励诊断")
    print("="*60)
    
    env_config = {
        "num_robots": 2,
        "grid_size_range": [6, 6],
        "obstacle_density_range": [0.0, 0.0],
        "obstacle_density_peak": 0.0,
        "fov_size": 3,
        "max_episode_steps": 100,
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
    
    env = MRPPEnv(env_config)
    obs, info = env.reset(seed=42)
    
    print("\n测试奖励计算...")
    print(f"起始位置: {env.grid_world.robot_positions}")
    print(f"目标位置: {env.grid_world.goal_positions}")
    
    # 测试一个机器人向目标移动
    robot_id = 0
    start = env.grid_world.robot_positions[robot_id]
    goal = env.grid_world.goal_positions[robot_id]
    
    print(f"\n机器人{robot_id}:")
    print(f"  起始: {start}, 目标: {goal}")
    
    # 计算距离
    dist = env.grid_world.manhattan_distance(start, goal)
    print(f"  曼哈顿距离: {dist}")
    
    # 模拟移动
    current_pos = start
    total_reward = 0
    
    for step in range(min(dist + 5, 20)):
        # 计算到目标的方向
        dx = goal[1] - current_pos[1]
        dy = goal[0] - current_pos[0]
        
        # 选择动作（优先移动）
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
        
        actions = [action, 4]  # 第一个机器人移动，第二个停留
        
        prev_pos = env.grid_world.robot_positions.copy()
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        reward = rewards[robot_id]
        total_reward += reward
        new_pos = env.grid_world.robot_positions[robot_id]
        
        print(f"  步{step+1}: {prev_pos[robot_id]} -> {new_pos}, 动作={action}, 奖励={reward:.2f}, 累计={total_reward:.2f}")
        
        if new_pos == goal:
            print(f"  ✓ 到达目标！总奖励={total_reward:.2f}")
            break
        
        current_pos = new_pos

def diagnose_agent():
    """诊断智能体"""
    print("\n" + "="*60)
    print("智能体诊断")
    print("="*60)
    
    agent_config = {
        "fov_size": 3,
        "encoder": {
            "lstm_hidden_dim": 128
        },
        "communication": {
            "hidden_dim": 128,
            "use_communication": True,
            "use_distance_encoding": True
        },
        "policy": {
            "hidden_dims": [128, 64]
        },
        "value": {
            "hidden_dims": [128, 64]
        }
    }
    
    agent = RMHAAgent(agent_config, device="cpu")
    agent.eval()
    
    print("\n测试智能体前向传播...")
    
    # 创建测试输入
    batch_size = 1
    num_robots = 2
    
    obs_image = torch.randn(batch_size, num_robots, 8, 3, 3)
    obs_vector = torch.randn(batch_size, num_robots, 7)
    distance_matrix = torch.randn(batch_size, num_robots, num_robots) * 5 + 5
    
    with torch.no_grad():
        output = agent.forward(obs_image, obs_vector, distance_matrix, deterministic=False)
    
    print(f"  输入形状: image={obs_image.shape}, vector={obs_vector.shape}")
    print(f"  输出形状: actions={output['actions'].shape}")
    print(f"  动作值: {output['actions']}")
    print(f"  动作概率: {torch.softmax(agent.policy.network(torch.randn(1, 256)), dim=-1)}")
    
    # 检查动作分布
    actions = output['actions'].cpu().numpy()[0]
    print(f"  采样动作: {actions}")
    print(f"  动作分布: UP={np.sum(actions==0)}, DOWN={np.sum(actions==1)}, LEFT={np.sum(actions==2)}, RIGHT={np.sum(actions==3)}, STAY={np.sum(actions==4)}")

if __name__ == "__main__":
    print("开始诊断训练问题...\n")
    
    diagnose_environment()
    diagnose_rewards()
    diagnose_agent()
    
    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)
    print("\n建议:")
    print("1. 检查环境设置是否合理")
    print("2. 检查奖励计算是否正确")
    print("3. 检查智能体是否能正常输出动作")
    print("4. 如果都正常，可能需要更激进的奖励设计")


