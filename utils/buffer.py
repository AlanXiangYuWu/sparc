"""
经验回放缓冲区
用于PPO算法
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class RolloutBuffer:
    """
    Rollout Buffer用于存储一个回合的经验
    """
    
    def __init__(
        self,
        num_robots: int,
        num_steps: int,
        obs_shape: Dict[str, Tuple],
        device: str = "cuda"
    ):
        """
        初始化Rollout Buffer
        
        Args:
            num_robots: 机器人数量
            num_steps: 每次rollout的步数
            obs_shape: 观测形状字典
            device: 设备
        """
        self.num_robots = num_robots
        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.device = device
        
        self.reset()
    
    def reset(self):
        """重置buffer"""
        # 观测
        self.observations = {
            "image": torch.zeros(
                (self.num_steps, self.num_robots) + self.obs_shape["image"], 
                dtype=torch.float32
            ),
            "vector": torch.zeros(
                (self.num_steps, self.num_robots) + self.obs_shape["vector"], 
                dtype=torch.float32
            )
        }
        
        # 动作
        self.actions = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.long
        )
        
        # 奖励
        self.rewards = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.float32
        )
        
        # 价值估计
        self.values = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.float32
        )
        
        # 对数概率
        self.log_probs = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.float32
        )
        
        # 完成标志
        self.dones = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.bool
        )
        
        # 优势
        self.advantages = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.float32
        )
        
        # 回报
        self.returns = torch.zeros(
            (self.num_steps, self.num_robots), 
            dtype=torch.float32
        )
        
        # 当前步数
        self.step = 0
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray
    ):
        """
        添加一步经验
        
        Args:
            obs: 观测字典
            actions: 动作
            rewards: 奖励
            values: 价值估计
            log_probs: 对数概率
            dones: 完成标志
        """
        if self.step >= self.num_steps:
            raise ValueError("Buffer已满")
        
        self.observations["image"][self.step] = torch.from_numpy(obs["image"])
        self.observations["vector"][self.step] = torch.from_numpy(obs["vector"])
        self.actions[self.step] = torch.from_numpy(actions)
        self.rewards[self.step] = torch.from_numpy(rewards)
        self.values[self.step] = torch.from_numpy(values)
        self.log_probs[self.step] = torch.from_numpy(log_probs)
        self.dones[self.step] = torch.from_numpy(dones)
        
        self.step += 1
    
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        计算回报和优势（使用GAE）
        
        Args:
            last_values: 最后一步的价值估计
            last_dones: 最后一步的完成标志
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        last_values = torch.from_numpy(last_values).float()
        last_dones = torch.from_numpy(last_dones).float()
        
        last_gae_lam = 0
        
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].float()
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step] 
                + gamma * next_values * next_non_terminal 
                - self.values[step]
            )
            
            last_gae_lam = (
                delta 
                + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )
            
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values
    
    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        获取所有经验
        
        Args:
            batch_size: 如果指定，返回随机批次
            
        Returns:
            经验字典
        """
        # 展平时间和机器人维度
        obs_image = self.observations["image"].view(-1, *self.obs_shape["image"])
        obs_vector = self.observations["vector"].view(-1, *self.obs_shape["vector"])
        actions = self.actions.view(-1)
        values = self.values.view(-1)
        log_probs = self.log_probs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        data = {
            "obs_image": obs_image.to(self.device),
            "obs_vector": obs_vector.to(self.device),
            "actions": actions.to(self.device),
            "values": values.to(self.device),
            "log_probs": log_probs.to(self.device),
            "advantages": advantages.to(self.device),
            "returns": returns.to(self.device)
        }
        
        return data
    
    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        采样小批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            小批次列表
        """
        data = self.get()
        total_size = data["actions"].size(0)
        
        indices = torch.randperm(total_size)
        
        batches = []
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = {
                key: value[batch_indices]
                for key, value in data.items()
            }
            batches.append(batch)
        
        return batches

