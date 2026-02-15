"""
策略网络（Actor）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    策略网络
    输出动作概率分布
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Tuple[int] = (256, 128),
        num_actions: int = 5,
        activation: str = "relu"
    ):
        """
        初始化策略网络
        
        Args:
            input_dim: 输入维度（来自LSTM+通信的特征）
            hidden_dims: 隐藏层维度
            num_actions: 动作数量
            activation: 激活函数
        """
        super().__init__()
        
        self.num_actions = num_actions
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 (batch_size, input_dim)
            
        Returns:
            action_logits: 动作logits (batch_size, num_actions)
        """
        action_logits = self.network(features)
        return action_logits
    
    def get_action_prob(self, features: torch.Tensor) -> torch.Tensor:
        """
        获取动作概率
        
        Args:
            features: 输入特征
            
        Returns:
            action_probs: 动作概率 (batch_size, num_actions)
        """
        action_logits = self.forward(features)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
    
    def sample_action(
        self, 
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            features: 输入特征
            deterministic: 是否确定性采样（选择最大概率动作）
            
        Returns:
            actions: 采样的动作 (batch_size,)
            log_probs: 动作的对数概率 (batch_size,)
            entropy: 策略熵 (batch_size,)
        """
        action_logits = self.forward(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 创建分布
        dist = Categorical(action_probs)
        
        if deterministic:
            actions = action_probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy
    
    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作
        
        Args:
            features: 输入特征
            actions: 要评估的动作
            
        Returns:
            log_probs: 动作的对数概率
            entropy: 策略熵
        """
        action_logits = self.forward(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class BlockingPredictor(nn.Module):
    """
    阻塞预测器
    预测机器人是否阻塞其他机器人
    """
    
    def __init__(self, input_dim: int = 256):
        """
        初始化阻塞预测器
        
        Args:
            input_dim: 输入维度
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 (batch_size, input_dim)
            
        Returns:
            blocking_prob: 阻塞概率 (batch_size, 1)
        """
        return self.network(features)

