"""
价值网络（Critic）
"""

import torch
import torch.nn as nn
from typing import Tuple


class ValueNetwork(nn.Module):
    """
    价值网络
    预测状态价值
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Tuple[int] = (256, 128),
        activation: str = "relu"
    ):
        """
        初始化价值网络
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
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
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 (batch_size, input_dim)
            
        Returns:
            values: 状态价值 (batch_size, 1)
        """
        return self.network(features)


class DualValueNetwork(nn.Module):
    """
    双价值网络
    分别预测外在奖励和内在奖励的价值
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Tuple[int] = (256, 128),
        activation: str = "relu"
    ):
        """
        初始化双价值网络
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        # 外在价值网络
        self.extrinsic_value = ValueNetwork(input_dim, hidden_dims, activation)
        
        # 内在价值网络
        self.intrinsic_value = ValueNetwork(input_dim, hidden_dims, activation)
    
    def forward(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征 (batch_size, input_dim)
            
        Returns:
            extrinsic_values: 外在价值 (batch_size, 1)
            intrinsic_values: 内在价值 (batch_size, 1)
        """
        extrinsic_values = self.extrinsic_value(features)
        intrinsic_values = self.intrinsic_value(features)
        
        return extrinsic_values, intrinsic_values

