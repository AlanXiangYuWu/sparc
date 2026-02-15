"""
RMHA智能体
集成所有模块的完整智能体实现
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np

from models.encoder import ObservationEncoder
from models.rmha import RMHACommunication
from models.policy import PolicyNetwork, BlockingPredictor
from models.value import DualValueNetwork


class RMHAAgent(nn.Module):
    """
    RMHA智能体
    
    完整的端到端模型，包括：
    - 观测编码
    - RMHA通信
    - 策略输出
    - 价值估计
    - 阻塞预测
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = "cuda"
    ):
        """
        初始化RMHA智能体
        
        Args:
            config: 配置字典
            device: 设备
        """
        super().__init__()
        
        self.config = config
        self.device = device
        
        # 提取配置
        encoder_config = config.get("encoder", {})
        comm_config = config.get("communication", {})
        policy_config = config.get("policy", {})
        value_config = config.get("value", {})
        
        # 观测编码器
        self.encoder = ObservationEncoder(
            image_channels=8,
            fov_size=config.get("fov_size", 3),
            vector_dim=7,
            conv_channels=tuple(encoder_config.get("conv_channels", [8, 16, 32, 64, 64, 32, 16])),
            conv_kernels=tuple(encoder_config.get("conv_kernels", [3, 3, 3, 3, 3, 3, 3])),
            pool_layers=tuple(encoder_config.get("pool_layers", [2, 4])),
            fc_dims=tuple(encoder_config.get("fc_dims", [256, 256, 256])),
            lstm_hidden_dim=encoder_config.get("lstm_hidden_dim", 256),
            device=device
        )
        
        # 通信模块
        # 只有当明确设置use_communication为True时才使用通信
        self.use_communication = comm_config.get("use_communication", False)
        
        if self.use_communication:
            self.communication = RMHACommunication(
                hidden_dim=comm_config.get("hidden_dim", 256),
                num_heads=comm_config.get("num_heads", 4),
                num_layers=comm_config.get("num_layers", 3),
                distance_embedding_dim=comm_config.get("distance_embedding_dim", 64),
                dropout=comm_config.get("dropout", 0.0),
                use_gru=comm_config.get("use_gru", True),
                comm_radius=comm_config.get("comm_radius", 40.0)
            )
            self.use_distance_encoding = comm_config.get("use_distance_encoding", True)
        else:
            self.communication = None
            self.use_distance_encoding = False
        
        # 策略网络
        input_dim = encoder_config.get("lstm_hidden_dim", 256)
        if self.use_communication:
            input_dim += comm_config.get("hidden_dim", 256)
        
        self.policy = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=tuple(policy_config.get("hidden_dims", [256, 128])),
            num_actions=5,
            activation=policy_config.get("activation", "relu")
        )
        
        # 价值网络
        self.value = DualValueNetwork(
            input_dim=input_dim,
            hidden_dims=tuple(value_config.get("hidden_dims", [256, 128])),
            activation=value_config.get("activation", "relu")
        )
        
        # 阻塞预测器
        self.blocking_predictor = BlockingPredictor(input_dim=input_dim)
        
        # 消息存储
        self.message_dim = comm_config.get("hidden_dim", 256)
        self.current_messages = None
        
        self.to(device)
    
    def forward(
        self,
        obs_image: torch.Tensor,
        obs_vector: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs_image: 图像观测 (batch_size, num_robots, 8, fov_size, fov_size)
            obs_vector: 向量观测 (batch_size, num_robots, 7)
            distance_matrix: 距离矩阵 (batch_size, num_robots, num_robots)
            hidden_states: LSTM隐藏状态列表
            deterministic: 是否确定性采样
            
        Returns:
            输出字典，包含：
            - actions: 动作
            - log_probs: 对数概率
            - values_ext: 外在价值
            - values_int: 内在价值
            - entropy: 熵
            - messages: 通信消息
            - new_hidden_states: 新的隐藏状态
            - blocking_probs: 阻塞概率
        """
        batch_size, num_robots = obs_image.size(0), obs_image.size(1)
        
        # 展平batch和robot维度进行编码
        obs_image_flat = obs_image.view(batch_size * num_robots, *obs_image.shape[2:])
        obs_vector_flat = obs_vector.view(batch_size * num_robots, obs_vector.size(-1))
        
        # 编码观测
        if hidden_states is None:
            hidden_states = [self.encoder.init_hidden_state(1) for _ in range(batch_size * num_robots)]
        
        encoded_features = []
        new_hidden_states = []
        
        for i in range(batch_size * num_robots):
            obs_img = obs_image_flat[i:i+1]
            obs_vec = obs_vector_flat[i:i+1]
            h_state = hidden_states[i]
            
            encoded, new_h = self.encoder(obs_img, obs_vec, h_state)
            encoded_features.append(encoded)
            new_hidden_states.append(new_h)
        
        encoded_features = torch.cat(encoded_features, dim=0)  # (B*N, lstm_hidden_dim)
        encoded_features = encoded_features.view(batch_size, num_robots, -1)
        
        # 通信模块
        if self.use_communication:
            # 如果distance_matrix为None，创建默认距离矩阵
            if distance_matrix is None:
                distance_matrix = torch.ones(
                    (batch_size, num_robots, num_robots),
                    device=encoded_features.device,
                    dtype=torch.float32
                ) * 10.0  # 默认距离值（在通信半径内）
                # 对角线设为0
                for i in range(num_robots):
                    distance_matrix[:, i, i] = 0.0
            
            # 使用编码特征作为初始消息
            messages = encoded_features
            
            # RMHA通信
            if self.use_distance_encoding:
                updated_messages = self.communication(messages, distance_matrix)
            else:
                # 无距离编码的图通信（消融基线）
                # 使用全1距离矩阵（所有距离相同）
                dummy_distance = torch.ones_like(distance_matrix)
                updated_messages = self.communication(messages, dummy_distance)
            
            # 拼接编码特征和通信消息
            combined_features = torch.cat([encoded_features, updated_messages], dim=-1)
            self.current_messages = updated_messages
        else:
            # 无通信
            combined_features = encoded_features
            self.current_messages = None
        
        # 展平以进行策略和价值计算
        combined_features_flat = combined_features.view(batch_size * num_robots, -1)
        
        # 策略网络
        actions, log_probs, entropy = self.policy.sample_action(
            combined_features_flat, 
            deterministic=deterministic
        )
        
        # 价值网络
        values_ext, values_int = self.value(combined_features_flat)
        
        # 阻塞预测
        blocking_probs = self.blocking_predictor(combined_features_flat)
        
        # 重塑回(batch_size, num_robots)
        actions = actions.view(batch_size, num_robots)
        log_probs = log_probs.view(batch_size, num_robots)
        entropy = entropy.view(batch_size, num_robots)
        values_ext = values_ext.view(batch_size, num_robots)
        values_int = values_int.view(batch_size, num_robots)
        blocking_probs = blocking_probs.view(batch_size, num_robots)
        
        return {
            "actions": actions,
            "log_probs": log_probs,
            "values_ext": values_ext,
            "values_int": values_int,
            "entropy": entropy,
            "messages": self.current_messages,
            "new_hidden_states": new_hidden_states,
            "blocking_probs": blocking_probs
        }
    
    def evaluate_actions(
        self,
        obs_image: torch.Tensor,
        obs_vector: torch.Tensor,
        actions: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        hidden_states: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:
        """
        评估给定动作
        用于PPO训练
        
        Args:
            obs_image: 图像观测
            obs_vector: 向量观测
            actions: 要评估的动作
            distance_matrix: 距离矩阵
            hidden_states: 隐藏状态
            
        Returns:
            评估结果字典
        """
        batch_size, num_robots = obs_image.size(0), obs_image.size(1)
        
        # 展平
        obs_image_flat = obs_image.view(batch_size * num_robots, *obs_image.shape[2:])
        obs_vector_flat = obs_vector.view(batch_size * num_robots, obs_vector.size(-1))
        actions_flat = actions.view(batch_size * num_robots)
        
        # 编码
        if hidden_states is None:
            hidden_states = [self.encoder.init_hidden_state(1) for _ in range(batch_size * num_robots)]
        
        encoded_features = []
        for i in range(batch_size * num_robots):
            encoded, _ = self.encoder(
                obs_image_flat[i:i+1], 
                obs_vector_flat[i:i+1], 
                hidden_states[i]
            )
            encoded_features.append(encoded)
        
        encoded_features = torch.cat(encoded_features, dim=0)
        encoded_features = encoded_features.view(batch_size, num_robots, -1)
        
        # 通信
        if self.use_communication:
            # 如果distance_matrix为None，创建默认距离矩阵（训练时）
            if distance_matrix is None:
                distance_matrix = torch.ones(
                    (batch_size, num_robots, num_robots),
                    device=encoded_features.device,
                    dtype=torch.float32
                ) * 10.0  # 默认距离值（在通信半径内）
                # 对角线设为0
                for i in range(num_robots):
                    distance_matrix[:, i, i] = 0.0
            
            messages = encoded_features
            if self.use_distance_encoding:
                updated_messages = self.communication(messages, distance_matrix)
            else:
                dummy_distance = torch.ones_like(distance_matrix)
                updated_messages = self.communication(messages, dummy_distance)
            combined_features = torch.cat([encoded_features, updated_messages], dim=-1)
        else:
            combined_features = encoded_features
        
        combined_features_flat = combined_features.view(batch_size * num_robots, -1)
        
        # 评估动作
        log_probs, entropy = self.policy.evaluate_actions(combined_features_flat, actions_flat)
        
        # 价值估计
        values_ext, values_int = self.value(combined_features_flat)
        
        # 重塑
        log_probs = log_probs.view(batch_size, num_robots)
        entropy = entropy.view(batch_size, num_robots)
        values_ext = values_ext.view(batch_size, num_robots)
        values_int = values_int.view(batch_size, num_robots)
        
        return {
            "log_probs": log_probs,
            "values_ext": values_ext,
            "values_int": values_int,
            "entropy": entropy
        }
    
    def get_value(
        self,
        obs_image: torch.Tensor,
        obs_vector: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        hidden_states: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅获取价值估计
        
        Args:
            obs_image: 图像观测
            obs_vector: 向量观测
            distance_matrix: 距离矩阵
            hidden_states: 隐藏状态
            
        Returns:
            values_ext, values_int: 外在和内在价值
        """
        batch_size, num_robots = obs_image.size(0), obs_image.size(1)
        
        # 展平
        obs_image_flat = obs_image.view(batch_size * num_robots, *obs_image.shape[2:])
        obs_vector_flat = obs_vector.view(batch_size * num_robots, obs_vector.size(-1))
        
        # 编码
        if hidden_states is None:
            hidden_states = [self.encoder.init_hidden_state(1) for _ in range(batch_size * num_robots)]
        
        encoded_features = []
        for i in range(batch_size * num_robots):
            encoded, _ = self.encoder(
                obs_image_flat[i:i+1],
                obs_vector_flat[i:i+1],
                hidden_states[i]
            )
            encoded_features.append(encoded)
        
        encoded_features = torch.cat(encoded_features, dim=0)
        encoded_features = encoded_features.view(batch_size, num_robots, -1)
        
        # 通信
        if self.use_communication:
            # 如果distance_matrix为None，创建默认距离矩阵（训练时）
            if distance_matrix is None:
                distance_matrix = torch.ones(
                    (batch_size, num_robots, num_robots),
                    device=encoded_features.device,
                    dtype=torch.float32
                ) * 10.0  # 默认距离值（在通信半径内）
                # 对角线设为0
                for i in range(num_robots):
                    distance_matrix[:, i, i] = 0.0
            
            messages = encoded_features
            if self.use_distance_encoding:
                updated_messages = self.communication(messages, distance_matrix)
            else:
                dummy_distance = torch.ones_like(distance_matrix)
                updated_messages = self.communication(messages, dummy_distance)
            combined_features = torch.cat([encoded_features, updated_messages], dim=-1)
        else:
            combined_features = encoded_features
        
        combined_features_flat = combined_features.view(batch_size * num_robots, -1)
        
        # 价值估计
        values_ext, values_int = self.value(combined_features_flat)
        
        values_ext = values_ext.view(batch_size, num_robots)
        values_int = values_int.view(batch_size, num_robots)
        
        return values_ext, values_int

