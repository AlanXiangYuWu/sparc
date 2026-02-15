"""
观测编码器
包括CNN编码器、LSTM时序编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ObservationEncoder(nn.Module):
    """
    观测编码器
    
    架构：
    - CNN: 7层卷积 + 2层最大池化
    - FC: 3层全连接
    - LSTM: 时序特征融合
    """
    
    def __init__(
        self,
        image_channels: int = 8,
        fov_size: int = 3,
        vector_dim: int = 7,
        conv_channels: Tuple[int] = (8, 16, 32, 64, 64, 32, 16),
        conv_kernels: Tuple[int] = (3, 3, 3, 3, 3, 3, 3),
        pool_layers: Tuple[int] = (2, 4),
        fc_dims: Tuple[int] = (256, 256, 256),
        lstm_hidden_dim: int = 256,
        device: str = "cuda"
    ):
        """
        初始化观测编码器
        
        Args:
            image_channels: 图像通道数
            fov_size: 视野大小
            vector_dim: 向量维度
            conv_channels: CNN通道数列表
            conv_kernels: 卷积核大小列表
            pool_layers: 池化层位置列表
            fc_dims: 全连接层维度列表
            lstm_hidden_dim: LSTM隐藏维度
            device: 设备
        """
        super().__init__()
        
        self.image_channels = image_channels
        self.fov_size = fov_size
        self.vector_dim = vector_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device
        
        # CNN编码器
        # 对于小FOV（3×3），使用轻量结构避免池化导致的尺寸问题
        conv_layers = []
        in_channels = image_channels
        
        # 如果FOV很小（<=3），使用简化的CNN结构
        if fov_size <= 3:
            # 使用较少的卷积层，避免池化
            num_conv_layers = min(4, len(conv_channels))
            for i in range(num_conv_layers):
                out_channels = conv_channels[i]
                kernel_size = conv_kernels[i]
                conv_layers.append(nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ))
                conv_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            
            # 使用AdaptivePooling将任意大小池化到1×1
            conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        else:
            # 对于大FOV，使用原始结构，但使用AdaptivePooling避免尺寸问题
            for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, conv_kernels)):
                conv_layers.append(nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ))
                conv_layers.append(nn.ReLU(inplace=True))
                
                # 在指定层添加自适应池化（避免固定池化的尺寸问题）
                if i in pool_layers:
                    # 计算池化后的目标尺寸
                    pool_idx = list(pool_layers).index(i) + 1
                    target_size = max(1, fov_size // (2 ** pool_idx))
                    conv_layers.append(nn.AdaptiveAvgPool2d((target_size, target_size)))
                
                in_channels = out_channels
            
            # 最后添加AdaptivePooling确保输出大小为1×1
            conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.cnn = nn.Sequential(*conv_layers)
        
        # 计算CNN输出维度
        self.cnn_output_dim = self._get_cnn_output_dim()
        
        # 向量编码器（简单的FC层）
        self.vector_fc = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(inplace=True)
        )
        
        # 融合后的全连接层
        fusion_input_dim = self.cnn_output_dim + 128
        
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(fusion_input_dim, fc_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fusion_input_dim = fc_dim
        
        self.fc = nn.Sequential(*fc_layers)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=fc_dims[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.to(device)
    
    def _get_cnn_output_dim(self) -> int:
        """计算CNN输出维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.image_channels, self.fov_size, self.fov_size)
            dummy_output = self.cnn(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def forward(
        self, 
        image_obs: torch.Tensor,
        vector_obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            image_obs: 图像观测 (batch_size, image_channels, fov_size, fov_size)
            vector_obs: 向量观测 (batch_size, vector_dim)
            hidden_state: LSTM隐藏状态 (h, c)
            
        Returns:
            encoded_obs: 编码后的观测 (batch_size, lstm_hidden_dim)
            new_hidden_state: 新的LSTM隐藏状态
        """
        batch_size = image_obs.size(0)
        
        # CNN编码图像
        cnn_features = self.cnn(image_obs)
        cnn_features = cnn_features.view(batch_size, -1)  # Flatten
        
        # FC编码向量
        vector_features = self.vector_fc(vector_obs)
        
        # 拼接特征
        fused_features = torch.cat([cnn_features, vector_features], dim=1)
        
        # 全连接层
        fc_output = self.fc(fused_features)
        
        # LSTM层（添加时间维度）
        fc_output_seq = fc_output.unsqueeze(1)  # (batch_size, 1, fc_dim)
        
        if hidden_state is None:
            lstm_output, new_hidden_state = self.lstm(fc_output_seq)
        else:
            lstm_output, new_hidden_state = self.lstm(fc_output_seq, hidden_state)
        
        # 移除时间维度
        encoded_obs = lstm_output.squeeze(1)  # (batch_size, lstm_hidden_dim)
        
        return encoded_obs, new_hidden_state
    
    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (h0, c0): 初始隐藏状态
        """
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device)
        return (h0, c0)

