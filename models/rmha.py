"""
空间关系增强的多头注意力通信模块（RMHA）
论文核心创新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DistanceEmbedding(nn.Module):
    """
    距离嵌入模块
    将曼哈顿距离编码为高维向量
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        初始化距离嵌入
        
        Args:
            embedding_dim: 嵌入维度
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 使用可学习的嵌入层
        # 假设最大距离为200（足够大）
        self.distance_embedding = nn.Embedding(201, embedding_dim)
        
        # 或者使用MLP
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
    
    def forward(self, distance_matrix: torch.Tensor, use_mlp: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            distance_matrix: 距离矩阵 (batch_size, num_robots, num_robots)
            use_mlp: 是否使用MLP（否则使用Embedding）
            
        Returns:
            embedded_distances: 嵌入后的距离 (batch_size, num_robots, num_robots, embedding_dim)
        """
        if use_mlp:
            # 使用MLP编码
            distances = distance_matrix.unsqueeze(-1)  # (B, N, N, 1)
            embedded = self.distance_mlp(distances)  # (B, N, N, D)
        else:
            # 使用Embedding编码
            # 裁剪距离到有效范围并转换为整数
            distances_int = distance_matrix.long().clamp(0, 200)
            embedded = self.distance_embedding(distances_int)  # (B, N, N, D)
        
        return embedded


class RMHALayer(nn.Module):
    """
    单层RMHA（空间关系增强的多头注意力）
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        distance_embedding_dim: int = 64,
        dropout: float = 0.0,
        use_gru: bool = True
    ):
        """
        初始化RMHA层
        
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            distance_embedding_dim: 距离嵌入维度
            dropout: Dropout率
            use_gru: 是否使用GRU门控
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.distance_embedding_dim = distance_embedding_dim
        self.use_gru = use_gru
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # 距离嵌入
        self.distance_embedding = DistanceEmbedding(distance_embedding_dim)
        
        # 按照论文公式11-13：W_1和W_2用于投影距离信息
        # Q = W_q(M^{t-1} + W_1 Distance)
        # K = W_k(M^{t-1} + W_2 Distance)
        self.distance_proj_q = nn.Linear(distance_embedding_dim, hidden_dim)  # W_1
        self.distance_proj_k = nn.Linear(distance_embedding_dim, hidden_dim)  # W_2
        
        # Query, Key, Value投影
        self.query = nn.Linear(hidden_dim, hidden_dim)  # W_q
        self.key = nn.Linear(hidden_dim, hidden_dim)   # W_k
        self.value = nn.Linear(hidden_dim, hidden_dim) # W_v
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # GRU门控（论文中提到）
        if use_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        messages: torch.Tensor,
        distance_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            messages: 消息矩阵 (batch_size, num_robots, hidden_dim)
            distance_matrix: 距离矩阵 (batch_size, num_robots, num_robots)
            mask: 注意力掩码 (batch_size, num_robots, num_robots)
                  1表示可通信，0表示不可通信
            
        Returns:
            updated_messages: 更新后的消息 (batch_size, num_robots, hidden_dim)
        """
        batch_size, num_robots, _ = messages.size()
        
        # 按照论文公式11-13实现
        # Q = W_q(M^{t-1} + W_1 Distance)
        # K = W_k(M^{t-1} + W_2 Distance)
        # V = W_v M^{t-1}
        
        # 1. 计算距离嵌入
        distance_emb = self.distance_embedding(distance_matrix)  # (B, N, N, D_emb)
        
        # 2. 按照论文公式5：s_ij = (o_i + d_i→j) W_q^T W_k (o_j + d_j→i)
        # 对于每个机器人i，需要聚合到所有其他机器人的距离信息
        # 这里使用平均聚合：对于机器人i，使用所有j的距离特征的平均值
        
        # 计算W_1 Distance和W_2 Distance
        distance_features_q = self.distance_proj_q(distance_emb)  # (B, N, N, H) - W_1 Distance
        distance_features_k = self.distance_proj_k(distance_emb)  # (B, N, N, H) - W_2 Distance
        
        # 3. 为每个机器人i聚合距离信息
        # 对于Q：机器人i需要考虑所有j的距离d_i→j，使用行方向平均
        # 对于K：机器人j需要考虑所有i的距离d_j→i，使用列方向平均
        distance_enhanced_q = distance_features_q.mean(dim=2)  # (B, N, H) - 对每个i，平均所有j的距离
        distance_enhanced_k = distance_features_k.mean(dim=1)  # (B, N, H) - 对每个j，平均所有i的距离
        
        # 4. 按照论文公式计算Q, K, V
        # Q = W_q(M^{t-1} + W_1 Distance)
        enhanced_messages_q = messages + distance_enhanced_q
        Q = self.query(enhanced_messages_q)  # (B, N, H)
        
        # K = W_k(M^{t-1} + W_2 Distance)
        enhanced_messages_k = messages + distance_enhanced_k
        K = self.key(enhanced_messages_k)    # (B, N, H)
        
        # V = W_v M^{t-1}（V不使用距离信息）
        V = self.value(messages)  # (B, N, H)
        
        # 重塑为多头
        Q = Q.view(batch_size, num_robots, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_robots, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_robots, self.num_heads, self.head_dim)
        
        # 转置以便计算注意力
        Q = Q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.transpose(1, 2)  # (B, num_heads, N, head_dim)
        V = V.transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # 融合距离特征到Q和K
        # 这是论文的核心：s_ij = (o_i + d_i->j) W_q^T W_k (o_j + d_j->i)
        # 简化实现：直接在attention score中加入距离的影响
        
        # 计算注意力分数（按照论文公式14）
        # 距离信息已经融合在Q和K中，直接计算点积即可
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, num_heads, N, N)
        
        # 应用掩码（限制通信范围）
        if mask is not None:
            # mask: (B, N, N)，需要扩展到 (B, num_heads, N, N)
            mask_expanded = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, num_robots, self.hidden_dim)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 残差连接 + 层归一化
        if self.use_gru:
            # 使用GRU更新（论文中提到）
            messages_flat = messages.view(-1, self.hidden_dim)
            attn_output_flat = attn_output.view(-1, self.hidden_dim)
            updated_messages = self.gru(attn_output_flat, messages_flat)
            updated_messages = updated_messages.view(batch_size, num_robots, self.hidden_dim)
        else:
            updated_messages = self.norm1(messages + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(updated_messages)
        ffn_output = self.dropout(ffn_output)
        
        # 残差连接 + 层归一化
        updated_messages = self.norm2(updated_messages + ffn_output)
        
        return updated_messages


class RMHACommunication(nn.Module):
    """
    RMHA通信模块
    包含多层RMHA
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        distance_embedding_dim: int = 64,
        dropout: float = 0.0,
        use_gru: bool = True,
        comm_radius: float = 40.0
    ):
        """
        初始化RMHA通信模块
        
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: RMHA层数
            distance_embedding_dim: 距离嵌入维度
            dropout: Dropout率
            use_gru: 是否使用GRU
            comm_radius: 通信半径
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.comm_radius = comm_radius
        
        # 位置编码（正弦编码机器人ID）
        self.position_encoding = nn.Parameter(
            torch.randn(1, 1000, hidden_dim) * 0.02  # 支持最多1000个机器人
        )
        
        # 多层RMHA
        self.layers = nn.ModuleList([
            RMHALayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                distance_embedding_dim=distance_embedding_dim,
                dropout=dropout,
                use_gru=use_gru
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        messages: torch.Tensor,
        distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            messages: 初始消息 (batch_size, num_robots, hidden_dim)
            distance_matrix: 距离矩阵 (batch_size, num_robots, num_robots)
            
        Returns:
            updated_messages: 更新后的消息 (batch_size, num_robots, hidden_dim)
        """
        batch_size, num_robots, _ = messages.size()
        
        # 添加位置编码
        pos_enc = self.position_encoding[:, :num_robots, :].expand(batch_size, -1, -1)
        messages = messages + pos_enc
        
        # 生成通信掩码（基于通信半径）
        mask = (distance_matrix <= self.comm_radius).float()  # (B, N, N)
        # 对角线设为1（自己总是可以"通信"）
        mask = mask + torch.eye(num_robots, device=messages.device).unsqueeze(0)
        mask = (mask > 0).float()
        
        # 通过多层RMHA
        updated_messages = messages
        for layer in self.layers:
            updated_messages = layer(updated_messages, distance_matrix, mask)
        
        return updated_messages

