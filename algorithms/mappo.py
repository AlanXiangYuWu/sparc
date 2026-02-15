"""
MAPPO算法实现
Multi-Agent Proximal Policy Optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np

from utils.buffer import RolloutBuffer


class MAPPO:
    """
    MAPPO训练器
    
    实现集中式训练分散式执行（CTDE）的PPO算法
    """
    
    def __init__(
        self,
        agent: nn.Module,
        config: Dict,
        device: str = "cuda"
    ):
        """
        初始化MAPPO
        
        Args:
            agent: RMHA智能体
            config: 算法配置
            device: 设备
        """
        self.agent = agent
        self.config = config
        self.device = device
        
        # PPO参数
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_param = config.get("clip_param", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        
        # 训练参数
        self.num_steps = config.get("num_steps", 256)
        self.num_mini_batch = config.get("num_mini_batch", 8)
        self.ppo_epoch = config.get("ppo_epoch", 4)
        
        # 优化器
        optimizer_config = config.get("optimization", {})
        self.optimizer = optim.Adam(
            agent.parameters(),
            lr=self.lr,
            eps=optimizer_config.get("adam_eps", 1e-5),
            weight_decay=optimizer_config.get("weight_decay", 0.0)
        )
        
        # 学习率调度器
        self.use_lr_scheduler = optimizer_config.get("use_lr_scheduler", True)
        if self.use_lr_scheduler:
            scheduler_config = optimizer_config.get("lr_scheduler", {})
            scheduler_type = scheduler_config.get("type", "cosine")
            
            if scheduler_type == "cosine":
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.get("total_timesteps", 10000000),
                    eta_min=self.lr * 0.1
                )
            elif scheduler_type == "linear":
                self.lr_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=config.get("total_timesteps", 10000000)
                )
            else:
                self.lr_scheduler = None
        else:
            self.lr_scheduler = None
        
        # 混合精度训练
        self.use_amp = optimizer_config.get("use_amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 正则化
        reg_config = config.get("regularization", {})
        self.normalize_advantage = reg_config.get("normalize_advantage", True)
        self.clip_observation = reg_config.get("clip_observation", True)
        
        # 统计
        self.update_count = 0
    
    def update(
        self,
        rollout_buffer: RolloutBuffer
    ) -> Dict[str, float]:
        """
        使用PPO更新策略
        
        Args:
            rollout_buffer: 回合缓冲区
            
        Returns:
            训练统计信息
        """
        # 获取所有数据
        data = rollout_buffer.get()
        
        # 计算批次大小
        total_samples = data["actions"].size(0)
        mini_batch_size = total_samples // self.num_mini_batch
        
        # 统计信息
        policy_losses = []
        value_ext_losses = []
        value_int_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        # PPO多轮更新
        for epoch in range(self.ppo_epoch):
            # 采样小批次
            batches = rollout_buffer.sample(mini_batch_size)
            
            for batch in batches:
                # 提取批次数据
                obs_image = batch["obs_image"]
                obs_vector = batch["obs_vector"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]
                
                # Buffer中的数据是展平的 (batch_size * num_robots, ...)
                # 需要重塑为 (batch_size, num_robots, ...) 格式
                # 从配置中获取num_robots（需要从agent或config传递）
                # 这里假设batch_size是mini_batch_size，num_robots需要从agent获取
                
                # 获取num_robots（从agent配置或buffer）
                # 简化：假设每个batch包含完整的机器人组
                # 实际应该从buffer或config中获取
                batch_size_samples = obs_image.size(0)
                # 假设每个样本对应一个机器人，需要知道num_robots来重塑
                # 临时方案：假设batch_size_samples能被某个数整除
                # 更好的方案：从agent或config传递num_robots
                
                # 由于buffer数据是展平的，我们需要知道num_robots
                # 临时解决方案：直接处理展平的数据，修改evaluate_actions来处理
                # 或者：从agent获取num_robots
                
                # 获取num_robots（从buffer）
                num_robots = rollout_buffer.num_robots
                
                # 重塑数据：batch中的数据是展平的 (batch_size * num_robots, ...)
                # 需要重塑为 (batch_size, num_robots, ...)
                batch_size = batch_size_samples // num_robots
                if batch_size_samples % num_robots != 0:
                    # 如果无法整除，说明数据格式有问题
                    raise ValueError(
                        f"Cannot reshape data: batch_size_samples={batch_size_samples} "
                        f"must be divisible by num_robots={num_robots}"
                    )
                
                # 重塑观测数据
                obs_image_reshaped = obs_image.view(batch_size, num_robots, *obs_image.shape[1:])
                obs_vector_reshaped = obs_vector.view(batch_size, num_robots, *obs_vector.shape[1:])
                actions_reshaped = actions.view(batch_size, num_robots)
                
                # 创建默认距离矩阵（训练时使用）
                # 使用全1矩阵表示所有机器人都可以通信（距离为1，在通信半径内）
                # 实际训练时应该从buffer中保存距离矩阵，这里使用简化方案
                distance_matrix = torch.ones(
                    (batch_size, num_robots, num_robots),
                    device=obs_image.device,
                    dtype=torch.float32
                ) * 10.0  # 设置一个合理的距离值（在通信半径内）
                
                # 对角线设为0（自己到自己的距离）
                for i in range(num_robots):
                    distance_matrix[:, i, i] = 0.0
                
                hidden_states = None
                
                # 前向传播
                if self.use_amp:
                    autocast_context = torch.amp.autocast('cuda', enabled=True)
                else:
                    from contextlib import nullcontext
                    autocast_context = nullcontext()
                
                with autocast_context:
                    # 评估动作
                    eval_results = self.agent.evaluate_actions(
                        obs_image_reshaped,
                        obs_vector_reshaped,
                        actions_reshaped,
                        distance_matrix,
                        hidden_states
                    )
                    
                    new_log_probs = eval_results["log_probs"].view(-1)
                    values_ext = eval_results["values_ext"].view(-1)
                    values_int = eval_results["values_int"].view(-1)
                    entropy = eval_results["entropy"].view(-1)
                    
                    # 计算比率
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # PPO裁剪目标
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值损失（使用裁剪）
                    value_pred_clipped = old_values + torch.clamp(
                        values_ext - old_values,
                        -self.clip_param,
                        self.clip_param
                    )
                    value_loss_1 = (values_ext - returns).pow(2)
                    value_loss_2 = (value_pred_clipped - returns).pow(2)
                    value_ext_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                    
                    # 内在价值损失（如果使用）
                    value_int_loss = 0.5 * values_int.pow(2).mean()  # 简化实现
                    
                    # 熵损失
                    entropy_loss = -entropy.mean()
                    
                    # 总损失
                    loss = (
                        policy_loss 
                        + self.value_loss_coef * (value_ext_loss + value_int_loss)
                        + self.entropy_coef * entropy_loss
                    )
                
                # 反向传播
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # 统计
                policy_losses.append(policy_loss.item())
                value_ext_losses.append(value_ext_loss.item())
                value_int_losses.append(value_int_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
                
                # KL散度（近似）
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kls.append(approx_kl.item())
                    
                    # 裁剪比例
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_param).float())
                    clip_fractions.append(clip_fraction.item())
        
        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.update_count += 1
        
        # 返回统计信息
        stats = {
            "policy_loss": np.mean(policy_losses),
            "value_ext_loss": np.mean(value_ext_losses),
            "value_int_loss": np.mean(value_int_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        return stats
    
    def save(self, path: str, full_config=None):
        """
        保存模型
        
        Args:
            path: 保存路径
            full_config: 完整配置（包括模型配置），如果提供则保存
        """
        save_dict = {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "config": self.config  # 算法配置
        }
        
        # 如果提供了完整配置，也保存
        if full_config is not None:
            save_dict["full_config"] = full_config
        
        torch.save(save_dict, path)
        
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        
        print(f"模型已从 {path} 加载")

