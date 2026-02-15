"""
日志记录工具
"""

import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import numpy as np


class Logger:
    """
    日志记录器
    支持TensorBoard和文件日志
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            use_tensorboard: 是否使用TensorBoard
            use_wandb: 是否使用Weights & Biases
            wandb_config: WandB配置
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(**wandb_config)
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb未安装，禁用WandB日志")
                self.use_wandb = False
        
        # 文件日志
        self.log_file = os.path.join(log_dir, "training_log.txt")
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        
        # 统计信息
        self.episode_stats = []
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """
        记录标量值
        
        Args:
            tag: 标签
            value: 值
            step: 步数
        """
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)
        
        if self.use_wandb:
            self.wandb.log({tag: value}, step=step)
    
    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: int
    ):
        """
        记录多个标量
        
        Args:
            tag: 主标签
            values: 值字典
            step: 步数
        """
        if self.use_tensorboard:
            self.writer.add_scalars(tag, values, step)
        
        if self.use_wandb:
            log_dict = {f"{tag}/{k}": v for k, v in values.items()}
            self.wandb.log(log_dict, step=step)
    
    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: int
    ):
        """
        记录直方图
        
        Args:
            tag: 标签
            values: 值数组
            step: 步数
        """
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, step)
    
    def log_episode(
        self,
        episode: int,
        stats: Dict[str, Any]
    ):
        """
        记录episode统计
        
        Args:
            episode: episode编号
            stats: 统计信息字典
        """
        self.episode_stats.append({
            "episode": episode,
            **stats
        })
        
        # 写入文件
        with open(self.log_file, "a") as f:
            f.write(f"Episode {episode}: {stats}\n")
        
        # 记录到TensorBoard
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"episode/{key}", value, episode)
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """
        记录训练步统计
        
        Args:
            step: 训练步数
            metrics: 指标字典
        """
        for key, value in metrics.items():
            self.log_scalar(f"train/{key}", value, step)
    
    def save_metrics(self):
        """保存所有metrics到JSON文件"""
        with open(self.metrics_file, "w") as f:
            json.dump(self.episode_stats, f, indent=2)
    
    def print_stats(
        self,
        episode: int,
        stats: Dict[str, Any]
    ):
        """
        打印统计信息
        
        Args:
            episode: episode编号
            stats: 统计信息
        """
        print(f"\n{'='*50}")
        print(f"Episode: {episode}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*50}\n")
    
    def close(self):
        """关闭日志记录器"""
        if self.writer is not None:
            self.writer.close()
        
        if self.use_wandb:
            self.wandb.finish()
        
        self.save_metrics()

