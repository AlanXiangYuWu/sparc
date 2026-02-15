"""
评估指标计算
"""

import numpy as np
from typing import List, Dict, Tuple


def compute_success_rate(
    all_done: List[bool],
    all_robots_on_goal: List[bool]
) -> float:
    """
    计算成功率
    
    Args:
        all_done: 是否完成列表
        all_robots_on_goal: 是否所有机器人到达目标列表
        
    Returns:
        成功率 [0, 1]
    """
    if len(all_done) == 0:
        return 0.0
    
    num_success = sum(1 for done, on_goal in zip(all_done, all_robots_on_goal) 
                      if done and on_goal)
    
    return num_success / len(all_done)


def compute_max_reached(
    robots_on_goal_history: List[int]
) -> int:
    """
    计算最大目标达成数
    
    Args:
        robots_on_goal_history: 每个时间步到达目标的机器人数列表
        
    Returns:
        最大目标达成数
    """
    if len(robots_on_goal_history) == 0:
        return 0
    
    return max(robots_on_goal_history)


def compute_collision_rate(
    collision_count: int,
    total_steps: int,
    num_robots: int
) -> float:
    """
    计算碰撞率
    
    Args:
        collision_count: 碰撞次数
        total_steps: 总步数
        num_robots: 机器人数量
        
    Returns:
        碰撞率 [0, 1]
    """
    if total_steps == 0 or num_robots == 0:
        return 0.0
    
    return collision_count / (total_steps * num_robots)


def compute_episode_metrics(
    episode_data: Dict
) -> Dict[str, float]:
    """
    计算单个episode的所有指标
    
    Args:
        episode_data: episode数据字典，包含：
            - done: 是否完成
            - all_robots_on_goal: 是否所有机器人到达目标
            - max_robots_reached: 最大到达目标的机器人数
            - collision_count: 碰撞次数
            - episode_length: episode长度
            - num_robots: 机器人数量
            - episode_return: 总回报
            
    Returns:
        指标字典
    """
    metrics = {}
    
    # 成功率（单个episode要么成功要么失败）
    metrics["success"] = 1.0 if (episode_data.get("done", False) and 
                                  episode_data.get("all_robots_on_goal", False)) else 0.0
    
    # 最大目标达成数
    metrics["max_reached"] = episode_data.get("max_robots_reached", 0)
    
    # 碰撞率
    collision_count = episode_data.get("collision_count", 0)
    episode_length = episode_data.get("episode_length", 1)
    num_robots = episode_data.get("num_robots", 1)
    metrics["collision_rate"] = compute_collision_rate(
        collision_count, episode_length, num_robots
    )
    
    # episode长度
    metrics["episode_length"] = episode_length
    
    # 总回报
    metrics["episode_return"] = episode_data.get("episode_return", 0.0)
    
    # 到达目标的机器人比例
    metrics["robots_on_goal_ratio"] = (
        episode_data.get("max_robots_reached", 0) / num_robots
    )
    
    return metrics


def compute_batch_metrics(
    episode_data_list: List[Dict]
) -> Dict[str, float]:
    """
    计算多个episode的平均指标
    
    Args:
        episode_data_list: episode数据列表
        
    Returns:
        平均指标字典
    """
    if len(episode_data_list) == 0:
        return {}
    
    # 计算每个episode的指标
    all_metrics = [compute_episode_metrics(data) for data in episode_data_list]
    
    # 计算平均值
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[f"{key}_mean"] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)
        avg_metrics[f"{key}_min"] = np.min(values)
        avg_metrics[f"{key}_max"] = np.max(values)
    
    return avg_metrics


def compute_metrics(
    episode_data_list: List[Dict],
    return_individual: bool = False
) -> Dict[str, float]:
    """
    计算评估指标（主函数）
    
    Args:
        episode_data_list: episode数据列表
        return_individual: 是否返回每个episode的指标
        
    Returns:
        指标字典
    """
    batch_metrics = compute_batch_metrics(episode_data_list)
    
    if return_individual:
        individual_metrics = [
            compute_episode_metrics(data) 
            for data in episode_data_list
        ]
        batch_metrics["individual"] = individual_metrics
    
    return batch_metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    打印指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if key != "individual":
            if isinstance(value, float):
                print(f"{key:30s}: {value:10.4f}")
            else:
                print(f"{key:30s}: {value:10}")
    
    print(f"{'='*60}\n")


def _convert_to_python_type(obj):
    """
    将numpy类型转换为Python原生类型（用于JSON序列化）
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的Python原生类型
    """
    import numpy as np
    
    # 兼容numpy 2.0（移除了np.float_和np.int_）
    if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8, np.uint16,
                       np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_python_type(item) for item in obj]
    else:
        return obj


def save_metrics_to_file(
    metrics: Dict[str, float],
    filepath: str
):
    """
    保存指标到文件
    
    Args:
        metrics: 指标字典
        filepath: 文件路径
    """
    import json
    
    # 移除individual（太大）
    metrics_to_save = {k: v for k, v in metrics.items() if k != "individual"}
    
    # 转换numpy类型为Python原生类型
    metrics_to_save = _convert_to_python_type(metrics_to_save)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"指标已保存到: {filepath}")

