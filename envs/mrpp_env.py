"""
多机器人路径规划（MRPP）环境
符合Gym接口的环境实现
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from envs.grid_world import GridWorld, Action


class MRPPEnv(gym.Env):
    """
    多机器人路径规划环境
    
    特性：
    - 支持多机器人同时规划
    - 部分可观测（局部FOV）
    - 奖励设计：移动惩罚、碰撞惩罚、目标奖励等
    - 内在奖励机制（探索激励）
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MRPP环境
        
        Args:
            config: 配置字典，包含：
                - num_robots: 机器人数量
                - grid_size_range: 网格大小范围 [min, max]
                - obstacle_density_range: 障碍物密度范围
                - fov_size: 视野范围
                - max_episode_steps: 最大回合步数
                - rewards: 奖励配置
                - intrinsic_reward: 内在奖励配置
        """
        super().__init__()
        
        # 环境参数
        self.num_robots = config.get("num_robots", 8)
        self.grid_size_range = config.get("grid_size_range", [10, 40])
        self.obstacle_density_range = config.get("obstacle_density_range", [0.0, 0.5])
        self.obstacle_density_peak = config.get("obstacle_density_peak", 0.33)
        self.fov_size = config.get("fov_size", 3)
        self.max_episode_steps = config.get("max_episode_steps", 256)
        
        # 奖励配置
        self.rewards = config.get("rewards", {
            "move": -0.3,
            "stay_on_goal": 0.0,
            "stay_off_goal": -0.3,
            "collision": -2.0,
            "blocking": -1.0,
            "reach_goal": 0.0
        })
        
        # 内在奖励配置
        self.use_intrinsic_reward = config.get("use_intrinsic_reward", True)
        self.intrinsic_config = config.get("intrinsic_reward", {
            "tau": 2.0,
            "phi": 0.1,
            "beta": 0.01,
            "buffer_size": 10
        })
        
        # 当前grid_size和obstacle_density将在reset时随机采样
        self.current_grid_size = None
        self.current_obstacle_density = None
        self.grid_world = None
        
        # 定义动作空间和观测空间
        # 动作空间：离散动作（UP, DOWN, LEFT, RIGHT, STAY）
        self.action_space = spaces.Discrete(5)
        
        # 观测空间：
        # 1. 图像观测 (8, fov_size, fov_size)
        #    - 前4个通道：启发式地图
        #    - 第5个通道：障碍物地图
        #    - 第6个通道：其他机器人地图
        #    - 第7个通道：自己的目标地图
        #    - 第8个通道：其他目标地图（如果在FOV内）
        # 2. 向量观测 (7,)
        #    - dx, dy, d: 到目标的归一化距离
        #    - re_t-1: 外在奖励
        #    - ri_t-1: 内在奖励
        #    - dmin_t-1: 与episode buffer中位置的最小距离
        #    - a_t-1: 前一动作
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=1, 
                shape=(8, self.fov_size, self.fov_size), 
                dtype=np.float32
            ),
            "vector": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,),
                dtype=np.float32
            )
        })
        
        # 环境状态
        self.current_step = 0
        self.episode_return = 0.0
        self.collision_count = 0
        
        # 前一时刻的信息
        self.prev_actions = [Action.STAY] * self.num_robots
        self.prev_extrinsic_rewards = [0.0] * self.num_robots
        self.prev_intrinsic_rewards = [0.0] * self.num_robots
        self.prev_min_distances = [0.0] * self.num_robots
        
        # Episode buffer（用于内在奖励）
        self.episode_buffers = [
            deque(maxlen=self.intrinsic_config["buffer_size"])
            for _ in range(self.num_robots)
        ]
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 可选配置
            
        Returns:
            observations: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # 随机采样网格大小和障碍物密度
        if self.grid_size_range[0] == self.grid_size_range[1]:
            # 固定网格大小
            self.current_grid_size = self.grid_size_range[0]
        else:
            self.current_grid_size = np.random.randint(
                self.grid_size_range[0], 
                self.grid_size_range[1] + 1
            )
        
        # 使用三角分布采样障碍物密度
        if self.obstacle_density_range[0] == self.obstacle_density_range[1]:
            # 固定障碍物密度（避免triangular分布错误）
            self.current_obstacle_density = self.obstacle_density_range[0]
        else:
            self.current_obstacle_density = np.random.triangular(
                self.obstacle_density_range[0],
                self.obstacle_density_peak,
                self.obstacle_density_range[1]
            )
        
        # 创建网格世界
        self.grid_world = GridWorld(
            grid_size=self.current_grid_size,
            obstacle_density=self.current_obstacle_density,
            num_robots=self.num_robots,
            fov_size=self.fov_size
        )
        
        # 重置网格世界
        self.grid_world.reset()
        
        # 重置环境状态
        self.current_step = 0
        self.episode_return = 0.0
        self.collision_count = 0
        
        # 重置前一时刻信息
        self.prev_actions = [Action.STAY] * self.num_robots
        self.prev_extrinsic_rewards = [0.0] * self.num_robots
        self.prev_intrinsic_rewards = [0.0] * self.num_robots
        self.prev_min_distances = [0.0] * self.num_robots
        
        # 重置episode buffer
        for buffer in self.episode_buffers:
            buffer.clear()
            
        # 获取初始观测
        observations = self._get_observations()
        
        info = {
            "grid_size": self.current_grid_size,
            "obstacle_density": self.current_obstacle_density,
            "num_robots": self.num_robots
        }
        
        return observations, info
    
    def step(
        self, 
        actions: List[int]
    ) -> Tuple[Dict[str, np.ndarray], List[float], bool, bool, Dict]:
        """
        执行一步环境交互
        
        Args:
            actions: 所有机器人的动作列表
            
        Returns:
            observations: 下一时刻观测
            rewards: 奖励列表
            terminated: 是否达到终止条件
            truncated: 是否达到最大步数
            info: 额外信息
        """
        self.current_step += 1
        
        # 保存前一时刻的位置
        prev_positions = self.grid_world.robot_positions.copy()
        
        # 计算下一时刻的位置
        next_positions = []
        for robot_id, action in enumerate(actions):
            next_pos = self.grid_world.get_next_position(
                prev_positions[robot_id],
                Action(action)
            )
            next_positions.append(next_pos)
        
        # 检测碰撞
        vertex_collisions = self.grid_world.check_vertex_collision(next_positions)
        edge_collisions = self.grid_world.check_edge_collision(prev_positions, next_positions)
        
        collided_robots = set()
        for i, j in vertex_collisions + edge_collisions:
            collided_robots.add(i)
            collided_robots.add(j)
            self.collision_count += 1
        
        # 处理碰撞：碰撞的机器人保持不动
        for robot_id in collided_robots:
            next_positions[robot_id] = prev_positions[robot_id]
        
        # 更新机器人位置
        self.grid_world.robot_positions = next_positions
        
        # 计算奖励
        rewards, info_dict = self._compute_rewards(
            actions, 
            prev_positions, 
            next_positions, 
            collided_robots
        )
        
        # 更新前一时刻信息
        self.prev_actions = actions
        self.prev_extrinsic_rewards = [r["extrinsic"] for r in rewards]
        self.prev_intrinsic_rewards = [r["intrinsic"] for r in rewards]
        
        # 总奖励
        total_rewards = [r["total"] for r in rewards]
        self.episode_return += sum(total_rewards)
        
        # 获取新观测
        observations = self._get_observations()
        
        # 检查终止条件
        all_on_goal = all(self.grid_world.robots_on_goal)
        terminated = all_on_goal
        truncated = self.current_step >= self.max_episode_steps
        
        # 如果成功完成，添加成功奖励
        if terminated and "success_bonus" in self.rewards:
            success_bonus = self.rewards["success_bonus"]
            total_rewards = [r + success_bonus for r in total_rewards]
        
        # 构建info
        info = {
            "collision_count": self.collision_count,
            "robots_on_goal": sum(self.grid_world.robots_on_goal),
            "max_robots_reached": max(
                info_dict.get("robots_on_goal_history", [0])
            ) if "robots_on_goal_history" in info_dict else sum(self.grid_world.robots_on_goal),
            "episode_return": self.episode_return,
            "current_step": self.current_step
        }
        
        return observations, total_rewards, terminated, truncated, info
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        获取所有机器人的观测
        
        Returns:
            observations: 观测字典，包含image和vector
        """
        observations = {
            "image": [],
            "vector": []
        }
        
        for robot_id in range(self.num_robots):
            # 图像观测
            image_obs = self._get_image_observation(robot_id)
            observations["image"].append(image_obs)
            
            # 向量观测
            vector_obs = self._get_vector_observation(robot_id)
            observations["vector"].append(vector_obs)
        
        observations["image"] = np.array(observations["image"], dtype=np.float32)
        observations["vector"] = np.array(observations["vector"], dtype=np.float32)
        
        return observations
    
    def _get_image_observation(self, robot_id: int) -> np.ndarray:
        """
        获取单个机器人的图像观测
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            图像观测 (8, fov_size, fov_size)
        """
        # 启发式地图 (4通道)
        heuristic_maps = self.grid_world.get_fov_observation(robot_id)
        
        # 障碍物地图 (1通道)
        obstacle_map = self.grid_world.get_obstacle_map(robot_id)[np.newaxis, ...]
        
        # 其他机器人地图 (1通道)
        robot_map = self.grid_world.get_other_robots_map(robot_id)[np.newaxis, ...]
        
        # 目标地图 (1通道)
        goal_map = self.grid_world.get_goal_map(robot_id)[np.newaxis, ...]
        
        # 其他目标地图 (1通道) - 简化处理，暂时置零
        other_goals_map = np.zeros((1, self.fov_size, self.fov_size), dtype=np.float32)
        
        # 拼接所有通道
        image_obs = np.concatenate([
            heuristic_maps,
            obstacle_map,
            robot_map,
            goal_map,
            other_goals_map
        ], axis=0)
        
        return image_obs
    
    def _get_vector_observation(self, robot_id: int) -> np.ndarray:
        """
        获取单个机器人的向量观测
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            向量观测 (7,)
        """
        pos = self.grid_world.robot_positions[robot_id]
        goal = self.grid_world.goal_positions[robot_id]
        
        # 计算到目标的归一化距离
        dx = (goal[1] - pos[1]) / self.current_grid_size  # 列方向
        dy = (goal[0] - pos[0]) / self.current_grid_size  # 行方向
        d = self.grid_world.euclidean_distance(pos, goal) / (self.current_grid_size * np.sqrt(2))
        
        # 前一时刻的奖励
        re_prev = self.prev_extrinsic_rewards[robot_id]
        ri_prev = self.prev_intrinsic_rewards[robot_id]
        
        # 与episode buffer的最小距离
        dmin_prev = self.prev_min_distances[robot_id]
        
        # 前一动作
        a_prev = self.prev_actions[robot_id]
        
        vector_obs = np.array([dx, dy, d, re_prev, ri_prev, dmin_prev, a_prev], dtype=np.float32)
        
        return vector_obs
    
    def _compute_rewards(
        self,
        actions: List[int],
        prev_positions: List[Tuple[int, int]],
        next_positions: List[Tuple[int, int]],
        collided_robots: set
    ) -> Tuple[List[Dict[str, float]], Dict]:
        """
        计算所有机器人的奖励
        
        Args:
            actions: 动作列表
            prev_positions: 前一位置
            next_positions: 下一位置
            collided_robots: 发生碰撞的机器人集合
            
        Returns:
            rewards: 奖励字典列表，每个包含extrinsic、intrinsic、total
            info: 额外信息
        """
        rewards = []
        robots_on_goal_count = 0
        
        for robot_id in range(self.num_robots):
            extrinsic_reward = 0.0
            intrinsic_reward = 0.0
            
            action = Action(actions[robot_id])
            prev_pos = prev_positions[robot_id]
            next_pos = next_positions[robot_id]
            goal_pos = self.grid_world.goal_positions[robot_id]
            
            # 检查是否在目标上
            on_goal = (next_pos == goal_pos)
            self.grid_world.robots_on_goal[robot_id] = on_goal
            
            if on_goal:
                robots_on_goal_count += 1
            
            # 计算到目标的距离（用于基于距离的奖励）
            prev_dist = self.grid_world.manhattan_distance(prev_pos, goal_pos)
            next_dist = self.grid_world.manhattan_distance(next_pos, goal_pos)
            dist_improvement = prev_dist - next_dist  # 正数表示接近目标
            
            # 外在奖励
            if action == Action.STAY:
                if on_goal:
                    extrinsic_reward += self.rewards["stay_on_goal"]
                else:
                    extrinsic_reward += self.rewards["stay_off_goal"]
            else:
                extrinsic_reward += self.rewards["move"]
            
            # 基于距离的奖励（接近目标时给予奖励）
            # 每接近目标1步，给予奖励（可配置）
            if not on_goal and dist_improvement > 0:
                distance_reward = dist_improvement * self.rewards.get("distance_reward", 0.1)
                extrinsic_reward += distance_reward
            
            # 基于当前距离的奖励（proximity_reward）
            # 距离目标越近，奖励越高（即使不移动）
            if not on_goal and "proximity_reward" in self.rewards:
                max_dist = self.current_grid_size * 2  # 最大可能距离
                normalized_dist = next_dist / max_dist  # 归一化距离 [0, 1]
                proximity_reward = self.rewards["proximity_reward"] * (1.0 - normalized_dist)
                extrinsic_reward += proximity_reward
            
            # 碰撞惩罚
            if robot_id in collided_robots:
                extrinsic_reward += self.rewards["collision"]
            
            # 到达目标奖励（第一次到达时）
            was_on_goal = self.grid_world.robots_on_goal[robot_id]
            if on_goal and not was_on_goal:
                extrinsic_reward += self.rewards["reach_goal"]
            
            # 内在奖励（探索激励）
            if self.use_intrinsic_reward and not on_goal:
                intrinsic_reward = self._compute_intrinsic_reward(robot_id, next_pos)
            
            # 总奖励
            total_reward = extrinsic_reward + intrinsic_reward
            
            rewards.append({
                "extrinsic": extrinsic_reward,
                "intrinsic": intrinsic_reward,
                "total": total_reward
            })
        
        info = {
            "robots_on_goal_count": robots_on_goal_count
        }
        
        return rewards, info
    
    def _compute_intrinsic_reward(
        self, 
        robot_id: int, 
        current_pos: Tuple[int, int]
    ) -> float:
        """
        计算内在奖励（基于episode buffer）
        
        Args:
            robot_id: 机器人ID
            current_pos: 当前位置
            
        Returns:
            内在奖励
        """
        buffer = self.episode_buffers[robot_id]
        tau = self.intrinsic_config["tau"]
        phi = self.intrinsic_config["phi"]
        
        if len(buffer) == 0:
            # Buffer为空，添加当前位置
            buffer.append(current_pos)
            self.prev_min_distances[robot_id] = 0.0
            return phi
        
        # 计算与buffer中所有位置的最小曼哈顿距离
        min_distance = min(
            self.grid_world.manhattan_distance(current_pos, pos)
            for pos in buffer
        )
        
        self.prev_min_distances[robot_id] = min_distance
        
        # 如果距离大于等于阈值，给予内在奖励并添加到buffer
        if min_distance >= tau:
            if len(buffer) < buffer.maxlen:
                buffer.append(current_pos)
            return phi
        
        return 0.0
    
    def render(self, mode="human"):
        """渲染环境"""
        if self.grid_world is not None:
            self.grid_world.render()
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        获取机器人间的距离矩阵（曼哈顿距离）
        
        Returns:
            距离矩阵 (num_robots, num_robots)
        """
        positions = self.grid_world.robot_positions
        distance_matrix = np.zeros((self.num_robots, self.num_robots), dtype=np.float32)
        
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if i != j:
                    distance_matrix[i, j] = self.grid_world.manhattan_distance(
                        positions[i], positions[j]
                    )
        
        return distance_matrix

