"""
网格世界基础类
用于多机器人路径规划环境
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from enum import IntEnum


class Action(IntEnum):
    """动作枚举"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class GridWorld:
    """
    网格世界环境
    
    支持：
    - 随机网格大小生成
    - 障碍物随机分布
    - 多机器人初始化
    - 碰撞检测
    """
    
    def __init__(
        self,
        grid_size: int = 10,
        obstacle_density: float = 0.3,
        num_robots: int = 8,
        fov_size: int = 3,
        seed: Optional[int] = None
    ):
        """
        初始化网格世界
        
        Args:
            grid_size: 网格大小 (grid_size x grid_size)
            obstacle_density: 障碍物密度 [0, 1]
            num_robots: 机器人数量
            fov_size: 视野范围大小
            seed: 随机种子
        """
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.num_robots = num_robots
        self.fov_size = fov_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化地图
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.obstacles = set()  # 障碍物位置集合
        
        # 机器人状态
        self.robot_positions = []  # 当前位置
        self.start_positions = []  # 起始位置
        self.goal_positions = []   # 目标位置
        self.robots_on_goal = [False] * num_robots  # 是否到达目标
        
        # 动作映射
        self.action_to_direction = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
            Action.STAY: (0, 0)
        }
        
    def reset(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        重置环境
        
        Returns:
            start_positions: 机器人起始位置列表
            goal_positions: 机器人目标位置列表
        """
        # 生成障碍物
        self._generate_obstacles()
        
        # 生成起始和目标位置
        self._generate_start_goal_positions()
        
        # 初始化机器人位置
        self.robot_positions = self.start_positions.copy()
        self.robots_on_goal = [False] * self.num_robots
        
        return self.start_positions, self.goal_positions
    
    def _generate_obstacles(self):
        """生成随机障碍物"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.obstacles = set()
        
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        
        for _ in range(num_obstacles):
            pos = self._get_random_free_position()
            if pos is not None:
                self.obstacles.add(pos)
                self.grid[pos[0], pos[1]] = -1  # -1表示障碍物
    
    def _generate_start_goal_positions(self):
        """
        生成机器人的起始和目标位置
        确保所有位置都在同一连通区域内
        """
        free_positions = self._get_all_free_positions()
        
        if len(free_positions) < 2 * self.num_robots:
            raise ValueError(f"没有足够的自由空间放置{self.num_robots}个机器人")
        
        # 随机选择起始和目标位置
        selected_positions = np.random.choice(
            len(free_positions), 
            size=2 * self.num_robots, 
            replace=False
        )
        
        self.start_positions = [
            free_positions[selected_positions[i]] 
            for i in range(self.num_robots)
        ]
        
        self.goal_positions = [
            free_positions[selected_positions[i + self.num_robots]] 
            for i in range(self.num_robots)
        ]
        
    def _get_all_free_positions(self) -> List[Tuple[int, int]]:
        """获取所有自由位置"""
        free_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.obstacles:
                    free_positions.append((i, j))
        return free_positions
    
    def _get_random_free_position(self) -> Optional[Tuple[int, int]]:
        """获取随机自由位置"""
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            i = np.random.randint(0, self.grid_size)
            j = np.random.randint(0, self.grid_size)
            
            if (i, j) not in self.obstacles:
                return (i, j)
            
            attempts += 1
        
        return None
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        检查位置是否有效
        
        Args:
            pos: (row, col) 位置
            
        Returns:
            是否为有效位置
        """
        i, j = pos
        
        # 检查边界
        if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size:
            return False
        
        # 检查障碍物
        if pos in self.obstacles:
            return False
        
        return True
    
    def get_next_position(
        self, 
        current_pos: Tuple[int, int], 
        action: Action
    ) -> Tuple[int, int]:
        """
        根据动作获取下一个位置
        
        Args:
            current_pos: 当前位置
            action: 动作
            
        Returns:
            下一个位置
        """
        direction = self.action_to_direction[action]
        next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
        
        # 如果下一个位置无效，保持不动
        if not self.is_valid_position(next_pos):
            return current_pos
        
        return next_pos
    
    def check_vertex_collision(
        self, 
        positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        检查顶点碰撞
        
        Args:
            positions: 所有机器人的位置
            
        Returns:
            发生碰撞的机器人索引对
        """
        collisions = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i] == positions[j]:
                    collisions.append((i, j))
        
        return collisions
    
    def check_edge_collision(
        self,
        prev_positions: List[Tuple[int, int]],
        next_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        检查边碰撞（交换碰撞）
        
        Args:
            prev_positions: 前一时刻的位置
            next_positions: 下一时刻的位置
            
        Returns:
            发生碰撞的机器人索引对
        """
        collisions = []
        
        for i in range(len(prev_positions)):
            for j in range(i + 1, len(prev_positions)):
                # 检查是否交换位置
                if (prev_positions[i] == next_positions[j] and 
                    prev_positions[j] == next_positions[i]):
                    collisions.append((i, j))
        
        return collisions
    
    def manhattan_distance(
        self, 
        pos1: Tuple[int, int], 
        pos2: Tuple[int, int]
    ) -> int:
        """
        计算曼哈顿距离
        
        Args:
            pos1: 位置1
            pos2: 位置2
            
        Returns:
            曼哈顿距离
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(
        self, 
        pos1: Tuple[int, int], 
        pos2: Tuple[int, int]
    ) -> float:
        """
        计算欧几里得距离
        
        Args:
            pos1: 位置1
            pos2: 位置2
            
        Returns:
            欧几里得距离
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_fov_observation(
        self, 
        robot_id: int
    ) -> np.ndarray:
        """
        获取机器人的局部视野观测
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            FOV观测 (4, fov_size, fov_size)
            - 通道0-3: 启发式地图（四个方向）
        """
        pos = self.robot_positions[robot_id]
        goal = self.goal_positions[robot_id]
        
        fov_radius = self.fov_size // 2
        obs = np.zeros((4, self.fov_size, self.fov_size), dtype=np.float32)
        
        # 生成启发式地图
        for i in range(self.fov_size):
            for j in range(self.fov_size):
                # 计算全局位置
                global_i = pos[0] + (i - fov_radius)
                global_j = pos[1] + (j - fov_radius)
                
                if not self._is_in_bounds(global_i, global_j):
                    continue
                
                current_dist = self.manhattan_distance((global_i, global_j), goal)
                
                # 四个方向的启发式
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
                for action_id, (di, dj) in enumerate(directions):
                    next_i = global_i + di
                    next_j = global_j + dj
                    
                    if self._is_in_bounds(next_i, next_j):
                        next_dist = self.manhattan_distance((next_i, next_j), goal)
                        if next_dist < current_dist:
                            obs[action_id, i, j] = 1.0
        
        return obs
    
    def _is_in_bounds(self, i: int, j: int) -> bool:
        """检查位置是否在边界内"""
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size
    
    def get_obstacle_map(self, robot_id: int) -> np.ndarray:
        """
        获取障碍物地图（FOV内）
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            障碍物地图 (fov_size, fov_size)
        """
        pos = self.robot_positions[robot_id]
        fov_radius = self.fov_size // 2
        
        obs_map = np.zeros((self.fov_size, self.fov_size), dtype=np.float32)
        
        for i in range(self.fov_size):
            for j in range(self.fov_size):
                global_i = pos[0] + (i - fov_radius)
                global_j = pos[1] + (j - fov_radius)
                
                if not self._is_in_bounds(global_i, global_j):
                    obs_map[i, j] = 1.0  # 边界视为障碍物
                elif (global_i, global_j) in self.obstacles:
                    obs_map[i, j] = 1.0
        
        return obs_map
    
    def get_other_robots_map(self, robot_id: int) -> np.ndarray:
        """
        获取其他机器人位置地图（FOV内）
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            其他机器人地图 (fov_size, fov_size)
        """
        pos = self.robot_positions[robot_id]
        fov_radius = self.fov_size // 2
        
        robot_map = np.zeros((self.fov_size, self.fov_size), dtype=np.float32)
        
        for other_id in range(self.num_robots):
            if other_id == robot_id:
                continue
            
            other_pos = self.robot_positions[other_id]
            relative_i = other_pos[0] - pos[0] + fov_radius
            relative_j = other_pos[1] - pos[1] + fov_radius
            
            if 0 <= relative_i < self.fov_size and 0 <= relative_j < self.fov_size:
                robot_map[relative_i, relative_j] = 1.0
        
        return robot_map
    
    def get_goal_map(self, robot_id: int) -> np.ndarray:
        """
        获取目标地图（FOV内）
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            目标地图 (fov_size, fov_size)
        """
        pos = self.robot_positions[robot_id]
        goal = self.goal_positions[robot_id]
        fov_radius = self.fov_size // 2
        
        goal_map = np.zeros((self.fov_size, self.fov_size), dtype=np.float32)
        
        relative_i = goal[0] - pos[0] + fov_radius
        relative_j = goal[1] - pos[1] + fov_radius
        
        if 0 <= relative_i < self.fov_size and 0 <= relative_j < self.fov_size:
            goal_map[relative_i, relative_j] = 1.0
        
        return goal_map
    
    def render(self):
        """简单的文本渲染"""
        grid_str = np.full((self.grid_size, self.grid_size), '.', dtype='<U2')
        
        # 障碍物
        for obs in self.obstacles:
            grid_str[obs[0], obs[1]] = '█'
        
        # 目标位置
        for i, goal in enumerate(self.goal_positions):
            grid_str[goal[0], goal[1]] = f'G{i}'[:1]
        
        # 机器人位置
        for i, pos in enumerate(self.robot_positions):
            grid_str[pos[0], pos[1]] = f'R{i}'[:1]
        
        # 打印
        for row in grid_str:
            print(' '.join(row))
        print()

