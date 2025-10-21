import numpy as np
import torch
import heapq
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import random
from collections import deque
from tqdm import tqdm
import os
from enum import Enum
import time
import math
import noise

# 设置matplotlib后端和样式
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 设置随机种子确保结果可重现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class TerrainType(Enum):
    """地形类型枚举"""
    GRASS = 0  # 草地
    TREE = 1  # 树木
    ROCK = 2  # 岩石
    WATER = 3  # 水域
    ANIMAL = 4  # 动物（动态障碍物）


class AStarNode:
    """A*算法节点"""

    def __init__(self, x, y, g_cost=0, h_cost=0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # 从起点到当前节点的实际代价
        self.h_cost = h_cost  # 从当前节点到目标的启发式代价
        self.f_cost = g_cost + h_cost  # 总代价
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class LocalAStarPlanner:
    """基于激光雷达的局部A*规划器"""

    def __init__(self, resolution=0.2, max_range=10.0):
        self.resolution = resolution  # 栅格分辨率
        self.max_range = max_range  # 规划范围
        self.grid_size = int(max_range * 2 / resolution)  # 栅格大小

    def build_local_grid(self, laser_scan, laser_angles, car_pos, laser_range):
        """基于激光雷达构建局部栅格地图"""
        # 创建局部栅格地图
        local_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 车辆在栅格中的位置（中心）
        car_grid_x = self.grid_size // 2
        car_grid_y = self.grid_size // 2

        # 标记障碍物
        for angle, distance in zip(laser_angles, laser_scan):
            if distance < 0.98:  # 检测到障碍物
                actual_distance = distance * laser_range

                # 计算障碍物在栅格中的位置
                obs_world_x = car_pos[1] + actual_distance * np.cos(angle)
                obs_world_y = car_pos[0] + actual_distance * np.sin(angle)

                # 转换为栅格坐标
                obs_grid_x = int(car_grid_x + (obs_world_x - car_pos[1]) / self.resolution)
                obs_grid_y = int(car_grid_y + (obs_world_y - car_pos[0]) / self.resolution)

                # 标记障碍物及其安全区域
                safety_radius = int(2.0 / self.resolution)  # 2米安全距离
                for dx in range(-safety_radius, safety_radius + 1):
                    for dy in range(-safety_radius, safety_radius + 1):
                        if np.sqrt(dx ** 2 + dy ** 2) <= safety_radius:
                            grid_x = obs_grid_x + dx
                            grid_y = obs_grid_y + dy
                            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                                local_grid[grid_y, grid_x] = 1.0  # 标记为障碍物

        return local_grid, car_grid_x, car_grid_y

    def heuristic(self, node, goal):
        """启发式函数（曼哈顿距离）"""
        return abs(node.x - goal.x) + abs(node.y - goal.y)

    def get_neighbors(self, node, grid):
        """获取节点的邻居"""
        neighbors = []
        # 8方向移动
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dx, dy in directions:
            new_x = node.x + dx
            new_y = node.y + dy

            # 检查边界
            if 0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0]:
                # 检查是否为障碍物
                if grid[new_y, new_x] < 0.5:  # 可通行
                    # 计算移动代价
                    move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0  # 对角线移动代价更高
                    neighbors.append((new_x, new_y, move_cost))

        return neighbors

    def plan_global_direction(self, car_pos, target_pos, laser_scan, laser_angles, laser_range):
        """规划全局方向"""
        # 构建局部栅格地图
        local_grid, car_grid_x, car_grid_y = self.build_local_grid(
            laser_scan, laser_angles, car_pos, laser_range
        )

        # 计算目标在栅格中的位置
        target_local_x = target_pos[1] - car_pos[1]
        target_local_y = target_pos[0] - car_pos[0]

        target_grid_x = int(car_grid_x + target_local_x / self.resolution)
        target_grid_y = int(car_grid_y + target_local_y / self.resolution)

        # 如果目标在栅格范围外，选择边界上最接近目标的点
        if not (0 <= target_grid_x < self.grid_size and 0 <= target_grid_y < self.grid_size):
            # 将目标点投影到栅格边界
            target_grid_x = np.clip(target_grid_x, 0, self.grid_size - 1)
            target_grid_y = np.clip(target_grid_y, 0, self.grid_size - 1)

        # 如果目标点是障碍物，寻找最近的可通行点
        if local_grid[target_grid_y, target_grid_x] > 0.5:
            target_grid_x, target_grid_y = self._find_nearest_free_cell(
                local_grid, target_grid_x, target_grid_y
            )

        # 执行A*搜索
        path = self._astar_search(local_grid,
                                  (car_grid_x, car_grid_y),
                                  (target_grid_x, target_grid_y))

        if len(path) > 1:
            # 计算全局方向（选择路径上的第2-3个点作为目标方向）
            next_point_idx = min(3, len(path) - 1)
            next_grid_x, next_grid_y = path[next_point_idx]

            # 转换回世界坐标
            direction_x = (next_grid_x - car_grid_x) * self.resolution
            direction_y = (next_grid_y - car_grid_y) * self.resolution

            direction = np.array([direction_y, direction_x])  # [y, x]格式
            direction_magnitude = np.linalg.norm(direction)

            if direction_magnitude > 0:
                return direction / direction_magnitude, len(path)
            else:
                return self._fallback_direction(car_pos, target_pos), 0
        else:
            # 如果A*失败，返回直接朝向目标的方向
            return self._fallback_direction(car_pos, target_pos), 0

    def _find_nearest_free_cell(self, grid, target_x, target_y):
        """找到最近的可通行单元格"""
        min_distance = float('inf')
        best_x, best_y = target_x, target_y

        for radius in range(1, 10):  # 搜索半径
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x = target_x + dx
                    y = target_y + dy

                    if (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and
                            grid[y, x] < 0.5):  # 可通行
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            best_x, best_y = x, y

            if min_distance < float('inf'):
                break

        return best_x, best_y

    def _astar_search(self, grid, start, goal):
        """A*搜索算法"""
        start_node = AStarNode(start[0], start[1], 0,
                               self.heuristic(AStarNode(start[0], start[1]), AStarNode(goal[0], goal[1])))
        goal_node = AStarNode(goal[0], goal[1])

        open_list = [start_node]
        closed_set = set()

        while open_list:
            # 取出f_cost最小的节点
            current_node = heapq.heappop(open_list)

            # 如果到达目标
            if current_node == goal_node:
                path = []
                while current_node:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]  # 反转路径

            closed_set.add((current_node.x, current_node.y))

            # 检查邻居
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current_node, grid):
                if (neighbor_x, neighbor_y) in closed_set:
                    continue

                g_cost = current_node.g_cost + move_cost
                h_cost = self.heuristic(AStarNode(neighbor_x, neighbor_y), goal_node)
                neighbor_node = AStarNode(neighbor_x, neighbor_y, g_cost, h_cost, current_node)

                # 检查是否已在open_list中
                existing_node = None
                for node in open_list:
                    if node.x == neighbor_x and node.y == neighbor_y:
                        existing_node = node
                        break

                if existing_node is None:
                    heapq.heappush(open_list, neighbor_node)
                elif g_cost < existing_node.g_cost:
                    # 找到更好的路径
                    existing_node.g_cost = g_cost
                    existing_node.f_cost = g_cost + h_cost
                    existing_node.parent = current_node

        return []  # 未找到路径

    def _fallback_direction(self, car_pos, target_pos):
        """备用方向（直接朝向目标）"""
        direction = np.array([target_pos[0] - car_pos[0], target_pos[1] - car_pos[1]])
        direction_magnitude = np.linalg.norm(direction)

        if direction_magnitude > 0:
            return direction / direction_magnitude
        else:
            return np.array([0.0, 0.0])


class DWALocalPlanner:
    """动态窗口法局部规划器（A*引导版）"""

    def __init__(self, max_speed=2.0, max_accel=1.5, dt=0.2, predict_time=5.0, resolution=0.1):
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.dt = dt
        self.predict_time = predict_time
        self.resolution = resolution

        # 评估权重
        self.alpha = 0.8  # A*方向一致性权重
        self.beta = 0.2  # 速度权重
        self.gamma = 0.2  # 避障权重
        self.delta = 0.2  # 平滑度权重

    def plan_optimal_velocity(self, current_pos, current_vel, global_direction,
                              laser_scan, laser_angles, laser_range):
        """在A*全局方向指导下计算最优速度"""

        # 生成动态窗口内的速度采样
        velocity_samples = self._generate_velocity_window(current_vel)

        if not velocity_samples:
            return [0.0, 0.0], 0.0

        best_velocity = None
        best_score = float('-inf')

        # 构建障碍物信息
        obstacles = self._build_obstacle_list(laser_scan, laser_angles, laser_range, current_pos)

        for vel_sample in velocity_samples:
            # 预测轨迹
            trajectory = self._predict_trajectory(current_pos, vel_sample)

            # 检查安全性
            if self._is_trajectory_safe(trajectory, obstacles):
                # 评估轨迹质量
                score = self._evaluate_trajectory_with_astar(
                    trajectory, vel_sample, current_vel, global_direction
                )

                if score > best_score:
                    best_score = score
                    best_velocity = vel_sample

        if best_velocity is None:
            # 如果没有安全轨迹，尝试减速
            best_velocity = [current_vel[0] * 0.5, current_vel[1] * 0.5]
            best_score = 0.0

        return best_velocity, best_score

    def _generate_velocity_window(self, current_vel):
        """生成动态窗口内的速度采样"""
        velocity_samples = []

        current_speed_y, current_speed_x = current_vel[0], current_vel[1]
        max_delta_v = self.max_accel * self.dt

        # 动态窗口约束
        min_vy = max(-self.max_speed, current_speed_y - max_delta_v)
        max_vy = min(self.max_speed, current_speed_y + max_delta_v)
        min_vx = max(-self.max_speed, current_speed_x - max_delta_v)
        max_vx = min(self.max_speed, current_speed_x + max_delta_v)

        # 速度采样
        vy_samples = np.arange(min_vy, max_vy + self.resolution, self.resolution)
        vx_samples = np.arange(min_vx, max_vx + self.resolution, self.resolution)

        for vy in vy_samples:
            for vx in vx_samples:
                speed = np.sqrt(vy ** 2 + vx ** 2)
                if speed <= self.max_speed:
                    velocity_samples.append([vy, vx])

        return velocity_samples

    def _predict_trajectory(self, start_pos, velocity):
        """预测轨迹"""
        trajectory = []
        pos = np.array(start_pos)

        num_steps = int(self.predict_time / self.dt)
        for i in range(num_steps):
            pos = pos + np.array(velocity) * self.dt
            trajectory.append(pos.copy())

        return trajectory

    def _build_obstacle_list(self, laser_scan, laser_angles, laser_range, current_pos):
        """构建障碍物列表"""
        obstacles = []

        for angle, distance in zip(laser_angles, laser_scan):
            if distance < 0.98:  # 检测到障碍物
                actual_distance = distance * laser_range
                obs_x = current_pos[1] + actual_distance * np.cos(angle)
                obs_y = current_pos[0] + actual_distance * np.sin(angle)
                obstacles.append([obs_y, obs_x])

        return np.array(obstacles) if obstacles else np.array([]).reshape(0, 2)

    def _is_trajectory_safe(self, trajectory, obstacles):
        """检查轨迹安全性"""
        if len(obstacles) == 0:
            return True

        safety_radius = 1.5  # 安全半径

        for traj_point in trajectory:
            distances = cdist([traj_point], obstacles).flatten()
            if np.min(distances) < safety_radius:
                return False

        return True

    def _evaluate_trajectory_with_astar(self, trajectory, velocity, current_vel, global_direction):
        """基于A*方向评估轨迹质量"""
        if not trajectory:
            return float('-inf')

        # 1. A*方向一致性评估
        velocity_direction = np.array(velocity)
        velocity_magnitude = np.linalg.norm(velocity_direction)

        if velocity_magnitude > 0.1:
            velocity_direction = velocity_direction / velocity_magnitude
            # 计算与A*全局方向的一致性
            astar_consistency = np.dot(velocity_direction, global_direction)
            astar_score = max(0, astar_consistency)  # 只奖励正向一致性
        else:
            astar_score = 0.0

        # 2. 速度评估
        speed = velocity_magnitude
        optimal_speed = min(1.5, np.linalg.norm(global_direction) * 2.0)
        speed_score = 1.0 - abs(speed - optimal_speed) / self.max_speed
        speed_score = max(0, speed_score)

        # 3. 避障评估（已通过安全检查的轨迹给基础分）
        obstacle_score = 1.0

        # 4. 平滑度评估
        acceleration = np.array(velocity) - np.array(current_vel)
        accel_magnitude = np.linalg.norm(acceleration)
        smoothness_score = np.exp(-accel_magnitude * 0.5)

        # 综合评估
        total_score = (self.alpha * astar_score +
                       self.beta * speed_score +
                       self.gamma * obstacle_score +
                       self.delta * smoothness_score)

        return total_score


class HybridNavigationSystem:
    """A*+DWA混合导航系统"""

    def __init__(self):
        self.astar_planner = LocalAStarPlanner(resolution=0.2, max_range=10.0)
        self.dwa_planner = DWALocalPlanner(max_speed=2.0, max_accel=1.5, dt=0.2)

        # 更新控制
        self.astar_update_interval = 8  # A*更新间隔（较慢）
        self.dwa_update_interval = 2  # DWA更新间隔（较快）

        self.astar_counter = 0
        self.dwa_counter = 0

        # 缓存的规划结果
        self.global_direction = np.array([0.0, 0.0])
        self.optimal_velocity = [0.0, 0.0]
        self.dwa_score = 0.0
        self.path_quality = 0.0  # A*路径质量评估

    def plan_navigation(self, current_pos, current_vel, target_pos,
                        laser_scan, laser_angles, laser_range):
        """混合导航规划"""

        # 更新A*全局方向
        self.astar_counter += 1
        if self.astar_counter >= self.astar_update_interval:
            self.global_direction, path_length = self.astar_planner.plan_global_direction(
                current_pos, target_pos, laser_scan, laser_angles, laser_range
            )

            # 路径质量评估
            if path_length > 0:
                direct_distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
                self.path_quality = min(1.0, direct_distance / (path_length * 0.5))  # 路径效率
            else:
                self.path_quality = 0.5  # 中等质量

            self.astar_counter = 0

        # 更新DWA局部规划
        self.dwa_counter += 1
        if self.dwa_counter >= self.dwa_update_interval:
            self.optimal_velocity, self.dwa_score = self.dwa_planner.plan_optimal_velocity(
                current_pos, current_vel, self.global_direction,
                laser_scan, laser_angles, laser_range
            )
            self.dwa_counter = 0

        return self.optimal_velocity, self.dwa_score, self.global_direction, self.path_quality


class AStarDWAGuidanceReward:
    """A*+DWA混合奖励计算器"""

    def __init__(self):
        self.navigation_system = HybridNavigationSystem()

    def calculate_navigation_reward(self, current_pos, current_vel, target_pos, action,
                                    laser_scan, laser_angles, laser_range):
        """计算基于A*+DWA的导航奖"""

        # 获取混合导航规划结果
        optimal_velocity, dwa_score, global_direction, path_quality = \
            self.navigation_system.plan_navigation(
                current_pos, current_vel, target_pos, laser_scan, laser_angles, laser_range
            )

        if optimal_velocity is None:
            return -5.0  # 🔥 规划失败给负奖励

        # 计算动作与最优速度的一致性
        action_velocity = np.array(action) * 2.0  # 转换为速度（max_velocity=2.0）
        optimal_vel = np.array(optimal_velocity)

        # 1. 速度一致性奖励（改为可正可负，保持原权重）
        velocity_diff = np.linalg.norm(action_velocity - optimal_vel)
        # 🔥 关键修改：使用双曲正切函数，让差距大的动作获得负奖励
        velocity_consistency = np.tanh(2.0 - velocity_diff)  # 范围：[-1, 1]

        # 2. 方向一致性奖励（改为可正可负，保持原权重）
        action_speed = np.linalg.norm(action_velocity)
        optimal_speed = np.linalg.norm(optimal_vel)

        if action_speed > 0.1 and optimal_speed > 0.1:
            action_dir = action_velocity / action_speed
            optimal_dir = optimal_vel / optimal_speed
            direction_dot = np.dot(action_dir, optimal_dir)
            # 🔥 关键修改：不再加1除2，让反向动作获得负值
            direction_consistency = direction_dot  # 范围：[-1, 1]
        else:
            direction_consistency = 0.0 if action_speed < 0.1 and optimal_speed < 0.1 else -0.5

        # 3. A*路径质量奖励（保持原权重3.0）
        path_quality_bonus = path_quality * 3.0  # 🔥 保持原权重

        # 4. DWA评估分数奖励（保持原权重5.0）
        dwa_quality_bonus = dwa_score * 5.0  # 🔥 保持原权重

        # 🔥 关键修改：新的奖励计算公式（可正可负，保持原权重）
        navigation_reward = (
                velocity_consistency * 4.0 +  # 范围：[-4, 4] (原权重)
                direction_consistency * 8.0 +  # 范围：[-8, 8] (原权重)
                path_quality_bonus +  # 范围：[0, 5.0]
                dwa_quality_bonus  # 范围：[0, 5.0]
        )
        # 总理论范围：[-14, 24]
        return np.clip(navigation_reward, -20.0, 25.0)

    def reset(self):
        """重置导航系统状态"""
        self.navigation_system.astar_counter = 0
        self.navigation_system.dwa_counter = 0
        self.navigation_system.global_direction = np.array([0.0, 0.0])
        self.navigation_system.optimal_velocity = [0.0, 0.0]
        self.navigation_system.dwa_score = 0.0
        self.navigation_system.path_quality = 0.0

class ImprovedForestEnvironment:
    """改进版森林环境仿真类（位移控制版本）"""

    def __init__(self, width=100, height=50, num_trees=20, num_rocks=10,
                 num_water=2, num_animals=1, dt=0.2, fixed_seed=42,
                 stage=1, randomize_start=False, randomize_target=False):
        self.width = width
        self.height = height
        self.num_trees = num_trees
        self.num_rocks = num_rocks
        self.num_water = num_water
        self.num_animals = num_animals
        self.fixed_seed = fixed_seed
        self._original_seed = fixed_seed
        self._env_random = np.random.RandomState(fixed_seed)
        self.stage = stage
        self.randomize_start = randomize_start
        self.randomize_target = randomize_target

        # 环境网格（静态部分）
        self.static_grid = np.zeros((height, width), dtype=np.float32)
        self.static_terrain_type = np.zeros((height, width), dtype=np.int32)
        self.height_map = np.zeros((height, width), dtype=np.float32)

        # 动态网格（包含动物）
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.terrain_type = np.zeros((height, width), dtype=np.int32)

        # 初始化起点和终点
        self._initialize_start_target_positions()

        # 无人车状态 - 简化为位置和速度
        self.car_velocity = [0.0, 0.0]  # 线速度[vy, vx]
        self.max_velocity = 2.0  # 最大速度
        self.dt = dt

        # 动态障碍物（动物）
        self.animals = []
        self.initial_animals = []

        # 动作空间：[y速度, x速度]
        self.action_space = 2

        # 状态空间：相对位置(2) + 距离(1) + 速度(2) + 激光雷达(32) = 37
        self.observation_space = 37

        # 激光雷达参数
        self.laser_range = 10.0
        self.laser_angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)


        # 实时可视化相关
        self.visualization_enabled = False
        self.trajectory = []
        self.fig = None
        self.ax = None
        self.car_circle = None
        self.trajectory_line = None
        self.animal_circles = []

        # 可视化控制
        self.viz_update_counter = 0
        self.viz_update_interval = 3

        # 历史记录
        self.position_history = deque(maxlen=15)
        self.displacement_history = deque(maxlen=8)  # 记录位移历史用于平滑度计算

        # 性能优化系统
        self.step_count = 0

        # 障碍物检测
        self.terrain_causes_collision = {
            TerrainType.GRASS.value: False,  # 草地：不碰撞
            TerrainType.TREE.value: True,  # 树木：碰撞
            TerrainType.ROCK.value: True,  # 岩石：碰撞
            TerrainType.WATER.value: True,  # 水域：碰撞
            TerrainType.ANIMAL.value: True  # 动物：碰撞
        }
        # 激光雷达优化
        self.laser_cache = {}
        self.laser_step_size = 0.2

        # 动物系统优化
        self.animal_update_interval = 1
        # A * +DWA导航系统
        self.astar_dwa_guidance = AStarDWAGuidanceReward()
        # 进度追踪
        self.initial_distance_to_target = None
        self.previous_distance_to_target = None
        self.best_distance_to_target = None
        self.progress_stagnation_counter = 0
        # 奖励统计
        self.reward_components = {
            'navigation': 0,  # A*+DWA导航奖励
            'progress': 0,  # 进度奖励
            'trajectory_quality': 0,  # 🆕 轨迹质量
            'time_efficiency': 0  # 🆕 时间效率
        }

        # 生成固定地形
        self._generate_fixed_terrain()


    def _initialize_start_target_positions(self):
        """根据随机化设置初始化起点和终点位置"""
        if not self.randomize_start and not self.randomize_target:
            self._set_fixed_start_and_target()
        elif not self.randomize_start and self.randomize_target:
            self._set_fixed_start_random_target()
        elif self.randomize_start and not self.randomize_target:
            self._set_random_start_fixed_target()
        else:
            self._set_random_start_and_target()

    def _set_fixed_start_and_target(self):
        """设置固定的起点和终点"""
        self.car_pos = [self.height * 0.5 , 5.0]
        self.target_pos = [self.height * 0.5, self.width - 5.0]

    def _set_fixed_start_random_target(self):
        """设置固定起点，随机终点"""
        self.car_pos = [self.height * 0.5 , 5.0]
        self._generate_safe_random_target()

    def _set_random_start_fixed_target(self):
        """设置随机起点，固定终点"""
        self.target_pos = [self.height * 0.5, self.width - 5.0]
        self._generate_safe_random_start()

    def _set_random_start_and_target(self):
        """设置随机的起点和终点"""
        self._generate_safe_random_start()
        self._generate_safe_random_target()

    def _generate_safe_random_start(self):
        """生成安全的随机起点"""
        safety_radius = 5.0
        max_attempts = 100

        start_y_min = self.height * 0.5 - 5.0
        start_y_max = self.height * 0.5 + 5.0
        start_x_min = 5.0
        start_x_max = 15.0

        start_found = False
        start_attempts = 0

        temp_random = np.random.RandomState(int(time.time() * 1000) % 10000)

        while not start_found and start_attempts < max_attempts:
            candidate_start = [
                temp_random.uniform(start_y_min, start_y_max),
                temp_random.uniform(start_x_min, start_x_max)
            ]

            if self._is_position_safe(candidate_start[0], candidate_start[1], safety_radius):
                self.car_pos = candidate_start
                start_found = True

            start_attempts += 1

        if not start_found:
            print("Warning: Could not find safe start position, using default")
            self.car_pos = [self.height * 0.8, self.width * 0.5]

        np.random.seed(self.fixed_seed)
        random.seed(self.fixed_seed)

    def _generate_safe_random_target(self):
        """生成安全的随机终点"""
        safety_radius = 5.0
        max_attempts = 100

        target_y_min = self.height * 0.5 - 5.0
        target_y_max = self.height * 0.5 + 5.0
        target_x_min = self.width  - 15.0
        target_x_max = self.width  - 5.0

        min_distance = self.height * 0.3

        target_found = False
        target_attempts = 0

        current_time_seed = int(time.time() * 1000) % 10000
        random.seed(current_time_seed)
        np.random.seed(current_time_seed)

        while not target_found and target_attempts < max_attempts:
            candidate_target = [
                np.random.uniform(target_y_min, target_y_max),
                np.random.uniform(target_x_min, target_x_max)
            ]

            if self._is_position_safe(candidate_target[0], candidate_target[1], safety_radius):
                distance = np.sqrt((self.car_pos[0] - candidate_target[0]) ** 2 +
                                   (self.car_pos[1] - candidate_target[1]) ** 2)

                if distance >= min_distance:
                    self.target_pos = candidate_target
                    target_found = True

            target_attempts += 1

        if not target_found:
            print("Warning: Could not find safe target position, using default")
            self.target_pos = [self.height * 0.5, self.width - 10.0]

        np.random.seed(self.fixed_seed)
        random.seed(self.fixed_seed)

    def _is_position_safe(self, y, x, safety_radius=6.0):
        """检查位置是否安全可通行"""
        if (y < safety_radius or y >= self.height - safety_radius or
                x < safety_radius or x >= self.width - safety_radius):
            return False

        center_y, center_x = int(y), int(x)
        if (center_y < 0 or center_y >= self.height or
                center_x < 0 or center_x >= self.width):
            return False

        terrain_type = self.static_terrain_type[center_y, center_x]
        if terrain_type in [TerrainType.TREE.value, TerrainType.ROCK.value,
                            TerrainType.WATER.value]:
            return False

        if self.static_grid[center_y, center_x] > 0.2:
            return False

        # 检查周围安全区域
        immediate_radius = 2.0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                check_y = center_y + dy
                check_x = center_x + dx

                if (check_y < 0 or check_y >= self.height or
                        check_x < 0 or check_x >= self.width):
                    return False

                distance = np.sqrt(dy ** 2 + dx ** 2)
                if distance <= immediate_radius:
                    terrain = self.static_terrain_type[check_y, check_x]
                    if (terrain in [TerrainType.TREE.value, TerrainType.ROCK.value,
                                    TerrainType.WATER.value] or
                            self.static_grid[check_y, check_x] > 0.2):
                        return False

        return True

    def _clear_start_end_areas(self):
        """清理起点和终点区域"""
        clear_radius = 6

        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                distance = np.sqrt(dy ** 2 + dx ** 2)
                if distance <= clear_radius:
                    y = int(self.car_pos[0]) + dy
                    x = int(self.car_pos[1]) + dx
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.static_grid[y, x] = 0
                        self.static_terrain_type[y, x] = TerrainType.GRASS.value

        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                distance = np.sqrt(dy ** 2 + dx ** 2)
                if distance <= clear_radius:
                    y = int(self.target_pos[0]) + dy
                    x = int(self.target_pos[1]) + dx
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.static_grid[y, x] = 0
                        self.static_terrain_type[y, x] = TerrainType.GRASS.value

    def _generate_fixed_terrain(self):
        """生成固定的地形（使用固定随机种子）"""
        current_state = np.random.get_state()
        current_torch_state = torch.get_rng_state()
        current_random_state = random.getstate()

        np.random.seed(self.fixed_seed)
        torch.manual_seed(self.fixed_seed)
        random.seed(self.fixed_seed)

        try:
            self.static_grid.fill(0)
            self.static_terrain_type.fill(TerrainType.GRASS.value)

            self._generate_height_map()
            self._generate_individual_trees()
            self._generate_rocks()
            self._generate_water_bodies()
            self._initialize_animals()
            self._clear_start_end_areas()

        finally:
            np.random.set_state(current_state)
            torch.set_rng_state(current_torch_state)
            random.setstate(current_random_state)

    def _generate_height_map(self):
        """使用Perlin噪声生成平滑地形高度图"""
        shape = (self.height, self.width)
        scale = 50.0
        octaves = 3
        persistence = 0.5
        lacunarity = 2.0
        seed = np.random.randint(0, 100)
        world = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                world[i][j] = noise.pnoise2(
                    i / scale, j / scale,
                    octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity, repeatx=shape[0],
                    repeaty=shape[1], base=seed
                )
        self.height_map = (world - np.min(world)) / (np.max(world) - np.min(world))
        self.height_map = gaussian_filter(self.height_map, sigma=1.2)

    def _generate_individual_trees(self):
        """生成单独的树木"""
        trees_generated = 0
        attempts = 0
        max_attempts = self.num_trees * 10

        while trees_generated < self.num_trees and attempts < max_attempts:
            # 树木边缘限制 ≥8
            tree_y = np.random.uniform(8, self.height - 8)
            tree_x = np.random.uniform(8, self.width - 8)
            tree_radius = np.random.uniform(1.0, 1.5)

            # 使用5米最小间距
            if self._is_valid_obstacle_position(tree_y, tree_x, tree_radius, min_distance=3):
                self._draw_circular_obstacle(tree_y, tree_x, tree_radius,
                                             TerrainType.TREE.value, intensity=0.9)
                trees_generated += 1

            attempts += 1

    def _generate_rocks(self):
        """生成岩石障碍物"""
        rocks_generated = 0
        attempts = 0
        max_attempts = self.num_rocks * 20

        while rocks_generated < self.num_rocks and attempts < max_attempts:
            # 岩石边缘限制 ≥5
            y = np.random.uniform(5, self.height - 5)
            x = np.random.uniform(5, self.width - 5)
            rock_radius = np.random.uniform(1.0, 2.0)

            # 使用5米最小间距，启用地形检查避免重叠
            if self._is_valid_obstacle_position(y, x, rock_radius, min_distance=3, check_terrain=True):
                self._draw_irregular_obstacle(y, x, rock_radius, TerrainType.ROCK.value)
                rocks_generated += 1

            attempts += 1

    def _generate_water_bodies(self):
        """生成水域"""
        water_generated = 0
        attempts = 0
        max_attempts = self.num_water * 20

        while water_generated < self.num_water and attempts < max_attempts:
            # 水域边缘限制 ≥10
            y = np.random.uniform(10, self.height - 10)
            x = np.random.uniform(10, self.width - 10)
            water_radius = np.random.uniform(2.5, 5)

            # 使用5米最小间距，启用地形检查避免重叠
            if self._is_valid_obstacle_position(y, x, water_radius, min_distance=3, check_terrain=True):
                self._draw_water_body(y, x, water_radius)
                water_generated += 1

            attempts += 1

    def _initialize_animals(self):
        """初始化动物（动态障碍物）"""
        self.animals = []
        self.initial_animals = []

        for animal_idx in range(self.num_animals):
            attempts = 0
            max_attempts = 100
            animal_placed = False

            while attempts < max_attempts and not animal_placed:
                y = np.random.uniform(15, self.height - 15)
                x = np.random.uniform(15, self.width - 15)

                if self._is_valid_obstacle_position(y, x, 1.0, min_distance=3, check_terrain=True):
                    animal_data = {
                        'id': animal_idx,
                        'pos': [y, x],
                        'velocity': [np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)],
                        'radius': 0.5,
                        'behavior': np.random.choice(['wander', 'graze', 'flee']),
                        'stuck_counter': 0,
                        'last_pos': [y, x]
                    }
                    self.animals.append(animal_data.copy())
                    self.initial_animals.append(animal_data.copy())
                    animal_placed = True

                attempts += 1

    def _draw_circular_obstacle(self, center_y, center_x, radius, terrain_type, intensity=1.0):
        """绘制圆形障碍物"""
        y_min = max(0, int(center_y - radius - 1))
        y_max = min(self.height, int(center_y + radius + 2))
        x_min = max(0, int(center_x - radius - 1))
        x_max = min(self.width, int(center_x + radius + 2))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                if distance <= radius:
                    if terrain_type == TerrainType.TREE.value:
                        if distance <= radius * 0.7:
                            value = intensity
                        else:
                            value = intensity * 0.8
                    else:
                        value = intensity * (1.0 - distance / radius * 0.3)

                    self.static_grid[y, x] = max(self.static_grid[y, x], value)
                    self.static_terrain_type[y, x] = terrain_type

    def _draw_irregular_obstacle(self, center_y, center_x, radius, terrain_type):
        """绘制不规则形状的障碍物（如岩石）"""
        for _ in range(3):
            offset_y = np.random.uniform(-radius / 2, radius / 2)
            offset_x = np.random.uniform(-radius / 2, radius / 2)
            ellipse_a = np.random.uniform(radius * 0.7, radius * 1.3)
            ellipse_b = np.random.uniform(radius * 0.7, radius * 1.3)
            rotation = np.random.uniform(0, np.pi)
            self._draw_ellipse(center_y + offset_y, center_x + offset_x,
                               ellipse_a, ellipse_b, rotation, terrain_type)

    def _draw_ellipse(self, center_y, center_x, a, b, rotation, terrain_type):
        """绘制椭圆"""
        y_min = max(0, int(center_y - max(a, b) - 1))
        y_max = min(self.height, int(center_y + max(a, b) + 2))
        x_min = max(0, int(center_x - max(a, b) - 1))
        x_max = min(self.width, int(center_x + max(a, b) + 2))

        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dy = y - center_y
                dx = x - center_x
                y_rot = dy * cos_r - dx * sin_r
                x_rot = dy * sin_r + dx * cos_r

                if (y_rot / a) ** 2 + (x_rot / b) ** 2 <= 1:
                    self.static_grid[y, x] = 1.0
                    self.static_terrain_type[y, x] = terrain_type

    def _draw_water_body(self, center_y, center_x, radius):
        """绘制水域"""
        y_min = max(0, int(center_y - radius - 2))
        y_max = min(self.height, int(center_y + radius + 3))
        x_min = max(0, int(center_x - radius - 2))
        x_max = min(self.width, int(center_x + radius + 3))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                noise_val = np.random.uniform(-0.3, 0.3)
                if distance <= radius + noise_val:
                    self.static_grid[y, x] = 1.0
                    self.static_terrain_type[y, x] = TerrainType.WATER.value

    def _is_valid_obstacle_position(self, y, x, radius, min_distance=5, check_terrain=False):
        """检查障碍物位置是否有效"""
        # 检查与起点和终点的距离
        start_dist = np.sqrt((y - self.car_pos[0]) ** 2 + (x - self.car_pos[1]) ** 2)
        end_dist = np.sqrt((y - self.target_pos[0]) ** 2 + (x - self.target_pos[1]) ** 2)

        # 确保障碍物不会干扰起点和终点
        if start_dist < radius + min_distance or end_dist < radius + min_distance:
            return False

        # 检查是否与现有障碍物重叠
        if check_terrain:
            check_radius = max(2, int(radius + min_distance))

            for dy in range(-check_radius, check_radius + 1):
                for dx in range(-check_radius, check_radius + 1):
                    check_y = int(y) + dy
                    check_x = int(x) + dx

                    if (check_y < 0 or check_y >= self.height or
                            check_x < 0 or check_x >= self.width):
                        continue

                    distance = np.sqrt(dy ** 2 + dx ** 2)
                    # 检查最小间距范围内是否有障碍物
                    if distance <= radius + min_distance:
                        terrain = self.static_terrain_type[check_y, check_x]

                        # 如果该位置已有障碍物，则不能放置新障碍物
                        if terrain in [TerrainType.TREE.value, TerrainType.ROCK.value,
                                       TerrainType.WATER.value]:
                            return False

                        # 检查障碍物强度
                        if self.static_grid[check_y, check_x] > 0.2:
                            return False

        return True

    def _update_dynamic_grid(self):
        """更新动态网格（静态环境 + 动物）"""
        self.grid = self.static_grid.copy()
        self.terrain_type = self.static_terrain_type.copy()

        for animal in self.animals:
            animal_y, animal_x = animal['pos']
            y_min = max(0, int(animal_y - animal['radius']))
            y_max = min(self.height, int(animal_y + animal['radius']) + 1)
            x_min = max(0, int(animal_x - animal['radius']))
            x_max = min(self.width, int(animal_x + animal['radius']) + 1)

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    distance = np.sqrt((y - animal_y) ** 2 + (x - animal_x) ** 2)
                    if distance <= animal['radius']:
                        self.grid[y, x] = max(self.grid[y, x], 0.8)
                        self.terrain_type[y, x] = TerrainType.ANIMAL.value

    def reset(self):
        """重置环境"""
        self._initialize_start_target_positions()

        # 重置无人车状态
        self.car_velocity = [0.0, 0.0]
        self.trajectory = []

        # 重置动物位置
        self.animals = []
        for initial_animal in self.initial_animals:
            self.animals.append(initial_animal.copy())

        self._update_dynamic_grid()

        self.steps = 0
        self.viz_update_counter = 0
        self.position_history.clear()
        self.displacement_history.clear()
        # 重置A*+DWA系统
        self.astar_dwa_guidance.reset()
        self.initial_distance_to_target = None
        self.previous_distance_to_target = None
        self.best_distance_to_target = None
        self.progress_stagnation_counter = 0

        # 重置奖励组件
        self.reward_components = {
            'navigation': 0,
            'progress': 0
        }
        for key in self.reward_components:
            self.reward_components[key] = 0

        if self.visualization_enabled and hasattr(self, 'fig') and self.fig is not None:
            if self.stage >= 4:
                self._redraw_environment()

        return self._get_state()

    def _redraw_environment(self):
        """重新绘制环境（用于起终点位置改变时）"""
        if not self.visualization_enabled or self.fig is None:
            return

        try:
            self.ax.clear()
            self._draw_environment_smooth()
            self.car_circle = patches.Circle((self.car_pos[1], self.car_pos[0]), 0.3,
                                             color='blue', alpha=0.8, zorder=10)
            self.ax.add_patch(self.car_circle)
            self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=5)
            self.animal_circles = []
            for animal in self.animals:
                animal_circle = patches.Circle((animal['pos'][1], animal['pos'][0]),
                                               animal['radius'], color='orange', alpha=0.7, zorder=8)
                self.ax.add_patch(animal_circle)
                self.animal_circles.append(animal_circle)

            self.ax.set_xlim(-1, self.width)
            self.ax.set_ylim(-1, self.height)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)

            stage_info = f"Stage {self.stage}"
            if self.stage == 4:
                stage_info += " (Fixed start, Random target)"
            elif self.stage == 5:
                stage_info += " (Random start and target)"

            self.ax.set_title(f'SAC Forest Navigation - {stage_info}', fontsize=14)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')

            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"重绘环境时出错: {e}")

    def update_animals(self):
        """修改后的动物更新方法 - 完全随机移动"""
        if not self.animals:
            return
        animal_maxspeed = 0.6
        for animal in self.animals:
            # 给动物的速度添加随机扰动
            noise_scale = 0.05
            animal['velocity'][0] += np.random.uniform(-noise_scale, noise_scale)
            animal['velocity'][1] += np.random.uniform(-noise_scale, noise_scale)

            # 限制速度大小
            speed = np.linalg.norm(animal['velocity'])
            if speed > animal_maxspeed:
                animal['velocity'][0] = animal['velocity'][0] / speed * animal_maxspeed
                animal['velocity'][1] = animal['velocity'][1] / speed * animal_maxspeed

            self._simple_move_animal(animal)

    def _simple_move_animal(self, animal):
        """简化的动物移动"""
        old_pos = animal['pos'].copy()

        new_pos = [
            animal['pos'][0] + animal['velocity'][0] * self.dt,
            animal['pos'][1] + animal['velocity'][1] * self.dt
        ]

        width_margin = 15  # 动物移动左右边界
        high_margin = 15  # 动物移动上下边界
        if (new_pos[0] < high_margin or new_pos[0] >= self.height - high_margin or
                new_pos[1] < width_margin or new_pos[1] >= self.width - width_margin):
            if new_pos[0] < high_margin or new_pos[0] >= self.height - high_margin:
                animal['velocity'][0] *= -0.8
            if new_pos[1] < width_margin or new_pos[1] >= self.width - width_margin:
                animal['velocity'][1] *= -0.8
            new_pos = old_pos

        elif self._is_simple_obstacle_at(new_pos):
            animal['velocity'][0] *= -0.7
            animal['velocity'][1] *= -0.7
            animal['velocity'][0] += np.random.uniform(-0.1, 0.1)
            animal['velocity'][1] += np.random.uniform(-0.1, 0.1)
            new_pos = old_pos

        animal['pos'] = new_pos

    def _is_simple_obstacle_at(self, pos):
        """快速障碍物检测"""
        y, x = int(pos[0]), int(pos[1])
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.static_grid[y, x] > 0.5
        return True

    def _get_laser_scan(self):
        """优化的激光雷达扫描"""
        car_y, car_x = self.car_pos

        cache_key = (int(car_y * 4), int(car_x * 4))
        if (cache_key in self.laser_cache and
                self.step_count - self.laser_cache[cache_key]['step'] < 2):
            return self.laser_cache[cache_key]['scan']

        scan_results = []

        animal_positions = None
        animal_radii = None
        if self.animals:
            animal_positions = np.array([animal['pos'] for animal in self.animals])
            animal_radii = np.array([animal['radius'] for animal in self.animals])

        for angle in self.laser_angles:
            direction = [np.cos(angle), np.sin(angle)]
            distance = self._cast_ray_optimized(
                car_y, car_x, direction,
                animal_positions, animal_radii
            )

            # 添加轻微噪声
            noise_val = np.random.normal(0, 0.005)
            distance = max(0, distance + noise_val)
            normalized_distance = min(distance / self.laser_range, 1.0)
            scan_results.append(normalized_distance)

        self.laser_cache[cache_key] = {
            'scan': scan_results,
            'step': self.step_count
        }

        # 清理缓存
        if len(self.laser_cache) > 50:
            oldest_key = min(self.laser_cache.keys(),
                             key=lambda k: self.laser_cache[k]['step'])
            del self.laser_cache[oldest_key]

        return scan_results

    def _cast_ray_optimized(self, start_y, start_x, direction, animal_positions, animal_radii):
        """优化的激光雷达射线投射"""
        distance = 0.0

        while distance < self.laser_range:
            detect_y = start_y + distance * direction[0]
            detect_x = start_x + distance * direction[1]

            # 边界检查
            if (detect_y < 0 or detect_y >= self.height or
                    detect_x < 0 or detect_x >= self.width):
                break

            grid_y, grid_x = int(detect_y), int(detect_x)

            # 基于地形类型判断碰撞
            terrain_type = self.terrain_type[grid_y, grid_x]
            if self.terrain_causes_collision.get(terrain_type, True):
                break

            # 检查动物碰撞
            if animal_positions is not None:
                detect_pos = np.array([[detect_y, detect_x]])
                distances_to_animals = cdist(detect_pos, animal_positions).flatten()
                if np.any(distances_to_animals < animal_radii):
                    break

            distance += self.laser_step_size

        return distance

    def _calculate_obstacle_distances(self):
        """基于激光雷达数据计算最近障碍物距离"""
        laser_scan = self._get_laser_scan()
        min_normalized_distance = min(laser_scan)

        if min_normalized_distance >= 0.99:
            return 8.0

        obstacle_distance = min_normalized_distance * self.laser_range
        return obstacle_distance

    def _get_state(self):
        """获取状态信息（37维）"""
        car_y, car_x = self.car_pos

        # 相对位置信息 (2维)
        relative_y = (self.target_pos[0] - car_y) / self.height
        relative_x = (self.target_pos[1] - car_x) / self.width

        # 距离信息 (1维)
        distance = np.sqrt((self.target_pos[0] - car_y) ** 2 + (self.target_pos[1] - car_x) ** 2)
        distance_norm = distance / np.sqrt(self.height ** 2 + self.width ** 2)
        # 速度信息 (2维)
        velocity_y = self.car_velocity[0]
        velocity_x = self.car_velocity[1]

        # 激光雷达扫描 (32维)
        laser_scan = self._get_laser_scan()

        # 组合所有状态信息 (总计37维: 2 + 1 + 2 + 32)
        state = [
                    # 基础状态 (5维)
                    relative_y, relative_x, distance_norm, velocity_y,velocity_x
                ] + laser_scan # 32维激光雷达

        state_array = np.array(state, dtype=np.float32)

        if len(state_array) != self.observation_space:
            print(f"Warning: State dimension mismatch! Expected {self.observation_space}, got {len(state_array)}")

        return state_array

    def _check_collision_circle(self, y, x, radius=0.3, num_points=16):
        """优化的碰撞检测"""
        center_y, center_x = int(y), int(x)

        # 边界检查
        if (center_y < 0 or center_y >= self.height or
                center_x < 0 or center_x >= self.width):
            return True

        # 检查中心点地形类型
        center_terrain = self.terrain_type[center_y, center_x]
        if self.terrain_causes_collision.get(center_terrain, True):
            return True

        # 使用圆形碰撞检测
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            check_y = y + radius * np.cos(angle)
            check_x = x + radius * np.sin(angle)

            grid_y, grid_x = int(check_y), int(check_x)

            if (grid_y < 0 or grid_y >= self.height or
                    grid_x < 0 or grid_x >= self.width):
                return True

            terrain_type = self.terrain_type[grid_y, grid_x]
            if self.terrain_causes_collision.get(terrain_type, True):
                return True

        return False

    def _calculate_astar_dwa_navigation_reward(self, action):
        """计算A*+DWA导航奖励"""
        laser_scan = self._get_laser_scan()

        navigation_reward = self.astar_dwa_guidance.calculate_navigation_reward(
            self.car_pos, self.car_velocity, self.target_pos,
            action, laser_scan, self.laser_angles, self.laser_range
        )

        return navigation_reward

    def step(self, action):
        """执行动作（改进的防刷分版本）"""
        self.step_count += 1

        # 更新动物
        self.update_animals()
        self._update_dynamic_grid()

        # 记录旧状态
        old_velocity = np.array(self.car_velocity.copy())
        old_pos = np.array(self.car_pos.copy())
        old_distance_to_target = np.linalg.norm(old_pos - np.array(self.target_pos))

        # 解析动作：[y位移, x位移]
        self.car_velocity[0] = np.clip(action[0], -1, 1) * self.max_velocity
        self.car_velocity[1] = np.clip(action[1], -1, 1) * self.max_velocity

        # 计算当前加速度和位移
        current_velocity = np.array(self.car_velocity)
        current_acceleration = (current_velocity - old_velocity) / self.dt
        displacement_y = self.car_velocity[0] * self.dt
        displacement_x = self.car_velocity[1] * self.dt

        # 计算新位置
        new_y = self.car_pos[0] + displacement_y
        new_x = self.car_pos[1] + displacement_x

        # 初始化奖励和标志
        reward = 0
        done = False
        collision = False
        info = {}

        # 检查边界和碰撞
        if (new_y < 1 or new_y >= self.height - 1 or
                new_x < 1 or new_x >= self.width - 1):
            collision = True
            reward = -100  # 降低边界惩罚
            done = True
            info['out_of_bounds'] = True
        else:
            collision = self._check_collision_circle(new_y, new_x, radius=0.3)
            if collision:
                reward = -100  # 降低碰撞惩罚
                done = True
                info['collision'] = True
            else:
                # 更新位置
                self.car_pos = [new_y, new_x]
                self.trajectory.append(self.car_pos.copy())
                self.position_history.append(self.car_pos.copy())
                self.displacement_history.append([displacement_y, displacement_x])

        # 🔥 改进的奖励计算（防刷分版本）
        if not done:
            # 1. A*+DWA导航奖励（保持创新点，但限制数值范围）
            navigation_reward = self._calculate_astar_dwa_navigation_reward(action)
            navigation_reward = np.clip(navigation_reward * 0.6, -15, 15)  # 大幅降低权重和范围
            self.reward_components['navigation'] = navigation_reward

            # 2. 改进的进度奖励（强化目标导向）
            progress_reward = self._calculate_improved_progress_reward()
            self.reward_components['progress'] = progress_reward

            # 3. 🆕 轨迹质量奖励（防止刷分的关键）
            trajectory_quality_reward = self._calculate_trajectory_quality_reward(action, current_acceleration)
            self.reward_components['trajectory_quality'] = trajectory_quality_reward

            # 4. 🆕 时间效率奖励（鼓励快速到达）
            time_efficiency_reward = self._calculate_time_efficiency_reward()
            self.reward_components['time_efficiency'] = time_efficiency_reward

            # 🔥 新的权重分配（防刷分优化）
            reward = (
                    navigation_reward * 1.5 +  # A*+DWA导航（保持创新）
                    progress_reward * 1.0 +  # 进度主导
                    trajectory_quality_reward * 1.5 +  # 轨迹质量约束
                    time_efficiency_reward * 1.0  # 时间效率
            )

            # 严格限制奖励范围
            reward = np.clip(reward, -40, 40)

            # 检查到达目标
            current_distance = np.linalg.norm(np.array(self.car_pos) - np.array(self.target_pos))
            if current_distance < 1.0:
                # 🔥 成功奖励基于效率（防止刷分）
                efficiency_ratio = (1200 - self.steps) / 1200
                directness_ratio = self._calculate_path_directness()

                success_base_reward = 100  # 大幅降低基础成功奖励
                efficiency_bonus = success_base_reward * efficiency_ratio
                directness_bonus = success_base_reward * directness_ratio

                total_success_reward = success_base_reward + efficiency_bonus + directness_bonus
                reward += total_success_reward
                done = True
                info['success'] = True

        # 步数更新和限制
        self.steps += 1
        if self.steps >= 1200:
            done = True
            info['timeout'] = True
            # 超时惩罚基于当前距离
            timeout_penalty = -50-50 * (current_distance / self.initial_distance_to_target)
            reward += timeout_penalty

        # 获取新状态和更新可视化
        next_state = self._get_state()
        if self.visualization_enabled:
            self.viz_update_counter += 1
            if self.viz_update_counter >= self.viz_update_interval:
                self._update_visualization()
                self.viz_update_counter = 0

        return next_state, reward, done, info

    def _calculate_improved_progress_reward(self):
        """改进的进度奖励（强化目标导向，防刷分）"""
        current_distance = np.linalg.norm(np.array(self.car_pos) - np.array(self.target_pos))

        if self.initial_distance_to_target is None:
            self.initial_distance_to_target = current_distance
            self.previous_distance_to_target = current_distance
            return 0.0

        # 计算进度
        total_progress = (self.initial_distance_to_target - current_distance) / self.initial_distance_to_target
        step_progress = self.previous_distance_to_target - current_distance
        self.previous_distance_to_target = current_distance

        # 🔥 严格的进度奖励
        base_progress_reward = total_progress * 0.0  # 降低基础奖励
        step_progress_reward = step_progress * 20.0  # 保持步进奖励

        # 🔥 更严厉的停滞和后退惩罚
        if step_progress < -0.01:  # 后退
            penalty_multiplier = 4.0 if step_progress < -0.05 else 3.0
            step_progress_reward *= penalty_multiplier
        elif -0.01 <= step_progress <= 0.01:  # 停滞
            # 🆕 基于距离的停滞惩罚
            if current_distance < 2.0:
                stagnation_penalty = -15.0  # 目标附近停滞
            elif current_distance < 5.0:
                stagnation_penalty = -10.0  # 中距离停滞
            else:
                stagnation_penalty = -5.0  # 远距离停滞
            step_progress_reward += stagnation_penalty

        total_reward = base_progress_reward + step_progress_reward
        return np.clip(total_reward, -25.0, 25.0)

    def _calculate_trajectory_quality_reward(self, action, acceleration):
        """🆕 轨迹质量奖励（防刷分的关键机制）"""
        quality_reward = 0.0

        # 1. 动作平滑性约束（防止抖动刷分）
        action_magnitude = np.linalg.norm(action)
        acceleration_magnitude = np.linalg.norm(acceleration)

        # 惩罚过度加速和急转弯
        if acceleration_magnitude > 2.0:
            quality_reward -= acceleration_magnitude * 2.0

        # 惩罚过度微小动作（可能的刷分行为）
        if action_magnitude < 0.1:
            quality_reward -= 3.0

        # 2. 🆕 轨迹直线性评估（防止绕行刷分）
        if len(self.displacement_history) >= 5:
            recent_displacements = list(self.displacement_history)[-5:]
            displacement_consistency = self._calculate_displacement_consistency(recent_displacements)
            quality_reward += displacement_consistency * 2.0

        # 3. 🆕 防止原地转圈
        if len(self.position_history) >= 8:
            recent_positions = list(self.position_history)[-8:]
            circle_penalty = self._detect_circular_motion(recent_positions)
            quality_reward -= circle_penalty

        # 4. 🆕 目标导向性约束
        target_direction = np.array(self.target_pos) - np.array(self.car_pos)
        if np.linalg.norm(target_direction) > 0:
            target_direction_norm = target_direction / np.linalg.norm(target_direction)
            action_direction_norm = action / max(np.linalg.norm(action), 1e-6)

            directional_alignment = np.dot(target_direction_norm, action_direction_norm)

            # 奖励朝向目标的动作，惩罚背离目标的动作
            if directional_alignment > 0.5:
                quality_reward += directional_alignment * 3.0
            elif directional_alignment < -0.3:
                quality_reward -= abs(directional_alignment) * 4.0

        return np.clip(quality_reward, -15.0, 10.0)

    def _calculate_displacement_consistency(self, recent_displacements):
        """计算位移一致性（检测是否在直线前进）"""
        if len(recent_displacements) < 3:
            return 0.0

        displacements = np.array(recent_displacements)

        # 计算方向变化
        directions = []
        for disp in displacements:
            if np.linalg.norm(disp) > 0.01:
                directions.append(disp / np.linalg.norm(disp))

        if len(directions) < 2:
            return 0.0

        # 计算方向一致性
        consistency_sum = 0.0
        for i in range(1, len(directions)):
            dot_product = np.dot(directions[i - 1], directions[i])
            consistency_sum += max(0, dot_product)  # 只奖励一致的方向

        return consistency_sum / (len(directions) - 1)

    def _detect_circular_motion(self, recent_positions):
        """检测圆周运动（防刷分）"""
        if len(recent_positions) < 6:
            return 0.0

        positions = np.array(recent_positions)

        # 计算位置变化
        position_changes = []
        for i in range(1, len(positions)):
            change = np.linalg.norm(positions[i] - positions[i - 1])
            position_changes.append(change)

        # 如果移动距离很小但一直在动，可能是原地转圈
        avg_change = np.mean(position_changes)
        total_displacement = np.linalg.norm(positions[-1] - positions[0])

        if avg_change > 0.1 and total_displacement < 2.0:
            # 在动但总位移很小，可能是转圈
            return 8.0

        return 0.0

    def _calculate_time_efficiency_reward(self):
        """🆕 时间效率奖励（鼓励快速导航）"""
        current_distance = np.linalg.norm(np.array(self.car_pos) - np.array(self.target_pos))

        # 基于当前距离和已用时间的效率评估
        expected_time = current_distance / 1.5  # 假设理想速度
        time_efficiency = max(0, (expected_time - self.steps * self.dt) / expected_time)

        # 时间压力（随着步数增加，奖励减少）
        time_pressure = max(0, (1200 - self.steps) / 1200)

        efficiency_reward = time_efficiency * time_pressure * 3.0

        # 🆕 步数惩罚（防止拖延刷分）
        if self.steps > 600:
            step_penalty = -(self.steps - 600) * 0.05
            efficiency_reward += step_penalty

        return np.clip(efficiency_reward, -30.0, 5.0)

    def _calculate_path_directness(self):
        """计算路径直线性（用于成功奖励）"""
        if len(self.trajectory) < 2:
            return 0.5

        # 计算实际路径长度
        actual_path_length = 0.0
        for i in range(1, len(self.trajectory)):
            segment_length = np.linalg.norm(
                np.array(self.trajectory[i]) - np.array(self.trajectory[i - 1])
            )
            actual_path_length += segment_length

        # 计算直线距离
        start_pos = np.array(self.trajectory[0])
        end_pos = np.array(self.trajectory[-1])
        direct_distance = np.linalg.norm(end_pos - start_pos)

        if actual_path_length < 1e-6:
            return 0.1

        # 直线性比率
        directness = direct_distance / actual_path_length
        return np.clip(directness, 0.1, 1.0)

    # 可视化方法
    def enable_visualization(self):
        """启用实时可视化"""
        self.visualization_enabled = True
        self._setup_visualization()

    def disable_visualization(self):
        """禁用实时可视化"""
        self.visualization_enabled = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _setup_visualization(self):
        """设置可视化界面"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self._draw_environment_smooth()

        self.car_circle = patches.Circle((self.car_pos[1], self.car_pos[0]), 0.3,
                                         color='blue', alpha=0.8, zorder=10)
        self.ax.add_patch(self.car_circle)

        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=5)

        self.animal_circles = []
        for i, animal in enumerate(self.animals):
            animal_circle = patches.Circle((animal['pos'][1], animal['pos'][0]),
                                           animal['radius'], color='orange', alpha=0.7, zorder=8)
            self.ax.add_patch(animal_circle)
            self.animal_circles.append(animal_circle)

        self.ax.set_xlim(-1, self.width)
        self.ax.set_ylim(-1, self.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        stage_info = f"Stage {self.stage}"
        if self.stage == 4:
            stage_info += " (Fixed start, Random target)"
        elif self.stage == 5:
            stage_info += " (Random start and target)"

        self.ax.set_title(f'SAC Forest Navigation - {stage_info}', fontsize=14)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)

    def _draw_environment_smooth(self):
        """绘制平滑的环境"""
        self.ax.clear()
        self._draw_height_contours()
        self._draw_smooth_terrain_features()
        self._draw_start_target_points()

    def _draw_height_contours(self):
        """绘制高度等高线作为背景"""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        contour_levels = np.linspace(0, 1, 10)
        cs = self.ax.contourf(X, Y, self.height_map, levels=contour_levels,
                              cmap='terrain', alpha=0.3, zorder=0)
        self.ax.contour(X, Y, self.height_map, levels=contour_levels,
                        colors='gray', alpha=0.2, linewidths=0.5, zorder=1)

    def _draw_smooth_terrain_features(self):
        """绘制平滑的地形特征"""
        terrain_masks = {}
        for terrain_type in [TerrainType.TREE, TerrainType.ROCK, TerrainType.WATER]:
            mask = (self.static_terrain_type == terrain_type.value).astype(float)
            smooth_mask = gaussian_filter(mask, sigma=1.0)
            terrain_masks[terrain_type] = smooth_mask

        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # 树木
        if np.any(terrain_masks[TerrainType.TREE] > 0.05):
            tree_levels = [0.05, 0.3, 0.6, 1.0]
            tree_colors = ['lightgreen', 'forestgreen', 'darkgreen']
            self.ax.contourf(X, Y, terrain_masks[TerrainType.TREE],
                             levels=tree_levels, colors=tree_colors,
                             alpha=0.8, zorder=2)

        # 岩石
        if np.any(terrain_masks[TerrainType.ROCK] > 0.1):
            self.ax.contourf(X, Y, terrain_masks[TerrainType.ROCK],
                             levels=[0.1, 0.5, 1.0], colors=['lightgray', 'gray'],
                             alpha=0.8, zorder=2)

        # 水域
        if np.any(terrain_masks[TerrainType.WATER] > 0.1):
            self.ax.contourf(X, Y, terrain_masks[TerrainType.WATER],
                             levels=[0.1, 0.5, 1.0], colors=['burlywood', 'saddlebrown'],
                             alpha=0.6, zorder=2)

    def _draw_start_target_points(self):
        """绘制起点和终点"""
        start_circle1 = patches.Circle((self.car_pos[1], self.car_pos[0]), 2.5,
                                       color='green', alpha=0.2, zorder=3)
        start_circle2 = patches.Circle((self.car_pos[1], self.car_pos[0]), 1.8,
                                       color='green', alpha=0.4, zorder=3)
        start_circle3 = patches.Circle((self.car_pos[1], self.car_pos[0]), 1.0,
                                       color='green', alpha=0.8, zorder=3)

        self.ax.add_patch(start_circle1)
        self.ax.add_patch(start_circle2)
        self.ax.add_patch(start_circle3)
        self.ax.text(self.car_pos[1], self.car_pos[0], 'START', ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=4)

        target_circle1 = patches.Circle((self.target_pos[1], self.target_pos[0]), 3.0,
                                        color='red', alpha=0.2, zorder=3)
        target_circle2 = patches.Circle((self.target_pos[1], self.target_pos[0]), 2.2,
                                        color='red', alpha=0.4, zorder=3)
        target_circle3 = patches.Circle((self.target_pos[1], self.target_pos[0]), 1.5,
                                        color='red', alpha=0.8, zorder=3)

        self.ax.add_patch(target_circle1)
        self.ax.add_patch(target_circle2)
        self.ax.add_patch(target_circle3)
        self.ax.text(self.target_pos[1], self.target_pos[0], 'TARGET', ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=4)

    def _update_visualization(self):
        """更新可视化"""
        if not self.visualization_enabled or self.fig is None:
            return

        try:
            if self.car_circle is not None:
                self.car_circle.center = (self.car_pos[1], self.car_pos[0])

            if len(self.trajectory) > 1 and self.trajectory_line is not None:
                trajectory_array = np.array(self.trajectory)
                self.trajectory_line.set_data(trajectory_array[:, 1], trajectory_array[:, 0])

            # 更新动物位置
            for i, animal in enumerate(self.animals):
                if i < len(self.animal_circles):
                    self.animal_circles[i].center = (animal['pos'][1], animal['pos'][0])

            distance_to_target = np.sqrt((self.car_pos[0] - self.target_pos[0]) ** 2 +
                                         (self.car_pos[1] - self.target_pos[1]) ** 2)
            speed = np.sqrt(self.car_velocity[0] ** 2 + self.car_velocity[1] ** 2)

            stage_info = f"Stage {self.stage}"
            if self.stage == 4:
                stage_info += " (Fixed-Random)"
            elif self.stage == 5:
                stage_info += " (Random-Random)"

            self.ax.set_title(
                f'SAC - {stage_info} - Step: {self.steps}, Distance: {distance_to_target:.1f}, '
                f'Speed: {speed:.2f}, Animals: {len(self.animals)}', fontsize=12)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"Visualization update error: {e}")


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, state_dim, action_dim, max_size=300000, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, next_state, reward, done):
        """添加经验，使用最大优先级"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """根据优先级采样"""
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            torch.FloatTensor(self.state[indices]),
            torch.FloatTensor(self.action[indices]),
            torch.FloatTensor(self.next_state[indices]),
            torch.FloatTensor(self.reward[indices]),
            torch.FloatTensor(self.done[indices]),
            torch.FloatTensor(weights.reshape(-1, 1)),
            indices
        )

    def update_priorities(self, indices, td_errors):
        """更新经验的优先级"""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class ImprovedActor(nn.Module):
    """改进的SAC Actor网络（位移控制版本）"""

    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(ImprovedActor, self).__init__()

        # 状态特征分解
        self.basic_features = 5  # 基础状态特征
        self.laser_features = 32  # 激光雷达

        # 基础特征编码器
        self.basic_encoder = nn.Sequential(
            nn.Linear(self.basic_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 激光雷达CNN编码器
        self.laser_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 主干网络
        total_features = 64 + 32  # 98
        self.fc1 = nn.Linear(total_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)

        # 输出层
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        self.max_action = max_action
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        # 特征分解
        basic_state = state[:, :self.basic_features]
        laser_state = state[:, self.basic_features:self.basic_features + self.laser_features]

        # 基础特征编码
        basic_encoded = self.basic_encoder(basic_state)

        # 激光雷达CNN编码
        laser_input = laser_state.unsqueeze(1)  # 添加通道维度 [batch, 1, 32]
        laser_encoded = self.laser_cnn(laser_input)

        # 特征融合
        all_features = torch.cat([basic_encoded, laser_encoded], dim=1)

        # 主干网络
        x = F.relu(self.fc1(all_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.max_action
        return action, log_prob, mean


class ImprovedCritic(nn.Module):
    """改进的SAC Critic网络（位移控制版本）"""

    def __init__(self, state_dim, action_dim):
        super(ImprovedCritic, self).__init__()

        # 状态特征分解
        self.basic_features = 5
        self.laser_features = 32

        # 状态特征提取器
        self.state_basic_encoder = nn.Sequential(
            nn.Linear(self.basic_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 激光雷达CNN编码器
        self.state_laser_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        state_feature_dim = 64 + 32  # 98
        total_input_dim = state_feature_dim + 32  # 128

        # Q1 网络
        self.q1_fc1 = nn.Linear(total_input_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 128)
        self.q1_fc4 = nn.Linear(128, 64)
        self.q1_fc5 = nn.Linear(64, 1)

        # Q2 网络
        self.q2_fc1 = nn.Linear(total_input_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 128)
        self.q2_fc4 = nn.Linear(128, 64)
        self.q2_fc5 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)

    def encode_state(self, state):
        """编码状态特征"""
        basic_state = state[:, :self.basic_features]
        laser_state = state[:, self.basic_features:self.basic_features + self.laser_features]

        basic_encoded = self.state_basic_encoder(basic_state)

        laser_input = laser_state.unsqueeze(1)
        laser_encoded = self.state_laser_cnn(laser_input)

        state_features = torch.cat([basic_encoded, laser_encoded], dim=1)
        return state_features

    def forward(self, state, action):
        state_features = self.encode_state(state)
        action_features = self.action_encoder(action)
        sa_features = torch.cat([state_features, action_features], dim=1)

        # Q1 网络
        q1 = F.relu(self.q1_fc1(sa_features))
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.dropout(q1)
        q1 = F.relu(self.q1_fc3(q1))
        q1 = F.relu(self.q1_fc4(q1))
        q1 = self.q1_fc5(q1)

        # Q2 网络
        q2 = F.relu(self.q2_fc1(sa_features))
        q2 = self.dropout(q2)
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.dropout(q2)
        q2 = F.relu(self.q2_fc3(q2))
        q2 = F.relu(self.q2_fc4(q2))
        q2 = self.q2_fc5(q2)

        return q1, q2

    def Q1(self, state, action):
        """只返回Q1的值"""
        state_features = self.encode_state(state)
        action_features = self.action_encoder(action)
        sa_features = torch.cat([state_features, action_features], dim=1)

        q1 = F.relu(self.q1_fc1(sa_features))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = F.relu(self.q1_fc4(q1))
        q1 = self.q1_fc5(q1)

        return q1


class ImprovedSAC:
    def __init__(self, state_dim, action_dim, max_action=1.0, device='cuda', lr=3e-4):
        self.device = device
        self.max_action = max_action

        # 确保state_dim为37
        assert state_dim == 37, f"Expected state_dim=37, got {state_dim}"

        # 使用改进的网络
        self.actor = ImprovedActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)

        self.critic = ImprovedCritic(state_dim, action_dim).to(device)
        self.critic_target = ImprovedCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr * 0.8, weight_decay=1e-5)

        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.total_it = 0

        # 损失记录
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []
        self.q_values = []
        self.alpha_values = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005):
        """训练函数 - 支持优先经验回放"""
        self.total_it += 1

        # 采样批次（包含权重和索引）
        state, action, next_state, reward, done, weights, indices = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        weights = weights.to(self.device)

        # 目标Q值计算
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + (1 - done) * discount * min_qf_next_target

        # Critic损失
        qf1, qf2 = self.critic(state, action)
        self.q_values.append(qf1.mean().item())

        # 计算TD误差用于更新优先级
        td_errors_1 = torch.abs(qf1 - next_q_value)
        td_errors_2 = torch.abs(qf2 - next_q_value)
        td_errors = torch.min(td_errors_1, td_errors_2).detach().cpu().numpy().flatten()

        # 使用重要性采样权重
        qf1_loss = (weights * F.mse_loss(qf1, next_q_value, reduction='none')).mean()
        qf2_loss = (weights * F.mse_loss(qf2, next_q_value, reduction='none')).mean()
        qf_loss = qf1_loss + qf2_loss

        self.critic_losses.append(qf_loss.item())

        # 更新Critic
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor损失
        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_losses.append(policy_loss.item())

        # 更新Actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 温度参数更新
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_losses.append(alpha_loss.item())
        self.alpha_values.append(self.alpha.item())

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 更新经验回放的优先级
        replay_buffer.update_priorities(indices, td_errors)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.critic_target.load_state_dict(self.critic.state_dict())


class FiveStageCurriculumTraining:
    """五阶段课程学习训练策略"""

    def __init__(self, base_env_config):
        self.base_config = base_env_config
        self.current_stage = 0
        self.stage_episodes = 0
        self.success_history = deque(maxlen=100)

        # 定义五个阶段的课程（删除泥地相关配置）
        self.curriculum_stages = [
            # 阶段1：简单环境，固定起终点
            {
                'num_trees': 20, 'num_rocks': 10, 'num_water': 0, 'num_animals': 5,
                'randomize_start': False, 'randomize_target': False,
                'success_threshold': 0.7, 'min_episodes': 60, 'max_episodes': 200,
                'description': 'Simple environment, fixed start and target, learning basic navigation'
            },
            # 阶段2：中等复杂度，固定起终点
            {
                'num_trees': 30, 'num_rocks': 10, 'num_water': 1, 'num_animals': 5,
                'randomize_start': False, 'randomize_target': True,
                'success_threshold': 0.8, 'min_episodes': 80, 'max_episodes': 300,
                'description': 'Medium complexity, fixed start and target, more obstacles and animals'
            },
            # 阶段3：高复杂度，固定起终点
            {
                'num_trees': 35, 'num_rocks': 10, 'num_water': 2 , 'num_animals': 10,
                  'randomize_start': False, 'randomize_target': True,
                'success_threshold': 0.8, 'min_episodes': 100, 'max_episodes': 300,
                'description': 'High complexity, fixed start and target, precise navigation in complex environment'
            },
            # 阶段4：最高复杂度，固定起点，随机终点
            {
                'num_trees': 35, 'num_rocks': 10, 'num_water': 2, 'num_animals': 15,
                'randomize_start': True, 'randomize_target': True,
                'success_threshold': 0.8, 'min_episodes': 300, 'max_episodes': 500,
                'description': 'Highest complexity, fixed start, random target, learning to adapt to different targets'
            },
            # 阶段5：最高难度，随机起点和终点
            {
                'num_trees': 35, 'num_rocks': 10, 'num_water': 2, 'num_animals': 15,
                'randomize_start': True, 'randomize_target': True,
                'success_threshold': 0.8, 'min_episodes': 100, 'max_episodes': 300,
                'description': 'Highest difficulty, random start and target, enhancing model generalization'
            }
        ]

    def get_current_env_config(self):
        """获取当前阶段的环境配置"""
        if self.current_stage >= len(self.curriculum_stages):
            return self.curriculum_stages[-1]
        return self.curriculum_stages[self.current_stage]

    def update(self, success):
        """更新课程学习状态"""
        self.success_history.append(success)
        self.stage_episodes += 1

        current_config = self.get_current_env_config()

        # 检查是否可以进入下一阶段
        if (len(self.success_history) >= 50 and
                self.stage_episodes >= current_config['min_episodes']):

            recent_success_rate = sum(list(self.success_history)[-50:]) / 50

            if (recent_success_rate >= current_config['success_threshold'] or
                    self.stage_episodes >= current_config['max_episodes']):

                if self.current_stage < len(self.curriculum_stages) - 1:
                    self.current_stage += 1
                    self.stage_episodes = 0
                    self.success_history.clear()
                    print(f"\n🎓 Upgraded to curriculum stage {self.current_stage + 1}/5")
                    print(f"📝 {self.curriculum_stages[self.current_stage]['description']}")
                    return True

        return False

    def get_stage_info(self):
        """获取当前阶段信息"""
        current_config = self.get_current_env_config()
        return {
            'stage': self.current_stage + 1,
            'total_stages': len(self.curriculum_stages),
            'stage_episodes': self.stage_episodes,
            'recent_success_rate': sum(self.success_history) / len(self.success_history) if self.success_history else 0,
            'description': current_config['description'],
            'randomize_start': current_config.get('randomize_start', False),
            'randomize_target': current_config.get('randomize_target', False)
        }


def improved_train_sac_five_stage(env_class, agent, num_episodes=1000, max_timesteps=1200,
                                  start_timesteps=10000, batch_size=256, save_freq=100,
                                  visualize=False):
    """五阶段课程学习SAC训练函数（位移控制版本）"""

    replay_buffer = PrioritizedReplayBuffer(37, 2, max_size=300000, alpha=0.6, beta=0.4)

    # 基础环境配置
    base_config = {
        'width': 100, 'height': 50,
        'dt': 0.2, 'fixed_seed': 42
    }

    # 定义环境参数的键
    env_param_keys = {'width', 'height', 'num_trees', 'num_rocks', 'num_water',
                      'num_animals', 'dt', 'fixed_seed',
                      'randomize_start', 'randomize_target', 'stage'}

    # 初始化五阶段课程学习
    curriculum = FiveStageCurriculumTraining(base_config)
    curriculum_config = curriculum.get_current_env_config()

    # 构建环境配置
    env_config = {k: v for k, v in curriculum_config.items() if k in env_param_keys}
    current_config = {**base_config, **env_config, 'stage': curriculum.current_stage + 1}

    env = env_class(**current_config)

    # 训练记录
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    episode_success = []
    stage_transitions = []

    # 自适应参数
    exploration_noise_scale = 1.0
    min_noise_scale = 0.1
    noise_decay = 0.9995

    state = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    os.makedirs('models', exist_ok=True)

    if visualize:
        env.enable_visualization()

    # 显示初始阶段信息
    stage_info = curriculum.get_stage_info()
    print(f"\n🚀 Starting five-stage curriculum learning training (Optimized)")
    print(f"📚 Current stage: {stage_info['stage']}/5")
    print(f"📝 {stage_info['description']}")
    print(f"🎯 Target success rate: {curriculum_config['success_threshold']:.1%}")

    pbar = tqdm(total=num_episodes, desc="Five-stage curriculum SAC training")

    try:
        for t in range(int(num_episodes * max_timesteps)):
            episode_timesteps += 1

            # 优化的动作选择
            if t < start_timesteps:
                # 改进的初始探索策略
                if t % 100 < 50:
                    # 目标导向的结构化探索
                    target_direction = np.array([
                        env.target_pos[0] - env.car_pos[0],
                        env.target_pos[1] - env.car_pos[1]
                    ])
                    direction_norm = np.linalg.norm(target_direction)
                    if direction_norm > 0:
                        target_direction = target_direction / direction_norm
                        action = target_direction * np.random.uniform(0.3, 0.8) + np.random.normal(0, 0.2, 2)
                        action = np.clip(action, -1, 1)
                    else:
                        action = np.random.uniform(-0.8, 0.8, 2)
                else:
                    action = np.random.uniform(-0.8, 0.8, 2)
            else:
                action = agent.select_action(state)
                # 改进的探索噪声策略
                if curriculum.current_stage <= 2:
                    noise_std = exploration_noise_scale
                else:
                    obstacle_distance = env._calculate_obstacle_distances()
                    if obstacle_distance < 2.0:
                        noise_std = exploration_noise_scale * 0.3
                    else:
                        noise_std = exploration_noise_scale

                noise = np.random.normal(0, noise_std, action.shape)
                action = np.clip(action + noise, -1, 1)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储经验
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            # 训练智能体
            if t >= start_timesteps and replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

            if done:
                # 记录结果
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_timesteps)
                success = info.get('success', False)
                episode_success.append(success)

                # 更新课程学习
                stage_changed = curriculum.update(success)
                if stage_changed:
                    # 记录阶段转换
                    stage_transitions.append({
                        'episode': episode_num,
                        'old_stage': curriculum.current_stage,
                        'new_stage': curriculum.current_stage + 1
                    })

                    # 重新创建环境
                    env.disable_visualization() if visualize else None
                    curriculum_config = curriculum.get_current_env_config()
                    env_config = {k: v for k, v in curriculum_config.items() if k in env_param_keys}
                    current_config = {**base_config, **env_config, 'stage': curriculum.current_stage + 1}
                    env = env_class(**current_config)
                    if visualize:
                        env.enable_visualization()

                # 计算成功率
                if len(episode_success) >= 100:
                    recent_success = sum(episode_success[-100:]) / 100
                else:
                    recent_success = sum(episode_success) / len(episode_success)
                success_rate.append(recent_success)

                # 优化的噪声衰减策略
                if curriculum.current_stage >= 3:
                    exploration_noise_scale = max(min_noise_scale * 1.5, exploration_noise_scale * 0.9985)
                else:
                    exploration_noise_scale = max(min_noise_scale, exploration_noise_scale * noise_decay)

                # 基于成功率的自适应噪声调整
                if len(episode_success) >= 50:
                    recent_success_rate_for_noise = sum(episode_success[-50:]) / 50
                    if recent_success_rate_for_noise > 0.8:
                        exploration_noise_scale *= 0.95
                    elif recent_success_rate_for_noise < 0.3:
                        exploration_noise_scale = min(exploration_noise_scale * 1.05, 0.4)

                # 更新进度条
                pbar.update(1)
                stage_info = curriculum.get_stage_info()
                info_dict = {
                    'Episode': episode_num,
                    'Reward': f"{episode_reward:.2f}",
                    'Length': episode_timesteps,
                    'Success Rate': f"{recent_success:.2%}",
                    'Alpha': f"{agent.alpha.item():.3f}",
                    'Noise': f"{exploration_noise_scale:.3f}"
                }

                info_dict['Stage'] = f"{stage_info['stage']}/5"
                if stage_info['stage'] == 4:
                    info_dict['Mode'] = "Fixed-Random"
                elif stage_info['stage'] == 5:
                    info_dict['Mode'] = "Random-Random"

                pbar.set_postfix(info_dict)

                # 重置环境
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                # 保存模型
                if episode_num % save_freq == 0:
                    agent.save(f'models/optimized_sac_displacement_ep{episode_num}.pth')

                if episode_num >= num_episodes:
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        pbar.close()
        if visualize:
            env.disable_visualization()

    # 保存最终模型
    agent.save('models/optimized_sac_displacement_final.pth')

    # 打印课程学习总结
    print(f"\n🎓 Five-stage curriculum learning training completed!")
    print(f"📊 Training statistics:")
    print(f"   Total training episodes: {episode_num}")
    print(f"   Final success rate: {recent_success:.2%}")
    print(f"   Final stage: {stage_info['stage']}/5")

    if stage_transitions:
        print(f"🔄 Stage transition record:")
        for transition in stage_transitions:
            print(f"   Episode {transition['episode']}: Stage {transition['old_stage']} → {transition['new_stage']}")

    return (episode_rewards, episode_lengths, success_rate,
            agent.critic_losses, agent.actor_losses, agent.alpha_losses,
            agent.q_values, agent.alpha_values, stage_transitions)


def test_sac_five_stage(env_class, agent, num_episodes=10, visualize=True, test_stage=5):
    """测试五阶段训练的SAC智能体（位移控制版本）- 只统计成功episode的数据"""

    # 根据测试阶段设置环境配置
    stage_configs = {
        1: {
            'width': 100, 'height': 50, 'num_trees': 30, 'num_rocks': 10,
            'num_water': 2, 'num_animals': 3, 'dt': 0.2,
            'fixed_seed': 42, 'stage': 1,
            'randomize_start': False, 'randomize_target': False
        },
        2: {
            'width': 100, 'height': 50, 'num_trees': 40, 'num_rocks': 10,
            'num_water': 2, 'num_animals': 5, 'dt': 0.2,
            'fixed_seed': 42, 'stage': 2,
            'randomize_start': True, 'randomize_target': False
        },
        3: {
            'width': 100, 'height': 50, 'num_trees': 40, 'num_rocks': 10,
            'num_water': 2, 'num_animals': 5, 'dt': 0.2,
            'fixed_seed': 42, 'stage': 3,
            'randomize_start': False, 'randomize_target': False
        },
        4: {
            'width': 100, 'height': 50, 'num_trees': 40, 'num_rocks': 20,
            'num_water': 4, 'num_animals': 20, 'dt': 0.2,
            'fixed_seed': 42, 'stage': 4,
            'randomize_start': False, 'randomize_target': False
        },
        5: {
            'width': 100, 'height': 50, 'num_trees': 35, 'num_rocks': 10,
            'num_water': 2, 'num_animals': 15, 'dt': 0.2,
            'fixed_seed': 42, 'stage': 5,
            'randomize_start': True, 'randomize_target': True
        }
    }

    env_config = stage_configs.get(test_stage, stage_configs[5])
    env = env_class(**env_config)

    # 🔥 修改1: 分别记录所有episode和成功episode的数据
    all_episode_rewards = []
    all_episode_lengths = []
    all_successes = []

    # 🔥 新增: 只记录成功episode的数据
    success_episode_rewards = []
    success_episode_lengths = []

    if visualize:
        env.enable_visualization()

    stage_desc = {
        1: "Stage 1: Simple environment, fixed start and target",
        2: "Stage 2: Medium complexity, fixed start and target",
        3: "Stage 3: High complexity, fixed start and target",
        4: "Stage 4: Highest difficulty, fixed start, random target",
        5: "Stage 5: Highest difficulty, random start and target"
    }

    print(f"\n🧪 Testing {stage_desc[test_stage]} (Optimized)")

    pbar = tqdm(total=num_episodes, desc=f"Testing Stage {test_stage}")

    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < 1200:
                # 选择动作（确定性策略）
                action = agent.select_action(state, evaluate=True)

                # 执行动作
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                state = next_state

                # 添加延迟以便观察
                if visualize:
                    time.sleep(0.03)

            # 🔥 修改2: 记录所有episode的结果
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            success = info.get('success', False)
            all_successes.append(success)

            # 🔥 新增: 只有成功的episode才记录到成功数据中
            if success:
                success_episode_rewards.append(episode_reward)
                success_episode_lengths.append(episode_length)

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'Episode': episode + 1,
                'Reward': f"{episode_reward:.2f}",
                'Length': episode_length,
                'Success': "✅" if success else "❌",
                'Avg Success': f"{sum(all_successes) / (episode + 1):.2%}"
            })

            # episode间暂停
            if visualize and episode < num_episodes - 1:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    finally:
        pbar.close()
        if visualize:
            env.disable_visualization()

    # 🔥 修改3: 打印统计信息 - 区分成功和整体数据
    print(f"\n📊 {stage_desc[test_stage]} Test Results:")

    # 整体成功率（基于所有episode）
    overall_success_rate = sum(all_successes) / len(all_successes)
    print(f"Overall success rate: {overall_success_rate:.2%} ({sum(all_successes)}/{len(all_successes)} episodes)")

    # 🔥 关键修改: 只统计成功episode的平均奖励和步数
    if success_episode_rewards:  # 如果有成功的episode
        success_avg_reward = np.mean(success_episode_rewards)
        success_std_reward = np.std(success_episode_rewards)
        success_avg_length = np.mean(success_episode_lengths)
        success_std_length = np.std(success_episode_lengths)

        print(f"📈 Success Episodes Only Statistics:")
        print(f"  Average reward: {success_avg_reward:.2f} ± {success_std_reward:.2f}")
        print(f"  Average steps: {success_avg_length:.2f} ± {success_std_length:.2f}")
        print(f"  Number of successful episodes: {len(success_episode_rewards)}")
    else:
        print(f"❌ No successful episodes to analyze!")

    return success_episode_rewards, success_episode_lengths, all_successes


def comprehensive_stage_test(env_class, agent):
    """对所有五个阶段进行综合测试（位移控制版本）- 修改为只统计成功数据"""
    print(f"\n🔬 Starting comprehensive five-stage test (Optimized)")

    results = {}

    for stage in range(1, 6):
        print(f"\n{'=' * 50}")
        # 🔥 修改: 接收的是成功episode的数据
        success_episode_rewards, success_episode_lengths, all_successes = test_sac_five_stage(
            env_class, agent, num_episodes=20, visualize=False, test_stage=stage
        )

        # 🔥 修改: 计算结果时使用成功episode的数据
        if success_episode_rewards:  # 如果有成功的episode
            avg_reward = np.mean(success_episode_rewards)
            std_reward = np.std(success_episode_rewards)
            avg_length = np.mean(success_episode_lengths)
        else:
            # 如果没有成功的episode，使用默认值
            avg_reward = 0.0
            std_reward = 0.0
            avg_length = 1200.0  # 最大步数

        results[f'Stage_{stage}'] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'success_rate': sum(all_successes) / len(all_successes),
            'episodes': len(all_successes),
            'successful_episodes': len(success_episode_rewards)  # 🔥 新增: 成功episode数量
        }

    # 打印总结
    print(f"\n{'=' * 60}")
    print(f"🎯 Five-stage comprehensive test summary (Success Episodes Only)")
    print(f"{'=' * 60}")

    for stage in range(1, 6):
        result = results[f'Stage_{stage}']
        if stage == 4:
            random_indicator = " (Fixed start, Random target)"
        elif stage == 5:
            random_indicator = " (Random start and target)"
        else:
            random_indicator = " (Fixed start and target)"

        print(f"Stage {stage}{random_indicator}:")
        print(
            f"  Success rate:     {result['success_rate']:7.1%} ({result['successful_episodes']}/{result['episodes']})")

        # 🔥 修改: 明确标注这是成功episode的统计
        if result['successful_episodes'] > 0:
            print(f"  Avg reward (success): {result['avg_reward']:7.1f} ± {result['std_reward']:5.1f}")
            print(f"  Avg steps (success):  {result['avg_length']:7.1f}")
        else:
            print(f"  Avg reward (success): No successful episodes")
            print(f"  Avg steps (success):  No successful episodes")
        print()

    return results


def plot_five_stage_training_curves(episode_rewards, episode_lengths, success_rate,
                                    critic_losses, actor_losses, alpha_losses,
                                    q_values, alpha_values, stage_transitions,
                                    save_data=True, save_dir='./training_data'):
    """绘制五阶段训练曲线（优化版本）"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    # 1. 奖励曲线（标记阶段转换）
    axes[0].plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5)
    window = min(100, len(episode_rewards) // 10)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')

    # 标记阶段转换
    for transition in stage_transitions:
        axes[0].axvline(x=transition['episode'], color='green', linestyle='--', alpha=0.7)
        axes[0].text(transition['episode'], max(episode_rewards) * 0.9,
                     f"Stage {transition['new_stage']}", rotation=90, fontsize=8)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards (Optimized Five-stage)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. 步数曲线
    axes[1].plot(episode_lengths, alpha=0.3, color='green', linewidth=0.5)
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[1].plot(range(window - 1, len(episode_lengths)), moving_avg, 'r-', linewidth=2, label='Moving Average')

    # 标记阶段转换
    for transition in stage_transitions:
        axes[1].axvline(x=transition['episode'], color='green', linestyle='--', alpha=0.7)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length')
    axes[1].set_title('Episode Lengths')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. 成功率曲线
    if success_rate:
        axes[2].plot(success_rate, 'b-', linewidth=2, marker='o', markersize=3)

        # 标记阶段转换
        for transition in stage_transitions:
            if transition['episode'] < len(success_rate):
                axes[2].axvline(x=transition['episode'], color='green', linestyle='--', alpha=0.7)

        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Success Rate')
        axes[2].set_title('Success Rate (Optimized Five-stage)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)

    # 4. Critic Loss曲线
    if critic_losses:
        axes[3].plot(critic_losses, alpha=0.7, color='red', linewidth=1)
        window_loss = min(1000, len(critic_losses) // 10)
        if len(critic_losses) >= window_loss:
            moving_avg = np.convolve(critic_losses, np.ones(window_loss) / window_loss, mode='valid')
            axes[3].plot(range(window_loss - 1, len(critic_losses)), moving_avg, 'b-', linewidth=2,
                         label='Moving Average')
        axes[3].set_xlabel('Training Step')
        axes[3].set_ylabel('Critic Loss')
        axes[3].set_title('Critic Loss')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

    # 5. Actor Loss曲线
    if actor_losses:
        axes[4].plot(actor_losses, alpha=0.7, color='orange', linewidth=1)
        window_loss = min(500, len(actor_losses) // 10)
        if len(actor_losses) >= window_loss:
            moving_avg = np.convolve(actor_losses, np.ones(window_loss) / window_loss, mode='valid')
            axes[4].plot(range(window_loss - 1, len(actor_losses)), moving_avg, 'b-', linewidth=2,
                         label='Moving Average')
        axes[4].set_xlabel('Training Step')
        axes[4].set_ylabel('Actor Loss')
        axes[4].set_title('Actor Loss')
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()

    # 6. Q值均值曲线
    if q_values:
        axes[5].plot(q_values, alpha=0.7, color='purple', linewidth=1)
        window_q = min(1000, len(q_values) // 10)
        if len(q_values) >= window_q:
            moving_avg = np.convolve(q_values, np.ones(window_q) / window_q, mode='valid')
            axes[5].plot(range(window_q - 1, len(q_values)), moving_avg, 'b-', linewidth=2, label='Moving Average')
        axes[5].set_xlabel('Training Step')
        axes[5].set_ylabel('Mean Q Value')
        axes[5].set_title('Q Value Mean')
        axes[5].grid(True, alpha=0.3)
        axes[5].legend()

    # 7. Alpha值曲线
    if alpha_values:
        axes[6].plot(alpha_values, 'c-', linewidth=1)
        axes[6].set_xlabel('Training Step')
        axes[6].set_ylabel('Alpha')
        axes[6].set_title('Temperature Parameter (Alpha)')
        axes[6].grid(True, alpha=0.3)

    # 8. 阶段进展总结
    axes[7].axis('off')
    if stage_transitions:
        stage_text = "Stage transition record (Optimized):\n"
        for i, transition in enumerate(stage_transitions):
            stage_text += f"Episode {transition['episode']}: stage {transition['old_stage']} → {transition['new_stage']}\n"
        axes[7].text(0.1, 0.9, stage_text, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('optimized_sac_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存数据到txt文件
    if save_data:
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving training data to {save_dir}...")

        # 保存奖励数据
        reward_file = os.path.join(save_dir, 'episode_rewards.txt')
        with open(reward_file, 'w') as f:
            f.write("# Episode Rewards\n")
            f.write("# Episode_Index\tReward\n")
            for i, reward in enumerate(episode_rewards):
                f.write(f"{i}\t{reward}\n")

        # 保存成功率数据
        if success_rate:
            success_file = os.path.join(save_dir, 'success_rate.txt')
            with open(success_file, 'w') as f:
                f.write("# Success Rate\n")
                f.write("# Episode_Index\tSuccess_Rate\n")
                for i, rate in enumerate(success_rate):
                    f.write(f"{i}\t{rate}\n")

        print(f"Data saved successfully to {save_dir}/")
        print(f"Files created:")
        print(f"  - episode_rewards.txt")
        if success_rate:
            print(f"  - success_rate.txt")


def visualize_stage_environment(env_class, stage=5):
    """可视化指定阶段的环境（优化版本）"""
    stage_configs = {
        1: {'num_trees': 30, 'num_rocks': 10, 'num_water': 2, 'num_animals': 3,
            'randomize_start': False, 'randomize_target': False},
        2: {'num_trees': 30, 'num_rocks': 10, 'num_water': 2, 'num_animals': 3,
            'randomize_start': False, 'randomize_target': False},
        3: {'num_trees': 30, 'num_rocks': 10, 'num_water': 2, 'num_animals': 3,
            'randomize_start': False, 'randomize_target': False},
        4: {'num_trees': 30, 'num_rocks': 10, 'num_water': 2, 'num_animals': 3,
            'randomize_start': False, 'randomize_target': True},
        5: {'num_trees': 30, 'num_rocks': 10, 'num_water': 2, 'num_animals': 3,
            'randomize_start': True, 'randomize_target': True}
    }

    base_config = {
        'width': 100, 'height': 100, 'dt': 0.2,
        'fixed_seed': 42, 'stage': stage
    }

    config = {**base_config, **stage_configs[stage]}
    env = env_class(**config)

    fig, ax = plt.subplots(figsize=(12, 12))

    # 绘制高度等高线背景
    x = np.arange(env.width)
    y = np.arange(env.height)
    X, Y = np.meshgrid(x, y)

    contour_levels = np.linspace(0, 1, 15)
    cs = ax.contourf(X, Y, env.height_map, levels=contour_levels,
                     cmap='terrain', alpha=0.4, zorder=0)
    ax.contour(X, Y, env.height_map, levels=contour_levels,
               colors='gray', alpha=0.3, linewidths=0.5, zorder=1)

    # 绘制地形特征
    terrain_masks = {}
    for terrain_type in [TerrainType.TREE, TerrainType.ROCK, TerrainType.WATER]:
        mask = (env.static_terrain_type == terrain_type.value).astype(float)
        smooth_mask = gaussian_filter(mask, sigma=1.2)
        terrain_masks[terrain_type] = smooth_mask

    if np.any(terrain_masks[TerrainType.TREE] > 0.1):
        ax.contourf(X, Y, terrain_masks[TerrainType.TREE],
                    levels=[0.1, 0.5, 1.0], colors=['lightgreen', 'forestgreen'],
                    alpha=0.8, zorder=2)

    if np.any(terrain_masks[TerrainType.ROCK] > 0.1):
        ax.contourf(X, Y, terrain_masks[TerrainType.ROCK],
                    levels=[0.1, 0.5, 1.0], colors=['lightgray', 'dimgray'],
                    alpha=0.9, zorder=2)

    if np.any(terrain_masks[TerrainType.WATER] > 0.1):
        ax.contourf(X, Y, terrain_masks[TerrainType.WATER],
                    levels=[0.1, 0.5, 1.0], colors=['lightblue', 'steelblue'],
                    alpha=0.7, zorder=2)

    # 绘制起点和终点
    start_circle = patches.Circle((env.car_pos[1], env.car_pos[0]), 2.0,
                                  color='green', alpha=0.8, zorder=3)
    ax.add_patch(start_circle)
    ax.text(env.car_pos[1], env.car_pos[0], 'START', ha='center', va='center',
            fontsize=6, fontweight='bold', color='white', zorder=4)

    target_circle = patches.Circle((env.target_pos[1], env.target_pos[0]), 2.0,
                                   color='red', alpha=0.8, zorder=3)
    ax.add_patch(target_circle)
    ax.text(env.target_pos[1], env.target_pos[0], 'TARGET', ha='center', va='center',
            fontsize=6, fontweight='bold', color='white', zorder=4)

    # 绘制动物
    for animal in env.initial_animals:
        animal_circle = patches.Circle((animal['pos'][1], animal['pos'][0]),
                                       animal['radius'], color='orange', alpha=0.8, zorder=5)
        ax.add_patch(animal_circle)

    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if stage == 4:
        position_info = "Fixed start, Random target"
    elif stage == 5:
        position_info = "Random start and target"
    else:
        position_info = "Fixed start and target"

    ax.set_title(f'Stage {stage} Environment layout ({position_info}) - Optimized', fontsize=16,
                 fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='forestgreen', alpha=0.8, label='tree'),
        patches.Patch(facecolor='dimgray', alpha=0.9, label='stone'),
        patches.Patch(facecolor='saddlebrown', alpha=0.7, label='building'),
        patches.Circle((0, 0), 1, facecolor='orange', alpha=0.8, label='animal'),
        patches.Circle((0, 0), 1, facecolor='green', alpha=0.8, label='start'),
        patches.Circle((0, 0), 1, facecolor='red', alpha=0.8, label='target')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(f'stage_{stage}_environment_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_comparison_plot(results_dict):
    """创建不同训练方法的对比图（优化版本）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    stages = ['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4', 'Stage_5']
    stage_labels = ['stage1\n(easy)', 'stage2\n(medium)', 'stage3\n(hard)', 'stage4\n(fixed-random)',
                    'stage5\n(random-random)']

    # 成功率对比
    success_rates = [results_dict[stage]['success_rate'] for stage in stages]
    axes[0, 0].bar(stage_labels, success_rates, color=['green', 'blue', 'orange', 'red', 'purple'])
    axes[0, 0].set_ylabel('Success rate')
    axes[0, 0].set_title('Comparison of success rate in each stage (Optimized)')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center')

    # 平均奖励对比
    avg_rewards = [results_dict[stage]['avg_reward'] for stage in stages]
    std_rewards = [results_dict[stage]['std_reward'] for stage in stages]
    axes[0, 1].bar(stage_labels, avg_rewards, yerr=std_rewards,
                   color=['green', 'blue', 'orange', 'red', 'purple'], capsize=5)
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Comparison of average rewards in each stage (Optimized)')

    # 平均步数对比
    avg_lengths = [results_dict[stage]['avg_length'] for stage in stages]
    axes[1, 0].bar(stage_labels, avg_lengths, color=['green', 'blue', 'orange', 'red', 'purple'])
    axes[1, 0].set_ylabel('Average Steps')
    axes[1, 0].set_title('Comparison of average steps in each stage (Optimized)')

    # 综合性能雷达图
    axes[1, 1].set_title('Comprehensive performance radar chart (Optimized)')

    # 准备雷达图数据
    categories = ['Success rate', 'Reward efficiency', 'step count efficiency', 'Stability']

    # 归一化数据
    normalized_data = []
    for stage in stages:
        success_norm = results_dict[stage]['success_rate']
        reward_norm = (results_dict[stage]['avg_reward'] + 200) / 700
        length_norm = 1 - (results_dict[stage]['avg_length'] / 2000)
        stability_norm = 1 - (results_dict[stage]['std_reward'] / 200)

        normalized_data.append([success_norm, reward_norm, length_norm, stability_norm])

    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    ax = plt.subplot(2, 2, 4, projection='polar')
    colors = ['green', 'blue', 'orange', 'red', 'purple']

    for i, (data, color, label) in enumerate(zip(normalized_data, colors, stage_labels)):
        data = data + [data[0]]
        ax.plot(angles, data, 'o-', linewidth=2, label=label.replace('\n', ' '), color=color)
        ax.fill(angles, data, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Comprehensive Performance Comparison (Optimized)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('optimized_sac_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数：交互式五阶段课程学习SAC训练系统（优化版本）"""

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建SAC智能体
    state_dim = 37 # 修改为37维
    action_dim = 2
    agent = ImprovedSAC(state_dim, action_dim, device=device)

    # 主菜单循环
    while True:
        print("\n" + "=" * 60)
        print("🚀 Optimized Five-Stage Curriculum SAC Navigation System")
        print("=" * 60)
        choice = input(
            "\nPlease select an option:\n"
            "1. Train new optimized five-stage SAC model\n"
            "2. Test existing optimized five-stage SAC model\n"
            "3. Comprehensive five-stage performance test\n"
            "4. Visualize stage environments\n"
            "5. Exit\n"
            "Enter your choice (1-5): "
        ).strip()

        if choice == '1':
            # 训练新模型
            train_episodes = int(input("\nInput training episodes (max:1600): ") or "1100")

            results = improved_train_sac_five_stage(
                ImprovedForestEnvironment,
                agent,
                num_episodes=train_episodes,
                max_timesteps=1200,
                start_timesteps=10000,
                batch_size=256,
                save_freq=100,
                visualize=False,
            )

            # 解包结果
            (episode_rewards, episode_lengths, success_rate,
             critic_losses, actor_losses, alpha_losses,
             q_values, alpha_values, stage_transitions) = results

            # 绘制训练曲线
            print("\n📊 Plotting training curves...")
            plot_five_stage_training_curves(
                episode_rewards, episode_lengths, success_rate,
                critic_losses, actor_losses, alpha_losses,
                q_values, alpha_values, stage_transitions
            )
            print("✅ Optimized five-stage training completed!")

        elif choice == '2':
            # 测试现有模型
            model_path = input(
                "\nInput model path (default: models/optimized_sac_displacement_final.pth): ").strip()
            if not model_path:
                model_path = 'models/optimized_sac_displacement_final.pth'

            if not os.path.exists(model_path):
                print(f"❌ Model file {model_path} does not exist!")
                continue

            print(f"Loading model: {model_path}...")
            agent.load(model_path)

            # 选择测试阶段
            test_stage = input("\nSelect test stage (1-5, default 5): ").strip()
            test_stage = int(test_stage) if test_stage else 5
            test_stage = max(1, min(5, test_stage))

            test_episodes = int(input("Input test episodes (default 10): ") or "10")

            stage_desc = {
                1: "Stage 1: Simple environment, fixed start and target",
                2: "Stage 2: Medium complexity, fixed start and target",
                3: "Stage 3: High complexity, fixed start and target",
                4: "Stage 4: Highest difficulty, fixed start, random target",
                5: "Stage 5: Highest difficulty, random start and target"
            }

            print(f"\nStarting test {stage_desc[test_stage]}...")
            test_rewards, test_lengths, test_successes = test_sac_five_stage(
                ImprovedForestEnvironment,
                agent,
                num_episodes=test_episodes,
                visualize=True,
                test_stage=test_stage
            )

        elif choice == '3':
            # 五阶段综合测试
            model_path = input(
                "\nInput model path (default: models/optimized_sac_displacement_final.pth): ").strip()
            if not model_path:
                model_path = 'models/optimized_sac_displacement_final.pth'

            if not os.path.exists(model_path):
                print(f"❌ Model file {model_path} does not exist!")
                continue

            print(f"Loading model: {model_path}...")
            agent.load(model_path)

            print("\nStarting comprehensive five-stage performance test...")
            comprehensive_results = comprehensive_stage_test(ImprovedForestEnvironment, agent)

            # 创建对比图
            print("\n📊 Creating comprehensive comparison chart...")
            create_comparison_plot(comprehensive_results)

        elif choice == '4':
            # 可视化各阶段环境
            print("\n🎨 Visualizing stage environments...")
            for stage in range(1, 6):
                print(f"Generating stage {stage} environment diagram...")
                visualize_stage_environment(ImprovedForestEnvironment, stage)
            print("✅ Environment visualization completed! Images saved as stage_1-5_environment_optimized.png")

        elif choice == '5':
            print("\n👋 Exiting program...")
            break

        else:
            print("❌ Invalid selection, please enter 1-5.")

    return agent


if __name__ == "__main__":
    try:
        agent = main()
        print("\n🎉 Thank you for using the Optimized Five-Stage Curriculum SAC System!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()