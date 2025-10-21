# Learning Date : 2025/10/13
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
import math
from matplotlib.colors import ListedColormap
from collections import deque
import copy
# 设置matplotlib后端和样式
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
from enum import Enum
# 尝试导入所需类，兼容不同运行环境
try:
    from project_env import ForestEnvironmentVisualizer, TerrainType
except ImportError:
    # 若无法导入，定义占位类（实际使用时需确保project_env可用）
    class TerrainType(Enum):
        GRASS = 0
        TREE = 1
        ROCK = 2
        MUD = 3
        BUILDING = 4


    class ForestEnvironmentVisualizer:
        def __init__(self, width, height, seed):
            self.width = width
            self.height = height
            self.start_pos = (height // 2, width // 2)
            self.static_grid = np.random.rand(height, width)
            self.static_terrain_type = np.random.randint(0, 5, size=(height, width))
            self.height_map = np.random.rand(height, width)

# 在代码开头设置全局字体
# plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial Unicode MS"]


class RadarSensor:
    """雷达传感器类"""

    def __init__(self, max_range=8, fov_angle=360, resolution=5):
        self.max_range = max_range
        self.fov_angle = fov_angle  # 视野角度（度）
        self.resolution = resolution  # 角度分辨率

    def scan(self, robot_pos, true_environment):
        """
        模拟雷达扫描
        返回探测到的障碍物信息和自由空间
        """
        y, x = robot_pos
        detected_obstacles = set()
        free_space = set()

        # 根据视野角度计算扫描范围
        start_angle = (360 - self.fov_angle) // 2
        end_angle = 360 - start_angle

        # 雷达模型：扇形探测（根据视野角度）
        for angle in range(start_angle, end_angle, self.resolution):
            for r in range(1, self.max_range + 1):
                # 计算扫描点坐标并确保为整数
                scan_y = int(round(y + r * np.sin(np.radians(angle))))
                scan_x = int(round(x + r * np.cos(np.radians(angle))))

                # 检查边界
                if not (0 <= scan_y < true_environment.height and
                        0 <= scan_x < true_environment.width):
                    break

                # 标记为自由空间
                free_space.add((scan_y, scan_x))

                # 如果检测到障碍物，停止这条射线
                if true_environment.static_grid[scan_y, scan_x] > 0.3:
                    detected_obstacles.add((scan_y, scan_x))
                    break

        return detected_obstacles, free_space


class UnknownMapAStarPlanner:
    """未知环境下的A*全覆盖路径规划器"""

    def __init__(self, environment, radar_range=5, inflation_radius=1):
        self.true_env = environment  # 真实环境（用于雷达模拟）
        self.height = environment.height
        self.width = environment.width

        # 新增：膨胀区参数（周围多少格视为不可行区域）
        self.inflation_radius = inflation_radius  # 例如1表示障碍物周围1格为膨胀区

        # 创建已知地图（初始为未知）
        self.known_grid = np.zeros((self.height, self.width), dtype=np.float32) - 1  # -1表示未知
        self.known_terrain = np.zeros((self.height, self.width), dtype=np.int32) - 1  # -1表示未知
        self.obstacle_grid = np.zeros((self.height, self.width), dtype=bool)  # 已知障碍物

        # 雷达传感器
        self.radar = RadarSensor(max_range=radar_range)

        # 起点信息（确保为整数坐标）
        self.start_pos = (int(round(environment.start_pos[0])),
                          int(round(environment.start_pos[1])))

        # 初始化集合
        self.coverage_path = [self.start_pos]
        self.visited = set([self.start_pos])
        self.frontier = set()  # 边界点（已知自由空间与未知区域的边界）

        # 初始化起点周围为已知
        self._update_known_area(self.start_pos)

        # 移动方向
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
        ]

        # 重合点数

    def _inflate_obstacles(self, original_obstacles):
        """
        对障碍物进行膨胀处理，返回包含膨胀区的障碍物集合
        original_obstacles: 原始障碍物位置集合
        """
        inflated = set()
        # 先添加原始障碍物
        for (y, x) in original_obstacles:
            inflated.add((y, x))

        # 对每个障碍物，膨胀周围区域
        for (y, x) in original_obstacles:
            # 遍历膨胀半径内的所有点（包括对角线）
            for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                    # 跳过原点（已添加原始障碍物）
                    if dy == 0 and dx == 0:
                        continue
                    # 计算膨胀点坐标
                    ny = y + dy
                    nx = x + dx
                    # 检查是否在地图范围内
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # 检查是否已标记为自由空间（避免覆盖已确认的自由区域）
                        if self.known_grid[ny, nx] < 0 or self.obstacle_grid[ny, nx]:
                            inflated.add((ny, nx))
        return inflated

    def _update_known_area(self, robot_pos):
        """更新已知地图信息"""
        # 使用雷达扫描
        obstacles, free_space = self.radar.scan(robot_pos, self.true_env)
        # 新增：对障碍物进行膨胀处理
        inflated_obstacles = self._inflate_obstacles(obstacles)

        # 更新已知地图
        for pos in free_space:
            y, x = pos
            self.known_grid[y, x] = self.true_env.static_grid[y, x]
            self.known_terrain[y, x] = self.true_env.static_terrain_type[y, x]
            self.obstacle_grid[y, x] = False

        for pos in inflated_obstacles:
            y, x = pos
            self.known_grid[y, x] = self.true_env.static_grid[y, x]
            self.known_terrain[y, x] = self.true_env.static_terrain_type[y, x]
            self.obstacle_grid[y, x] = True

        # 更新边界点
        self._update_frontier()

    def _update_frontier(self):
        """更新边界点（已知自由空间与未知区域的边界）"""
        new_frontier = set()

        for y in range(self.height):
            for x in range(self.width):
                # 如果是已知自由空间
                if self.known_grid[y, x] >= 0 and not self.obstacle_grid[y, x]:
                    # 检查邻居是否有未知区域
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width and
                                self.known_grid[ny, nx] < 0):  # 邻居是未知区域
                            new_frontier.add((y, x))
                            break

        self.frontier = new_frontier

    def heuristic(self, a, b):
        """启发式函数：曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_min_obstacle_distance(self, pos):
        """计算当前位置与最近障碍物的距离"""
        y, x = pos
        min_dist = float('inf')
        # 检查周围一定范围内的障碍物（范围可根据需要调整）
        check_range = 3
        for dy in range(-check_range, check_range + 1):
            for dx in range(-check_range, check_range + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.obstacle_grid[ny, nx]:
                        dist = math.hypot(dy, dx)
                        if dist < min_dist:
                            min_dist = dist
        return min_dist if min_dist != float('inf') else -1

    def get_movement_cost(self, current_pos, next_pos):
        """获取移动代价（只基于已知信息）"""
        y, x = next_pos

        # 检查边界
        if not (0 <= y < self.height and 0 <= x < self.width):
            return float('inf')

        # 检查已知障碍物
        if self.obstacle_grid[y, x]:
            return float('inf')

        # 新增：计算与障碍物的距离，增加安全距离代价
        obstacle_distance = self._get_min_obstacle_distance(next_pos)
        safety_distance = 2  # 安全距离（可调整）
        if obstacle_distance < safety_distance and obstacle_distance >= 0:
            # 距离障碍物越近，代价越高（指数增长）
            distance_penalty = (safety_distance - obstacle_distance) * 5
        else:
            distance_penalty = 0

        # 如果是未知区域，给予较高代价（鼓励探索已知区域）
        if self.known_grid[y, x] < 0:
            return 5.0  # 探索未知的代价

        # 基础移动代价
        base_cost = 1.0

        # 地形代价（基于已知地形）
        if self.known_terrain[y, x] >= 0:
            terrain_costs = {0: 1.0, 1: 5.0, 2: 8.0, 3: 3.0, 4: 6.0}
            terrain_cost = terrain_costs.get(self.known_terrain[y, x], 1.0)
            base_cost *= terrain_cost

        # 对角线移动代价
        dy = abs(next_pos[0] - current_pos[0])
        dx = abs(next_pos[1] - current_pos[1])
        if dy == 1 and dx == 1:
            base_cost *= 1.414

        # return base_cost + distance_penalty
        return base_cost

    def a_star_search(self, start, goal):
        open_set = []
        # 存储格式：(优先级, 位置, 上一步方向)
        heapq.heappush(open_set, (0, start, None))

        came_from = {}  # 仅用位置作为键
        dir_record = {}  # 记录到达每个位置的方向
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            _, current, prev_dir = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                # 正确重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dy, dx in self.directions:
                neighbor = (current[0] + dy, current[1] + dx)
                curr_dir = (dy, dx)

                # 边界检查
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue

                # 障碍物检查
                move_cost = self.get_movement_cost(current, neighbor)
                if move_cost == float('inf'):
                    continue

                # 转向惩罚计算
                turn_penalty = 0
                if prev_dir is not None and curr_dir != prev_dir:
                    if curr_dir[0] == prev_dir[0] and (curr_dir[1] + prev_dir[1] == 0) :
                        turn_penalty = move_cost * 0.5
                    elif (curr_dir[0] + prev_dir[0] == 0) and curr_dir[1] == prev_dir[1]:
                        turn_penalty = move_cost * 0.5
                    elif (curr_dir[0] + prev_dir[0] == 0) and (curr_dir[1] + prev_dir[1] == 0):
                        turn_penalty = move_cost * 0.5
                    else:
                        turn_penalty = move_cost * 0.35  # 可调整惩罚系数

                # 总代价计算
                tentative_g_score = g_score[current] + move_cost + turn_penalty

                # 更新节点信息
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current  # 仅记录位置关系
                    dir_record[neighbor] = curr_dir  # 单独记录方向
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor, curr_dir))
                        open_set_hash.add(neighbor)

        return None  # 找不到路径时返回None

    def select_next_goal(self):
        """选择下一个探索目标（最近的边界点）"""
        if not self.frontier:
            return None  # 没有更多边界点可探索

        current_pos = self.coverage_path[-1]
        min_distance = float('inf')
        best_goal = None

        for frontier_point in self.frontier:
            # 计算到边界点的代价估计
            distance = self.heuristic(current_pos, frontier_point)

            # 如果边界点已经被访问过，跳过
            if frontier_point in self.visited:
                continue

            if distance < min_distance:
                min_distance = distance
                best_goal = frontier_point

        return best_goal

    def smooth_path(self, path, max_range=2):
        """
        路径平滑：在指定范围内寻找可直线到达的点（而非最远点）
        max_range: 最大检查范围（最多跳过的点数）
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # 计算当前位置能检查的最远索引（不超过路径长度和i+max_range）
            max_check_index = min(i + max_range, len(path) - 1)
            farthest = i + 1  # 至少前进1步

            # 在i+1到max_check_index范围内寻找最远的可直线到达点
            for j in range(i + 2, max_check_index + 1):
                if self.is_straight_line_clear(path[i], path[j]):
                    farthest = j
                else:
                    # 遇到障碍物则停止当前方向的检查（避免跳过障碍物）
                    break

            smoothed.append(path[farthest])
            i = farthest
        return smoothed

    def is_straight_line_clear(self, start, end):
        """检查两点之间直线是否无障碍物"""
        y0, x0 = start
        y1, x1 = end

        #  Bresenham算法判断直线上的点是否有障碍物
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy

        while (x, y) != (x1, y1):
            if self.obstacle_grid[y, x]:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def plan_coverage_path(self, max_steps=2000):
        """执行未知环境下的全覆盖路径规划"""
        print("开始未知环境全覆盖路径规划...")
        print(f"雷达探测范围: {self.radar.max_range} 单位")

        current_pos = self.start_pos
        step_count = 0

        while step_count < max_steps and self.frontier:
            step_count += 1

            # 选择下一个目标点
            next_goal = self.select_next_goal()

            if next_goal is None:
                print("所有可探索区域已完成覆盖!")
                break

            # 使用A*规划到目标点的路径
            path_to_goal = self.a_star_search(current_pos, next_goal)

            if path_to_goal is None:
                # 无法到达该边界点，从边界点集合中移除
                self.frontier.discard(next_goal)
                continue

            if path_to_goal:
                path_to_goal = self.smooth_path(path_to_goal)  # 增加平滑步骤

            # 执行路径（模拟移动和探测）
            for i, next_pos in enumerate(path_to_goal[1:], 1):  # 跳过起点
                # 更新当前位置
                current_pos = next_pos

                # 更新已知地图
                self._update_known_area(current_pos)

                # 记录路径和访问点
                self.coverage_path.append(current_pos)
                self.visited.add(current_pos)

                # 显示进度
                if step_count % 50 == 0:
                    known_ratio = np.sum(self.known_grid >= 0) / (self.height * self.width) * 100
                    visited_ratio = len(self.visited) / (self.height * self.width) * 100
                    print(
                        f"步数: {step_count}, 已知区域: {known_ratio:.1f}%, 访问率: {visited_ratio:.1f}%, 边界点: {len(self.frontier)}")

            # 检查是否完成探索
            if not self.frontier:
                print("探索完成!")
                break

        # 统计结果
        known_ratio = np.sum(self.known_grid >= 0) / (self.height * self.width) * 100
        visited_ratio = len(self.visited) / (self.height * self.width) * 100

        print(f"\n规划完成!")
        print(f"总步数: {len(self.coverage_path)}")
        print(f"已知区域比例: {known_ratio:.2f}%")
        print(f"访问区域比例: {visited_ratio:.2f}%")
        print(f"剩余边界点: {len(self.frontier)}")

        return self.coverage_path


class UnknownMapVisualizer:
    """未知环境路径规划可视化器"""

    def __init__(self, environment, planner):
        self.env = environment
        self.planner = planner
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        # 动画元素存储
        self.robot_pos1 = None
        self.robot_pos2 = None
        self.path_line1 = None
        self.path_line2 = None
        self.radar_circle1 = None
        self.radar_circle2 = None
        self.known_map = np.ones((self.env.height, self.env.width, 3)) * 0.8
        # 新增：存储历史雷达位置 (x, y)
        self.radar_history = []  # 记录所有历史雷达中心
        self.radar_history_step = 1  # 每2步记录一次，减少数据量

    def setup_visualization(self):
        """设置可视化界面"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 左侧：真实环境
        self._plot_environment(self.ax1, self.env, "true env")

        # 右侧：已知地图
        self._plot_known_map(self.ax2, "known map and searching path")

        # 初始化动画元素
        self.robot_pos1, = self.ax1.plot([], [], 'ro', markersize=10, alpha=0.9, zorder=5, label='robot')
        self.robot_pos2, = self.ax2.plot([], [], 'ro', markersize=10, alpha=0.9, zorder=5, label='robot')

        self.path_line1, = self.ax1.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=3, label='path')
        self.path_line2, = self.ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=3, label='path')

        self.radar_circle1 = patches.Circle((0, 0), self.planner.radar.max_range,
                                            fill=False, edgecolor='red', linestyle='--',
                                            linewidth=2, alpha=0.6, zorder=4, label='Radar Range')
        self.radar_circle2 = patches.Circle((0, 0), self.planner.radar.max_range,
                                            fill=False, edgecolor='red', linestyle='--',
                                            linewidth=2, alpha=0.6, zorder=4, label='Radar Range')

        self.ax1.add_patch(self.radar_circle1)
        self.ax2.add_patch(self.radar_circle2)

        # 添加图例
        self.ax1.legend(loc='upper right')
        self.ax2.legend(loc='upper right')

        plt.tight_layout()
        return self.fig, (self.ax1, self.ax2)

    def _plot_environment(self, ax, environment, title):
        """绘制环境地图"""
        x = np.arange(environment.width)
        y = np.arange(environment.height)
        X, Y = np.meshgrid(x, y)

        # 绘制高度等高线背景
        contour_levels = np.linspace(0, 1, 15)
        cs = ax.contourf(X, Y, environment.height_map, levels=contour_levels,
                         cmap='terrain', alpha=0.6, zorder=0)

        # 绘制地形特征
        terrain_masks = {}
        for terrain_type in [TerrainType.TREE, TerrainType.ROCK, TerrainType.MUD]:
            mask = (environment.static_terrain_type == terrain_type.value).astype(float)
            smooth_mask = gaussian_filter(mask, sigma=1.0)
            terrain_masks[terrain_type] = smooth_mask

        if np.any(terrain_masks[TerrainType.TREE] > 0.05):
            tree_levels = [0.03, 0.4, 0.8, 1.2]
            tree_colors = ['lightgreen', 'forestgreen', 'darkgreen']
            ax.contourf(X, Y, terrain_masks[TerrainType.TREE],
                        levels=tree_levels, colors=tree_colors, alpha=0.8, zorder=2)

        if np.any(terrain_masks[TerrainType.ROCK] > 0.1):
            ax.contourf(X, Y, terrain_masks[TerrainType.ROCK],
                        levels=[0.2, 0.4, 1.1], colors=['lightgray', 'gray'],
                        alpha=0.8, zorder=2)

        if np.any(terrain_masks[TerrainType.MUD] > 0.1):
            ax.contourf(X, Y, terrain_masks[TerrainType.MUD],
                        levels=[0.1, 0.5, 1.0], colors=['burlywood', 'saddlebrown'],
                        alpha=0.7, zorder=2)

        # 绘制起点
        start_circle = patches.Circle((self.env.start_pos[1], self.env.start_pos[0]), 1.0,
                                      color='green', alpha=0.8, zorder=4)
        ax.add_patch(start_circle)
        ax.text(self.env.start_pos[1], self.env.start_pos[0], 'S',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=5)

        ax.set_xlim(-1, self.env.width)
        ax.set_ylim(-1, self.env.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def _plot_known_map(self, ax, title):
        """绘制已知地图"""
        # 创建已知地图可视化
        self.known_map = np.ones((self.env.height, self.env.width, 3)) * [0.8, 1.0, 0.8]
        # 已知障碍物 - 黑色
        obstacle_mask_pengzhang = self.planner.obstacle_grid
        obstacle_mask = (self.planner.known_grid > 0.3) & (self.planner.obstacle_grid)
        self.known_map[obstacle_mask] = [0.1, 0.1, 0.1]  # 黑色

        # 第0层已知区域简单图
        ax.imshow(self.known_map, extent=[0, self.env.width, 0, self.env.height],
                  origin='lower', alpha=0.9, zorder=0)

        # 绘制起点
        start_circle = patches.Circle((self.env.start_pos[1], self.env.start_pos[0]), 1.0,
                                      color='green', alpha=0.8, zorder=4)
        ax.add_patch(start_circle)
        ax.text(self.env.start_pos[1], self.env.start_pos[0], 'S',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=5)

        # 绘制边界点
        if hasattr(self.planner, 'frontier') and self.planner.frontier:
            frontier_y, frontier_x = zip(*self.planner.frontier)
            ax.scatter(frontier_x, frontier_y, c='red', s=20, alpha=0.6,
                       marker='.', zorder=3, label='Boundary point')

        ax.set_xlim(-1, self.env.width)
        ax.set_ylim(-1, self.env.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # 添加图例
        legend_elements = [
            patches.Patch(facecolor='gray', alpha=0.8, label='unknown area'),
            patches.Patch(facecolor='black', alpha=0.8, label='known obstacle'),
            patches.Patch(facecolor='lightgreen', alpha=0.8, label='grass'),
            patches.Patch(facecolor='darkgreen', alpha=0.8, label='tree'),
            patches.Patch(facecolor='gray', alpha=0.8, label='rock'),
            patches.Patch(facecolor='brown', alpha=0.8, label='building'),
            patches.Patch(facecolor='red', alpha=0.6, label='boundary point'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    def animate_exploration(self, interval=50):
        """创建探索过程动画"""
        if not self.planner.coverage_path:
            print("请先执行路径规划!")
            return

        self.setup_visualization()

        def init():
            self.robot_pos1.set_data([], [])
            self.robot_pos2.set_data([], [])
            self.path_line1.set_data([], [])
            self.path_line2.set_data([], [])
            self.radar_circle1.center = (0, 0)
            self.radar_circle2.center = (0, 0)
            return (self.robot_pos1, self.robot_pos2,
                    self.path_line1, self.path_line2,
                    self.radar_circle1, self.radar_circle2)

        def update(frame):
            if frame >= len(self.planner.coverage_path):
                return (self.robot_pos1, self.robot_pos2,
                        self.path_line1, self.path_line2,
                        self.radar_circle1, self.radar_circle2)

            current_pos = self.planner.coverage_path[frame]
            x_pos = current_pos[1]
            y_pos = current_pos[0]

            # 更新机器人位置
            self.robot_pos1.set_data([x_pos], [y_pos])
            self.robot_pos2.set_data([x_pos], [y_pos])

            # 更新雷达范围
            self.radar_circle1.center = (x_pos, y_pos)
            self.radar_circle2.center = (x_pos, y_pos)

            self.radar_history.append((x_pos, y_pos))  # 存储当前雷达中心
            # 控制历史记录数量，避免内存占用过大
            if len(self.radar_history) > 1000:
                self.radar_history.pop(0)

            path_x = []
            path_y = []
            # 更新路径线
            if frame > 0:
                path_x = [p[1] for p in self.planner.coverage_path[:frame + 1]]
                path_y = [p[0] for p in self.planner.coverage_path[:frame + 1]]
                self.path_line1.set_data(path_x, path_y)
                self.path_line2.set_data(path_x, path_y)

            # 更新右侧地图（每5帧更新一次以提高性能）
            if frame % 3 == 0 or frame == len(self.planner.coverage_path) - 1:
                self.ax2.clear()
                unknown_map = np.ones((self.env.height, self.env.width, 3)) * 0.8
                # 转为RGBA格式（增加alpha通道控制透明度）
                unknown_map_rgba = np.concatenate(
                    [unknown_map, np.ones((self.env.height, self.env.width, 1))],  # alpha初始为1（完全不透明）
                    axis=2
                )
                # 生成网格坐标（y为行，x为列）
                y_grid, x_grid = np.mgrid[0:self.env.height, 0:self.env.width]
                radar_range = self.planner.radar.max_range + 0.5
                # 雷达扫过区域显示为已知地图
                for (hx, hy) in self.radar_history:
                    distance = np.sqrt((y_grid - hy) ** 2 + (x_grid - hx) ** 2)
                    radar_mask = distance <= radar_range
                    unknown_map_rgba[radar_mask, 3] = 0

                self.ax2.imshow(
                    unknown_map_rgba,
                    extent=[0, self.env.height, 0, self.env.width],
                    origin='lower',
                    zorder=1  # 上层（覆盖底层，但透明区域露出底层）
                )

                # 重新创建右侧动态元素
                self.robot_pos2, = self.ax2.plot([x_pos], [y_pos], 'ro', markersize=20, alpha=0.9, zorder=5)
                self.path_line2, = self.ax2.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, zorder=3)

                # 绘制当前雷达范围
                self.radar_circle2 = patches.Circle(
                    (x_pos, y_pos), self.planner.radar.max_range,
                    fill=False, edgecolor='red', linestyle='--',  # 当前范围用深绿色
                    linewidth=2, alpha=0.6, zorder=4  # 层级高于历史痕迹
                )
                self.ax2.add_patch(self.radar_circle2)
                self._plot_known_map(self.ax2,
                                     f"known map (process: {(frame + 1) / len(self.planner.coverage_path) * 100:.1f}%)")



            progress = (frame + 1) / len(self.planner.coverage_path) * 100
            self.ax1.set_title(f'True_Env (Process: {progress:.1f}%)', fontsize=12, fontweight='bold')

            return (self.robot_pos1, self.robot_pos2,
                    self.path_line1, self.path_line2,
                    self.radar_circle1, self.radar_circle2)

        anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.planner.coverage_path),
            init_func=init, interval=interval, blit=False, repeat=False
        )

        plt.show()
        return anim

    def clear_radar_history(self):
        self.radar_history.pop(0)


def main():
    """主函数"""
    print("🌲 未知环境森林探索 - 带雷达的全覆盖路径规划")
    print("=" * 60)

    # 创建环境（使用较小地图提高演示速度）
    env = ForestEnvironmentVisualizer(width=55, height=55, seed=43)

    # 创建未知环境规划器
    planner = UnknownMapAStarPlanner(env, radar_range=4, inflation_radius=1)

    # 执行路径规划
    coverage_path = planner.plan_coverage_path(max_steps=1000)

    # 可视化
    visualizer = UnknownMapVisualizer(env, planner)

    print("\n🎬 开始探索动画演示...")
    anim = visualizer.animate_exploration(interval=20)

    visualizer.clear_radar_history()

    return env, planner, anim


if __name__ == "__main__":
    main()
