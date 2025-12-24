# Learning Date : 2025/11/10
# Learning Date : 2025/10/25
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
import noise
import random
from enum import Enum
from collections import deque
import heapq
import math
from sklearn.cluster import DBSCAN  # 用于将散乱的边界点聚类成区域

# 设置matplotlib后端和样式
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class TerrainType(Enum):
    """地形类型枚举"""
    GRASS = 0  # 草地
    TREE = 1  # 树木
    ROCK = 2  # 岩石
    MUD = 3  # 泥地/建筑
    ANIMAL = 4  # 动物（动态障碍物）


class PathSmoother:
    """路径平滑器"""

    def __init__(self, smoothing_factor=0.1, smooth_points_density=2.0):
        self.smoothing_factor = smoothing_factor
        self.smooth_points_density = smooth_points_density

    def smooth_path_b_spline(self, path, s=0.0):
        """
        使用B样条曲线平滑路径
        Args:
            path: 原始路径点列表 [(y1, x1), (y2, x2), ...]
            s: 平滑因子，0表示完全平滑，值越大越接近原始路径
        Returns:
            smoothed_path: 平滑后的路径
        """
        if len(path) < 4:
            return path  # 点太少无法进行B样条拟合

        # 将路径点转换为numpy数组 (注意坐标顺序)
        path_array = np.array(path)
        y_coords = path_array[:, 0]
        x_coords = path_array[:, 1]

        try:
            # 使用B样条曲线拟合
            tck, u = splprep([x_coords, y_coords], s=s, per=False)

            # 生成更密集的插值点
            num_points = max(50, int(len(path) * self.smooth_points_density))
            u_new = np.linspace(0, 1, num_points)

            # 计算平滑后的路径
            x_smooth, y_smooth = splev(u_new, tck)

            # 重新组合为(y, x)格式
            smoothed_path = list(zip(y_smooth, x_smooth))

            print(f"   - 路径平滑: {len(path)} → {len(smoothed_path)} 个点")
            return smoothed_path

        except Exception as e:
            print(f"   - B样条平滑失败: {e}, 使用原始路径")
            return path

    def smooth_path_simple(self, path, window_size=3):
        """
        使用移动平均简单平滑路径
        Args:
            path: 原始路径
            window_size: 滑动窗口大小
        Returns:
            smoothed_path: 平滑后的路径
        """
        if len(path) < window_size:
            return path

        smoothed_path = []
        for i in range(len(path)):
            # 计算滑动窗口内的平均位置
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)

            window_points = path[start_idx:end_idx]
            avg_y = np.mean([p[0] for p in window_points])
            avg_x = np.mean([p[1] for p in window_points])

            smoothed_path.append((avg_y, avg_x))

        print(f"   - 简单平滑: 窗口大小 {window_size}")
        return smoothed_path


class ForestEnvironmentVisualizer:
    """森林环境可视化器"""

    def __init__(self, width=100, height=100, seed=30):
        self.width = width
        self.height = height
        self.seed = seed

        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)

        # 初始化网格
        self.static_grid = np.zeros((height, width), dtype=np.float32)
        self.static_terrain_type = np.zeros((height, width), dtype=np.int32)
        self.height_map = np.zeros((height, width), dtype=np.float32)

        # 起点和终点
        self.start_pos = [height * 0.05, width * 0.05]  # 左下角

        # 障碍物列表
        self.animals = []

        # 生成环境
        self._generate_environment()

    def _generate_environment(self):
        """生成完整的环境"""
        self._generate_height_map()
        self._generate_trees()
        self._generate_rocks()
        self._generate_mud_areas()
        self._clear_start_end_areas()

    def _generate_height_map(self):
        """使用Perlin噪声生成平滑地形高度图"""
        scale = 50.0
        octaves = 3
        persistence = 0.5
        lacunarity = 2.0
        seed = np.random.randint(0, 100)

        world = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                world[i][j] = noise.pnoise2(
                    i / scale, j / scale,
                    octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity, repeatx=self.height,
                    repeaty=self.width, base=seed
                )

        # 归一化到0-1范围
        self.height_map = (world - np.min(world)) / (np.max(world) - np.min(world))
        # 平滑处理
        self.height_map = gaussian_filter(self.height_map, sigma=1.2)

    def _generate_trees(self, num_trees=30):
        """生成树木"""
        for _ in range(num_trees):
            attempts = 0
            while attempts < 100:
                tree_y = np.random.uniform(8, self.height - 8)
                tree_x = np.random.uniform(8, self.width - 8)
                tree_radius = np.random.uniform(1.5, 3.0)

                if self._is_valid_position(tree_y, tree_x, tree_radius):
                    self._draw_circular_obstacle(tree_y, tree_x, tree_radius,
                                                 TerrainType.TREE.value, intensity=0.9)
                    break
                attempts += 1

    def _generate_rocks(self, num_rocks=12):
        """生成岩石"""
        for _ in range(num_rocks):
            attempts = 0
            while attempts < 100:
                rock_y = np.random.uniform(5, self.height - 5)
                rock_x = np.random.uniform(5, self.width - 5)
                rock_radius = np.random.uniform(1.5, 3.0)

                if self._is_valid_position(rock_y, rock_x, rock_radius):
                    self._draw_irregular_obstacle(rock_y, rock_x, rock_radius,
                                                  TerrainType.ROCK.value)
                    break
                attempts += 1

    def _generate_mud_areas(self, num_mud=3):
        """生成泥地/建筑区域"""
        for _ in range(num_mud):
            attempts = 0
            while attempts < 100:
                mud_y = np.random.uniform(8, self.height - 8)
                mud_x = np.random.uniform(8, self.width - 8)
                mud_radius = np.random.uniform(2.5, 4.5)

                if self._is_valid_position(mud_y, mud_x, mud_radius):
                    self._draw_circular_obstacle(mud_y, mud_x, mud_radius,
                                                 TerrainType.MUD.value, intensity=0.8)
                    break
                attempts += 1

    def _generate_animals(self, num_animals=10):
        """生成动物"""
        for _ in range(num_animals):
            attempts = 0
            while attempts < 100:
                animal_y = np.random.uniform(15, self.height - 15)
                animal_x = np.random.uniform(15, self.width - 15)

                if self._is_valid_position(animal_y, animal_x, 1.0):
                    self.animals.append({
                        'pos': [animal_y, animal_x],
                        'radius': 0.8
                    })
                    break
                attempts += 1

    def _is_valid_position(self, y, x, radius, min_distance=4):
        """检查位置是否有效"""
        # 检查与起点和终点的距离
        start_dist = np.sqrt((y - self.start_pos[0]) ** 2 + (x - self.start_pos[1]) ** 2)

        if start_dist < radius + min_distance:
            return False

        # 检查是否与现有障碍物重叠
        check_radius = int(radius + min_distance)
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                check_y = int(y) + dy
                check_x = int(x) + dx

                if (check_y < 0 or check_y >= self.height or
                        check_x < 0 or check_x >= self.width):
                    continue

                distance = np.sqrt(dy ** 2 + dx ** 2)
                if distance <= radius + min_distance:
                    terrain = self.static_terrain_type[check_y, check_x]
                    if terrain in [TerrainType.TREE.value, TerrainType.ROCK.value,
                                   TerrainType.MUD.value]:
                        return False
                    if self.static_grid[check_y, check_x] > 0.2:
                        return False

        return True

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
                        # 树木有核心和边缘
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
        for _ in range(3):  # 多个椭圆组合
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

    def _clear_start_end_areas(self):
        """清理起点和终点区域"""
        clear_radius = 6

        # 清理起点
        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                distance = np.sqrt(dy ** 2 + dx ** 2)
                if distance <= clear_radius:
                    y = int(self.start_pos[0]) + dy
                    x = int(self.start_pos[1]) + dx
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.static_grid[y, x] = 0
                        self.static_terrain_type[y, x] = TerrainType.GRASS.value

    def visualize(self, save_path="forest_environment.png", figsize=(12, 10)):
        """可视化森林环境"""
        fig, ax = plt.subplots(figsize=figsize)

        # 创建坐标网格
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # 1. 绘制高度等高线背景
        contour_levels = np.linspace(0, 1, 15)
        cs = ax.contourf(X, Y, self.height_map, levels=contour_levels,
                         cmap='terrain', alpha=0.6, zorder=0)

        # 添加等高线
        ax.contour(X, Y, self.height_map, levels=contour_levels,
                   colors='gray', alpha=0.3, linewidths=0.5, zorder=1)

        # 添加颜色条
        cbar = plt.colorbar(cs, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Terrain Height', rotation=270, labelpad=20)

        # 2. 绘制各种地形特征
        terrain_masks = {}
        for terrain_type in [TerrainType.TREE, TerrainType.ROCK, TerrainType.MUD]:
            mask = (self.static_terrain_type == terrain_type.value).astype(float)
            smooth_mask = gaussian_filter(mask, sigma=1.0)
            terrain_masks[terrain_type] = smooth_mask

        # 绘制树木
        if np.any(terrain_masks[TerrainType.TREE] > 0.05):
            tree_levels = [0.05, 0.3, 0.6, 1.0]
            tree_colors = ['lightgreen', 'forestgreen', 'darkgreen']
            ax.contourf(X, Y, terrain_masks[TerrainType.TREE],
                        levels=tree_levels, colors=tree_colors,
                        alpha=0.8, zorder=2)

        # 绘制岩石
        if np.any(terrain_masks[TerrainType.ROCK] > 0.1):
            ax.contourf(X, Y, terrain_masks[TerrainType.ROCK],
                        levels=[0.1, 0.5, 1.0], colors=['lightgray', 'gray'],
                        alpha=0.8, zorder=2)

        # 绘制建筑
        if np.any(terrain_masks[TerrainType.MUD] > 0.1):
            ax.contourf(X, Y, terrain_masks[TerrainType.MUD],
                        levels=[0.1, 0.5, 1.0], colors=['burlywood', 'saddlebrown'],
                        alpha=0.7, zorder=2)

        # 3. 绘制起点和终点
        # 起点 (绿色)
        start_circle1 = patches.Circle((self.start_pos[1], self.start_pos[0]), 3.0,
                                       color='green', alpha=0.3, zorder=3)
        start_circle2 = patches.Circle((self.start_pos[1], self.start_pos[0]), 2.2,
                                       color='green', alpha=0.5, zorder=3)
        start_circle3 = patches.Circle((self.start_pos[1], self.start_pos[0]), 1.5,
                                       color='green', alpha=0.8, zorder=3)

        ax.add_patch(start_circle1)
        ax.add_patch(start_circle2)
        ax.add_patch(start_circle3)
        ax.text(self.start_pos[1], self.start_pos[0], 'S', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)

        # 4. 绘制动物
        for animal in self.animals:
            animal_circle = patches.Circle((animal['pos'][1], animal['pos'][0]),
                                           animal['radius'], color='orange',
                                           alpha=0.8, zorder=5)
            ax.add_patch(animal_circle)

        # 5. 设置图形属性
        ax.set_xlim(-1, self.width)
        ax.set_ylim(-1, self.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)

        # 6. 添加图例
        legend_elements = [
            patches.Patch(facecolor='forestgreen', alpha=0.8, label='Tree'),
            patches.Patch(facecolor='gray', alpha=0.8, label='Rock'),
            patches.Patch(facecolor='saddlebrown', alpha=0.7, label='Building'),
            patches.Circle((0, 0), 1, facecolor='orange', alpha=0.8, label='Animal')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.show()

        return fig, ax


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


class GridCell:
    """Grid cell class"""

    def __init__(self, row, col, center_y, center_x, cell_size):
        self.row = row
        self.col = col
        self.center_y = center_y
        self.center_x = center_x
        self.cell_size = cell_size
        self.is_covered = False
        self.is_explored = False
        self.is_obstacle = False
        self.visit_count = 0
        self.distance_to_uncovered = float('inf')


class ImprovedWavefrontPlanner:
    """Improved wavefront coverage planner"""

    def __init__(self, environment, radar_range=10, cell_size=2):
        self.env = environment
        self.height = environment.height
        self.width = environment.width
        self.radar_range = radar_range
        self.cell_size = cell_size

        # Grid parameters
        self.grid_rows = int(np.ceil(self.height / self.cell_size))
        self.grid_cols = int(np.ceil(self.width / self.cell_size))

        print(f"\nGrid Info:")
        print(f"   - Map size: {self.height} x {self.width} m")
        print(f"   - Cell size: {self.cell_size} x {self.cell_size} m")
        print(f"   - Grid dimensions: {self.grid_rows} x {self.grid_cols} = {self.grid_rows * self.grid_cols} cells")

        # Initialize grid
        self.grid = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                center_y = (row + 0.5) * self.cell_size
                center_x = (col + 0.5) * self.cell_size
                if center_y >= self.height:
                    center_y = self.height - 0.5
                if center_x >= self.width:
                    center_x = self.width - 0.5
                self.grid[row][col] = GridCell(row, col, center_y, center_x, self.cell_size)

        # Radar
        self.radar = RadarSensor(max_range=radar_range)

        # Path smoother
        self.path_smoother = PathSmoother(smoothing_factor=0.1, smooth_points_density=2.0)

        # Start position
        start_y, start_x = environment.start_pos
        self.current_cell = self._get_cell_from_pos(start_y, start_x)
        self.current_pos = (int(self.current_cell.center_y), int(self.current_cell.center_x))

        # Path record
        self.raw_path = [self.current_pos]  # 原始路径
        self.smooth_path = []  # 平滑后的路径
        self.path = self.raw_path  # 当前使用的路径
        self.use_astar = False

        # Initial scan
        self._scan_and_update()
        self.current_cell.is_covered = True
        self.current_cell.visit_count = 1

        print(f"   - Start: Cell({self.current_cell.row}, {self.current_cell.col})")
        print(f"   - Radar range: {self.radar_range} m")

    def _get_cell_from_pos(self, y, x):
        """Get cell from position"""
        row = min(int(y / self.cell_size), self.grid_rows - 1)
        col = min(int(x / self.cell_size), self.grid_cols - 1)
        return self.grid[row][col]

    # def _scan_and_update(self):
    #     """Scan and update map"""
    #     obstacles, free_space = self.radar.scan(self.current_pos, self.env)
    #
    #     # Update all scanned cells as "explored"
    #     all_scanned = obstacles | free_space
    #     for y, x in all_scanned:
    #         cell = self._get_cell_from_pos(y, x)
    #         if not cell.is_explored:
    #             cell.is_explored = True
    #
    #     # Mark obstacle cells
    #     for y, x in obstacles:
    #         cell = self._get_cell_from_pos(y, x)
    #         cell.is_obstacle = True
    #
    #     # Ensure free space cells are not obstacles
    #     for y, x in free_space:
    #         cell = self._get_cell_from_pos(y, x)
    #         if not cell.is_obstacle:
    #             cell.is_obstacle = False

    def _scan_and_update(self):
        """直接基于真实环境更新单元格障碍，跳过雷达模拟的不完整扫描"""
        # 遍历所有单元格，直接读取真实环境的障碍状态
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = self.grid[row][col]
                # 取单元格中心像素的真实环境值
                cy, cx = int(cell.center_y), int(cell.center_x)
                if 0 <= cy < self.env.height and 0 <= cx < self.env.width:
                    # 标记为已探索
                    cell.is_explored = True
                    # 真实环境中该像素是障碍（static_grid>0.3）→ 单元格设为障碍
                    if self.env.static_grid[cy, cx] > 0.3:
                        cell.is_obstacle = True
                    else:
                        cell.is_obstacle = False


    def _compute_distance_field(self):
        """Compute distance field to nearest uncovered cell"""
        # Reset distances
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                self.grid[row][col].distance_to_uncovered = float('inf')

        # Find all uncovered passable cells as seeds
        queue = deque()
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = self.grid[row][col]
                # Explored, passable, uncovered cells as seeds
                if cell.is_explored and not cell.is_obstacle and not cell.is_covered:
                    cell.distance_to_uncovered = 0
                    queue.append(cell)
                # Unexplored cells also as potential targets (boundary exploration)
                elif not cell.is_explored:
                    cell.distance_to_uncovered = 0
                    queue.append(cell)

        # BFS to compute distance field
        while queue:
            cell = queue.popleft()

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = cell.row + dr, cell.col + dc
                    if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                        continue

                    neighbor = self.grid[nr][nc]

                    # Skip obstacles
                    if neighbor.is_explored and neighbor.is_obstacle:
                        continue

                    # Diagonal distance is 1.414, straight is 1
                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    new_distance = cell.distance_to_uncovered + move_cost

                    if new_distance < neighbor.distance_to_uncovered:
                        neighbor.distance_to_uncovered = new_distance
                        queue.append(neighbor)

    def _get_next_cell_improved(self):
        """Improved wavefront: select direction guided to uncovered area"""
        # Compute distance field
        self._compute_distance_field()

        candidates = []

        # Check 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = self.current_cell.row + dr, self.current_cell.col + dc
                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue

                neighbor = self.grid[nr][nc]

                # Skip known obstacles
                if neighbor.is_explored and neighbor.is_obstacle:
                    continue

                # Priority calculation
                if not neighbor.is_covered:
                    priority = (0, 0, neighbor.visit_count)
                else:
                    priority = (1, neighbor.distance_to_uncovered, neighbor.visit_count)

                candidates.append((priority, neighbor))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return None

    def _find_nearest_uncovered(self):
        """BFS to find nearest uncovered cell"""
        visited = set()
        queue = deque([self.current_cell])
        visited.add((self.current_cell.row, self.current_cell.col))

        while queue:
            cell = queue.popleft()

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = cell.row + dr, cell.col + dc
                    if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                        continue
                    if (nr, nc) in visited:
                        continue

                    neighbor = self.grid[nr][nc]
                    visited.add((nr, nc))

                    if neighbor.is_explored and neighbor.is_obstacle:
                        continue

                    if not neighbor.is_covered:
                        return neighbor

                    queue.append(neighbor)

        return None

    def _astar_plan(self, target_cell):
        """A* path planning"""

        def heuristic(c1, c2):
            return abs(c1.row - c2.row) + abs(c1.col - c2.col)

        open_set = []
        heapq.heappush(open_set, (0, id(self.current_cell), self.current_cell))

        came_from = {}
        g_score = {(self.current_cell.row, self.current_cell.col): 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == target_cell:
                path = []
                while (current.row, current.col) in came_from:
                    path.append(current)
                    current = came_from[(current.row, current.col)]
                path.reverse()
                return path

            current_pos = (current.row, current.col)

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = current.row + dr, current.col + dc
                    if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                        continue

                    neighbor = self.grid[nr][nc]

                    if neighbor != target_cell:
                        if neighbor.is_explored and neighbor.is_obstacle:
                            continue

                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    tentative_g = g_score[current_pos] + move_cost

                    neighbor_pos = (nr, nc)
                    if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                        came_from[neighbor_pos] = current
                        g_score[neighbor_pos] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, target_cell)
                        heapq.heappush(open_set, (f_score, id(neighbor), neighbor))

        return []

    def step(self):
        """Execute one step"""
        # Scan and update map
        self._scan_and_update()

        # Select next step
        next_cell = self._get_next_cell_improved()
        self.use_astar = False

        if next_cell is None:
            # Dead end, activate A*
            self.use_astar = True
            target = self._find_nearest_uncovered()

            if target is None:
                return False

            path = self._astar_plan(target)
            if not path:
                return False

            next_cell = path[0]

        if next_cell is None:
            return False

        # Move to next cell
        self.current_cell = next_cell
        self.current_pos = (int(next_cell.center_y), int(next_cell.center_x))
        self.raw_path.append(self.current_pos)

        next_cell.is_covered = True
        next_cell.visit_count += 1

        return True

    def smooth_final_path(self, method='b_spline'):
        """
        平滑最终路径
        Args:
            method: 平滑方法 'b_spline' 或 'simple'
        """
        print(f"\n开始路径平滑...")
        print(f"   - 原始路径点数: {len(self.raw_path)}")

        if method == 'b_spline':
            self.smooth_path = self.path_smoother.smooth_path_b_spline(self.raw_path, s=0.1)
        else:
            self.smooth_path = self.path_smoother.smooth_path_simple(self.raw_path, window_size=3)

        # 更新当前使用的路径为平滑后的路径
        self.path = self.smooth_path
        print(f"   - 平滑后路径点数: {len(self.smooth_path)}")

    def run_coverage(self, max_steps=5000, target_coverage=0.95, enable_smoothing=True):
        """Run coverage planning"""
        print(f"\nStart coverage planning...")
        print(f"   Strategy: Improved Wavefront (distance field guided) + A* backtrack")

        step = 0
        while step < max_steps:
            if not self.step():
                print(f"   Cannot continue planning")
                break

            step += 1

            # Statistics
            if step % 100 == 0:
                explored = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                               if self.grid[r][c].is_explored and not self.grid[r][c].is_obstacle)
                covered = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                              if self.grid[r][c].is_covered)
                total_visits = sum(self.grid[r][c].visit_count for r in range(self.grid_rows)
                                   for c in range(self.grid_cols))

                coverage = covered / explored if explored > 0 else 0
                repeat = (total_visits - covered) / total_visits if total_visits > 0 else 0

                print(f"   Steps: {step}, Explored: {explored}, Covered: {covered}, "
                      f"Coverage: {coverage * 100:.1f}%, Repeat: {repeat * 100:.1f}%")

            # Check if target coverage reached
            if step % 50 == 0:
                explored = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                               if self.grid[r][c].is_explored and not self.grid[r][c].is_obstacle)
                covered = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                              if self.grid[r][c].is_covered)
                coverage = covered / explored if explored > 0 else 0

                if coverage >= target_coverage:
                    print(f"\nTarget coverage {target_coverage * 100:.1f}% reached!")
                    break

        # Final statistics
        explored = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                       if self.grid[r][c].is_explored and not self.grid[r][c].is_obstacle)
        covered = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                      if self.grid[r][c].is_covered)
        total_visits = sum(self.grid[r][c].visit_count for r in range(self.grid_rows)
                           for c in range(self.grid_cols))

        coverage = covered / explored if explored > 0 else 0
        repeat = (total_visits - covered) / total_visits if total_visits > 0 else 0

        print(f"\nPlanning complete:")
        print(f"   - Total steps: {len(self.raw_path)}")
        print(f"   - Explored cells: {explored}")
        print(f"   - Covered cells: {covered}")
        print(f"   - Coverage rate: {coverage * 100:.2f}%")
        print(f"   - Repeat rate: {repeat * 100:.2f}%")

        # 路径平滑
        if enable_smoothing and len(self.raw_path) > 10:
            self.smooth_final_path(method='b_spline')


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
        self.smooth_path_line1 = None  # 平滑路径线
        self.smooth_path_line2 = None
        self.radar_circle1 = None
        self.radar_circle2 = None
        self.known_map = np.ones((self.env.height, self.env.width, 3)) * 0.8
        # 新增：存储历史雷达位置 (x, y)
        self.radar_history = []
        self.radar_history_step = 1

    def setup_visualization(self):
        """设置可视化界面"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 左侧：真实环境
        self._plot_environment(self.ax1, self.env, "true env")

        # 右侧：已知地图
        self._plot_known_map(self.ax2, "known map and searching path")

        # 初始化动画元素
        self.robot_pos1, = self.ax1.plot([], [], 'ro', markersize=15, alpha=0.9, zorder=5, label='robot')
        self.robot_pos2, = self.ax2.plot([], [], 'ro', markersize=15, alpha=0.9, zorder=5, label='robot')

        self.path_line1, = self.ax1.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=3, label='raw path')
        self.path_line2, = self.ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=3, label='raw path')

        # 平滑路径线（红色）
        self.smooth_path_line1, = self.ax1.plot([], [], 'b-', linewidth=3, alpha=0.8, zorder=3, label='smooth path')
        self.smooth_path_line2, = self.ax2.plot([], [], 'b-', linewidth=3, alpha=0.8, zorder=3, label='smooth path')

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
        """绘制已知地图 - 适配ImprovedWavefrontPlanner"""
        # 创建已知地图可视化
        self.known_map = np.ones((self.env.height, self.env.width, 3)) * [0.8, 1.0, 0.8]

        # 绘制网格单元状态
        for row in range(self.planner.grid_rows):
            for col in range(self.planner.grid_cols):
                cell = self.planner.grid[row][col]
                y_start = int(cell.center_y - cell.cell_size / 2)
                x_start = int(cell.center_x - cell.cell_size / 2)
                y_end = int(cell.center_y + cell.cell_size / 2)
                x_end = int(cell.center_x + cell.cell_size / 2)

                # 确保在边界内
                y_start = max(0, min(y_start, self.env.height - 1))
                y_end = max(0, min(y_end, self.env.height - 1))
                x_start = max(0, min(x_start, self.env.width - 1))
                x_end = max(0, min(x_end, self.env.width - 1))


                if cell.is_explored and cell.is_obstacle:
                    # 黑色：障碍物
                    self.known_map[y_start:y_end, x_start:x_end] = [0.1, 0.1, 0.1]
                elif cell.is_covered:
                    # 绿色：已覆盖
                    self.known_map[y_start:y_end, x_start:x_end] = [0.8, 1.0, 0.8]

        ax.imshow(self.known_map, extent=[0, self.env.width, 0, self.env.height],
                  origin='lower', alpha=0.9, zorder=0)

        # 绘制起点
        start_circle = patches.Circle((self.env.start_pos[1], self.env.start_pos[0]), 1.0,
                                      color='green', alpha=0.8, zorder=4)
        ax.add_patch(start_circle)
        ax.text(self.env.start_pos[1], self.env.start_pos[0], 'S',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=5)

        # 绘制边界点（frontier）的适配
        frontier_points = []
        for row in range(self.planner.grid_rows):
            for col in range(self.planner.grid_cols):
                cell = self.planner.grid[row][col]
                if cell.is_explored and not cell.is_obstacle:
                    # 检查邻居是否有未探索区域
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = row + dr, col + dc
                            if (0 <= nr < self.planner.grid_rows and
                                    0 <= nc < self.planner.grid_cols):
                                neighbor = self.planner.grid[nr][nc]
                                if not neighbor.is_explored:
                                    frontier_points.append((cell.center_y, cell.center_x))
                                    break

        ax.set_xlim(-1, self.env.width)
        ax.set_ylim(-1, self.env.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # 添加图例
        legend_elements = [
            patches.Patch(facecolor='grey', alpha=0.8, label='Unknown area'),
            patches.Patch(facecolor='black', alpha=0.8, label='Obstacle'),
            patches.Patch(facecolor=[0.8, 1.0, 0.8], alpha=0.8, label='Covered'),
            patches.Patch(facecolor='white', alpha=0.8, label='Explored free'),
            patches.Patch(facecolor='red', alpha=0.6, label='Boundary point'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    def animate_exploration(self, interval=50, show_smooth_path=True):
        """创建探索过程动画"""
        if not self.planner.path:
            print("请先执行路径规划!")
            return

        self.setup_visualization()

        def init():
            self.robot_pos1.set_data([], [])
            self.robot_pos2.set_data([], [])
            self.smooth_path_line1.set_data([], [])
            self.smooth_path_line2.set_data([], [])
            self.radar_circle1.center = (0, 0)
            self.radar_circle2.center = (0, 0)
            return (self.robot_pos1, self.robot_pos2,
                    self.smooth_path_line1, self.smooth_path_line2,
                    self.radar_circle1, self.radar_circle2)

        def update(frame):
            if frame >= len(self.planner.path):
                return (self.robot_pos1, self.robot_pos2,
                        self.smooth_path_line1, self.smooth_path_line2,
                        self.radar_circle1, self.radar_circle2)

            current_pos = self.planner.path[frame]
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
            if len(self.radar_history) > 2000:
                self.radar_history.pop(0)

            smooth_path_x = []
            smooth_path_y = []

            # 更新原始路径线

            # 更新平滑路径线（如果存在且需要显示）
            if show_smooth_path and hasattr(self.planner, 'smooth_path') and self.planner.smooth_path:
                # 计算当前帧对应的平滑路径点
                if frame < len(self.planner.smooth_path):
                    # 逐步显示平滑路径：从起点到当前帧位置
                    current_smooth_segment = self.planner.smooth_path[:frame + 1]
                    smooth_path_x = [p[1] for p in current_smooth_segment]
                    smooth_path_y = [p[0] for p in current_smooth_segment]
                else:
                    # 如果帧数超过平滑路径长度，显示完整路径
                    smooth_path_x = [p[1] for p in self.planner.smooth_path]
                    smooth_path_y = [p[0] for p in self.planner.smooth_path]

                self.smooth_path_line1.set_data(smooth_path_x, smooth_path_y)
                self.smooth_path_line2.set_data(smooth_path_x, smooth_path_y)

            # 更新右侧地图（每5帧更新一次以提高性能）
            if frame % 5 == 0 or frame == len(self.planner.path) - 1:
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

                # 创建总的雷达掩码
                total_radar_mask = np.zeros((self.env.height, self.env.width), dtype=bool)

                # 雷达扫过区域显示为已知地图
                for (hx, hy) in self.radar_history:
                    distance = np.sqrt((y_grid - hy) ** 2 + (x_grid - hx) ** 2)
                    radar_mask = distance <= radar_range
                    unknown_map_rgba[radar_mask, 3] = 0
                    total_radar_mask = total_radar_mask | radar_mask  # 合并所有雷达扫描区域

                # 计算灰色区域（未扫描区域）的比例
                total_cells = self.env.height * self.env.width
                scanned_cells = np.sum(total_radar_mask)
                unscanned_cells = total_cells - scanned_cells
                scanned_ratio = 1.0 - unscanned_cells / total_cells

                self.ax2.imshow(
                    unknown_map_rgba,
                    extent=[0, self.env.height, 0, self.env.width],
                    origin='lower',
                    zorder=1  # 上层（覆盖底层，但透明区域露出底层）
                )

                # 重新创建右侧动态元素
                self.robot_pos2, = self.ax2.plot([x_pos], [y_pos], 'ro', markersize=15, alpha=0.9, zorder=5)

                if show_smooth_path and smooth_path_x and smooth_path_y:
                    if frame < len(self.planner.smooth_path):
                        current_smooth_segment = self.planner.smooth_path[:frame + 1]
                        smooth_display_x = [p[1] for p in current_smooth_segment]
                        smooth_display_y = [p[0] for p in current_smooth_segment]
                    else:
                        smooth_display_x = [p[1] for p in self.planner.smooth_path]
                        smooth_display_y = [p[0] for p in self.planner.smooth_path]

                    self.smooth_path_line2, = self.ax2.plot(smooth_display_x, smooth_display_y, 'b-',
                                                            linewidth=3, alpha=0.8, zorder=4)

                # 绘制当前雷达范围
                self.radar_circle2 = patches.Circle(
                    (x_pos, y_pos), self.planner.radar.max_range,
                    fill=False, edgecolor='red', linestyle='--',  # 当前范围用深绿色
                    linewidth=2, alpha=0.6, zorder=4  # 层级高于历史痕迹
                )
                self.ax2.add_patch(self.radar_circle2)
                self._plot_known_map(self.ax2,
                                     f"known map (process: {scanned_ratio * 100:.1f}%)")

                progress = (frame + 1) / len(self.planner.path) * 100
                self.ax1.set_title(f'True_Env (Process: {scanned_ratio * 100:.1f}%)', fontsize=12, fontweight='bold')

            return (self.robot_pos1, self.robot_pos2,
                    self.smooth_path_line1, self.smooth_path_line2,
                    self.radar_circle1, self.radar_circle2)

        anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.planner.path),
            init_func=init, interval=interval, blit=False, repeat=False
        )

        plt.show()
        return anim

    def clear_radar_history(self):
        self.radar_history.pop(0)


def main():
    """主函数：创建并可视化森林环境"""
    print("🌲 森林环境可视化器")
    print("=" * 50)

    # 创建环境可视化器
    env = ForestEnvironmentVisualizer(width=70, height=70, seed=40)
    # 波前法规划
    planner = ImprovedWavefrontPlanner(env, radar_range=5, cell_size=2)
    # 执行覆盖规划
    planner.run_coverage(max_steps=3000, target_coverage=0.99, enable_smoothing=True)

    # 可视化
    visualizer = UnknownMapVisualizer(env, planner)
    print("\n🎬 开始探索动画演示...")
    anim = visualizer.animate_exploration(interval=20, show_smooth_path=True)

    return env, planner, anim


if __name__ == "__main__":
    main()