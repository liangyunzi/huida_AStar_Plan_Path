# Learning Date : 2025/12/22
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

    def _calculate_curvature(self, path):
        """
        计算路径上每个点的曲率
        Args:
            path: 路径点列表 [(y1, x1), (y2, x2), ...]
        Returns:
            curvature: 每个点的曲率值
        """
        if len(path) < 3:
            return np.zeros(len(path))

        curvature = np.zeros(len(path))

        for i in range(1, len(path) - 1):
            # 前一个点、当前点、后一个点
            p0 = np.array(path[i - 1])
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])

            # 计算向量
            v1 = p1 - p0
            v2 = p2 - p1

            # 计算角度变化（转弯程度）
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 > 0 and norm_v2 > 0:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                # 限制在[-1, 1]范围内，防止浮点误差
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                # 角度变化（弧度）
                angle_change = np.arccos(cos_angle)
                curvature[i] = angle_change

        # 首尾点复制相邻点的曲率
        if len(path) > 1:
            curvature[0] = curvature[1]
            curvature[-1] = curvature[-2]

        return curvature

    def _detect_turn_regions(self, path, curvature_threshold=0.3):
        """
        检测转弯区域
        Args:
            path: 路径点
            curvature_threshold: 曲率阈值，大于此值被认为是转弯
        Returns:
            turn_regions: 转弯区域的索引列表 [(start1, end1), (start2, end2), ...]
        """
        curvature = self._calculate_curvature(path)

        turn_regions = []
        in_turn = False
        start_idx = 0

        # 添加扩展区域，确保平滑过渡
        extend_len = 2  # 在转弯区域前后扩展的点数

        for i in range(len(path)):
            if curvature[i] > curvature_threshold and not in_turn:
                in_turn = True
                start_idx = max(0, i - extend_len)
            elif curvature[i] <= curvature_threshold and in_turn:
                in_turn = False
                end_idx = min(len(path), i + extend_len)
                if end_idx - start_idx >= 4:  # 至少需要4个点才能进行B样条
                    turn_regions.append((start_idx, end_idx))

        # 处理最后一个区域
        if in_turn:
            end_idx = min(len(path), len(path) + extend_len)
            if end_idx - start_idx >= 4:
                turn_regions.append((start_idx, end_idx))

        # 合并重叠的区域
        if turn_regions:
            merged_regions = []
            current_start, current_end = turn_regions[0]

            for start, end in turn_regions[1:]:
                if start <= current_end:  # 重叠
                    current_end = max(current_end, end)
                else:
                    merged_regions.append((current_start, current_end))
                    current_start, current_end = start, end

            merged_regions.append((current_start, current_end))
            turn_regions = merged_regions

        return turn_regions

    def smooth_path_selective(self, path, s=0.3, curvature_threshold=0.3):
        """
        选择性平滑路径：只在转弯区域使用B样条，直线区域保持原样或轻微平滑
        Args:
            path: 原始路径点列表
            s: B样条平滑因子
            curvature_threshold: 曲率阈值，决定什么是转弯
        Returns:
            smoothed_path: 平滑后的路径
        """
        if len(path) < 4:
            return self.smooth_path_simple(path)  # 点太少，使用简单平滑

        # 检测转弯区域
        turn_regions = self._detect_turn_regions(path, curvature_threshold)

        if not turn_regions:
            print(f"   - 未检测到明显转弯，使用简单平滑")
            return self.smooth_path_simple(path, window_size=3)

        # 创建结果路径副本
        smoothed_path = path.copy()

        # 对每个转弯区域进行B样条平滑
        for region_idx, (start, end) in enumerate(turn_regions):
            print(f"   - 平滑转弯区域 {region_idx + 1}: 点 {start}-{end} ({end - start}个点)")

            # 提取转弯区域
            turn_region = path[start:end]

            if len(turn_region) >= 4:
                # 对转弯区域进行B样条平滑
                smoothed_region = self.smooth_path_b_spline(turn_region, s)

                # 将平滑后的区域替换回原路径
                # 保留首尾点以保持连接平滑
                if start > 0 and end < len(path):
                    # 内部区域：保留第一个和最后一个点
                    smoothed_path[start + 1:end - 1] = smoothed_region[1:-1]
                elif start == 0:
                    # 起始区域：保留最后一个点
                    smoothed_path[start:end - 1] = smoothed_region[:-1]
                elif end == len(path):
                    # 结束区域：保留第一个点
                    smoothed_path[start + 1:end] = smoothed_region[1:]

        # 对整个路径进行轻微的整体平滑，确保过渡自然
        final_path = self._blend_smooth_regions(smoothed_path, turn_regions)

        print(f"   - 选择性平滑完成: {len(path)} → {len(final_path)} 个点")
        print(f"   - 检测到 {len(turn_regions)} 个转弯区域")
        return final_path

    def _blend_smooth_regions(self, path, turn_regions, blend_window=3):
        """
        混合平滑区域，确保过渡自然
        Args:
            path: 路径
            turn_regions: 转弯区域
            blend_window: 混合窗口大小
        Returns:
            blended_path: 混合后的路径
        """
        if not turn_regions or len(path) < blend_window * 2:
            return path

        blended_path = list(path)  # 转换为列表以便修改

        # 对每个转弯区域的边界进行混合
        for start, end in turn_regions:
            # 混合起始边界
            if start > 0:
                blend_start = max(0, start - blend_window)
                for i in range(blend_start, start):
                    if i < len(blended_path) - 1:
                        alpha = (i - blend_start) / (start - blend_start)
                        # 线性混合
                        y1, x1 = blended_path[i]
                        y2, x2 = path[i]
                        blended_path[i] = (
                            y1 * alpha + y2 * (1 - alpha),
                            x1 * alpha + x2 * (1 - alpha)
                        )

            # 混合结束边界
            if end < len(path):
                blend_end = min(len(path), end + blend_window)
                for i in range(end, blend_end):
                    if i < len(blended_path):
                        alpha = (i - end) / (blend_end - end)
                        y1, x1 = blended_path[i]
                        y2, x2 = path[i]
                        blended_path[i] = (
                            y1 * (1 - alpha) + y2 * alpha,
                            x1 * (1 - alpha) + x2 * alpha
                        )

        return blended_path

    def smooth_path_b_spline(self, path, s=0.3):
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
            num_points = max(15, int(len(path) * self.smooth_points_density))
            u_new = np.linspace(0, 1, num_points)

            # 计算平滑后的路径
            x_smooth, y_smooth = splev(u_new, tck)

            # 重新组合为(y, x)格式
            smoothed_path = list(zip(y_smooth, x_smooth))

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


class AStarSolver:
    """改进后的 A* 算法：处理空心障碍物并支持软膨胀"""

    def __init__(self, rows, cols, robot_radius=2):
        self.rows = rows
        self.cols = cols
        self.robot_radius = robot_radius  # 安全避障半径

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_cell_cost(self, node, grid_map):
        """
        计算进入该格子的总代价
        """
        r, c = node
        cell_status = grid_map[r, c]

        # 1. 绝对障碍物：返回极大值
        if cell_status == 2:
            return float('inf')

        # 2. 基础代价
        # status 1 (Free) = 1.0
        # status 0 (Unknown) = 10.0 (惩罚项，防止钻进空心的障碍物内部)
        base_cost = 1.0 if cell_status == 1 else 10.0

        # 3. 软膨胀代价 (靠近障碍物的惩罚)
        inflation_penalty = 0
        if self.robot_radius > 0:
            margin = int(self.robot_radius)
            # 检查周围的一个小窗口
            r_min, r_max = max(0, r - margin), min(self.rows, r + margin + 1)
            c_min, c_max = max(0, c - margin), min(self.cols, c + margin + 1)

            region = grid_map[r_min:r_max, c_min:c_max]
            if np.any(region == 2):
                # 附近有障碍物，增加额外代价，离得越近代价越高（这里简化为固定惩罚）
                inflation_penalty = 15.0

        return base_cost + inflation_penalty

    def plan(self, start, goal, grid_map):
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols):
            return []

        # 即使目标点在膨胀层内，我们也允许规划（只要不是直接在障碍物上）
        if grid_map[goal] == 2:
            return []

        open_set = []
        # (f_score, current_node)
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dy, dx in neighbors:
                neighbor = (current[0] + dy, current[1] + dx)

                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue

                # 计算这一步的移动权重
                step_cost = self.get_cell_cost(neighbor, grid_map)

                # 如果是绝对碰撞，跳过
                if step_cost == float('inf'):
                    continue

                # 基础移动代价（欧几里得距离）
                move_dist = 1.414 if abs(dy) + abs(dx) == 2 else 1.0

                # 总代价 = 当前G + 距离 * 该格子的危险权重
                tentative_g_score = g_score[current] + move_dist * step_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []


class FALCONPlanner:
    def __init__(self, environment, radar_range=8, cell_size=1):
        """
        初始化 FALCON 规划器
        """
        self.env = environment
        self.grid_rows = environment.height
        self.grid_cols = environment.width
        self.cell_size = cell_size

        # 1. 初始化雷达
        self.radar = RadarSensor(max_range=radar_range)

        # 2. 初始化双重地图表示
        # (a) 给可视化用的 GridCell 对象列表
        self.grid = [[GridCell(r, c, r, c, cell_size)
                      for c in range(self.grid_cols)]
                     for r in range(self.grid_rows)]

        # (b) 给算法用的 Numpy 数组 (0: Unknown, 1: Free, 2: Obstacle)
        # 初始化全为 0 (Unknown)
        self.np_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        # 3. 初始化工具
        self.a_star = AStarSolver(self.grid_rows, self.grid_cols, robot_radius=1.5)
        self.smoother = PathSmoother()
        self.vals = {'unknown': 0, 'free': 1, 'obs': 2}

        # FALCON 参数
        self.cluster_eps = 5.0
        self.cluster_min_samples = 3

        # 路径记录
        self.path = []  # 历史走过的所有点 (用于动画)
        self.smooth_path = []  # 当前规划的平滑路径
        self.current_pos = tuple(map(int, environment.start_pos))

        # 初始状态
        self.path.append(self.current_pos)
        self._update_map_at(self.current_pos[0], self.current_pos[1], status='free')

    def _update_map_at(self, r, c, status):
        """同时更新 GridCell 和 NumpyGrid"""
        if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
            return

        # 更新 GridCell (可视化用)
        cell = self.grid[r][c]
        cell.is_explored = True

        # 更新 NumpyGrid (算法用)
        if status == 'obs':
            cell.is_obstacle = True
            self.np_grid[r, c] = self.vals['obs']
        elif status == 'free':
            cell.is_obstacle = False
            # 只有当原本不是障碍物时才标记为 Free (避免覆盖)
            if self.np_grid[r, c] != self.vals['obs']:
                self.np_grid[r, c] = self.vals['free']

        # 标记覆盖 (对于覆盖率统计)
        cell.is_covered = True

    def _perform_scan(self):
        """执行雷达扫描并更新地图"""
        obs, free = self.radar.scan(self.current_pos, self.env)

        for (r, c) in free:
            self._update_map_at(r, c, 'free')

        for (r, c) in obs:
            self._update_map_at(r, c, 'obs')

    def _get_frontiers(self):
        """FALCON: 提取边界点 (numpy 优化版)"""
        # 只有 Free 的点才有资格作为边界的 "这一侧"
        free_mask = (self.np_grid == self.vals['free'])

        # 定义卷积核查找 4 邻域内的 Unknown
        # 这里用简单的切片操作代替卷积以减少依赖
        # 边界点定义：本身是 Free，且上下左右至少有一个是 Unknown

        rows, cols = self.np_grid.shape
        frontiers = []

        # 获取所有 Free 点的坐标
        free_indices = np.argwhere(free_mask)

        if len(free_indices) == 0:
            return np.array([])

        # 遍历 Free 点 (量大时可优化，但对于 100x100 地图尚可)
        for r, c in free_indices:
            is_frontier = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self.np_grid[nr, nc] == self.vals['unknown']:
                        is_frontier = True
                        break
            if is_frontier:
                frontiers.append((r, c))

        return np.array(frontiers)

    def _decompose_and_cluster(self, frontiers):
        """FALCON: 空间聚类 (DBSCAN)"""
        if len(frontiers) == 0:
            return []

        # 调用 DBSCAN
        try:
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(frontiers)
        except Exception:
            return []  # 异常处理

        labels = clustering.labels_
        zones = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1: continue  # 噪声点忽略

            points = frontiers[labels == label]
            if len(points) == 0: continue

            centroid = np.mean(points, axis=0).astype(int)
            zones.append({
                'id': label,
                'centroid': tuple(centroid),
                'points': points,
                'score': len(points)  # 区域大小作为评分
            })

        return zones

    def _solve_global_goal(self, zones):
        """FALCON: 全局目标选择 (Greedy ATSP)"""
        if not zones: return None

        best_zone = None
        min_cost = float('inf')

        for zone in zones:
            dist = np.sqrt((zone['centroid'][0] - self.current_pos[0]) ** 2 +
                           (zone['centroid'][1] - self.current_pos[1]) ** 2)

            # 代价函数：距离越短越好，区域包含的边界点越多越好
            cost = dist / (zone['score'] + 0.1)

            if cost < min_cost:
                min_cost = cost
                best_zone = zone

        return best_zone

    def _solve_local_path(self, target_zone):
        """FALCON: 局部优化与寻路"""
        # 从 Zone 中选择一个最佳进入点 (离机器人最近的点)
        candidate_points = target_zone['points']

        # 简单的最近邻
        dists = np.linalg.norm(candidate_points - np.array(self.current_pos), axis=1)
        best_idx = np.argmin(dists)
        target_pt = tuple(candidate_points[best_idx])

        # A* 寻路
        path = self.a_star.plan(self.current_pos, target_pt, self.np_grid)
        return path

    def run_coverage(self, max_steps=2000, target_coverage=0.98, enable_smoothing=True):
        """
        [主接口] 执行完整的覆盖探测循环
        """
        print("🚀 开始 FALCON 覆盖路径规划...")
        steps = 0

        while steps < max_steps:
            # 1. 扫描环境
            self._perform_scan()

            # 2. 检查覆盖率 (可选，这里简化逻辑)
            # if coverage > target: break
            # 计算已探索（非 0）的格子数量
            total_cells = self.grid_rows * self.grid_cols
            explored_count = np.sum(self.np_grid != self.vals['unknown'])
            current_coverage = explored_count / total_cells

            if current_coverage >= target_coverage:
                print(f"🎯 已达到目标覆盖率 ({current_coverage * 100:.2f}%)，停止探索。")
                break

            # 3. 规划逻辑
            # 获取边界
            frontiers = self._get_frontiers()

            if len(frontiers) < 4:
                print("✅ 没有更多边界，探索完成。")
                break

            # 空间聚类
            zones = self._decompose_and_cluster(frontiers)

            current_plan_path = []

            if not zones:
                # Fallback: 如果聚类失败(点太散)，直接去最近的边界点
                dists = np.linalg.norm(frontiers - np.array(self.current_pos), axis=1)
                nearest_pt = tuple(frontiers[np.argmin(dists)])
                current_plan_path = self.a_star.plan(self.current_pos, nearest_pt, self.np_grid)
            else:
                # 全局规划 + 局部规划
                target_zone = self._solve_global_goal(zones)
                if target_zone:
                    current_plan_path = self._solve_local_path(target_zone)

            # 4. 执行移动 (如果规划出路径)
            if not current_plan_path:
                print("⚠️ 无法规划路径，尝试随机移动摆脱困境...")
                # 简单的随机游走策略
                attempts = 0
                moved = False
                while attempts < 10:
                    rr = self.current_pos[0] + random.randint(-2, 2)
                    cc = self.current_pos[1] + random.randint(-2, 2)
                    if (0 <= rr < self.grid_rows and 0 <= cc < self.grid_cols and
                            self.np_grid[rr, cc] != self.vals['obs']):
                        path_seg = self.a_star.plan(self.current_pos, (rr, cc), self.np_grid)
                        if path_seg:
                            current_plan_path = path_seg
                            moved = True
                            break
                    attempts += 1
                if not moved:
                    break  # 彻底卡死

            # 5. 记录与平滑
            if current_plan_path:
                # 为了动画效果，我们只走路径的一小段，然后重新扫描 (Receding Horizon)
                # 这样可以模拟实时发现障碍物
                step_size = min(len(current_plan_path), 5)  # 每次规划只走前5步
                execute_path = current_plan_path[:step_size]

                for pt in execute_path:
                    self.current_pos = pt
                    self.path.append(pt)
                    # 每一步都需要扫描，确保遇到动态障碍物能停下(简单模拟)
                    self._perform_scan()
                    steps += 1

                # 存储当前的完整规划路径用于可视化展示 (平滑)
                if enable_smoothing:
                    self.smooth_path.extend(self.smoother.smooth_path_simple(execute_path))
                else:
                    self.smooth_path.extend(execute_path)
            else:
                break

            if steps % 100 == 0:
                print(f"Step {steps}: Frontiers={len(frontiers)}, Zones={len(zones)}")

        print(f"🏁 探索结束，总步数: {steps}\n")
        print(f"总平滑步数: {len(self.smooth_path)}")

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

        self.path_line1, = self.ax1.plot([], [], 'b-', linewidth=3, alpha=0.8, zorder=3, label='raw path')
        self.path_line2, = self.ax2.plot([], [], 'b-', linewidth=3, alpha=0.8, zorder=3, label='raw path')

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
            if frame >= len(self.planner.smooth_path):
                return (self.robot_pos1, self.robot_pos2,
                        self.smooth_path_line1, self.smooth_path_line2,
                        self.radar_circle1, self.radar_circle2)

            current_pos = self.planner.smooth_path[frame]
            x_pos = current_pos[1]
            y_pos = current_pos[0]

            # 更新机器人位置
            self.robot_pos1.set_data([x_pos], [y_pos])


            # 更新雷达范围
            self.radar_circle1.center = (x_pos, y_pos)


            self.radar_history.append((x_pos, y_pos))  # 存储当前雷达中心
            # 控制历史记录数量，避免内存占用过大
            if len(self.radar_history) > 3000:
                self.radar_history.pop(0)

            smooth_path_x = []
            smooth_path_y = []

            # 更新原始路径线
            if frame > 0:
                smooth_path_x = [p[1] for p in self.planner.smooth_path[:frame + 1]]
                smooth_path_y = [p[0] for p in self.planner.smooth_path[:frame + 1]]
                self.path_line1.set_data(smooth_path_x, smooth_path_y)
                self.path_line2.set_data(smooth_path_x, smooth_path_y)



            # 更新平滑路径线（如果存在且需要显示）
            # if show_smooth_path and hasattr(self.planner, 'smooth_path') and self.planner.smooth_path:
            #     # 计算当前帧对应的平滑路径点
            #     if frame < len(self.planner.smooth_path):
            #         # 逐步显示平滑路径：从起点到当前帧位置
            #         current_smooth_segment = self.planner.smooth_path[:frame + 1]
            #         smooth_path_x = [p[1] for p in current_smooth_segment]
            #         smooth_path_y = [p[0] for p in current_smooth_segment]
            #     else:
            #         # 如果帧数超过平滑路径长度，显示完整路径
            #         smooth_path_x = [p[1] for p in self.planner.smooth_path]
            #         smooth_path_y = [p[0] for p in self.planner.smooth_path]
            #
            #     self.smooth_path_line1.set_data(smooth_path_x, smooth_path_y)
            #     self.smooth_path_line2.set_data(smooth_path_x, smooth_path_y)

            # 更新右侧地图（每5帧更新一次以提高性能）
            if frame % 2 == 0 or frame == len(self.planner.smooth_path) - 1:
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
                self.smooth_path_line2, = self.ax2.plot(smooth_path_x, smooth_path_y, 'b-',
                                                        linewidth=3, alpha=0.8, zorder=4)
                self.robot_pos2, = self.ax2.plot([x_pos], [y_pos], 'ro', markersize=15, alpha=0.9, zorder=5)
                self.smooth_path_line2, = self.ax2.plot([x_pos], [y_pos], 'b-',
                                                        linewidth=3, alpha=0.8, zorder=4)

                # if show_smooth_path and smooth_path_x and smooth_path_y:
                #     if frame < len(self.planner.smooth_path):
                #         current_smooth_segment = self.planner.smooth_path[:frame + 1]
                #         smooth_display_x = [p[1] for p in current_smooth_segment]
                #         smooth_display_y = [p[0] for p in current_smooth_segment]
                #     else:
                #         smooth_display_x = [p[1] for p in self.planner.smooth_path]
                #         smooth_display_y = [p[0] for p in self.planner.smooth_path]
                #
                #     self.smooth_path_line2, = self.ax2.plot(smooth_display_x, smooth_display_y, 'b-',
                #                                             linewidth=3, alpha=0.8, zorder=4)

                # 绘制当前雷达范围
                self.radar_circle2 = patches.Circle(
                    (x_pos, y_pos), self.planner.radar.max_range,
                    fill=False, edgecolor='red', linestyle='--',  # 当前范围用深绿色
                    linewidth=2, alpha=0.6, zorder=4  # 层级高于历史痕迹
                )
                self.ax2.add_patch(self.radar_circle2)
                self._plot_known_map(self.ax2, f"known map (process: {scanned_ratio * 100:.1f}%)")

                progress = (frame + 1) / len(self.planner.path) * 100
                self.ax1.set_title(f'True_Env (Process: {scanned_ratio * 100:.1f}%)', fontsize=12, fontweight='bold')

            return (self.robot_pos1, self.robot_pos2,
                    self.smooth_path_line1, self.smooth_path_line2,
                    self.radar_circle1, self.radar_circle2)

        anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.planner.smooth_path),
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

    # 1. 创建环境 (调大一点看起来更爽)
    env = ForestEnvironmentVisualizer(width=100, height=100, seed=40)

    # 2. 初始化 FALCON 规划器 (注意接口变化)
    planner = FALCONPlanner(env, radar_range=10, cell_size=1)

    # 3. 执行覆盖规划 (生成路径数据)
    # 这一步会跑完整个模拟循环，把路径存在 planner.path 里
    planner.run_coverage(max_steps=3000, target_coverage=0.985, enable_smoothing=True)

    # 4. 可视化回放
    visualizer = UnknownMapVisualizer(env, planner)
    print("\n🎬 开始探索动画演示...")
    # interval 越小动画越快
    anim = visualizer.animate_exploration(interval=10, show_smooth_path=True)

    return env, planner, anim

if __name__ == "__main__":
    main()