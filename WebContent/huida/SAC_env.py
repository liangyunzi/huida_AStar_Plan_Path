import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import noise
import random
from enum import Enum
import matplotlib as mpl
import matplotlib.font_manager as fm


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


class ForestEnvironmentVisualizer:
    """森林环境可视化器"""

    def __init__(self, width=100, height=100, seed=42):
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
#        self.target_pos = [height * 0.95, width * 0.95]  # 右上角

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
        self._generate_animals()
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
#        target_dist = np.sqrt((y - self.target_pos[0]) ** 2 + (x - self.target_pos[1]) ** 2)

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

        # 清理终点
        # for dy in range(-clear_radius, clear_radius + 1):
        #     for dx in range(-clear_radius, clear_radius + 1):
        #         distance = np.sqrt(dy ** 2 + dx ** 2)
        #         if distance <= clear_radius:
        #             y = int(self.target_pos[0]) + dy
        #             x = int(self.target_pos[1]) + dx
        #             if 0 <= y < self.height and 0 <= x < self.width:
        #                 self.static_grid[y, x] = 0
        #                 self.static_terrain_type[y, x] = TerrainType.GRASS.value

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
        ax.text(self.start_pos[1], self.start_pos[0], 'START', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)

        # 终点 (红色)
        # target_circle1 = patches.Circle((self.target_pos[1], self.target_pos[0]), 3.5,
        #                                 color='red', alpha=0.3, zorder=3)
        # target_circle2 = patches.Circle((self.target_pos[1], self.target_pos[0]), 2.5,
        #                                 color='red', alpha=0.5, zorder=3)
        # target_circle3 = patches.Circle((self.target_pos[1], self.target_pos[0]), 1.8,
        #                                 color='red', alpha=0.8, zorder=3)
        #
        # ax.add_patch(target_circle1)
        # ax.add_patch(target_circle2)
        # ax.add_patch(target_circle3)
        # ax.text(self.target_pos[1], self.target_pos[0], 'TARGET', ha='center', va='center',
        #         fontsize=8, fontweight='bold', color='white', zorder=4)

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


def main():
    """主函数：创建并可视化森林环境"""
    print("🌲 森林环境可视化器")
    print("=" * 50)

    # 创建环境可视化器
    visualizer = ForestEnvironmentVisualizer(
        width=100,
        height=100,
        seed=42
    )

    # 可视化环境
    visualizer.visualize(save_path="forest_environment_layout.png")



if __name__ == "__main__":
    main()