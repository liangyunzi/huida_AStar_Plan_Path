import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import heapq
from collections import deque
import copy


class AStarCoveragePlanner:
    """基于A*算法的全覆盖路径规划器"""

    def __init__(self, environment):
        self.env = environment
        self.grid = copy.deepcopy(environment.static_grid)
        self.terrain_type = environment.static_terrain_type
        self.height = environment.height
        self.width = environment.width
        self.start_pos = (int(environment.start_pos[0]), int(environment.start_pos[1]))

        # 定义移动方向（8个方向）
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
        ]

        # 地形代价权重
        self.terrain_costs = {
            0: 1.0,  # 草地 - 基础代价
            1: 5.0,  # 树木 - 高代价
            2: 8.0,  # 岩石 - 更高代价
            3: 3.0,  # 泥地/建筑 - 中等代价
            4: 6.0  # 动物 - 高代价（动态障碍物）
        }

        # 规划结果
        self.path = []
        self.visited = set()
        self.coverage_path = []

    def heuristic(self, a, b):
        """启发式函数：曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_movement_cost(self, current_pos, next_pos):
        """获取移动代价"""
        y, x = next_pos

        # 检查边界
        if not (0 <= y < self.height and 0 <= x < self.width):
            return float('inf')

        # 检查障碍物
        if self.grid[y, x] > 0.3:  # 障碍物阈值
            return float('inf')

        # 基础移动代价
        base_cost = 1.0

        # 地形代价
        terrain = self.terrain_type[y, x]
        terrain_cost = self.terrain_costs.get(terrain, 1.0)

        # 高度代价（如果可用）
        height_cost = 0
        if hasattr(self.env, 'height_map'):
            height_cost = self.env.height_map[y, x] * 0.5

        # 对角线移动代价稍高
        dy = abs(next_pos[0] - current_pos[0])
        dx = abs(next_pos[1] - current_pos[1])
        if dy == 1 and dx == 1:
            base_cost *= 1.414  # √2

        total_cost = base_cost * terrain_cost + height_cost
        return total_cost

    def a_star_search(self, start, goal):
        """A*算法搜索最短路径"""
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dy, dx in self.directions:
                neighbor = (current[0] + dy, current[1] + dx)

                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue

                tentative_g_score = g_score[current] + self.get_movement_cost(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # 没有找到路径

    def find_nearest_unvisited(self, current_pos, visited):
        """找到最近的未访问点"""
        min_distance = float('inf')
        nearest_point = None

        # 使用BFS寻找最近的未访问点
        queue = deque([(current_pos, 0)])
        visited_bfs = set([current_pos])

        while queue:
            pos, distance = queue.popleft()

            if pos not in visited and distance < min_distance:
                min_distance = distance
                nearest_point = pos
                if distance < 5:  # 找到较近的点就返回，提高效率
                    break

            # 添加邻居
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (pos[0] + dy, pos[1] + dx)

                if (0 <= neighbor[0] < self.height and
                        0 <= neighbor[1] < self.width and
                        neighbor not in visited_bfs and
                        self.grid[neighbor[0], neighbor[1]] <= 0.3):  # 不是障碍物

                    visited_bfs.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return nearest_point

    def plan_coverage_path(self):
        """规划全覆盖路径"""
        print("开始规划全覆盖路径...")

        visited = set()
        current_pos = self.start_pos
        full_path = [current_pos]
        visited.add(current_pos)

        step_count = 0
        max_steps = self.height * self.width * 2  # 防止无限循环

        while len(visited) < self.height * self.width and step_count < max_steps:
            step_count += 1

            # 找到最近的未访问点
            next_goal = self.find_nearest_unvisited(current_pos, visited)

            if next_goal is None:
                # 所有可达点都已访问
                break

            # 使用A*规划到目标点的路径
            path_to_goal = self.a_star_search(current_pos, next_goal)

            if path_to_goal is None:
                # 无法到达该点，标记为已访问（障碍物）
                visited.add(next_goal)
                continue

            # 将路径添加到完整路径中（跳过第一个点，因为已经是当前位置）
            for point in path_to_goal[1:]:
                if point not in visited:
                    full_path.append(point)
                    visited.add(point)
                current_pos = point

            # 进度显示
            if step_count % 50 == 0:
                coverage_ratio = len(visited) / (self.height * self.width) * 100
                print(f"进度: {coverage_ratio:.1f}%, 已访问: {len(visited)}/{self.height * self.width}")

        self.coverage_path = full_path
        self.visited = visited

        coverage_ratio = len(visited) / (self.height * self.width) * 100
        print(f"路径规划完成!")
        print(f"总步数: {len(full_path)}")
        print(f"覆盖率: {coverage_ratio:.2f}%")
        print(f"访问点数: {len(visited)}")

        return full_path


class CoverageVisualizer:
    """全覆盖路径可视化器"""

    def __init__(self, environment, planner):
        self.env = environment
        self.planner = planner
        self.fig = None
        self.ax = None

    def setup_visualization(self):
        """设置可视化界面"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # 创建坐标网格
        x = np.arange(self.env.width)
        y = np.arange(self.env.height)
        X, Y = np.meshgrid(x, y)

        # 绘制高度等高线背景
        contour_levels = np.linspace(0, 1, 15)
        cs = self.ax.contourf(X, Y, self.env.height_map, levels=contour_levels,
                              cmap='terrain', alpha=0.6, zorder=0)

        # 添加等高线
        self.ax.contour(X, Y, self.env.height_map, levels=contour_levels,
                        colors='gray', alpha=0.3, linewidths=0.5, zorder=1)

        # 绘制地形特征
        terrain_masks = {}
        for terrain_type in [TerrainType.TREE, TerrainType.ROCK, TerrainType.MUD]:
            mask = (self.env.static_terrain_type == terrain_type.value).astype(float)
            smooth_mask = gaussian_filter(mask, sigma=1.0)
            terrain_masks[terrain_type] = smooth_mask

        # 绘制树木
        if np.any(terrain_masks[TerrainType.TREE] > 0.05):
            tree_levels = [0.05, 0.3, 0.6, 1.0]
            tree_colors = ['lightgreen', 'forestgreen', 'darkgreen']
            self.ax.contourf(X, Y, terrain_masks[TerrainType.TREE],
                             levels=tree_levels, colors=tree_colors,
                             alpha=0.8, zorder=2)

        # 绘制岩石
        if np.any(terrain_masks[TerrainType.ROCK] > 0.1):
            self.ax.contourf(X, Y, terrain_masks[TerrainType.ROCK],
                             levels=[0.1, 0.5, 1.0], colors=['lightgray', 'gray'],
                             alpha=0.8, zorder=2)

        # 绘制建筑
        if np.any(terrain_masks[TerrainType.MUD] > 0.1):
            self.ax.contourf(X, Y, terrain_masks[TerrainType.MUD],
                             levels=[0.1, 0.5, 1.0], colors=['burlywood', 'saddlebrown'],
                             alpha=0.7, zorder=2)

        # 绘制起点
        start_circle = patches.Circle((self.env.start_pos[1], self.env.start_pos[0]), 1.0,
                                      color='green', alpha=0.8, zorder=4)
        self.ax.add_patch(start_circle)
        self.ax.text(self.env.start_pos[1], self.env.start_pos[0], 'S',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white', zorder=5)

        self.ax.set_xlim(-1, self.env.width)
        self.ax.set_ylim(-1, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_title('全覆盖路径规划 - A*算法', fontsize=14, fontweight='bold')

        # 添加图例
        legend_elements = [
            patches.Patch(facecolor='forestgreen', alpha=0.8, label='树木'),
            patches.Patch(facecolor='gray', alpha=0.8, label='岩石'),
            patches.Patch(facecolor='saddlebrown', alpha=0.7, label='建筑'),
            patches.Patch(facecolor='green', alpha=0.8, label='起点'),
            patches.Patch(facecolor='red', alpha=0.8, label='机器人位置'),
            patches.Patch(facecolor='blue', alpha=0.3, label='已覆盖区域'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()

        return self.fig, self.ax

    def animate_coverage(self, interval=10):
        """创建覆盖动画"""
        if not self.planner.coverage_path:
            print("请先执行路径规划!")
            return

        fig, ax = self.setup_visualization()

        # 初始化动画元素
        robot_pos, = ax.plot([], [], 'ro', markersize=8, alpha=0.8, zorder=5)
        path_line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.5, zorder=3)
        coverage_area = np.zeros((self.env.height, self.env.width), dtype=bool)
        coverage_image = ax.imshow(coverage_area, cmap='Blues', alpha=0.3,
                                   extent=[0, self.env.width, 0, self.env.height],
                                   zorder=2, origin='lower')

        def init():
            robot_pos.set_data([], [])
            path_line.set_data([], [])
            coverage_image.set_array(coverage_area)
            return robot_pos, path_line, coverage_image

        def update(frame):
            if frame >= len(self.planner.coverage_path):
                return robot_pos, path_line, coverage_image

            # 更新机器人位置
            current_pos = self.planner.coverage_path[frame]
            x_pos = current_pos[1]
            y_pos = current_pos[0]
            robot_pos.set_data([x_pos], [y_pos])

            # 更新路径线
            if frame > 0:
                path_x = [p[1] for p in self.planner.coverage_path[:frame + 1]]
                path_y = [p[0] for p in self.planner.coverage_path[:frame + 1]]
                path_line.set_data(path_x, path_y)

            # 更新覆盖区域
            coverage_area[y_pos, x_pos] = True
            coverage_image.set_array(coverage_area)

            # 更新标题显示进度
            progress = (frame + 1) / len(self.planner.coverage_path) * 100
            ax.set_title(f'全覆盖路径规划 - A*算法 (进度: {progress:.1f}%)',
                         fontsize=14, fontweight='bold')

            return robot_pos, path_line, coverage_image

        # 创建动画
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.planner.coverage_path),
            init_func=init, interval=interval, blit=True, repeat=False
        )

        plt.show()

        return anim


def main():
    """主函数：执行全覆盖路径规划并显示动画"""
    print("🌲 森林环境全覆盖路径规划")
    print("=" * 50)

    # 创建环境
    env = ForestEnvironmentVisualizer(width=60, height=60, seed=42)  # 减小尺寸提高速度

    # 创建路径规划器
    planner = AStarCoveragePlanner(env)

    # 执行全覆盖路径规划
    coverage_path = planner.plan_coverage_path()

    # 可视化结果
    visualizer = CoverageVisualizer(env, planner)

    print("\n🎬 开始动画演示...")
    anim = visualizer.animate_coverage(interval=10)  # 调整间隔控制速度

    # 保存动画（可选）
    # print("保存动画中...")
    # anim.save('coverage_path_animation.gif', writer='pillow', fps=20)
    # print("动画已保存为 coverage_path_animation.gif")

    return env, planner, anim


if __name__ == "__main__":
    # 重新导入必要的模块
    from scipy.ndimage import gaussian_filter
    import matplotlib.patches as patches
    from project_env import ForestEnvironmentVisualizer, TerrainType

    main()
