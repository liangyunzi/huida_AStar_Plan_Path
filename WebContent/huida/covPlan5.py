# Learning Date : 2025/10/29
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from collections import deque
import heapq

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class ContinuousGridCell:
    """连续空间下的网格单元类（用于覆盖状态跟踪）"""

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


class ForestEnvironment:
    """森林环境类保持不变"""
    def __init__(self, width, height, seed):
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.start_pos = (0.0, 0.0)  # 改为浮点数表示

        # 生成地形
        base_noise = np.random.rand(height, width)
        self.static_grid = gaussian_filter(base_noise, sigma=0.8)
        self.static_grid = (self.static_grid - self.static_grid.min()) / (
                self.static_grid.max() - self.static_grid.min())

        # 清除起始位置附近的障碍物
        start_y, start_x = self.start_pos
        clear_radius = 8
        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                ny, nx = int(start_y + dy), int(start_x + dx)
                if 0 <= ny < height and 0 <= nx < width:
                    dist = np.sqrt(dy **2 + dx** 2)
                    if dist <= clear_radius:
                        self.static_grid[ny, nx] = min(self.static_grid[ny, nx], 0.3 * (dist / clear_radius))

    def is_obstacle(self, y, x):
        """检查连续坐标位置是否为障碍物"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.static_grid[int(y), int(x)] > 0.7


class RadarSensor:
    """雷达传感器类保持不变"""
    def __init__(self, max_range=10, fov_angle=360, resolution=5):
        self.max_range = max_range
        self.fov_angle = fov_angle
        self.resolution = resolution

    def scan(self, robot_pos, environment):
        """扫描并返回障碍物和自由空间"""
        y, x = robot_pos
        detected_obstacles = set()
        free_space = set()

        start_angle = (360 - self.fov_angle) // 2
        end_angle = 360 - start_angle

        for angle in range(start_angle, end_angle, self.resolution):
            hit_obstacle = False
            for r in np.linspace(0.1, self.max_range, 50):  # 使用更多采样点实现连续扫描
                scan_y = y + r * np.sin(np.radians(angle))
                scan_x = x + r * np.cos(np.radians(angle))

                if not (0 <= scan_y < environment.height and 0 <= scan_x < environment.width):
                    break

                if environment.is_obstacle(scan_y, scan_x):
                    detected_obstacles.add((scan_y, scan_x))
                    hit_obstacle = True
                    break
                else:
                    if not hit_obstacle:
                        free_space.add((scan_y, scan_x))

        return detected_obstacles, free_space


class ContinuousWavefrontPlanner:
    """连续空间下的波前覆盖规划器"""

    def __init__(self, environment, radar_range=10, cell_size=2, move_step=0.5):
        self.env = environment
        self.height = environment.height
        self.width = environment.width
        self.radar_range = radar_range
        self.cell_size = cell_size  # 网格单元大小（用于覆盖状态跟踪）
        self.move_step = move_step  # 连续移动的步长
        self.max_speed = 1.0  # 最大移动速度

        # 网格参数
        self.grid_rows = int(np.ceil(self.height / self.cell_size))
        self.grid_cols = int(np.ceil(self.width / self.cell_size))

        print(f"\nGrid Info:")
        print(f"   - Map size: {self.height} x {self.width} m")
        print(f"   - Cell size: {self.cell_size} x {self.cell_size} m")
        print(f"   - Grid dimensions: {self.grid_rows} x {self.grid_cols} = {self.grid_rows * self.grid_cols} cells")

        # 初始化网格（用于跟踪覆盖状态）
        self.grid = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                center_y = (row + 0.5) * self.cell_size
                center_x = (col + 0.5) * self.cell_size
                if center_y >= self.height:
                    center_y = self.height - 0.5
                if center_x >= self.width:
                    center_x = self.width - 0.5
                self.grid[row][col] = ContinuousGridCell(row, col, center_y, center_x, self.cell_size)

        # 雷达
        self.radar = RadarSensor(max_range=radar_range)

        # 起始位置（连续坐标）
        self.current_pos = environment.start_pos
        self.path = [self.current_pos]

        # 当前目标（连续坐标）
        self.current_target = None
        self.target_path = []  # 存储A*规划的连续路径点

        # 初始扫描
        self._scan_and_update()
        self._mark_covered_from_pos(self.current_pos)

        print(f"   - Start: {self.current_pos}")
        print(f"   - Radar range: {self.radar_range} m")

    def _get_cell_from_pos(self, y, x):
        """从连续位置获取对应的网格单元"""
        row = min(int(y / self.cell_size), self.grid_rows - 1)
        col = min(int(x / self.cell_size), self.grid_cols - 1)
        return self.grid[row][col]

    def _mark_covered_from_pos(self, pos):
        """标记位置周围的覆盖区域"""
        y, x = pos
        cell = self._get_cell_from_pos(y, x)
        if not cell.is_covered:
            cell.is_covered = True
            cell.visit_count += 1

    def _scan_and_update(self):
        """扫描并更新地图"""
        obstacles, free_space = self.radar.scan(self.current_pos, self.env)

        # 更新所有扫描到的单元格为"已探索"
        all_scanned = obstacles | free_space
        for y, x in all_scanned:
            cell = self._get_cell_from_pos(y, x)
            if not cell.is_explored:
                cell.is_explored = True

        # 标记障碍物单元格
        for y, x in obstacles:
            cell = self._get_cell_from_pos(y, x)
            cell.is_obstacle = True

        # 确保自由空间单元格不是障碍物
        for y, x in free_space:
            cell = self._get_cell_from_pos(y, x)
            if not cell.is_obstacle:
                cell.is_obstacle = False

    def _compute_distance_field(self):
        """计算到最近未覆盖单元格的距离场"""
        # 重置距离
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                self.grid[row][col].distance_to_uncovered = float('inf')

        # 找到所有未覆盖的可通行单元格作为种子
        queue = deque()
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = self.grid[row][col]
                # 已探索、可通行、未覆盖的单元格作为种子
                if cell.is_explored and not cell.is_obstacle and not cell.is_covered:
                    cell.distance_to_uncovered = 0
                    queue.append(cell)
                # 未探索的单元格也作为潜在目标（边界探索）
                elif not cell.is_explored:
                    cell.distance_to_uncovered = 0
                    queue.append(cell)

        # BFS计算距离场
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

                    # 跳过障碍物
                    if neighbor.is_explored and neighbor.is_obstacle:
                        continue

                    # 对角线距离为1.414，直线为1
                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    new_distance = cell.distance_to_uncovered + move_cost

                    if new_distance < neighbor.distance_to_uncovered:
                        neighbor.distance_to_uncovered = new_distance
                        queue.append(neighbor)

    def _get_next_target(self):
        """获取下一个连续空间目标点"""
        # 如果已有目标路径且未完成，则继续跟踪
        if self.target_path:
            return self.target_path.pop(0)

        # 计算距离场
        self._compute_distance_field()

        # 在连续空间中采样可能的目标点
        candidates = []
        sample_points = 36  # 采样角度数量
        max_distance = min(2.0, self.radar_range)  # 最大采样距离

        for angle in np.linspace(0, 2*np.pi, sample_points, endpoint=False):
            # 在不同距离上采样
            for dist in np.linspace(self.move_step, max_distance, 5):
                tx = self.current_pos[1] + dist * np.cos(angle)
                ty = self.current_pos[0] + dist * np.sin(angle)

                # 检查是否在环境范围内且不是障碍物
                if 0 <= tx < self.width and 0 <= ty < self.height and not self.env.is_obstacle(ty, tx):
                    cell = self._get_cell_from_pos(ty, tx)
                    # 计算优先级：优先选择未覆盖区域，然后是距离未覆盖区域的距离
                    if not cell.is_covered:
                        priority = (0, cell.distance_to_uncovered)
                    else:
                        priority = (1, cell.distance_to_uncovered)
                    candidates.append((priority, (ty, tx)))

        if candidates:
            # 按优先级排序
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        else:
            # 找不到直接目标，使用A*规划到更远的目标
            return self._find_distant_target()

    def _find_distant_target(self):
        """找到远处的未覆盖目标并规划路径"""
        # 找到最近的未覆盖单元格
        nearest_uncovered = self._find_nearest_uncovered_cell()
        if not nearest_uncovered:
            return None

        # 生成到目标单元格中心的路径
        target_pos = (nearest_uncovered.center_y, nearest_uncovered.center_x)
        self.target_path = self._astar_continuous_plan(target_pos)
        return self.target_path.pop(0) if self.target_path else None

    def _find_nearest_uncovered_cell(self):
        """找到最近的未覆盖单元格"""
        visited = set()
        queue = deque([self._get_cell_from_pos(*self.current_pos)])
        visited.add((queue[0].row, queue[0].col))

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

    def _astar_continuous_plan(self, target_pos):
        """连续空间A*路径规划"""
        def heuristic(p1, p2):
            return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

        open_set = []
        start_id = id(self.current_pos)
        heapq.heappush(open_set, (0, start_id, self.current_pos))

        came_from = {}
        g_score = {self.current_pos: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            # 如果到达目标附近
            if np.hypot(current[0]-target_pos[0], current[1]-target_pos[1]) < self.move_step:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                path.append(target_pos)  # 添加目标点
                return path

            # 生成邻居点（连续空间采样）
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                step = self.move_step * 2  # A*搜索步长可以大一些
                ny = current[0] + step * np.sin(angle)
                nx = current[1] + step * np.cos(angle)
                neighbor = (ny, nx)

                # 检查是否在环境范围内且不是障碍物
                if not (0 <= nx < self.width and 0 <= ny < self.height) or self.env.is_obstacle(ny, nx):
                    continue

                # 计算移动成本
                move_cost = np.hypot(ny-current[0], nx-current[1])
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, target_pos)
                    heapq.heappush(open_set, (f_score, id(neighbor), neighbor))

        return []

    def step(self):
        """执行一步连续移动"""
        # 扫描并更新地图
        self._scan_and_update()

        # 获取下一个目标位置
        next_target = self._get_next_target()
        if next_target is None:
            return False

        # 计算移动向量
        dy = next_target[0] - self.current_pos[0]
        dx = next_target[1] - self.current_pos[1]
        distance = np.hypot(dy, dx)

        # 如果距离小于步长，直接到达目标
        if distance <= self.move_step:
            new_pos = next_target
        else:
            # 否则按比例移动
            ratio = self.move_step / distance
            new_pos = (
                self.current_pos[0] + dy * ratio,
                self.current_pos[1] + dx * ratio
            )

        # 更新位置
        self.current_pos = new_pos
        self.path.append(self.current_pos)

        # 标记覆盖区域
        self._mark_covered_from_pos(self.current_pos)

        return True

    def run_coverage(self, max_steps=5000, target_coverage=0.95):
        """运行覆盖规划"""
        print(f"\nStart coverage planning...")
        print(f"   Strategy: Continuous Wavefront + A*")

        step = 0
        while step < max_steps:
            if not self.step():
                print(f"   Cannot continue planning")
                break

            step += 1

            # 统计信息
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

            # 检查是否达到目标覆盖率
            if step % 50 == 0:
                explored = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                               if self.grid[r][c].is_explored and not self.grid[r][c].is_obstacle)
                covered = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                              if self.grid[r][c].is_covered)
                coverage = covered / explored if explored > 0 else 0

                if coverage >= target_coverage:
                    print(f"\nTarget coverage {target_coverage * 100:.1f}% reached!")
                    break

        # 最终统计
        explored = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                       if self.grid[r][c].is_explored and not self.grid[r][c].is_obstacle)
        covered = sum(1 for r in range(self.grid_rows) for c in range(self.grid_cols)
                      if self.grid[r][c].is_covered)
        total_visits = sum(self.grid[r][c].visit_count for r in range(self.grid_rows)
                           for c in range(self.grid_cols))

        coverage = covered / explored if explored > 0 else 0
        repeat = (total_visits - covered) / total_visits if total_visits > 0 else 0

        print(f"\nPlanning complete:")
        print(f"   - Total steps: {len(self.path)}")
        print(f"   - Explored cells: {explored}")
        print(f"   - Covered cells: {covered}")
        print(f"   - Coverage rate: {coverage * 100:.2f}%")
        print(f"   - Repeat rate: {repeat * 100:.2f}%")


def run_animation(environment, planner, interval=30):
    """创建动画"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 转换网格为显示用的像素图
    grid_size = planner.grid_rows

    # 准备环境显示
    env_display = np.zeros((grid_size, grid_size))
    for row in range(grid_size):
        for col in range(grid_size):
            cell = planner.grid[row][col]
            y_start = cell.center_y - cell.cell_size / 2
            x_start = cell.center_x - cell.cell_size / 2

            # 采样该单元格内的环境
            obstacle_count = 0
            total_count = 0
            for dy in np.linspace(0, cell.cell_size-0.1, 5):
                for dx in np.linspace(0, cell.cell_size-0.1, 5):
                    py = min(y_start + dy, environment.height - 0.1)
                    px = min(x_start + dx, environment.width - 0.1)
                    if py >= 0 and px >= 0:
                        if environment.is_obstacle(py, px):
                            obstacle_count += 1
                        total_count += 1

            if obstacle_count > total_count * 0.5:
                env_display[row, col] = 1

    def update(frame):
        if frame >= len(planner.path):
            return

        # 清除轴
        ax1.clear()
        ax2.clear()

        # 左侧：真实环境（灰度图）
        ax1.imshow(env_display, cmap='Greys', origin='upper',
                  extent=[0, environment.width, environment.height, 0])
        ax1.set_title('Real Environment', fontsize=16)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 添加坐标轴范围限制（关键修改）
        # ax1.set_xlim(0, planner.grid_cols)
        # ax1.set_ylim(planner.grid_rows, 0)  # 保持与网格方向一致
        # ax2.set_xlim(0, planner.grid_cols)
        # ax2.set_ylim(planner.grid_rows, 0)  # 保持与网格方向一致

        # 左侧的机器人位置
        pos = planner.path[frame]
        circle1 = patches.Circle((pos[1], pos[0]), 0.5, color='red', zorder=10)
        ax1.add_patch(circle1)

        # 左侧的雷达范围
        radar_circle = patches.Circle((pos[1], pos[0]),
                                      planner.radar_range,
                                      color='blue', fill=False, linestyle='--',
                                      linewidth=2, alpha=0.5)
        ax1.add_patch(radar_circle)

        # 右侧：覆盖过程
        coverage_map = np.zeros((grid_size, grid_size, 3))

        for row in range(grid_size):
            for col in range(grid_size):
                cell = planner.grid[row][col]

                if cell.is_covered:
                    coverage_map[row, col] = [0, 1, 0]  # 绿色：已覆盖
                elif cell.is_explored and cell.is_obstacle:
                    coverage_map[row, col] = [0, 0, 0]  # 黑色：障碍物
                elif cell.is_explored:
                    coverage_map[row, col] = [1, 1, 1]  # 白色：可通行
                else:
                    coverage_map[row, col] = [0.6, 0.6, 0.6]  # 灰色：未探索

        ax2.imshow(coverage_map, origin='upper',
                  extent=[0, environment.width, environment.height, 0])

        # 绘制蓝色路径轨迹
        if frame > 0:
            path_x = [p[1] for p in planner.path[:frame+1]]
            path_y = [p[0] for p in planner.path[:frame+1]]
            ax2.plot(path_x, path_y, color='blue', linewidth=2, alpha=0.6, zorder=5)

        # 计算统计信息
        explored = sum(1 for r in range(grid_size) for c in range(grid_size)
                       if planner.grid[r][c].is_explored and not planner.grid[r][c].is_obstacle)
        covered = sum(1 for r in range(grid_size) for c in range(grid_size)
                      if planner.grid[r][c].is_covered)
        coverage = covered / explored if explored > 0 else 0

        ax2.set_title(f'Coverage Process (Steps: {frame + 1}, Coverage: {coverage * 100:.1f}%)',
                      fontsize=16)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # 右侧的机器人
        circle2 = patches.Circle((pos[1], pos[0]), 0.5, color='red', zorder=10)
        ax2.add_patch(circle2)

        # 图例
        legend_elements = [
            patches.Patch(color='green', label=f'Covered ({covered} cells)'),
            patches.Patch(color='white', label='Explored passable'),
            patches.Patch(color='black', label='Explored obstacle'),
            patches.Patch(color='gray', label='Unexplored'),
            patches.Patch(color='red', label='Robot'),
            patches.Patch(color='blue', label='Path trajectory')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')

    anim = FuncAnimation(fig, update, frames=len(planner.path),
                         interval=interval, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()
    return anim


def main():
    """主函数"""
    print("Robot Continuous Coverage Path Planning")
    print("=" * 70)

    # 创建环境
    env = ForestEnvironment(width=30, height=30, seed=20)

    obstacle_count = np.sum(env.static_grid > 0.7)
    total_pixels = env.width * env.height
    print(f"\nEnvironment Info:")
    print(f"   - Map size: {env.width} x {env.height} m")
    print(f"   - Obstacle ratio: {obstacle_count / total_pixels * 100:.1f}%")
    print(f"   - Start: {env.start_pos}")

    # 创建规划器（连续移动版本）
    planner = ContinuousWavefrontPlanner(env, radar_range=5, cell_size=2, move_step=0.5)

    # 规划路径
    planner.run_coverage(max_steps=5000, target_coverage=0.95)

    # 可视化
    print("\nStarting animation...")
    anim = run_animation(env, planner, interval=20)

    return env, planner, anim


if __name__ == "__main__":
    main()
