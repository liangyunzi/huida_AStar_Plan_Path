import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from collections import deque
import heapq

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


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


class ForestEnvironment:
    """Forest environment"""

    def __init__(self, width, height, seed):
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.start_pos = (0, 0)

        # Generate terrain
        base_noise = np.random.rand(height, width)
        self.static_grid = gaussian_filter(base_noise, sigma=3.0)
        self.static_grid = (self.static_grid - self.static_grid.min()) / (
                self.static_grid.max() - self.static_grid.min())

        # Clear obstacles near start position
        start_y, start_x = self.start_pos
        clear_radius = 8
        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                ny, nx = start_y + dy, start_x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    dist = np.sqrt(dy ** 2 + dx ** 2)
                    if dist <= clear_radius:
                        self.static_grid[ny, nx] = min(self.static_grid[ny, nx], 0.3 * (dist / clear_radius))


class RadarSensor:
    """Radar sensor"""

    def __init__(self, max_range=10, fov_angle=360, resolution=5):
        self.max_range = max_range
        self.fov_angle = fov_angle
        self.resolution = resolution

    def scan(self, robot_pos, environment):
        """Scan and return obstacles and free space"""
        y, x = robot_pos
        detected_obstacles = set()
        free_space = set()

        start_angle = (360 - self.fov_angle) // 2
        end_angle = 360 - start_angle

        for angle in range(start_angle, end_angle, self.resolution):
            hit_obstacle = False
            for r in range(1, self.max_range + 1):
                scan_y = int(round(y + r * np.sin(np.radians(angle))))
                scan_x = int(round(x + r * np.cos(np.radians(angle))))

                if not (0 <= scan_y < environment.height and 0 <= scan_x < environment.width):
                    break

                if environment.static_grid[scan_y, scan_x] > 0.7:
                    detected_obstacles.add((scan_y, scan_x))
                    hit_obstacle = True
                    break
                else:
                    if not hit_obstacle:
                        free_space.add((scan_y, scan_x))

        return detected_obstacles, free_space


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

        # Start position
        start_y, start_x = environment.start_pos
        self.current_cell = self._get_cell_from_pos(start_y, start_x)
        self.current_pos = (int(self.current_cell.center_y), int(self.current_cell.center_x))

        # Path record
        self.path = [self.current_pos]
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

    def _scan_and_update(self):
        """Scan and update map"""
        obstacles, free_space = self.radar.scan(self.current_pos, self.env)

        # Update all scanned cells as "explored"
        all_scanned = obstacles | free_space
        for y, x in all_scanned:
            cell = self._get_cell_from_pos(y, x)
            if not cell.is_explored:
                cell.is_explored = True

        # Mark obstacle cells
        for y, x in obstacles:
            cell = self._get_cell_from_pos(y, x)
            cell.is_obstacle = True

        # Ensure free space cells are not obstacles
        for y, x in free_space:
            cell = self._get_cell_from_pos(y, x)
            if not cell.is_obstacle:
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
        self.path.append(self.current_pos)

        next_cell.is_covered = True
        next_cell.visit_count += 1

        return True

    def run_coverage(self, max_steps=5000, target_coverage=0.95):
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
        print(f"   - Total steps: {len(self.path)}")
        print(f"   - Explored cells: {explored}")
        print(f"   - Covered cells: {covered}")
        print(f"   - Coverage rate: {coverage * 100:.2f}%")
        print(f"   - Repeat rate: {repeat * 100:.2f}%")


def run_animation(environment, planner, interval=30):
    """Create animation with improved visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Convert grid to pixel map for display
    grid_size = planner.grid_rows

    # Prepare environment display
    env_display = np.zeros((grid_size, grid_size))
    for row in range(grid_size):
        for col in range(grid_size):
            cell = planner.grid[row][col]
            y_start = int(cell.center_y - cell.cell_size / 2)
            x_start = int(cell.center_x - cell.cell_size / 2)

            # Sample environment in this cell
            obstacle_count = 0
            total_count = 0
            for dy in range(cell.cell_size):
                for dx in range(cell.cell_size):
                    py = min(y_start + dy, environment.height - 1)
                    px = min(x_start + dx, environment.width - 1)
                    if py >= 0 and px >= 0:
                        if environment.static_grid[py, px] > 0.7:
                            obstacle_count += 1
                        total_count += 1

            if obstacle_count > total_count * 0.5:
                env_display[row, col] = 1

    def update(frame):
        if frame >= len(planner.path):
            return

        # Clear axes
        ax1.clear()
        ax2.clear()

        # Left: Real environment (grayscale)
        ax1.imshow(env_display, cmap='Greys', origin='upper')
        ax1.set_title('Real Environment', fontsize=16)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Robot position on left
        pos = planner.path[frame]
        robot_row = min(int(pos[0] / planner.cell_size), grid_size - 1)
        robot_col = min(int(pos[1] / planner.cell_size), grid_size - 1)

        circle1 = patches.Circle((robot_col, robot_row), 0.5, color='red', zorder=10)
        ax1.add_patch(circle1)

        # Radar range on left
        radar_circle = patches.Circle((robot_col, robot_row),
                                      planner.radar_range / planner.cell_size,
                                      color='blue', fill=False, linestyle='--',
                                      linewidth=2, alpha=0.5)
        ax1.add_patch(radar_circle)

        # Right: Coverage process
        coverage_map = np.zeros((grid_size, grid_size, 3))

        for row in range(grid_size):
            for col in range(grid_size):
                cell = planner.grid[row][col]

                if cell.is_covered:
                    coverage_map[row, col] = [0, 1, 0]  # Green: covered
                elif cell.is_explored and cell.is_obstacle:
                    coverage_map[row, col] = [0, 0, 0]  # Black: obstacle
                elif cell.is_explored:
                    coverage_map[row, col] = [1, 1, 1]  # White: passable
                else:
                    coverage_map[row, col] = [0.6, 0.6, 0.6]  # Gray: unexplored

        ax2.imshow(coverage_map, origin='upper')

        # Draw path trajectory in blue
        if frame > 0:
            path_cols = []
            path_rows = []
            for i in range(frame + 1):
                pos = planner.path[i]
                row = min(int(pos[0] / planner.cell_size), grid_size - 1)
                col = min(int(pos[1] / planner.cell_size), grid_size - 1)
                path_cols.append(col)
                path_rows.append(row)
            ax2.plot(path_cols, path_rows, color='blue', linewidth=2, alpha=0.6, zorder=5)

        # Calculate statistics
        explored = sum(1 for r in range(grid_size) for c in range(grid_size)
                       if planner.grid[r][c].is_explored and not planner.grid[r][c].is_obstacle)
        covered = sum(1 for r in range(grid_size) for c in range(grid_size)
                      if planner.grid[r][c].is_covered)
        coverage = covered / explored if explored > 0 else 0

        algorithm = "A*" if planner.use_astar else "Wavefront"
        ax2.set_title(f'Coverage Process (Steps: {frame + 1}, Alg: {algorithm}, Coverage: {coverage * 100:.1f}%)',
                      fontsize=16)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # Robot on right
        circle2 = patches.Circle((robot_col, robot_row), 0.5, color='red', zorder=10)
        ax2.add_patch(circle2)

        # Legend
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
    """Main function"""
    print("Robot Coverage Path Planning")
    print("=" * 70)

    # Create environment
    env = ForestEnvironment(width=55, height=55, seed=43)

    obstacle_count = np.sum(env.static_grid > 0.7)
    total_pixels = env.width * env.height
    print(f"\nEnvironment Info:")
    print(f"   - Map size: {env.width} x {env.height} m")
    print(f"   - Obstacle ratio: {obstacle_count / total_pixels * 100:.1f}%")
    print(f"   - Start: {env.start_pos}")

    # Create planner
    planner = ImprovedWavefrontPlanner(env, radar_range=10, cell_size=2)

    # Plan path
    planner.run_coverage(max_steps=5000, target_coverage=0.95)

    # Visualize
    print("\nStarting animation...")
    anim = run_animation(env, planner, interval=20)

    return env, planner, anim


if __name__ == "__main__":
    main()