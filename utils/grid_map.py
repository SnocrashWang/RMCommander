import heapq
import math

class GridMap:
    def __init__(self, width, height, cell_size, robot_radius):
        self.cell_size = cell_size
        self.robot_radius = robot_radius
        self.grid_cols = int(width / cell_size)
        self.grid_rows = int(height / cell_size)
        self.grid_blocked = [[False for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]

    def world_to_grid(self, pos):
        """世界坐标转换为网格坐标"""
        x, y = pos
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return col, row

    def grid_to_world(self, col, row):
        """网格坐标转换为世界坐标"""
        x = col * self.cell_size
        y = row * self.cell_size
        return x, y

    def set_blocked(self, col, row):
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            self.grid_blocked[col][row] = True

    def is_blocked(self, col, row):
        """检查指定位置是否被阻塞"""
        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            return self.grid_blocked[col][row]
        else:
            raise ValueError(f"Invalid grid coordinates: ({col}, {row})")

    def mark_obstacles(self, obstacles):
        for obs in obstacles:
            start = obs.p1
            end = obs.p2
            thickness = getattr(obs, "thickness", 0.1)
            self._mark_line_blocked(start, end, thickness)

    def _mark_line_blocked(self, start, end, thickness):
        """标记线段为阻塞"""
        x1, y1 = start
        x2, y2 = end
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) / self.cell_size) + 1
        
        # 影响半径 = 墙体半厚度 + 机器人半径
        block_radius = int(math.ceil((thickness / 2 + self.robot_radius) / self.cell_size))
        
        for i in range(steps + 1):
            x = x1 + (x2 - x1) * i / steps
            y = y1 + (y2 - y1) * i / steps
            col, row = self.world_to_grid((x, y))
            
            for dx in range(-block_radius, block_radius + 1):
                for dy in range(-block_radius, block_radius + 1):
                    self.set_blocked(col + dx, row + dy)

    def clear(self):
        """清空地图"""
        self.grid_blocked = [[False for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]

def a_star(grid_map, start, goal):
    """A*算法，返回网格路径"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for d in dirs:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if grid_map.is_blocked(*neighbor):
                continue
            tentative_g = g_score[current] + math.hypot(d[0], d[1])
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path