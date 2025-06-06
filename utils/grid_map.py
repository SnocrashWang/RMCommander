import heapq
import math

import utils.robot_config as robot_config

class GridMap:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(math.ceil(width / cell_size))
        self.rows = int(math.ceil(height / cell_size))
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def world_to_grid(self, pos):
        x, y = pos
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (col, row)

    def grid_to_world(self, grid_pos):
        col, row = grid_pos
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return (x, y)

    def set_blocked(self, col, row):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = 1

    def is_blocked(self, col, row):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] == 1
        return True  # 越界视为阻挡

    def mark_obstacles(self, obstacles):
        for obs in obstacles:
            start = obs.p1
            end = obs.p2
            thickness = getattr(obs, "thickness", 0.1)
            self._mark_line_blocked(start, end, thickness, robot_config.TANK_RADIUS)

    def _mark_line_blocked(self, start, end, thickness, robot_radius):
        x1, y1 = start
        x2, y2 = end
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) / self.cell_size) + 1
        for i in range(steps + 1):
            x = x1 + (x2 - x1) * i / steps
            y = y1 + (y2 - y1) * i / steps
            col, row = self.world_to_grid((x, y))
            # 影响半径 = 墙体半厚度 + robot半径
            block_radius = int(math.ceil((thickness / 2 + robot_config.TANK_RADIUS) / self.cell_size))
            for dx in range(-block_radius, block_radius + 1):
                for dy in range(-block_radius, block_radius + 1):
                    self.set_blocked(col + dx, row + dy)

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