import heapq
import math
from typing import Tuple
from utils.utils import point_in_polygon, point_to_line_segment_distance, has_line_of_sight

GRID_CELL_SIZE = 0.1  # 栅格大小（米）

def world_to_grid(pos):
    """世界坐标转换为网格坐标"""
    x, y = pos
    col = int(x / GRID_CELL_SIZE - 0.5)
    row = int(y / GRID_CELL_SIZE - 0.5)
    return col, row

def grid_to_world(col, row):
    """网格坐标转换为世界坐标"""
    x = (col + 0.5) * GRID_CELL_SIZE
    y = (row + 0.5) * GRID_CELL_SIZE
    return x, y

class GridMap:
    def __init__(self, width, height, robot_radius):
        self.cell_size = GRID_CELL_SIZE
        self.robot_radius = robot_radius
        self.grid_cols = int(width / self.cell_size)
        self.grid_rows = int(height / self.cell_size)
        self.grid_blocked = [[False for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]

    def set_blocked(self, col, row):
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            self.grid_blocked[col][row] = True

    def is_blocked(self, col, row):
        """检查指定位置是否被阻塞"""
        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            return self.grid_blocked[col][row]
        else:
            raise ValueError(f"Invalid grid coordinates: ({col}, {row}) out of ({self.grid_cols}, {self.grid_rows})")

    def mark_obstacles(self, obstacles):
        for obs in obstacles:
            self._mark_obstacle_blocked(obs)

    def _mark_line_blocked(self, start, end, thickness):
        """标记线段为阻塞
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            thickness: 线段厚度
        """
        # 计算线段的方向向量
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
            
        # 单位方向向量
        dx, dy = dx / length, dy / length
        
        # 计算垂直向量（逆时针旋转90度）
        perp_dx, perp_dy = -dy, dx
        
        # 计算影响半径（墙体半厚度 + 机器人半径）
        block_radius = (thickness / 2 + self.robot_radius)
        
        # 计算网格范围
        min_x = min(start[0], end[0]) - block_radius
        max_x = max(start[0], end[0]) + block_radius
        min_y = min(start[1], end[1]) - block_radius
        max_y = max(start[1], end[1]) + block_radius
        
        # 转换为网格坐标
        min_col, min_row = world_to_grid((min_x, min_y))
        max_col, max_row = world_to_grid((max_x, max_y))
        min_col = max(0, min_col)
        max_col = min(self.grid_cols - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(self.grid_rows - 1, max_row)
        
        # 遍历可能受影响的网格
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                # 计算网格中心点坐标
                cell_x, cell_y = grid_to_world(col, row)
                
                # 计算点到线段的距离
                # 1. 计算点到线段起点的向量
                px = cell_x - start[0]
                py = cell_y - start[1]
                
                # 2. 计算投影长度
                proj = px * dx + py * dy
                
                # 3. 计算垂直距离
                if proj < 0:
                    # 点在起点之前
                    dist = math.hypot(px, py)
                elif proj > length:
                    # 点在终点之后
                    dist = math.hypot(cell_x - end[0], cell_y - end[1])
                else:
                    # 点到线段的垂直距离
                    dist = abs(px * perp_dx + py * perp_dy)
                
                # 如果距离小于影响半径，标记为阻塞
                if dist <= block_radius:
                    self.set_blocked(col, row)

    def _mark_obstacle_blocked(self, obstacle):
        """标记障碍物为阻塞
        Args:
            obstacle: 障碍物对象，包含start, end, thickness属性
        """
        # 计算障碍物的矩形区域
        dx = obstacle.p2[0] - obstacle.p1[0]
        dy = obstacle.p2[1] - obstacle.p1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
            
        # 计算矩形的四个角点
        half_thickness = obstacle.thickness / 2
        # 单位方向向量
        dir_x = dx / length
        dir_y = dy / length
        # 垂直向量（逆时针旋转90度）
        perp_x = -dir_y
        perp_y = dir_x
        
        # 计算矩形的四个角点
        corners = [
            (obstacle.p1[0] + perp_x * half_thickness, obstacle.p1[1] + perp_y * half_thickness),
            (obstacle.p1[0] - perp_x * half_thickness, obstacle.p1[1] - perp_y * half_thickness),
            (obstacle.p2[0] - perp_x * half_thickness, obstacle.p2[1] - perp_y * half_thickness),
            (obstacle.p2[0] + perp_x * half_thickness, obstacle.p2[1] + perp_y * half_thickness)
        ]
        
        # 计算矩形的边界框
        min_x = min(x for x, y in corners) - self.robot_radius
        max_x = max(x for x, y in corners) + self.robot_radius
        min_y = min(y for x, y in corners) - self.robot_radius
        max_y = max(y for x, y in corners) + self.robot_radius
        
        # 转换为网格坐标
        min_col, min_row = world_to_grid((min_x, min_y))
        max_col, max_row = world_to_grid((max_x, max_y))
        min_col = max(0, min_col)
        max_col = min(self.grid_cols - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(self.grid_rows - 1, max_row)
        
        # 遍历可能受影响的网格
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                # 计算网格中心点坐标
                cell_x, cell_y = grid_to_world(col, row)
                
                # 1. 检查点是否在矩形内
                if point_in_polygon((cell_x, cell_y), corners):
                    self.set_blocked(col, row)
                    continue
                
                # 2. 检查点到矩形边的距离
                min_dist = float('inf')
                for i in range(4):
                    p1 = corners[i]
                    p2 = corners[(i + 1) % 4]
                    dist = point_to_line_segment_distance((cell_x, cell_y), p1, p2)
                    min_dist = min(min_dist, dist)
                
                if min_dist <= self.robot_radius:
                    self.set_blocked(col, row)

    def clear(self):
        """清空地图"""
        self.grid_blocked = [[False for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]

def a_star(grid_map, start: Tuple[int, int], goal: Tuple[int, int]):
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
            try:
                if grid_map.is_blocked(*neighbor):
                    continue
            except ValueError:
                continue
            except Exception as e:
                raise e
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
    # print(path)
    return path

def bresenham_line(x0, y0, x1, y1):
    """
    使用Bresenham算法计算两点之间连线经过的所有栅格点
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    for _ in range(n):
        points.append((x, y))
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return points

def simplify_path(path, grid_map):
    """
    使用可行栅格对A*路径进行简化，只保留转折点或关键点。
    确保简化后的路径点仍然在可行栅格内，并检测两点之间连线触及的所有栅格是否可行。
    Args:
        path: A*返回的网格路径，列表[(col, row), ...]
        grid_map: GridMap实例，用于检查栅格是否可行
    Returns:
        简化后的路径点列表（与输入path同格式）
    """
    if not path or len(path) <= 2:
        return path

    simplified = [path[0]]
    last_idx = 0
    for i in range(2, len(path)+1):
        # 检查从last_idx到i-1是否有直线可行性
        col1, row1 = path[last_idx]
        col2, row2 = path[i-1]
        # 使用Bresenham算法检查两点连线经过的栅格是否可行
        line_points = bresenham_line(col1, row1, col2, row2)
        feasible = True
        for col, row in line_points:
            if grid_map.is_blocked(col, row):
                feasible = False
                break
        if i == len(path) or not feasible:
            # 上一个点是关键点
            simplified.append(path[i-2])
            last_idx = i-2
    # 保证终点在最后
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    # 确保简化后的路径点仍然在可行栅格内
    simplified = [p for p in simplified if not grid_map.is_blocked(*p)]
    return simplified
