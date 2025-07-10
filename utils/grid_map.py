import heapq
import math
import torch
from typing import Tuple

from config import DEVICE
from utils.utils import point_in_polygon, point_to_line_segment_distance, has_line_of_sight

GRID_CELL_SIZE = torch.tensor(0.1, dtype=torch.float, device=DEVICE)  # 栅格大小（米）

def world_to_grid(pos: torch.Tensor):
    """世界坐标转换为网格坐标"""
    col = torch.floor(pos[0] / GRID_CELL_SIZE - 0.5)
    row = torch.floor(pos[1] / GRID_CELL_SIZE - 0.5)
    return torch.stack([col, row], dim=0).to(torch.int)

def grid_to_world(col: torch.Tensor, row: torch.Tensor):
    """网格坐标转换为世界坐标"""
    x = (col + 0.5) * GRID_CELL_SIZE
    y = (row + 0.5) * GRID_CELL_SIZE
    return torch.stack([x, y], dim=0)

class GridMap:
    def __init__(self, width: torch.Tensor, height: torch.Tensor, robot_radius: torch.Tensor):
        self.cell_size = GRID_CELL_SIZE
        self.robot_radius = robot_radius
        # 仅在初始化时进行设备转换
        self.grid_cols = torch.floor(width / self.cell_size).int()
        self.grid_rows = torch.floor(height / self.cell_size).int()
        self.grid_blocked = torch.zeros((self.grid_rows, self.grid_cols), dtype=torch.bool, device=DEVICE)

    def set_blocked(self, col: torch.Tensor, row: torch.Tensor):
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            self.grid_blocked[col, row] = True

    def is_blocked(self, col: torch.Tensor, row: torch.Tensor):
        """检查指定位置是否被阻塞"""
        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            return self.grid_blocked[col, row]
        else:
            raise ValueError(f"Invalid grid coordinates: ({col}, {row})")

    def mark_obstacles(self, obstacles):
        for obs in obstacles:
            self._mark_obstacle_blocked(obs)

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
        min_grid = world_to_grid((min_x, min_y))
        min_col, min_row = min_grid[0].item(), min_grid[1].item()
        max_grid = world_to_grid((max_x, max_y))
        max_col, max_row = max_grid[0].item(), max_grid[1].item()
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

def a_star(grid_map: GridMap, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """
    grid_map: 二维布尔张量，True 表示障碍物
    start: 起点 (x, y)，tensor
    goal: 终点 (x, y)，tensor
    """
    # 8邻域方向
    dirs = torch.tensor([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)], device=DEVICE)

    # 初始化 g_score_map 和 f_score_map
    g_score_map = torch.full((grid_map.grid_cols * grid_map.grid_rows,), float('inf'), device=DEVICE)
    f_score_map = torch.full((grid_map.grid_cols * grid_map.grid_rows,), float('inf'), device=DEVICE)
    parent_map = torch.zeros((grid_map.grid_cols * grid_map.grid_rows, 2), dtype=torch.long, device=DEVICE)

    # 设置起点分数
    start_idx = start[1] + start[0] * grid_map.grid_cols
    g_score_map[start_idx] = 0
    f_score_map[start_idx] = torch.norm(start.float() - goal.float(), p=2)

    # 使用两个张量模拟 open_set
    frontier_scores = f_score_map.clone()
    frontier_positions = torch.stack(
        torch.meshgrid(torch.arange(grid_map.grid_cols, device=DEVICE),
                       torch.arange(grid_map.grid_rows, device=DEVICE)), dim=-1
    ).reshape(-1, 2)

    visited = torch.zeros_like(frontier_scores, dtype=torch.bool)

    path_found = False

    while True:
        # 找到当前最小 f_score 的节点索引
        min_mask = ~visited
        if not min_mask.any():
            break  # 没有可访问的节点了

        min_idx = (frontier_scores + (~min_mask) * 1e9).argmin()
        current_pos = frontier_positions[min_idx]

        # 如果找到目标，结束
        if torch.equal(current_pos, goal):
            path_found = True
            break

        # 标记为已访问
        visited[min_idx] = True

        # 扩展邻居
        neighbors = current_pos.expand(dirs.shape[0], 2) + dirs
        valid_mask = (
            (neighbors[:, 0] >= 0) &
            (neighbors[:, 0] < grid_map.grid_cols) &
            (neighbors[:, 1] >= 0) &
            (neighbors[:, 1] < grid_map.grid_rows)
        )
        neighbor_candidates = neighbors[valid_mask]

        for neighbor in neighbor_candidates:
            n_idx = neighbor[1] + neighbor[0] * grid_map.grid_cols

            if grid_map.grid_blocked[neighbor[0], neighbor[1]]:
                continue

            dx = neighbor[0] - current_pos[0]
            dy = neighbor[1] - current_pos[1]
            step_cost = torch.hypot(dx.float(), dy.float())

            tentative_g = g_score_map[min_idx] + step_cost

            if tentative_g < g_score_map[n_idx]:
                g_score_map[n_idx] = tentative_g
                f_score = tentative_g + torch.hypot(
                    (neighbor[0] - goal[0]).float(),
                    (neighbor[1] - goal[1]).float()
                )
                f_score_map[n_idx] = f_score
                frontier_scores[n_idx] = f_score
                parent_map[n_idx] = current_pos

    if path_found:
        return reconstruct_path(parent_map, goal, grid_map.grid_cols)
    else:
        return torch.empty((0, 2), dtype=torch.int, device=DEVICE)  # 无路径

def reconstruct_path(came_from: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    path = torch.tensor([current], device=DEVICE)
    while current in came_from:
        current = came_from[current.long()]
        path = torch.cat([path, current.unsqueeze(0)], dim=0)
    path = path.flip(0)
    return path

def bresenham_line(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """
    使用Bresenham算法计算两点之间连线经过的所有栅格点
    """
    # 确保输入是整数类型的Tensor
    x0, y0 = start[0], start[1]
    x1, y1 = end[0], end[1]
    
    # 计算差值和方向
    dx = torch.abs(x1 - x0)
    dy = torch.abs(y1 - y0)
    x_inc = torch.where(x1 > x0, 1, -1)
    y_inc = torch.where(y1 > y0, 1, -1)
    
    # 初始化参数
    x, y = x0, y0
    error = dx - dy
    dx2 = dx * 2
    dy2 = dy * 2
    
    # 预分配结果Tensor（最大可能点数）
    max_points = 1 + dx + dy
    points = torch.zeros((max_points.item(), 2), dtype=torch.int, device=DEVICE)
    count = 0
    
    # 主循环
    while True:
        points[count] = torch.tensor([x, y], device=DEVICE)
        count += 1
        
        if x == x1 and y == y1:
            break
            
        if error > 0:
            x += x_inc
            error -= dy2
        else:
            y += y_inc
            error += dx2
    
    # 裁剪实际使用的部分
    return points[:count]

def simplify_path(path: torch.Tensor, grid_map: GridMap) -> torch.Tensor:
    """
    使用可行栅格对A*路径进行简化，只保留转折点或关键点。
    确保简化后的路径点仍然在可行栅格内，并检测两点之间连线触及的所有栅格是否可行。
    Args:
        path: A*返回的网格路径，tensor
        grid_map: GridMap实例，用于检查栅格是否可行
    Returns:
        简化后的路径点列表（与输入path同格式）
    """
    if path.shape[0] <= 2:
        return path

    simplified = torch.tensor([path[0]], device=DEVICE)
    last_idx = 0
    for i in range(2, path.shape[0]+1):
        # 检查从last_idx到i-1是否有直线可行性
        col1, row1 = path[last_idx, 0], path[last_idx, 1]
        col2, row2 = path[i-1, 0], path[i-1, 1]
        # 使用Bresenham算法检查两点连线经过的栅格是否可行
        line_points = bresenham_line(col1, row1, col2, row2)
        feasible = True
        for col, row in line_points:
            if grid_map.is_blocked(col, row):
                feasible = False
                break
        if i == len(path) or not feasible:
            # 上一个点是关键点
            simplified = torch.cat([simplified, path[i-2].unsqueeze(0)], dim=0)
            last_idx = i-2
    # 保证终点在最后
    if simplified[-1] != path[-1]:
        simplified = torch.cat([simplified, path[-1].unsqueeze(0)], dim=0)
    # 确保简化后的路径点仍然在可行栅格内
    simplified = [p for p in simplified if not grid_map.is_blocked(*p)]
    return simplified
