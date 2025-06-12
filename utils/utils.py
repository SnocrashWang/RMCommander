import math
import pygame
from typing import List, Tuple

from utils.obstacle import Obstacle
from visualization.config import render_config

def meters_to_pixels(meters: float) -> int:
    """将米转换为像素"""
    return int(meters * render_config.SCALE)

def pixels_to_meters(pixels: int) -> float:
    """将像素转换为米"""
    return pixels / render_config.SCALE

def second2minute(seconds: int) -> Tuple[int, int]:
    min = seconds // 60
    sec = seconds % 60
    return min, sec

def draw_dashed_line(
    surface: pygame.Surface,
    color: Tuple[int, int, int],
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    width: int = 2,
    dash_length: int = 5,
    gap_length: int = 10
) -> None:
    """绘制虚线
    Args:
        surface: 要绘制的表面
        color: 线条颜色 (R, G, B)
        start_pos: 起点坐标 (x, y)
        end_pos: 终点坐标 (x, y)
        width: 线条宽度
        dash_length: 虚线段的长度
        gap_length: 间隔的长度
    """
    start_pos = (meters_to_pixels(start_pos[0]), meters_to_pixels(start_pos[1]))
    end_pos = (meters_to_pixels(end_pos[0]), meters_to_pixels(end_pos[1]))
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = math.hypot(dx, dy)
    
    if distance == 0:
        return
        
    # 计算单位向量
    dx, dy = dx / distance, dy / distance
    # 当前绘制位置
    current_pos = start_pos
    # 绘制虚线
    while distance > 0:
        # 计算当前段的长度
        current_length = min(dash_length, distance)
        
        # 计算当前段的终点
        next_pos = (
            current_pos[0] + dx * current_length,
            current_pos[1] + dy * current_length
        )
        
        # 绘制当前段
        pygame.draw.line(surface, color, current_pos, next_pos, width)
        
        # 移动到下一段的起点
        current_pos = (
            next_pos[0] + dx * min(gap_length, distance - current_length),
            next_pos[1] + dy * min(gap_length, distance - current_length)
        )
        
        # 更新剩余距离
        distance -= (current_length + gap_length)

def point_in_polygon(point, polygon):
    """判断点是否在多边形内
    Args:
        point: 点坐标 (x, y)
        polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
    Returns:
        bool: 点是否在多边形内
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def point_to_line_segment_distance(point, line_start, line_end):
    """计算点到线段的距离
    Args:
        point: 点坐标 (x, y)
        line_start: 线段起点 (x, y)
        line_end: 线段终点 (x, y)
    Returns:
        float: 点到线段的距离
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 计算线段向量
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(x - x1, y - y1)
        
    # 计算投影
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (length * length)))
    
    # 计算最近点
    px = x1 + t * dx
    py = y1 + t * dy
    
    # 返回距离
    return math.hypot(x - px, y - py)

def get_reverse_obstacle_config(obstacle_config, field_width, field_height):
    """获取障碍物的反向配置"""
    return {
        "x1": field_width - obstacle_config["x1"],
        "y1": field_height - obstacle_config["y1"],
        "x2": field_width - obstacle_config["x2"],
        "y2": field_height - obstacle_config["y2"],
        "thickness": obstacle_config["thickness"]
    }

def line_intersects_obstacle(p1: Tuple[float, float], p2: Tuple[float, float], obstacle: Obstacle) -> bool:
    """判断线段是否与障碍物相交
    
    Args:
        p1: 线段起点坐标 (x, y)
        p2: 线段终点坐标 (x, y)
        obstacle: 障碍物对象
        
    Returns:
        bool: 如果线段与障碍物相交返回True，否则返回False
    """
    # 计算障碍物的四个顶点
    dx = obstacle.p2[0] - obstacle.p1[0]
    dy = obstacle.p2[1] - obstacle.p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return False
        
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
    
    # 检查线段是否与矩形的任何边相交
    for i in range(4):
        q1 = corners[i]
        q2 = corners[(i + 1) % 4]
        if line_segments_intersect(p1, p2, q1, q2):
            return True
            
    return False

def line_segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], 
                          q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """判断两条线段是否相交
    
    Args:
        p1, p2: 第一条线段的两个端点
        q1, q2: 第二条线段的两个端点
        
    Returns:
        bool: 如果线段相交返回True，否则返回False
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def has_line_of_sight(p1: Tuple[float, float], p2: Tuple[float, float], obstacles: List[Obstacle]) -> bool:
    """判断两点之间是否有直线视野（无障碍物阻挡）
    
    Args:
        p1: 起点坐标 (x, y)
        p2: 终点坐标 (x, y)
        obstacles: 障碍物列表
        
    Returns:
        bool: 如果两点之间有直线视野返回True，否则返回False
    """
    for obstacle in obstacles:
        if line_intersects_obstacle(p1, p2, obstacle):
            return False
    return True
