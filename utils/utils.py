import pygame
from typing import Tuple
import math

def meters_to_pixels(meters, scale):
    """将米转换为像素"""
    return meters * scale

def pixels_to_meters(pixels, scale):
    """将像素转换为米"""
    return pixels / scale

def second2minute(seconds: int) -> Tuple[int, int]:
    min = seconds // 60
    sec = seconds % 60
    return min, sec

def draw_dashed_line(
    surface: pygame.Surface,
    scale: float,
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
    start_pos = (meters_to_pixels(start_pos[0], scale), meters_to_pixels(start_pos[1], scale))
    end_pos = (meters_to_pixels(end_pos[0], scale), meters_to_pixels(end_pos[1], scale))
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