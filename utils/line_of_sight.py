import math
from typing import List, Tuple
from utils.obstacle import Obstacle

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