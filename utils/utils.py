import math
import pygame
import time
from contextlib import contextmanager
from typing import List, Tuple

from utils.config.game_config import GameTeam
from utils.obstacle import Obstacle
from visualization.config import render_config

@contextmanager
def timer(stats_dict, key):
    """计时器上下文管理器"""
    start = time.perf_counter()
    try:
        yield
    finally:
        stats_dict[key].append(time.perf_counter() - start)

def opposite_team(team: GameTeam) -> GameTeam:
    if team == GameTeam.RED:
        return GameTeam.BLUE
    else:
        return GameTeam.RED

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

def calc_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点之间的距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def opposite_position(p: Tuple[float, float], field_width: float, field_height: float) -> Tuple[float, float]:
    """计算相反位置"""
    return field_width - p[0], field_height - p[1]

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

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """判断点是否在多边形内
    Args:
        point: 点坐标 (x, y)
        polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
    Returns:
        bool: 点是否在多边形内
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False  # 至少需要3个点才能形成多边形
    
    inside = False
    p1x, p1y = polygon[0]
    
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        
        # 处理水平边的情况
        if p1y == p2y:
            if y == p1y and x <= max(p1x, p2x) and x >= min(p1x, p2x):
                return True  # 点在水平边上
        else:
            # 检查射线是否与边相交
            if y > min(p1y, p2y) and y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    # 计算交点的x坐标
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
        "p1": opposite_position(obstacle_config["p1"], field_width, field_height),
        "p2": opposite_position(obstacle_config["p2"], field_width, field_height),
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

def get_tangent_points(A: Tuple[float, float], B: Tuple[float, float], r: float) -> List[Tuple[float, float]]:
    """获取圆的切线点
    
    Args:
        A: 圆心坐标 (x1, y1)
        B: 被切圆心坐标 (x2, y2)
        r: 半径
    Returns:
        List[Tuple[float, float]]: 切线点列表
    """
    x1, y1 = A
    x2, y2 = B
    dx, dy = x2 - x1, y2 - y1
    d_sq = dx**2 + dy**2
    d = math.hypot(dx, dy)
    if d <= r:
        return []  # 无切线
    # 单位向量
    vx, vy = dx / d, dy / d
    # 垂直单位向量
    perp_vx, perp_vy = -vy, vx
    # 切线长度
    l = math.sqrt(d_sq - r**2)
    # 切点
    mx, my = x2 + r**2 * (x1 - x2) / d_sq, y2 + r**2 * (y1 - y2) / d_sq
    factor = r * l / d_sq
    tx1 = mx + factor * (y1 - y2)
    ty1 = my - factor * (x1 - x2)
    tx2 = mx - factor * (y1 - y2)
    ty2 = my + factor * (x1 - x2)
    return [(tx1, ty1), (tx2, ty2)]

def attack_sight_clear(
    attacker_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    target_radius: float,
    obstacles: List[Obstacle],
    robots: List,
    ignore_robot_blocked: bool = False, # 是否忽略被机器人遮挡
) -> bool:
    """
    判断攻击路径是否无遮挡（障碍物/机器人）
    返回：视野是否无遮挡
    """
    # 1. 计算两条切线
    tangents = get_tangent_points(attacker_pos, target_pos, target_radius)
    if len(tangents) < 2:
        return True

    cut1, cut2 = tangents
    quad = [attacker_pos, cut1, target_pos, cut2]

    # 2. 判断切线是否被障碍物遮挡
    for tp in tangents:
        if not has_line_of_sight(attacker_pos, tp, obstacles):
            # print(f"Blocked tangents by obstacle")
            return False

    # 3. 判断障碍物端点是否在四边形内
    for obs in obstacles:
        for pt in [obs.p1, obs.p2]:
            if point_in_polygon(pt, quad):
                # print(f"Blocked by obstacle endpoint")
                return False
    
    if ignore_robot_blocked:
        return True
    robots = [robot for robot in robots if robot.get_position() != attacker_pos and robot.get_position() != target_pos]
    
    # 4. 判断切线是否被机器人遮挡
    for tp in tangents:
        for robot in robots:
            if point_to_line_segment_distance(robot.get_position(), attacker_pos, tp) < robot.radius:
                # print(f"Blocked tangents by robot: {robot.id}")
                return False

    # 5. 判断机器人坐标是否在四边形内
    for robot in robots:
        if point_in_polygon(robot.get_position(), quad):
            # print(f"Blocked by robot: {robot.id}")
            return False

    return True
