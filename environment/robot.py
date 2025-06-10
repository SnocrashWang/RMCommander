import pygame
import pymunk
import math
import time
from utils import robot_config
from utils import env_config
from utils.game_config import GameTeam
from utils.utils import meters_to_pixels
from utils.grid_map import a_star
from typing import Tuple, Optional
from utils.robot_config import ROBOT_COLORS
from utils.grid_map import GridMap

class Robot:
    def __init__(
        self,
        physics_engine: pymunk.Space,
        init_pos: Tuple[float, float],
        team: GameTeam,
        hp: int = 200,
        speed: float = 2.0,
        rotation_speed: float = 180.0,
        radius: float = 0.25
    ):
        self.physics_engine = physics_engine
        self.team = team
        self.hp = hp
        self.max_hp = hp
        self.speed = speed
        self.rotation_speed = rotation_speed
        self.radius = radius
        self.color = ROBOT_COLORS[team]

        # 创建物理实体
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, radius))
        self.body.position = init_pos

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7

        self.physics_engine.add(self.body, self.shape)

        self.angle = 0  # 角度（度）
        self.in_center_zone = False
        self.target_pos = None
        self.path_points = []
        self.current_path_idx = 0
        self.is_alive = True  # 机器人是否存活

        # 攻击相关
        self.is_attacking = False  # 是否正在攻击
        self.attack_start_time = 0  # 攻击开始时间
        self.attack_duration = 0.2  # 攻击持续时间（秒）

        # GridMap相关
        self.grid_map = None

    def destroy(self, physics_engine):
        """销毁物理体"""
        physics_engine.remove(self.shape, self.body)

    def get_position(self):
        """获取位置"""
        return self.body.position.x, self.body.position.y

    def set_target(self, target_pos):
        """设置目标位置并计算路径"""
        self.target_pos = target_pos
        print(self.target_pos)
        if self.grid_map is None:
            self.path_points = [target_pos]
            self.current_path_idx = 0
            return

        # 使用A*算法规划路径
        start_grid = self.grid_map.world_to_grid(self.get_position())
        goal_grid = self.grid_map.world_to_grid(target_pos)
        path_grids = a_star(self.grid_map, start_grid, goal_grid)
        
        # 将栅格坐标转换回世界坐标
        self.path_points = [self.grid_map.grid_to_world(gp[0], gp[1]) for gp in path_grids[1:]]
        self.current_path_idx = 0

    def update(self, center_zone_rect, scale):
        """沿路径点导航"""
        if not self.is_alive:
            self.body.velocity = (0, 0)
            self.in_center_zone = False
            return

        # 检查攻击持续时间
        if self.is_attacking and time.time() - self.attack_start_time >= self.attack_duration:
            self.is_attacking = False

        # 沿路径移动
        if self.path_points and self.current_path_idx < len(self.path_points):
            next_point = self.path_points[self.current_path_idx]
            current_pos = pygame.math.Vector2(self.body.position)
            direction = pygame.math.Vector2(next_point) - current_pos
            if direction.length() < 0.05:
                self.current_path_idx += 1
            else:
                direction = direction.normalize() * self.speed
                self.body.velocity = (direction.x, direction.y)
        else:
            self.body.velocity = (0, 0)
            self.target_pos = None

        # 检查是否在中心区域
        pos = self.body.position
        pixel_x = meters_to_pixels(pos.x, scale)
        pixel_y = meters_to_pixels(pos.y, scale)
        if isinstance(center_zone_rect, tuple):
            center_zone_rect = pygame.Rect(*center_zone_rect)
        self.in_center_zone = center_zone_rect.collidepoint(pixel_x, pixel_y)

    def attack(self, target_robot):
        """攻击目标机器人
        Args:
            target_robot: 目标机器人对象
        Returns:
            bool: 目标是否被击毁
        """
        if not self.is_alive or not target_robot.is_alive:
            return False
            
        # 计算攻击角度
        target_pos = target_robot.get_position()
        current_pos = self.get_position()
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # 设置炮台角度
        self.angle = target_angle
        
        # 设置攻击标记和时间
        self.is_attacking = True
        self.attack_start_time = time.time()
        
        # 造成伤害
        return target_robot.take_damage(10)  # 每次攻击造成10点伤害

    def get_attack_line(self, target_robot):
        """获取攻击线段的起点和终点
        Args:
            target_robot: 目标机器人对象
        Returns:
            tuple: (起点, 终点) 或 None（如果无法攻击）
        """
        if not self.is_alive or not target_robot.is_alive:
            return None
            
        # 检查攻击持续时间
        if self.is_attacking and time.time() - self.attack_start_time < self.attack_duration:
            return (self.get_position(), target_robot.get_position())
        else:
            self.is_attacking = False
            return None

    def take_damage(self, damage):
        """受到伤害
        Args:
            damage: 伤害值
        Returns:
            bool: 是否被击毁
        """
        if not self.is_alive:
            return False
            
        self.hp = max(0, self.hp - damage)
        if self.hp <= 0:
            self.is_alive = False
            return True
        return False

    def heal(self, amount):
        """恢复血量"""
        self.hp = min(self.max_hp, self.hp + amount)

    def get_hp_percentage(self):
        """获取血量百分比"""
        return self.hp / self.max_hp if self.is_alive else 0

    def set_grid_map(self, grid_map: GridMap):
        """设置机器人的网格地图"""
        self.grid_map = grid_map
