import pygame
import pymunk
from utils import robot_config
from utils.game_config import GameTeam
from utils.utils import meters_to_pixels
from utils.grid_map import a_star

class Robot:
    def __init__(self, physics_engine, position, team, radius=robot_config.TANK_RADIUS, env=None):
        self.radius = radius
        self.team = team
        self.color = robot_config.RED_COLOR if team == GameTeam.RED else robot_config.BLUE_COLOR
        self.speed = robot_config.TANK_SPEED
        self.rotation_speed = robot_config.TANK_ROTATION_SPEED
        self.angle = 0  # 角度（度）
        self.in_center_zone = False
        self.target_pos = None  # 新增：目标点
        self.path_points = []
        self.current_path_idx = 0
        self.env = env  # 传入环境对象以访问grid_map

        # 创建物理体
        mass = 10
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.5
        self.shape.friction = 0.9
        self.shape.filter = pymunk.ShapeFilter(categories=0b1)
        self.shape.collision_type = 1

        # 添加到物理引擎
        physics_engine.add_object(self.body, self.shape)

    def set_target(self, pos):
        """设置目标点（世界坐标）并规划路径"""
        self.target_pos = pygame.math.Vector2(pos)
        if self.env is None or not hasattr(self.env, "grid_map"):
            self.path_points = [pos]
            self.current_path_idx = 0
            return
        grid_map = self.env.grid_map
        start_grid = grid_map.world_to_grid(self.body.position)
        goal_grid = grid_map.world_to_grid(pos)
        path_grids = a_star(grid_map, start_grid, goal_grid)
        self.path_points = [grid_map.grid_to_world(gp) for gp in path_grids]
        self.current_path_idx = 0

    def update(self, center_zone_rect, scale):
        """沿路径点导航"""
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

    def get_position(self):
        """获取机器人位置（世界坐标）"""
        return self.body.position.x, self.body.position.y

    def destroy(self, physics_engine):
        """从物理引擎中移除机器人相关的物体和形状"""
        physics_engine.remove_object(self.body, self.shape)
