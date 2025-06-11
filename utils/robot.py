import pygame
import pymunk
import math
import time
from typing import Tuple
from utils.config.exp_prop_config import *
from utils.config.robot_config import ROBOT_ID, RobotType
from utils.config.game_config import GameTeam, HEAT_PER_17, DAMAGE_PER_17
from utils.grid_map import GridMap, a_star
from utils.utils import meters_to_pixels

class Robot:
    def __init__(
        self,
        physics_engine: pymunk.Space,
        team: GameTeam,
        robot_type: RobotType,
        init_pos: Tuple[float, float],
        chassis_property_type: CHASSIS_PROPERTY_TYPE,
        gimbal_property_type: GIMBAL_PROPERTY_TYPE,
        forward_speed_efficiency: float,
        rotation_speed_efficiency: float,
        radius: float
    ):
        # 全局属性
        self.physics_engine = physics_engine
        self.team = team
        self.robot_type = robot_type
        self.id = ROBOT_ID[team][robot_type]

        # 规则性能
        self.level = 1
        self.exp = 0

        # 英雄
        if self.robot_type == RobotType.HERO:
            self.chassis_property = CHASSIS_PROPERTY_HERO[chassis_property_type]
            self.gimbal_property = GIMBAL_PROPERTY_42[GIMBAL_PROPERTY_TYPE.DEFAULT]
        # 步兵
        elif self.robot_type in [RobotType.STANDARD_3, RobotType.STANDARD_4, RobotType.STANDARD_5]:
            self.chassis_property = CHASSIS_PROPERTY_STANDARD[chassis_property_type]
            self.gimbal_property = GIMBAL_PROPERTY_17[gimbal_property_type]
        # 哨兵
        elif self.robot_type == RobotType.SENTRY:
            self.level = 10
            self.exp = LEVEL_NEED_EXP[self.level]
            self.chassis_property = CHASSIS_PROPERTY_STANDARD[chassis_property_type]
            self.gimbal_property = GIMBAL_PROPERTY_17[GIMBAL_PROPERTY_TYPE.COOL_DOWN]

        # 更新性能
        self.update_property()
        self.hp = self.max_hp
        self.heat = 0
        
        # 物理属性
        self.radius = radius
        self.forward_speed = self.power * forward_speed_efficiency
        self.rotation_speed = self.power * rotation_speed_efficiency

        # 创建物理实体
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, radius))
        self.body.position = init_pos

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7

        self.physics_engine.add(self.body, self.shape)

        self.angle = 0  # 角度（度）
        self.target_pos = None
        self.path_points = []
        self.current_path_idx = 0
        self.is_alive = True  # 机器人是否存活

        # 攻击相关
        self.attack_target = None  # 攻击目标
        self.last_attack_time = 0  # 上次攻击的时间（秒）
        self.last_in_combat_time = 0  # 上次进入战斗的时间（秒）

        # GridMap相关
        self.grid_map = None

    def print_info(self):
        info = {
            "id": self.id,
            "position": self.get_position(),
            "angle": self.angle,
            "is_alive": self.is_alive,
            "level": self.level,
            "hp": self.hp,
            "heat": self.heat,
            "attack_target": self.attack_target.id if self.attack_target else None,
        }
        print("-" * 20 + "Robot Info" + "-" * 20)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("-" * 50)

    def update_exp(self, exp):
        """更新经验"""
        self.exp += exp
        # 升级
        if self.exp >= LEVEL_NEED_EXP[self.level + 1]:
            self.level += 1
            max_hp_before_upgrade = self.max_hp
            self.update_property()
            self.heal(self.max_hp / max_hp_before_upgrade * self.hp - self.hp)

    def update_property(self):
        """更新机器人属性"""
        self.max_hp = self.chassis_property[self.level]["HP"]
        self.power = self.chassis_property[self.level]["POWER"]
        self.max_heat = self.gimbal_property[self.level]["HEAT"]
        self.cool_down = self.gimbal_property[self.level]["COOL_DOWN"]

    def destroy(self, physics_engine):
        """销毁物理体"""
        physics_engine.remove(self.shape, self.body)

    def get_position(self):
        """获取位置"""
        return self.body.position.x, self.body.position.y

    def set_target(self, target_pos):
        """设置目标位置并计算路径"""
        self.target_pos = target_pos
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

    def step(self, dt):
        """沿路径点导航"""
        if not self.is_alive:
            self.body.velocity = (0, 0)
            return

        # 沿路径移动
        if self.path_points and self.current_path_idx < len(self.path_points):
            next_point = self.path_points[self.current_path_idx]
            current_pos = pygame.math.Vector2(self.body.position)
            direction = pygame.math.Vector2(next_point) - current_pos
            if direction.length() < 0.05:
                self.current_path_idx += 1
            else:
                direction = direction.normalize() * self.forward_speed
                self.body.velocity = (direction.x, direction.y)
        else:
            self.body.velocity = (0, 0)
            self.target_pos = None

        # 结算热量冷却
        self.heat = max(0, self.heat - self.cool_down * dt)

        # 结算脱战状态
        if time.time() - self.last_in_combat_time > 6:
            self.attack_target = None

    def attack(self, target_robot, num=1):
        """攻击目标机器人
        Args:
            target_robot: 目标机器人对象
            num: 攻击次数
        """
        if not self.is_alive or not target_robot or not target_robot.is_alive:
            return
        # 刷新战斗状态
        self.attack_target = target_robot
        self.last_attack_time = time.time()
        self.last_in_combat_time = time.time()

        # 计算攻击角度
        target_pos = self.attack_target.get_position()
        current_pos = self.get_position()
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        self.angle = math.degrees(math.atan2(dy, dx))

        # 计算可以攻击的次数
        try:
            num = int(min(num, (self.max_heat - self.heat) // HEAT_PER_17))
        except:
            print(num, self.max_heat, self.heat, DAMAGE_PER_17)
        # 造成伤害
        damage = self.attack_target.take_damage(DAMAGE_PER_17, num)
        # 增加热量
        self.heat += HEAT_PER_17 * num
        # 结算经验
        self.update_exp(1 * num) # 每发射1次增加1点经验
        self.update_exp(damage * 4) # 每造成1点伤害增加4点经验
        if not self.attack_target.is_alive:
            # 击杀经验
            # TODO: 经验分享
            kill_exp = 50 * target_robot.level * (1 + max(0, 0.2 * (target_robot.level - self.level)))
            self.update_exp(kill_exp)

    def get_attack_line(self):
        """获取攻击线段的起点和终点
        Returns:
            tuple: (起点, 终点) 或 None（如果无法攻击）
        """
        if not self.is_alive or not self.attack_target or not self.attack_target.is_alive:
            return None
        if time.time() - self.last_attack_time > 0.2:
            return None
        return (self.get_position(), self.attack_target.get_position())

    def take_damage(self, base_damage, num) -> int:
        """受到伤害
        Args:
            base_damage: 基础伤害值
            num: 攻击次数
        Returns:
            int: 实际受到的伤害值
        """
        if not self.is_alive:
            return False

        # 受击刷新战斗状态
        self.last_in_combat_time = time.time()

        # TODO: 考虑命中
        damage = min(base_damage * num, self.hp)
        self.hp = self.hp - damage
        if self.hp <= 0:
            self.is_alive = False
        return damage

    def heal(self, amount):
        """恢复血量"""
        self.hp = min(self.max_hp, self.hp + amount)

    def set_grid_map(self, grid_map: GridMap):
        """设置机器人的网格地图"""
        self.grid_map = grid_map
