import pygame
import pymunk
import math
import time
from typing import List, Dict, Tuple, Optional
from utils.config.exp_prop_config import *
from utils.config.bullet_config import *
from utils.config.robot_config import ROBOT_ID, RobotType
from utils.config.game_config import GameTeam
from utils.grid_map import GridMap, a_star, world_to_grid, grid_to_world, simplify_path

class Robot:
    def __init__(
        self,
        team: GameTeam,
        robot_type: RobotType,
        init_pos: Tuple[float, float],
        chassis_property_type: CHASSIS_PROPERTY_TYPE,
        gimbal_property_type: GIMBAL_PROPERTY_TYPE,
        forward_speed_efficiency: float,
        rotation_speed_efficiency: float,
        radius: float,
        max_ammo: int,
        ammo_allowed: int,
        physics_engine: Optional[pymunk.Space] = None,
    ):
        # 全局属性
        self.team : GameTeam = team
        self.robot_type : RobotType = robot_type
        self.id : str = ROBOT_ID[team][robot_type]

        # 规则性能
        self.level : int = 1
        self.exp : int = 0
        self.chassis_property_type : CHASSIS_PROPERTY_TYPE = chassis_property_type
        self.gimbal_property_type : GIMBAL_PROPERTY_TYPE = gimbal_property_type

        # 英雄
        if self.robot_type == RobotType.HERO:
            self.chassis_property = CHASSIS_PROPERTY_HERO[self.chassis_property_type]
            self.gimbal_property = GIMBAL_PROPERTY_42[GIMBAL_PROPERTY_TYPE.DEFAULT]
            self.bullet = LargeBullet()
        # 步兵
        elif self.robot_type in [RobotType.STANDARD_3, RobotType.STANDARD_4, RobotType.STANDARD_5]:
            self.chassis_property = CHASSIS_PROPERTY_STANDARD[self.chassis_property_type]
            self.gimbal_property = GIMBAL_PROPERTY_17[self.gimbal_property_type]
            self.bullet = SmallBullet()
        # 哨兵
        elif self.robot_type == RobotType.SENTRY:
            self.level = 10
            self.exp = LEVEL_NEED_EXP[self.level]
            self.chassis_property = CHASSIS_PROPERTY_STANDARD[CHASSIS_PROPERTY_TYPE.HP]
            self.gimbal_property = GIMBAL_PROPERTY_17[GIMBAL_PROPERTY_TYPE.COOL_DOWN]
            self.bullet = SmallBullet()

        # 更新性能
        self.update_property()
        self.hp : int = self.max_hp
        self.heat : float = 0
        
        # 物理属性
        self.radius : float = radius
        self.forward_speed : float = self.power * forward_speed_efficiency
        self.rotation_speed : float = self.power * rotation_speed_efficiency

        # 创建物理实体
        self._body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, radius))
        self._body.position = init_pos

        self._shape = pymunk.Circle(self._body, radius)
        self._shape.elasticity = 0.8
        self._shape.friction = 0.7

        if physics_engine:
            physics_engine.add(self._body, self._shape)

        self.angle : float = 0  # 角度（度）
        self.target_pos : Tuple[float, float] = None
        self.path_points : List[Tuple[float, float]] = []
        self.current_path_idx : int = 0
        self.is_alive : bool = True  # 机器人是否存活

        # 弹丸相关
        self.max_ammo : int = max_ammo      # 最大弹药量
        self.ammo : int = self.max_ammo     # 当前弹药量
        self.ammo_allowed : int = ammo_allowed # 允许发弹量

        # 攻击相关
        self.gun_locked : bool = False      # 发射机构锁定
        self.attack_target : Robot = None   # 攻击目标
        self.last_attack_time : float = 0   # 上次攻击的时间（秒）
        self.last_in_combat_time : float = 0 # 上次进入战斗的时间（秒）

        # 复活相关
        self.revive_progress : float = 0    # 复活进度
        self.revive_target : float = 10     # 复活所需进度
        self.revive_efficiency : float = 2  # 复活效率（每秒增加的进度）
        self.last_revive_time : float = 0.0 # 上次复活时间

        # TODO: 增益相关
        self.damage_buff_dict : Dict[str, float] = {}      # 伤害增益
        self.damage_buff : float = 0.0
        self.defense_buff_dict : Dict[str, float] = {}     # 防御增益
        self.defense_buff : float = 0.0
        self.defense_debuff_dict : Dict[str, float] = {}     # 防御减益
        self.defense_debuff : float = 0.0
        self.cool_down_buff_dict : Dict[str, float] = {}   # 冷却缩减增益
        self.cool_down_buff : float = 1.0
        self.power_buff_dict : Dict[str, float] = {}       # 功率增益
        self.power_buff : float = 1.0

        # GridMap相关
        self.grid_map : GridMap = None

    def destroy_physics_body(self, physics_engine: pymunk.Space):
        """从物理引擎中移除物理体"""
        if self._body is not None and self._shape is not None:
            physics_engine.remove(self._shape, self._body)

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

    def update_exp(self, exp: int):
        """更新经验"""
        self.exp = min(self.exp + exp, LEVEL_NEED_EXP[len(LEVEL_NEED_EXP)])
        # 升级
        if self.exp >= LEVEL_NEED_EXP[min(self.level + 1, len(LEVEL_NEED_EXP))] and self.level < len(LEVEL_NEED_EXP):
            self.level += 1
            max_hp_before_upgrade = self.max_hp
            self.update_property()
            self.heal(int(self.max_hp / max_hp_before_upgrade * self.hp - self.hp))

    def update_property(self):
        """更新机器人属性"""
        self.max_hp : int = self.chassis_property[self.level]["HP"]
        self.power : int = self.chassis_property[self.level]["POWER"]
        self.max_heat : int = self.gimbal_property[self.level]["HEAT"]
        self.cool_down : int = self.gimbal_property[self.level]["COOL_DOWN"]

    def get_position(self):
        """获取位置"""
        if self._body is None:
            return self._init_pos
        return self._body.position.x, self._body.position.y
    
    def get_velocity(self):
        """获取速度"""
        if self._body is None:
            return (0, 0)
        return self._body.velocity.x, self._body.velocity.y
    
    def set_velocity(self, velocity: Tuple[float, float]):
        """设置速度"""
        if self._body is None:
            return
        self._body.velocity = pymunk.Vec2d(velocity[0], velocity[1])

    def step(self, dt):
        """沿路径点导航"""
        # 若非存活
        if not self.is_alive:
            if self._body is not None:
                self._body.velocity = (0, 0)
            # 结算复活进度
            self.revive_progress = min(self.revive_progress + self.revive_efficiency * dt, self.revive_target)
            if self.revive_progress >= self.revive_target:
                self.is_alive = True
                self.last_revive_time = time.time()
                self.hp = int(self.max_hp * 0.2)
                self.heat = 0
                self.revive_progress = 0
                self.revive_target += 10
            return

        # 结算复活无敌时间
        if time.time() - self.last_revive_time < 10:
            self.defense_buff_dict["revive"] = 2.0
        else:
            self.defense_buff_dict.pop("revive", None)

        # # 沿路径移动
        # if self.path_points and self.current_path_idx < len(self.path_points):
        #     next_point = self.path_points[self.current_path_idx]
        #     current_pos = pygame.math.Vector2(self.get_position())
        #     direction = pygame.math.Vector2(next_point) - current_pos
        #     if direction.length() < 0.05:
        #         self.current_path_idx += 1
        #     else:
        #         direction = direction.normalize() * self.forward_speed
        #         self._body.velocity = (direction.x, direction.y)
        # else:
        #     self._body.velocity = (0, 0)

        # 结算热量冷却
        self.heat = max(0, self.heat - self.cool_down * dt)

        # 结算脱战状态
        if time.time() - self.last_in_combat_time > 6:
            self.attack_target = None

        # 计算最高增益
        self.damage_buff = max(self.damage_buff_dict.values(), default=0.0)
        self.defense_buff = max(self.defense_buff_dict.values(), default=0.0)
        self.defense_debuff = max(self.defense_debuff_dict.values(), default=0.0)
        self.cool_down_buff = min(self.cool_down_buff_dict.values(), default=1.0)
        self.power_buff = max(self.power_buff_dict.values(), default=1.0)

    def attack(self, target_robot) -> bool:
        """攻击目标机器人
        Args:
            target_robot: 目标机器人对象
            num: 攻击次数
        Returns:
            bool: 是否成功攻击
        """
        # 检查是否可以攻击
        if not self.is_alive or self.gun_locked or not target_robot or not target_robot.is_alive:
            return False

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

        # 检查是否可以攻击
        if self.ammo <= 0 or self.ammo_allowed <= 0 or self.heat + self.bullet.HEAT > self.max_heat:
            self.attack_target = None
            return False
        
        # 造成伤害
        damage = self.attack_target.take_damage(self.bullet.DAMAGE * (1 + self.damage_buff))
        # 增加热量
        self.heat += self.bullet.HEAT
        # 减少子弹
        self.ammo -= 1
        self.ammo_allowed -= 1
        # 结算经验
        self.update_exp(self.bullet.EXP)
        self.update_exp(damage * 4)

        return True

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

    def take_damage(self, damage) -> int:
        """受到伤害
        Args:
            damage: 伤害值
        Returns:
            int: 实际受到的伤害值
        """
        if not self.is_alive:
            return False

        # 受击刷新战斗状态
        self.last_in_combat_time = time.time()

        # TODO: 考虑命中
        damage = int(min(damage * max(0, 1 - self.defense_buff + self.defense_debuff), self.hp))
        self.hp = self.hp - damage
        if self.hp <= 0:
            self.is_alive = False
            self.gun_locked = True
        return damage

    def heal(self, amount: int):
        """恢复血量"""
        if self.is_alive:
            self.hp = int(min(self.max_hp, self.hp + amount))

    def set_grid_map(self, grid_map: GridMap):
        """设置网格地图"""
        self.grid_map = grid_map
