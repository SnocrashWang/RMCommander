import pygame
import pymunk
import math
import time
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from utils.config.exp_prop_config import *
from utils.config.bullet_config import *
from utils.config.robot_config import ROBOT_ID, RobotType
from utils.config.game_config import GameTeam
from utils.buff import BuffType, Buff, BuffManager
from utils.grid_map import GridMap, a_star, world_to_grid, grid_to_world, simplify_path

WAYPOINT_REACHED_DISTANCE = 0.05

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
        forward_rotation_allocation: float,
        shoot_frequency: int,
        radius: float,
        max_ammo: int,
        ammo_allowed: int,
        level: int,
        hp: int,
        heat: float,
        enable_exp: bool = True,
        physics_engine: Optional[pymunk.Space] = None,
    ):
        # 全局属性
        self.team : GameTeam = team
        self.robot_type : RobotType = robot_type
        self.id : str = ROBOT_ID[team][robot_type]

        # 规则性能
        self.level : int = np.clip(level, 1, 10) if level else 1
        self.exp : int = 0
        self.chassis_property_type : CHASSIS_PROPERTY_TYPE = chassis_property_type
        self.gimbal_property_type : GIMBAL_PROPERTY_TYPE = gimbal_property_type
        self.enable_exp = enable_exp
        self.forward_speed_efficiency = forward_speed_efficiency
        self.rotation_speed_efficiency = rotation_speed_efficiency
        self.forward_rotation_allocation = min(1.0, max(0.0, forward_rotation_allocation))
        self.spin_enabled : bool = False

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
            self.gimbal_property = GIMBAL_PROPERTY_17[GIMBAL_PROPERTY_TYPE.COOLDOWN]
            self.bullet = SmallBullet()

        # 更新性能
        self.update_property()
        self.hp : int = np.clip(hp, 0, self.max_hp) if hp else self.max_hp
        self.heat : float = np.clip(heat, 0, self.max_heat) if heat else 0
        
        # 物理属性
        self.radius : float = radius
        self.forward_speed : float = 0.0
        self.rotation_speed : float = 0.0
        self.shoot_frequency : int = shoot_frequency

        # 创建物理实体
        self._body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, radius))
        self._body.position = init_pos

        self._shape = pymunk.Circle(self._body, radius)
        self._shape.elasticity = 0.8
        self._shape.friction = 0.7
        self._shape.filter = pymunk.ShapeFilter(categories=0b1, mask=0b1)

        if physics_engine:
            physics_engine.add(self._body, self._shape)

        self.angle_gimbal : float = 0                       # 云台角度（弧度）
        self.angle_chassis : float = 0                      # 底盘角度（弧度）
        self.target_pos : Tuple[float, float] = (0, 0)      # 导航坐标
        self.path_points : List[Tuple[float, float]] = []   # 导航路径点集
        self.current_path_idx : int = 0                     # 当前目标导航路径点
        self.is_alive : bool = True                         # 机器人是否存活

        # 弹丸相关
        self.max_ammo : int = max_ammo                      # 最大弹药量
        self.ammo : int = self.max_ammo                     # 当前弹药量
        self.ammo_allowed : int = ammo_allowed              # 允许发弹量

        # 攻击相关
        self.gun_locked : bool = False                      # 发射机构锁定
        self.attack_target : Robot = None                   # 攻击目标
        self.last_attack_time_real : float = 0              # 上次攻击的真实时间（秒）
        self.last_attack_time_remain : float = math.inf     # 上次攻击的倒计时时间（秒）
        self.last_in_combat_time_remain : float = math.inf  # 上次进入战斗的倒计时时间（秒）

        # 复活相关
        self.revive_progress : float = 0                    # 复活进度
        self.revive_target : float = 10                     # 复活所需进度
        self.revive_efficiency : float = 2                  # 复活效率（每秒增加的进度）
        self.last_revive_time_remain : float = math.inf     # 上次复活的倒计时时间

        self.buff_manager = BuffManager()                  # 增益管理器

        # GridMap相关
        self._grid_map : GridMap = None

    def __deepcopy__(self, memo):
        """自定义深拷贝行为"""
        # 创建新实例（不调用__init__）
        new_robot = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_robot
        
        # 只拷贝不以_开头的属性
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # 忽略私有属性
                setattr(new_robot, key, deepcopy(value, memo))
            else:
                # 可选择设为None或保留原引用
                setattr(new_robot, key, None)  # 或者不设置
        
        return new_robot

    def random_start(self):
        pass

    def destroy_physics_body(self, physics_engine: pymunk.Space):
        """从物理引擎中移除物理体"""
        if self._body is not None and self._shape is not None:
            physics_engine.remove(self._shape, self._body)

    def print_info(self):
        info = {
            "id": self.id,
            "position": self.get_position(),
            "angle_gimbal": self.angle_gimbal,
            # "angle_chassis": self.angle_chassis,
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
        if not self.enable_exp:
            return
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
        self.cooldown : int = self.gimbal_property[self.level]["COOLDOWN"]

    def add_buff(self, buff: Buff):
        self.buff_manager.add_buff(buff)

    def remove_buff(self, buff: Buff, strict: bool = False):
        self.buff_manager.remove_buff(buff, strict)

    def set_spin(self, spin_enabled: bool):
        """设置是否启用自旋防御姿态。"""
        self.spin_enabled = bool(spin_enabled)

    def _resolve_motion_speed(self, moving: bool):
        """根据平移/自旋组合解算本帧平移速度和旋转速度。"""
        if moving and self.spin_enabled:
            forward_power = self.power * (1 - self.forward_rotation_allocation)
            rotation_power = self.power * self.forward_rotation_allocation
        elif moving:
            forward_power = self.power
            rotation_power = 0.0
        elif self.spin_enabled:
            forward_power = 0.0
            rotation_power = self.power
        else:
            forward_power = 0.0
            rotation_power = 0.0

        self.forward_speed = forward_power * self.forward_speed_efficiency
        self.rotation_speed = rotation_power * self.rotation_speed_efficiency

    def refresh_motion_speed(self):
        """刷新当前动作组合对应的速度，用于攻击结算前启用自旋防御。"""
        moving = False
        if self.is_alive and self.path_points and self.current_path_idx < len(self.path_points):
            next_point = self.path_points[self.current_path_idx]
            current_pos = pygame.math.Vector2(self.get_position())
            moving = (pygame.math.Vector2(next_point) - current_pos).length() > WAYPOINT_REACHED_DISTANCE
        self._resolve_motion_speed(moving)

    def get_position(self):
        """获取位置"""
        if self._body is None:
            return self._init_pos
        return self._body.position.x, self._body.position.y

    def has_buff(self, buff: Buff):
        return buff in self.buff_manager._buff_list

    def set_target(self, target_pos):
        """设置目标位置并计算路径"""
        # 如果目标位置与当前位置相同，则不再次计算路径
        if target_pos == self.target_pos:
            return
        else:
            self.target_pos = target_pos

        # 如果网格地图为空
        if self._grid_map is None:
            self.path_points = [target_pos]
            self.current_path_idx = 0
            raise ValueError("网格地图为空，无法计算路径")

        # 使用A*算法规划路径
        start_grid = world_to_grid(self.get_position())
        goal_grid = world_to_grid(target_pos)
        path_grids = a_star(self._grid_map, start_grid, goal_grid)
        path_grids = simplify_path(path_grids, self._grid_map)
        
        # 将栅格坐标转换回世界坐标
        self.path_points = [grid_to_world(gp[0], gp[1]) for gp in path_grids[1:]]
        self.current_path_idx = 0

    def step(self, dt, remaining_time):
        """沿路径点导航"""
        # 判断存活
        if self.hp <= 0:
            self.is_alive = False
            self.gun_locked = True
        # 若非存活
        if not self.is_alive:
            if self._body is not None:
                self._body.velocity = (0, 0)
                self._body.angular_velocity = 0
            self.forward_speed = 0.0
            self.rotation_speed = 0.0
            self.buff_manager.clear_buff()
            # 结算复活进度
            self.revive_progress = min(self.revive_progress + self.revive_efficiency * dt, self.revive_target)
            # 复活读条已满
            if self.revive_progress >= self.revive_target:
                self.is_alive = True
                self.last_revive_time_remain = remaining_time
                self.hp = int(self.max_hp * 0.2)
                self.heat = 0
                self.revive_progress = 0
                self.revive_target += 10
            else:
                return

        # 结算复活无敌时间
        if self.last_revive_time_remain - remaining_time < 10:
            self.buff_manager.add_buff(Buff(name="revive", defence=100.0))    # 给足饱和防御buff，防止被易伤抵消
        else:
            self.buff_manager.remove_buff(Buff(name="revive", defence=100.0), strict=False)

        # 结算增益
        self.buff_manager.update_buff_list()

        # 沿路径移动
        moving = False
        if self.path_points and self.current_path_idx < len(self.path_points):
            next_point = self.path_points[self.current_path_idx]
            current_pos = pygame.math.Vector2(self.get_position())
            direction = pygame.math.Vector2(next_point) - current_pos
            # 若距离目标点已经小于一帧将移动的距离，则停止
            distance = direction.length()
            if distance <= max(WAYPOINT_REACHED_DISTANCE, self.forward_speed * dt):
                self.current_path_idx += 1
                self._body.velocity = (0, 0)
            else:
                moving = True
                direction = direction.normalize() * self.forward_speed
                self._body.velocity = (direction.x, direction.y)
        else:
            self._body.velocity = (0, 0)

        # 分配当前功率
        self._resolve_motion_speed(moving)

        # 处理自旋
        self._body.angular_velocity = self.rotation_speed
        if self.rotation_speed > 0:
            self.angle_chassis = self.angle_chassis + self.rotation_speed * dt

        # 结算热量冷却
        cooldown = max(self.cooldown * self.buff_manager.get_buff(BuffType.COOLDOWN_RATE), self.cooldown + self.buff_manager.get_buff(BuffType.COOLDOWN_CONST))
        self.heat = max(0, self.heat - cooldown * dt)

        # 结算脱战状态
        if self.last_in_combat_time_remain - remaining_time > 6:
            self.attack_target = None

        # 结算回血增益
        if self.buff_manager.get_buff(BuffType.HEALING):
            # 为防止血量计算中出现小数，仅在整数秒时一次性回复血量
            if 0 < math.modf(remaining_time)[0] < dt:
                self.heal(int(self.max_hp * self.buff_manager.get_buff(BuffType.HEALING)))

        # TODO：结算禁区

    def attack(self, target_robot) -> bool:
        """攻击目标机器人
        Args:
            target_robot: 目标机器人对象
            num: 攻击次数
        Returns:
            bool: 是否成功攻击
        """
        # 检查是否可以攻击
        if not self.is_alive or self.gun_locked:
            return False
        if not target_robot or not target_robot.is_alive:
            return False

        # 刷新战斗状态
        self.attack_target = target_robot

        # 计算攻击角度
        target_pos = self.attack_target.get_position()
        current_pos = self.get_position()
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        self.angle_gimbal = math.atan2(dy, dx)

        # 检查是否可以攻击
        if self.ammo <= 0 or self.ammo_allowed <= 0 or self.heat + self.bullet.HEAT > self.max_heat:
            self.attack_target = None
            return False
        
        # 造成伤害
        damage = self.attack_target.take_damage(self.bullet.DAMAGE * (1 + self.buff_manager.get_buff(BuffType.ATTACK)))
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
        if time.time() - self.last_attack_time_real > 0.2:
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
            return 0

        # 自旋防御：旋转越快，实际受到伤害的概率越低。
        if self.rotation_speed > 0 and np.random.random() > math.exp(-self.rotation_speed / 10):
            return 0

        real_damage = int(min(
            damage * max(0, 1 - self.buff_manager.get_buff(BuffType.DEFENCE) + self.buff_manager.get_buff(BuffType.VULNERABILITY)),   # 不允许造成负伤害
            self.hp
        ))
        self.hp = self.hp - real_damage
        if self.hp <= 0:
            self.is_alive = False
            self.gun_locked = True
        return real_damage

    def heal(self, amount: int):
        """恢复血量"""
        if self.is_alive:
            self.hp = int(min(self.max_hp, self.hp + amount))
