import pygame
import math
import time
import torch
from typing import List, Dict, Tuple, Optional

from config import DEVICE
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
        init_pos: torch.Tensor,
        chassis_property_type: CHASSIS_PROPERTY_TYPE,
        gimbal_property_type: GIMBAL_PROPERTY_TYPE,
        forward_speed_efficiency: torch.Tensor,
        rotation_speed_efficiency: torch.Tensor,
        radius: torch.Tensor,
        max_ammo: torch.Tensor,
        ammo_allowed: torch.Tensor,
    ):
        # 全局属性
        self.team : GameTeam = team
        self.robot_type : RobotType = robot_type
        self.id : str = ROBOT_ID[team][robot_type]

        # 规则性能
        self.level : torch.Tensor = torch.tensor(1, dtype=torch.int, device=DEVICE)
        self.exp : torch.Tensor = torch.tensor(0, dtype=torch.int, device=DEVICE)
        self.chassis_property_type : CHASSIS_PROPERTY_TYPE = chassis_property_type
        self.gimbal_property_type : GIMBAL_PROPERTY_TYPE = gimbal_property_type

        # 英雄
        if self.robot_type == RobotType.HERO:
            self.chassis_property = CHASSIS_PROPERTY_HERO[self.chassis_property_type.value, :, :]
            self.gimbal_property = GIMBAL_PROPERTY_42[GIMBAL_PROPERTY_TYPE.DEFAULT.value, :, :]
            self.bullet = LargeBullet()
        # 步兵
        elif self.robot_type in [RobotType.STANDARD_3, RobotType.STANDARD_4, RobotType.STANDARD_5]:
            self.chassis_property = CHASSIS_PROPERTY_STANDARD[self.chassis_property_type.value, :, :]
            self.gimbal_property = GIMBAL_PROPERTY_17[self.gimbal_property_type.value, :, :]
            self.bullet = SmallBullet()
        # 哨兵
        elif self.robot_type == RobotType.SENTRY:
            self.level = torch.tensor(10, dtype=torch.int, device=DEVICE)
            self.exp = torch.tensor(LEVEL_NEED_EXP[self.level], dtype=torch.float, device=DEVICE)
            self.chassis_property = CHASSIS_PROPERTY_STANDARD[CHASSIS_PROPERTY_TYPE.HP.value, :, :]
            self.gimbal_property = GIMBAL_PROPERTY_17[GIMBAL_PROPERTY_TYPE.COOL_DOWN.value, :, :]
            self.bullet = SmallBullet()

        # 更新性能
        self.update_property()
        self.hp : torch.Tensor = self.max_hp
        self.heat : torch.Tensor = torch.tensor(0, dtype=torch.int, device=DEVICE)
        
        # 物理属性
        self.position : torch.Tensor = init_pos
        self.velocity : torch.Tensor = torch.tensor([0, 0], dtype=torch.float, device=DEVICE)
        self.radius : torch.Tensor = radius
        self.forward_speed : torch.Tensor = self.power * forward_speed_efficiency
        self.rotation_speed : torch.Tensor = self.power * rotation_speed_efficiency

        self.angle : torch.Tensor = torch.tensor(0, dtype=torch.float, device=DEVICE)  # 角度（度）
        self.target_pos : torch.Tensor = None
        self.path_points : torch.Tensor = torch.tensor([], dtype=torch.float, device=DEVICE)
        self.current_path_idx : int = 0
        self.is_alive : bool = True  # 机器人是否存活

        # 弹丸相关
        self.max_ammo : torch.Tensor = max_ammo      # 最大弹药量
        self.ammo : torch.Tensor = self.max_ammo     # 当前弹药量
        self.ammo_allowed : torch.Tensor = ammo_allowed # 允许发弹量

        # 攻击相关
        self.gun_locked : bool = False          # 发射机构锁定
        self.attack_target : Robot = None       # 攻击目标
        self.last_attack_time : float = 0       # 上次攻击的时间（秒）
        self.last_in_combat_time : float = 0    # 上次进入战斗的时间（秒）

        # 复活相关
        self.revive_progress : torch.Tensor = torch.tensor(0, dtype=torch.float, device=DEVICE)    # 复活进度
        self.revive_target : torch.Tensor = torch.tensor(10, dtype=torch.float, device=DEVICE)     # 复活所需进度
        self.revive_efficiency : torch.Tensor = torch.tensor(2, dtype=torch.float, device=DEVICE)  # 复活效率（每秒增加的进度）
        self.last_revive_time : torch.Tensor = torch.tensor(0.0, dtype=torch.float, device=DEVICE) # 上次复活时间

        # TODO: 增益相关
        self.damage_buff_dict : Dict[str, torch.Tensor] = {"default": torch.tensor(0.0, dtype=torch.float, device=DEVICE)}        # 伤害增益
        self.damage_buff : torch.Tensor = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
        self.defense_buff_dict : Dict[str, torch.Tensor] = {"default": torch.tensor(0.0, dtype=torch.float, device=DEVICE)}       # 防御增益
        self.defense_buff : torch.Tensor = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
        self.defense_debuff_dict : Dict[str, torch.Tensor] = {"default": torch.tensor(0.0, dtype=torch.float, device=DEVICE)}     # 防御减益
        self.defense_debuff : torch.Tensor = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
        self.cool_down_buff_dict : Dict[str, torch.Tensor] = {"default": torch.tensor(1.0, dtype=torch.float, device=DEVICE)}     # 冷却缩减增益
        self.cool_down_buff : torch.Tensor = torch.tensor(1.0, dtype=torch.float, device=DEVICE)
        self.power_buff_dict : Dict[str, torch.Tensor] = {"default": torch.tensor(1.0, dtype=torch.float, device=DEVICE)}         # 功率增益
        self.power_buff : torch.Tensor = torch.tensor(1.0, dtype=torch.float, device=DEVICE)

        # GridMap相关
        self.grid_map : GridMap = None

    def update_exp(self, exp: torch.Tensor):
        """更新经验"""
        self.exp = min(self.exp + exp, LEVEL_NEED_EXP.max())
        # 升级
        if self.level < LEVEL_NEED_EXP.shape[0] and self.exp >= LEVEL_NEED_EXP[self.level + 1]:
            self.level += 1
            max_hp_before_upgrade = self.max_hp
            self.update_property()
            self.heal(torch.floor(self.max_hp / max_hp_before_upgrade * self.hp - self.hp))

    def update_property(self):
        """更新机器人属性"""
        self.max_hp : torch.Tensor = self.chassis_property[self.level, 0]
        self.power : torch.Tensor = self.chassis_property[self.level, 1]
        self.max_heat : torch.Tensor = self.gimbal_property[self.level, 0]
        self.cool_down : torch.Tensor = self.gimbal_property[self.level, 1]

    def set_target(self, target_pos):
        """设置目标位置并计算路径"""
        # 如果目标位置与当前位置相同，则不再次计算路径
        if target_pos == self.target_pos:
            return
        else:
            self.target_pos = target_pos

        # 如果网格地图为空
        if self.grid_map is None:
            self.path_points = torch.tensor([target_pos], dtype=torch.float, device=DEVICE)
            self.current_path_idx = 0
            raise ValueError("网格地图为空，无法计算路径")

        # 使用A*算法规划路径
        start_grid = world_to_grid(self.position)
        goal_grid = world_to_grid(target_pos)
        path_grids = a_star(self.grid_map, start_grid, goal_grid)
        path_grids = simplify_path(path_grids, self.grid_map)
        
        # 将栅格坐标转换回世界坐标
        self.path_points = torch.stack([grid_to_world(gp) for gp in path_grids[1:]], dim=0)
        self.current_path_idx = 0

    def step(self, dt):
        """沿路径点导航"""
        # 若非存活
        if not self.is_alive:
            self.velocity = torch.tensor([0, 0], dtype=torch.float, device=DEVICE)
            # 结算复活进度
            self.revive_progress = min(self.revive_progress + self.revive_efficiency * dt, self.revive_target)
            if self.revive_progress >= self.revive_target:
                self.is_alive = True
                self.last_revive_time = time.time()
                self.hp = torch.floor(self.max_hp * 0.2)
                self.heat = 0
                self.revive_progress = 0
                self.revive_target += 10
            return

        # 结算复活无敌时间
        if time.time() - self.last_revive_time < 10:
            self.defense_buff_dict["revive"] = 2.0
        else:
            self.defense_buff_dict.pop("revive", None)

        # 沿路径移动
        if self.path_points.numel() > 0 and self.current_path_idx < self.path_points.shape[0]:
            next_point = self.path_points[self.current_path_idx]
            direction = next_point - self.position
            if direction.norm() < 0.05:
                self.current_path_idx += 1
            else:
                self.velocity = direction.normalize() * self.forward_speed
        else:
            self.velocity = torch.tensor([0, 0], dtype=torch.float, device=DEVICE)
        self.position = self.position + self.velocity * dt

        # 结算热量冷却
        self.heat = torch.maximum(self.heat - self.cool_down * dt, torch.tensor(0, dtype=torch.int, device=DEVICE))

        # 结算脱战状态
        if time.time() - self.last_in_combat_time > 6:
            self.attack_target = None

        # 计算最高增益
        self.damage_buff = torch.max(torch.stack(list(self.damage_buff_dict.values())))
        self.defense_buff = torch.max(torch.stack(list(self.defense_buff_dict.values())))
        self.defense_debuff = torch.max(torch.stack(list(self.defense_debuff_dict.values())))
        self.cool_down_buff = torch.min(torch.stack(list(self.cool_down_buff_dict.values())))
        self.power_buff = torch.max(torch.stack(list(self.power_buff_dict.values())))

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
        target_pos = self.attack_target.position
        current_pos = self.position
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
        return (self.position, self.attack_target.position)

    def take_damage(self, damage: torch.Tensor) -> torch.Tensor:
        """受到伤害
        Args:
            damage: 伤害值
        Returns:
            int: 实际受到的伤害值
        """
        if not self.is_alive:
            return torch.tensor(0, dtype=torch.int, device=DEVICE)

        # 受击刷新战斗状态
        self.last_in_combat_time = time.time()

        # TODO: 考虑命中
        damage = torch.floor(torch.min(damage * torch.max(0, 1 - self.defense_buff + self.defense_debuff), self.hp))
        self.hp = self.hp - damage
        if self.hp <= 0:
            self.is_alive = False
            self.gun_locked = True
        return damage

    def heal(self, amount: torch.Tensor):
        """恢复血量"""
        if self.is_alive:
            self.hp = torch.floor(torch.clamp(self.hp + amount, 0, self.max_hp))

    def set_grid_map(self, grid_map: GridMap):
        """设置网格地图"""
        self.grid_map = grid_map
