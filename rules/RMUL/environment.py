import pymunk
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from rules.base.environment import Environment
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, RobotType, ROBOT_ID
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.robot import Robot
from utils.utils import point_in_polygon, opposite_team, has_line_of_sight, attack_sight_clear

from rules.rmul.config import env_config as RMUL_ENV_CONFIG
from rules.rmul.config.robot_config import RMUL_ROBOT_CONFIGS


@dataclass
class ActionRMUL:
    navigation_target_norm: Tuple[float, float] = (0.0, 0.0)
    navigation_set: int = 0
    attack_target: int = 0
    spin: int = 0
    purchase: int = 0

    def __post_init__(self):
        """初始化"""
        try:
            if isinstance(self.navigation_target_norm, np.ndarray):
                self.navigation_target_norm = tuple(self.navigation_target_norm.astype(float))
            else:
                self.navigation_target_norm = tuple(self.navigation_target_norm)
        except:
            self.navigation_target_norm = (0.0, 0.0)
        try:
            self.navigation_set = int(self.navigation_set)
        except:
            self.navigation_set = 0
        try:
            self.attack_target = int(self.attack_target)
        except:
            self.attack_target = 0
        try:
            self.spin = int(self.spin)
        except:
            self.spin = 0
        try:
            self.purchase = int(self.purchase)
        except:
            self.purchase = 0

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            *self.navigation_target_norm,
            self.navigation_set,
            self.attack_target,
            self.spin,
            self.purchase,
        ])

    def get_target_position(self) -> Tuple[float, float]:
        """获取导航目标的实际坐标"""
        return (np.array(self.navigation_target_norm) + 1) * np.array([RMUL_ENV_CONFIG.FIELD_WIDTH, RMUL_ENV_CONFIG.FIELD_HEIGHT]) / 2

@dataclass
class GameObsRMUL:
    remaining_time_norm: float
    victory_progress_red_norm: float
    victory_progress_blue_norm: float
    # economics_red: int
    # economics_blue: int

    @classmethod
    def from_array(cls, array: np.ndarray):
        assert array.shape == (3,)
        return cls(
            remaining_time_norm=array[0],
            victory_progress_red_norm=array[1],
            victory_progress_blue_norm=array[2],
        )

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            self.remaining_time_norm,
            self.victory_progress_red_norm,
            self.victory_progress_blue_norm,
            # self.economics_red,
            # self.economics_blue,
        ])

@dataclass
class RobotObsRMUL:
    position: np.ndarray
    target_position_norm: np.ndarray
    chassis_property_type: int
    gimbal_property_type: int
    level: int
    exp_norm: float
    hp_norm: float
    heat_norm: float
        
    @classmethod
    def from_robot(cls, robot: Robot):
        return cls(
            position=np.array(robot.get_position()),
            target_position_norm=np.array(robot.target_pos) / np.array([RMUL_ENV_CONFIG.FIELD_WIDTH, RMUL_ENV_CONFIG.FIELD_HEIGHT]),
            chassis_property_type=robot.chassis_property_type.value,
            gimbal_property_type=robot.gimbal_property_type.value,
            level=robot.level,
            exp_norm=(robot.exp - LEVEL_NEED_EXP[robot.level]) / (LEVEL_NEED_EXP[robot.level + 1] - LEVEL_NEED_EXP[robot.level]) if robot.level < len(LEVEL_NEED_EXP) else 1,
            hp_norm=robot.hp / robot.max_hp,
            heat_norm=robot.heat / robot.max_heat,
        )

    @classmethod
    def from_array(cls, array: np.ndarray):
        assert array.shape == (10,)
        return cls(
            position=array[:2] * np.array([RMUL_ENV_CONFIG.FIELD_WIDTH, RMUL_ENV_CONFIG.FIELD_HEIGHT]),
            target_position_norm=array[2:4],
            chassis_property_type=math.ceil(array[4]),
            gimbal_property_type=math.ceil(array[5]),
            level=math.ceil(array[6]),
            exp_norm=array[7],
            hp_norm=array[8],
            heat_norm=array[9],
        )

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            *(self.position / np.array([RMUL_ENV_CONFIG.FIELD_WIDTH, RMUL_ENV_CONFIG.FIELD_HEIGHT])),
            *(self.target_position_norm),
            self.chassis_property_type,
            self.gimbal_property_type,
            self.level,
            self.exp_norm,
            self.hp_norm,
            self.heat_norm,
        ])

@dataclass
class ObservationRMUL:
    # TODO
    game_obs: GameObsRMUL
    robot_obs: Dict[str, RobotObsRMUL]
    
    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.concatenate([
            self.game_obs.to_array(),
            *[robot_obs.to_array() for _, robot_obs in sorted(
                self.robot_obs.items(),
                key=lambda x: (x[0].split('_')[0], -int(x[0].split('_')[1])),
                reverse=True
            )]
        ])

class EnvironmentRMUL(Environment):
    def __init__(self):
        # 创建物理引擎
        self.env_config = RMUL_ENV_CONFIG
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力
        self.dt = 1 / self.env_config.FPS

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.total_time = self.env_config.GAME_TIME_LIMIT       # 总时长
        self._remaining_time = self.env_config.GAME_TIME_LIMIT  # 剩余时间

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(self.env_config.OBSTACLES)

        # 创建增益区
        self.buff_zone = {}
        self.buff_zone["center"] = self.env_config.CENTER_ZONE_VERTICES
        self.buff_zone["boot_red"] = self.env_config.BOOT_ZONE_RED_VERTICES
        self.buff_zone["boot_blue"] = self.env_config.BOOT_ZONE_BLUE_VERTICES

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self.robot_configs = RMUL_ROBOT_CONFIGS
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(self.env_config)

        # 游戏机制
        self._economics = {GameTeam.RED: 0, GameTeam.BLUE: 0}           # 经济
        self._victory_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}    # 胜利进度
        self._laggard_bonus_taken = {                                   # 落后奖励
            "red_lag_70": False,
            "red_lag_140": False,
            "blue_lag_70": False,
            "blue_lag_140": False,
        }

    def step(self, dt: float, red_action: Dict[str, ActionRMUL], blue_action: Dict[str, ActionRMUL]):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)

        # 应用动作
        # 为了使结算效果与红蓝先后解耦，我们先统一应用运动动作，再应用攻击动作
        self._apply_team_motion(red_action)
        self._apply_team_motion(blue_action)
        self._apply_team_attack(GameTeam.RED, red_action)
        self._apply_team_attack(GameTeam.BLUE, blue_action)

        # 更新机器人状态
        for robot in self.robots.values():
            robot.step(dt)

        # 检查中心区域占领情况
        robots_in_center_zone = {GameTeam.RED: False, GameTeam.BLUE: False} # 机器人是否在中心区域
        for robot in self.robots.values():
            if robot.is_alive and point_in_polygon(robot.get_position(), self.buff_zone["center"]):
                robots_in_center_zone[robot.team] = True
        
        # 检查补给区占领情况
        for robot in self.robots.values():
            if robot.team == GameTeam.RED and point_in_polygon(robot.get_position(), self.buff_zone["boot_red"]) or \
                robot.team == GameTeam.BLUE and point_in_polygon(robot.get_position(), self.buff_zone["boot_blue"]):
                # 解锁发射机构
                robot.gun_locked = False
                # 为防止血量计算中出现小数，仅在整数秒时一次性回复血量
                if 0 < math.modf(time.time())[0] < dt:
                    robot.heal(int(robot.max_hp * 0.25))

        # 更新游戏状态# 倒计时减少
        self._remaining_time = max(0, self._remaining_time - dt)

        # 更新经济
        if 0 < 300 - self._remaining_time < dt:
            self._economics[GameTeam.RED] += 200
            self._economics[GameTeam.BLUE] += 200
        elif 0 < 240 - self._remaining_time < dt:
            self._economics[GameTeam.RED] += 200
            self._economics[GameTeam.BLUE] += 200
        elif 0 < 180 - self._remaining_time < dt:
            self._economics[GameTeam.RED] += 200
            self._economics[GameTeam.BLUE] += 200
        elif 0 < 120 - self._remaining_time < dt:
            self._economics[GameTeam.RED] += 300
            self._economics[GameTeam.BLUE] += 300
        elif 0 < 60 - self._remaining_time < dt:
            self._economics[GameTeam.RED] += 300
            self._economics[GameTeam.BLUE] += 300

        # 更新中心区域进度
        for team, has_robot in robots_in_center_zone.items():
            if has_robot:  # 如果该队伍有机器人在中心区域
                self._victory_progress[team] += dt  # 只要有一个机器人在区域中就增加进度

        # 检查胜利条件
        red_progress = self._victory_progress[GameTeam.RED]
        blue_progress = self._victory_progress[GameTeam.BLUE]
        target = self.env_config.OCCUPATION_TARGET

        # 结算落后奖励
        if blue_progress - red_progress >= 70 and not self._laggard_bonus_taken["red_lag_70"]:
            self._economics[GameTeam.RED] += 200
            self._laggard_bonus_taken["red_lag_70"] = True
        elif blue_progress - red_progress >= 140 and not self._laggard_bonus_taken["red_lag_140"]:
            self._economics[GameTeam.RED] += 200
            self._laggard_bonus_taken["red_lag_140"] = True
        elif red_progress - blue_progress >= 70 and not self._laggard_bonus_taken["blue_lag_70"]:
            self._economics[GameTeam.BLUE] += 200
            self._laggard_bonus_taken["blue_lag_70"] = True
        elif red_progress - blue_progress >= 140 and not self._laggard_bonus_taken["blue_lag_140"]:
            self._economics[GameTeam.BLUE] += 200
            self._laggard_bonus_taken["blue_lag_140"] = True

        # 1. 有一方率先积满
        if red_progress >= target > blue_progress:
            self.game_state = GameState.RED_TEAM_WIN
        elif blue_progress >= target > red_progress:
            self.game_state = GameState.BLUE_TEAM_WIN
        # 2. 同时积满
        elif min(red_progress, blue_progress) >= target:
            self.game_state = GameState.DRAW
        # 3. 时间到
        elif self._remaining_time <= 0:
            if red_progress > blue_progress:
                self.game_state = GameState.RED_TEAM_WIN
            elif blue_progress > red_progress:
                self.game_state = GameState.BLUE_TEAM_WIN
            else:
                self.game_state = GameState.DRAW

    def _apply_team_motion(self, action: Dict[str, ActionRMUL]):
        super()._apply_team_motion(action)

    def _apply_robot_attack(self, robot_attacker: Robot, robot_target: Robot):
        super()._apply_robot_attack(robot_attacker, robot_target)

    def _apply_team_attack(self, team: GameTeam, action: Dict[str, ActionRMUL]):
        """应用攻击动作。"""
        for robot_id, robot_action in action.items():
            # 攻击者
            robot_attacker = self.get_robot(robot_id)
            # 攻击目标类型
            target_type = RobotType(robot_action.attack_target)
            # 目标为空
            if target_type == RobotType.NONE:
                continue
            # 被攻击者
            robot_target = self.get_robot(ROBOT_ID[opposite_team(team)][target_type])
            self._apply_robot_attack(robot_attacker, robot_target)

            # 被攻击目标阵亡
            if not robot_target.is_alive:
                # 结算击杀经验
                if robot.robot_type == RobotType.SENTRY:
                    killer_level = self.get_robot(ROBOT_ID[team][RobotType.STANDARD_3]).level
                    kill_exp = 50 * robot_target.level * (1 + max(0, 0.2 * (robot_target.level - killer_level)))
                    robot_alive = [robot for robot in self.robots.values() if robot.team == team and robot.is_alive]
                    # 经验分享
                    for robot in robot_alive:
                        robot.update_exp(int(kill_exp / len(robot_alive)))
                else:
                    kill_exp = 50 * robot_target.level * (1 + max(0, 0.2 * (robot_target.level - robot_attacker.level)))
                    robot.update_exp(int(kill_exp))
                # 结算胜利进度
                self._victory_progress[team] += 20

        # 购买允许发弹量
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)
            if robot_action.purchase and robot.robot_type != RobotType.SENTRY:
                if self._economics[team] >= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM:
                    if point_in_polygon(robot.get_position(), self.buff_zone["boot_red"] if team == GameTeam.RED else self.buff_zone["boot_blue"]):
                        robot.ammo_allowed += robot.bullet.PURCHASE_NUM
                        self._economics[team] -= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM


    # # TODO: 奖励函数
    # def _calculate_reward(self) -> float:
    #     """计算奖励"""
    #     reward = 0.0
        
    #     # 获取红方机器人
    #     red_robot = self.get_robot("RED_3_STANDARD")
    #     if red_robot is None:
    #         return reward
        
    #     # # 存活奖励
    #     # if red_robot.is_alive:
    #     #     reward += 0.1
        
    #     # 中心区域距离奖励
    #     robot_pos = red_robot.get_position()
    #     distance = math.sqrt((robot_pos[0] - env_config.FIELD_WIDTH / 2) ** 2 + (robot_pos[1] - env_config.FIELD_HEIGHT / 2) ** 2)
    #     reward_distance = 10 * (1 - distance / env_config.FIELD_WIDTH)
    #     reward += reward_distance
    #     print(f"中心区域距离奖励: {reward_distance}")
        
    #     # 中心区域占领奖励
    #     if self.robots_in_zone[GameTeam.RED]:
    #         reward_zone = 0.2
    #     else:
    #         reward_zone = -0.2
    #     reward += reward_zone
    #     print(f"中心区域占领奖励: {reward_zone}")        
        
    #     # 击杀奖励
    #     blue_robots = [self.get_robot(f"BLUE_{i}") for i in [3, 4, 5, 7]]
    #     for robot in blue_robots:
    #         if robot is not None and not robot.is_alive:
    #             reward_kill = 1.0
    #             reward += reward_kill
    #             print(f"击杀奖励: {reward_kill}")
        
    #     # 胜利奖励
    #     if self.game_state_manager.state.value == 2:  # 红方胜利
    #         reward_win = 10.0
    #         reward += reward_win
    #         print(f"胜利奖励: {reward_win}")
        
    #     return reward

    def get_top_bar_info(self) -> Dict[str, Any]:
        """获取渲染顶部信息"""
        return {
            "remaining_time": self._remaining_time,
            "victory_progress": self._victory_progress,
            "economics": self._economics,
        }
