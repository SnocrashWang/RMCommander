import pymunk
import math
import numpy as np
import copy
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, ROBOT_ID, RobotType
from utils.grid_map import GridMap
from utils.robot import Robot
from utils.obstacle import Obstacle
from utils.utils import attack_sight_clear, opposite_team, timer

from rules.base.config import env_config as BASE_ENV_CONFIG
from rules.base.config.robot_config import BASE_ROBOT_CONFIGS, BASE_ROBOT_TYPE_LIST


@dataclass
class Action:
    navigation_target_norm: Tuple[float, float] = (0.0, 0.0)
    navigation_set: int = 0
    attack_target: int = 0
    spin: int = 0

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

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            *self.navigation_target_norm,
            self.navigation_set,
            self.attack_target,
            self.spin,
        ])

    def get_target_position(self) -> Tuple[float, float]:
        """获取导航目标的实际坐标"""
        return (np.array(self.navigation_target_norm) + 1) * np.array([BASE_ENV_CONFIG.FIELD_WIDTH, BASE_ENV_CONFIG.FIELD_HEIGHT]) / 2

@dataclass
class GameObs:
    remaining_time_norm: float

    @classmethod
    def from_array(cls, array: np.ndarray):
        assert array.shape == (1,)
        return cls(
            remaining_time_norm=array[0]
        )

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            self.remaining_time_norm,
        ])

@dataclass
class RobotObs:
    position_norm: np.ndarray
    chassis_property_type: int
    gimbal_property_type: int
    level: int
    exp_norm: float
    hp_norm: float
    heat_norm: float

    @classmethod
    def from_robot(cls, robot: Robot):
        return cls(
            position_norm=np.array(robot.get_position()) / np.array([BASE_ENV_CONFIG.FIELD_WIDTH, BASE_ENV_CONFIG.FIELD_HEIGHT]) * 2 - 1,
            chassis_property_type=robot.chassis_property_type.value,
            gimbal_property_type=robot.gimbal_property_type.value,
            level=robot.level,
            exp_norm=(robot.exp - LEVEL_NEED_EXP[robot.level]) / (LEVEL_NEED_EXP[robot.level + 1] - LEVEL_NEED_EXP[robot.level]) if robot.level < len(LEVEL_NEED_EXP) else 1,
            hp_norm=robot.hp / robot.max_hp,
            heat_norm=robot.heat / robot.max_heat,
        )

    @classmethod
    def from_array(cls, array: np.ndarray):
        assert array.shape == (8,)
        return cls(
            position_norm=array[:2],
            chassis_property_type=math.ceil(array[2]),
            gimbal_property_type=math.ceil(array[3]),
            level=math.ceil(array[4]),
            exp_norm=array[5],
            hp_norm=array[6],
            heat_norm=array[7],
        )

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            *self.position_norm,
            self.chassis_property_type,
            self.gimbal_property_type,
            self.level,
            self.exp_norm,
            self.hp_norm,
            self.heat_norm,
        ])

@dataclass
class Observation:
    game_obs: GameObs
    robot_obs: Dict[str, RobotObs]

    @classmethod
    def from_array(cls, array: np.ndarray):
        game_obs = GameObs.from_array(array[:1])
        robot_array = array[1:]
        robot_obs = {}
        for team in [GameTeam.RED, GameTeam.BLUE]:
            for robot_type in BASE_ROBOT_TYPE_LIST:
                robot_id = ROBOT_ID[team][robot_type]
                robot_obs[robot_id] = RobotObs.from_array(robot_array[:8])
                robot_array = robot_array[8:]
        return cls(game_obs, robot_obs)

    def to_array(self, team: GameTeam = GameTeam.RED) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        red_robot_obs = {robot_id: copy.deepcopy(robot_obs) for robot_id, robot_obs in self.robot_obs.items() if robot_id.startswith("RED")}
        blue_robot_obs = {robot_id: copy.deepcopy(robot_obs) for robot_id, robot_obs in self.robot_obs.items() if robot_id.startswith("BLUE")}
        if team == GameTeam.RED:
            # 先己方，后对方
            robot_obs = {**red_robot_obs, **blue_robot_obs}
        else:
            # 调换红蓝方的坐标方向
            for _, robot_obs in red_robot_obs.items():
                robot_obs.position_norm *= -1
            for _, robot_obs in blue_robot_obs.items():
                robot_obs.position_norm *= -1
            robot_obs = {**blue_robot_obs, **red_robot_obs}

        return np.concatenate([
            self.game_obs.to_array(),
            *[robot_obs.to_array() for _, robot_obs in robot_obs.items()]
        ])

class Environment:
    def __init__(self):
        # 创建物理引擎
        self.env_config = BASE_ENV_CONFIG
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力
        self.dt = 1 / BASE_ENV_CONFIG.FPS

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.total_time = self.env_config.GAME_TIME_LIMIT       # 总时长
        self._remaining_time = self.env_config.GAME_TIME_LIMIT  # 剩余时间

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(self.env_config.OBSTACLES)

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self._create_robots(BASE_ROBOT_CONFIGS)
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(self.env_config)

        # 性能统计
        self._time_stats = defaultdict(list)

    def _create_robots(self, robot_configs: List[RobotConfig]):
        """根据配置创建机器人"""
        for config in robot_configs:
            robot = Robot(
                **config.__dict__,
                physics_engine=self.physics_engine,
            )
            self.robots[robot.id] = robot

    def _init_robot_grid_maps(self, env_config):
        """初始化所有机器人的网格地图"""
        for robot in self.robots.values():
            grid_map = GridMap(
                width=env_config.FIELD_WIDTH,  # 场地宽度
                height=env_config.FIELD_HEIGHT,  # 场地高度
                robot_radius=robot.radius
            )
            # 标记所有障碍物
            grid_map.mark_obstacles(self.obstacles)
            # 设置机器人的网格地图
            robot.grid_map = grid_map

    def _create_obstacles(self, obstacles: List[Dict[str, Any]]):
        for obstacle_config in obstacles:
            self.obstacles.append(Obstacle(obstacle_config, self.physics_engine))

    def reset(self):
        """重置环境"""
        # 销毁现有机器人
        for robot in self.robots.values():
            robot.destroy_physics_body(self.physics_engine)
        self.robots.clear()
        
        # 创建新机器人
        self._create_robots(BASE_ROBOT_CONFIGS)
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(self.env_config)
        
        # 重置游戏状态
        self.game_state = GameState.PLAYING
        self._remaining_time = self.env_config.GAME_TIME_LIMIT

    def step(self, red_action: Dict[str, Action], blue_action: Dict[str, Action]):
        """推进环境仿真"""
        # 更新物理引擎
        with timer(self._time_stats, 'physics_engine_step'):
            self.physics_engine.step(self.dt)
        
        # 应用动作
        with timer(self._time_stats, 'apply_team_action'):
            # 为了使结算效果与红蓝先后解耦，我们先统一应用运动动作，再应用攻击动作
            self._apply_team_motion(red_action)
            self._apply_team_motion(blue_action)
            self._apply_team_attack(GameTeam.RED, red_action)
            self._apply_team_attack(GameTeam.BLUE, blue_action)

        # 更新机器人状态
        with timer(self._time_stats, 'robot_step'):
            for robot in self.robots.values():
                robot.step(self.dt, self._remaining_time)

        # 更新游戏状态
        with timer(self._time_stats, 'game_state_update'):
            # 倒计时减少
            self._remaining_time = max(0, self._remaining_time - self.dt)
            # 检查胜利条件
            red_hp = self.robots["RED_3_STANDARD"].hp
            blue_hp = self.robots["BLUE_3_STANDARD"].hp
            # 1. 有一方率先阵亡
            if red_hp <= 0:
                self.game_state = GameState.BLUE_TEAM_WIN
            elif blue_hp <= 0:
                self.game_state = GameState.RED_TEAM_WIN
            # 2. 时间到
            elif self._remaining_time <= 0:
                if red_hp > blue_hp:
                    self.game_state = GameState.RED_TEAM_WIN
                elif blue_hp > red_hp:
                    self.game_state = GameState.BLUE_TEAM_WIN
                else:
                    self.game_state = GameState.DRAW

        # # 打印性能统计
        # print("\n性能统计:")
        # for key, times in self._time_stats.items():
        #     if times:  # 确保有数据
        #         avg_time = sum(times) / len(times)
        #         print(f"{key}: {avg_time:.6f}s -- {times}")
        # print("=" * 50)
        # self._time_stats.clear()

    def _apply_team_motion(self, action: Dict[str, Action]):
        """应用移动和自旋姿态"""
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)

            robot.set_spin(robot_action.spin == 1)

            # 设置导航点
            if robot_action.navigation_set == 1:
                navigation_target = robot_action.get_target_position()
                robot.set_target(tuple(navigation_target))

            robot.refresh_motion_speed()

    def _apply_robot_attack(self, robot_attacker: Robot, robot_target: Robot):
        # 目标不存在
        if robot_target is None:
            return False
        # 判断完整视野
        if not attack_sight_clear(robot_attacker.get_position(), robot_target.get_position(), robot_target.radius, self.obstacles, self.robots.values()):
            return False
        # 攻击
        if not robot_attacker.attack(robot_target):
            return False
        print("no jump")
        return True

    def _apply_team_attack(self, team: GameTeam, action: Dict[str, Action]):
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
            if not self._apply_robot_attack(robot_attacker, robot_target):
                continue

    def apply_observation(self, observation: Observation):
        """
        【注意！】这是一个非常危险的函数，非特殊情况不要使用！
        直接将指定的观察值赋值到当前环境中
        """
        # 游戏状态
        self._remaining_time = observation.game_obs.remaining_time_norm * self.env_config.GAME_TIME_LIMIT
    
    def get_top_bar_info(self) -> Dict[str, Any]:
        """获取渲染顶部信息"""
        return {
            "remaining_time": self._remaining_time,
        }

    def get_robot(self, id: str) -> Robot:
        if id not in self.robots:
            return None
        return self.robots[id]
