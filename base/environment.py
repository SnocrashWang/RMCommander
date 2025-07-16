import pymunk
import math
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, ROBOT_ID, RobotType
from utils.grid_map import GridMap, world_to_grid
from utils.robot import Robot
from utils.obstacle import Obstacle
from utils.utils import attack_sight_clear, calc_distance, opposite_team, opposite_position, timer

from base.config import env_config
from base.config.robot_config import BASE_ROBOT_CONFIGS, BASE_ROBOT_TYPE_LIST


@dataclass
class Action:
    velocity: Tuple[float, float] = (0.0, 0.0)
    attack_target: int = 0

    def __post_init__(self):
        """初始化"""
        try:
            if isinstance(self.velocity, np.ndarray):
                self.velocity = tuple(self.velocity.astype(float))
            else:
                self.velocity = tuple(self.velocity)
        except:
            self.velocity = (0.0, 0.0)
        try:
            self.attack_target = int(self.attack_target)
        except:
            self.attack_target = 0

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        return np.array([
            *self.velocity,
            self.attack_target
        ])

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
    position: np.ndarray
    velocity_norm: np.ndarray
    chassis_property_type: int
    gimbal_property_type: int
    level: int
    exp_norm: float
    hp_norm: float
    heat_norm: float

    @classmethod
    def from_robot(cls, robot: Robot):
        velocity = robot.get_velocity()
        return cls(
            position=np.array(robot.get_position()),
            velocity_norm=np.array(velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) != 0 else np.array([0, 0])),
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
            position=tuple(array[:2] * np.array([env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT])),
            velocity_norm=tuple(array[2:4]),
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
            *(self.position / np.array([env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT])),
            *self.velocity_norm,
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
                robot_obs[robot_id] = RobotObs.from_array(robot_array[:10])
                robot_array = robot_array[10:]
        return cls(game_obs, robot_obs)

    def to_array(self, team: GameTeam = GameTeam.RED) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        red_robot_obs = {robot_id: robot_obs for robot_id, robot_obs in self.robot_obs.copy().items() if robot_id.startswith("RED")}
        blue_robot_obs = {robot_id: robot_obs for robot_id, robot_obs in self.robot_obs.copy().items() if robot_id.startswith("BLUE")}
        if team == GameTeam.RED:
            robot_obs = {**red_robot_obs, **blue_robot_obs}
        else:
            for _, robot_obs in red_robot_obs.items():
                robot_obs.position = opposite_position(robot_obs.position, env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT)
                robot_obs.velocity_norm *= -1
            for _, robot_obs in blue_robot_obs.items():
                robot_obs.position = opposite_position(robot_obs.position, env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT)
                robot_obs.velocity_norm *= -1
            robot_obs = {**blue_robot_obs, **red_robot_obs}

        return np.concatenate([
            self.game_obs.to_array(),
            *[robot_obs.to_array() for _, robot_obs in robot_obs.items()]
        ])

class Environment:
    def __init__(
            self,
            env_config = env_config,
            obstacle_configs: Optional[List[Dict[str, Any]]] = env_config.OBSTACLES,
            robot_configs: Optional[Dict[str, RobotConfig]] = BASE_ROBOT_CONFIGS,
        ):
        # 创建物理引擎
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力
        self.dt = 1 / env_config.FPS

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.total_time = env_config.GAME_TIME_LIMIT       # 总时长
        self._remaining_time = env_config.GAME_TIME_LIMIT  # 剩余时间

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(obstacle_configs)

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self.robot_configs = robot_configs
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(env_config)

        # 性能统计
        self._time_stats = defaultdict(list)

    def _create_robots(self):
        """根据配置创建机器人"""
        for config in self.robot_configs:
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
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(env_config)
        
        # 重置游戏状态
        self.game_state = GameState.PLAYING
        self._remaining_time = env_config.GAME_TIME_LIMIT

    def step(self, red_action: Dict[str, Action], blue_action: Dict[str, Action]):
        """推进环境仿真"""
        # 更新物理引擎
        with timer(self._time_stats, 'physics_engine_step'):
            self.physics_engine.step(self.dt)
        
        # 应用动作
        with timer(self._time_stats, 'apply_team_action'):
            self._apply_team_action(GameTeam.RED, red_action)
            self._apply_team_action(GameTeam.BLUE, blue_action)

        # 更新机器人状态
        with timer(self._time_stats, 'robot_step'):
            for robot in self.robots.values():
                robot.step(self.dt)

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

    def _apply_team_action(self, team: GameTeam, action: Dict[str, Action]):
        """应用动作"""
        def apply_robot_action_attack(robot: Robot, robot_action: Action):
            target_type = RobotType(robot_action.attack_target)
            # 目标为空
            if target_type == RobotType.NONE:
                return
            target_robot = self.get_robot(ROBOT_ID[opposite_team(team)][target_type])
            # 目标不存在
            if target_robot is None:
                return
            # 判断完整视野
            if not attack_sight_clear(robot.get_position(), target_robot.get_position(), target_robot.radius, self.obstacles, self.robots.values()):
                return
            # 攻击
            if not robot.attack(target_robot) or target_robot.is_alive:
                return
            
            # 结算击杀经验（虽然1v1没有经验一说，此处仅做测试）
            if robot.robot_type == RobotType.SENTRY:
                killer_level = np.mean([robot.level for robot in self.robots.values() if robot.team == team])
                kill_exp = 50 * target_robot.level * (1 + max(0, 0.2 * (target_robot.level - killer_level)))
                robot_alive = [robot for robot in self.robots.values() if robot.team == team and robot.is_alive]
                # 经验分享
                for robot in robot_alive:
                    robot.update_exp(int(kill_exp / len(robot_alive)))
            else:
                kill_exp = 50 * target_robot.level * (1 + max(0, 0.2 * (target_robot.level - robot.level)))
                robot.update_exp(int(kill_exp))

        # 对所有机器人应用动作
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)

            # 设置速度
            robot.set_velocity(robot_action.velocity)

            # 攻击
            apply_robot_action_attack(robot, robot_action)

    def apply_observation(self, observation: Observation):
        """
        【注意！】这是一个非常危险的函数，非特殊情况不要使用！
        直接将指定的观察值赋值到当前环境中
        """
        # 游戏状态
        self._remaining_time = observation.game_obs.remaining_time_norm * env_config.GAME_TIME_LIMIT
    
    def get_top_bar_info(self) -> Dict[str, Any]:
        """获取渲染顶部信息"""
        return {
            "remaining_time": self._remaining_time,
        }

    def get_robot(self, id: str) -> Robot:
        if id not in self.robots:
            return None
        return self.robots[id]
