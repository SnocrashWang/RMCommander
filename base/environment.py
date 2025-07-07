import pymunk
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
from utils.utils import attack_sight_clear, calc_distance, opposite_team, timer

from base.config import env_config
from base.config.robot_config import BASE_ROBOT_CONFIGS


@dataclass
class Action:
    navigation_target: Tuple[float, float] = (0.0, 0.0)
    navigation_set: int = 0
    attack_target: int = 0

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        # 将navigation_target元组展开，然后与其他属性合并
        return np.array([
            *self.navigation_target,
            self.navigation_set,
            self.attack_target
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
        self.time_stats = defaultdict(list)

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
            robot.set_grid_map(grid_map)

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

    def step(self, dt: float, red_action: Dict[str, Action], blue_action: Dict[str, Action]):
        """推进环境仿真"""
        # 更新物理引擎
        with timer(self.time_stats, 'physics_engine_step'):
            self.physics_engine.step(dt)
        
        # 应用动作
        with timer(self.time_stats, 'apply_team_action'):
            self._apply_team_action(GameTeam.RED, red_action)
            self._apply_team_action(GameTeam.BLUE, blue_action)

        # 更新机器人状态
        with timer(self.time_stats, 'robot_step'):
            for robot in self.robots.values():
                robot.step(dt)

        # 更新游戏状态
        with timer(self.time_stats, 'game_state_manager_update'):
            # 倒计时减少
            self._remaining_time = max(0, self._remaining_time - dt)
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

    def _apply_team_action(self, team: GameTeam, action: Dict[str, Action]):
        """应用动作"""
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)
            if robot_action.navigation_set:
                robot.set_target(robot_action.navigation_target)
            target_type = RobotType(robot_action.attack_target)
            if target_type != RobotType.NONE:
                target_robot = self.get_robot(ROBOT_ID[opposite_team(team)][target_type])
                if target_robot is not None:
                    # 判断完整视野
                    if attack_sight_clear(robot.get_position(), target_robot.get_position(), target_robot.radius, self.obstacles, self.robots.values()):
                        if robot.attack(target_robot) and not target_robot.is_alive:
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
    
    def get_top_bar_info(self) -> Dict[str, Any]:
        """获取渲染顶部信息"""
        return {
            "remaining_time": self._remaining_time,
        }

    def get_robot(self, id: str) -> Robot:
        if id not in self.robots:
            return None
        return self.robots[id]
