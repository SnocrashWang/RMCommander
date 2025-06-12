import pymunk
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, ROBOT_ID, RobotType
from utils.grid_map import GridMap
from utils.robot import Robot
from utils.obstacle import Obstacle
from utils.utils import has_line_of_sight, calc_distance, opposite_team

from base.config import env_config
from base.config.robot_config import DEFAULT_ROBOT_CONFIGS, BASE_ROBOT_TYPE_LIST
from base.game import GameStateManager

@dataclass
class Action():
    navigation: Tuple[float, float]
    attack: bool
    target: RobotType

class Environment:
    def __init__(
            self,
            env_config = env_config,
            obstacle_configs: Optional[List[Dict[str, Any]]] = env_config.OBSTACLES,
            robot_configs: Optional[Dict[str, RobotConfig]] = DEFAULT_ROBOT_CONFIGS,
        ):
        # 创建物理引擎
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力
        self.dt = 1 / env_config.FPS

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(obstacle_configs)

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self.robot_configs = robot_configs
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(env_config)

        # 创建游戏状态管理器
        self.game_state_manager = GameStateManager()

        # 状态记录，仅用于计算奖励
        self.last_team_state = {
            GameTeam.RED: self._get_team_state(GameTeam.RED),
            GameTeam.BLUE: self._get_team_state(GameTeam.BLUE)
        }

    def _create_robots(self):
        """根据配置创建机器人"""
        for config in self.robot_configs:
            robot = Robot(
                self.physics_engine,
                team=config.team,
                robot_type=config.robot_type,
                init_pos=config.init_pos,
                chassis_property_type=config.chassis_property_type,
                gimbal_property_type=config.gimbal_property_type,
                forward_speed_efficiency=config.forward_speed_efficiency,
                rotation_speed_efficiency=config.rotation_speed_efficiency,
                radius=config.radius
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
            self.obstacles.append(Obstacle(self.physics_engine, obstacle_config))

    def reset(self):
        """重置环境"""
        # 销毁现有机器人
        for robot in self.robots.values():
            robot.destroy(self.physics_engine)
        self.robots.clear()
        # 创建新机器人
        self._create_robots()
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(env_config)
        # 重置游戏状态
        self.game_state_manager.reset()

    def step(self, dt: float, red_action: Dict[str, Action], blue_action: Dict[str, Action]):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)

        # 应用动作
        self._apply_team_action(GameTeam.RED, red_action)
        self._apply_team_action(GameTeam.BLUE, blue_action)

        # 更新机器人状态
        for robot in self.robots.values():
            robot.step(dt)

        # 更新游戏状态
        robot_hp = {
            robot.id: robot.hp for robot in self.robots.values()
        }
        self.game_state_manager.update(robot_hp, dt)
    
    def _encode_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码状态"""
        # 全局状态向量
        game_state = [
            self.game_state_manager.remaining_time / env_config.GAME_TIME_LIMIT,
        ]
        
        # 机器人状态向量
        red_robot_state = []
        blue_robot_state = []
        for robot in self.robots.values():
            robot_state = [
                robot.body.position.x / env_config.FIELD_WIDTH,
                robot.body.position.y / env_config.FIELD_HEIGHT,
                # robot.angle / 360,
                robot.chassis_property_type.value,
                robot.gimbal_property_type.value,
                robot.level,
                robot.exp / (LEVEL_NEED_EXP[robot.level + 1] - LEVEL_NEED_EXP[robot.level]) if robot.level < len(LEVEL_NEED_EXP) else 1,
                robot.hp / robot.max_hp,
                robot.heat / robot.max_heat,
            ]
            if robot.team == GameTeam.RED:
                red_robot_state.extend(robot_state)
            else:
                blue_robot_state.extend(robot_state)
        return np.array(game_state), np.array(red_robot_state), np.array(blue_robot_state)
    
    def _decode_state(self, state: np.ndarray, team: GameTeam) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解码状态"""
        game_state = state[:1]
        red_robots = [robot for robot in self.robots.values() if robot.team == GameTeam.RED]
        blue_robots = [robot for robot in self.robots.values() if robot.team == GameTeam.BLUE]
        if team == GameTeam.RED:
            red_robot_state = state[1:1+8*len(red_robots)]
            blue_robot_state = state[1+8*len(red_robots):]
        else:
            blue_robot_state = state[1:1+8*len(blue_robots)]
            red_robot_state = state[1+8*len(blue_robots):]

        game_state_dict = {
            "remaining_time": game_state[0] * env_config.GAME_TIME_LIMIT,
        }
        red_robot_state_dict = {
            robot.id: {
                "position": (red_robot_state[i * 8 + 0] * env_config.FIELD_WIDTH, red_robot_state[i * 8 + 1] * env_config.FIELD_HEIGHT),
                "chassis_property_type": red_robot_state[i * 8 + 2],
                "gimbal_property_type": red_robot_state[i * 8 + 3],
                "level": red_robot_state[i * 8 + 4],
                "exp": red_robot_state[i * 8 + 5],
                "hp": red_robot_state[i * 8 + 6],
                "heat": red_robot_state[i * 8 + 7],
            } for i, robot in enumerate(red_robots)
        }
        blue_robot_state_dict = {
            robot.id: {
                "position": (blue_robot_state[i * 8 + 0] * env_config.FIELD_WIDTH, blue_robot_state[i * 8 + 1] * env_config.FIELD_HEIGHT),
                "chassis_property_type": blue_robot_state[i * 8 + 2],
                "gimbal_property_type": blue_robot_state[i * 8 + 3],
                "level": blue_robot_state[i * 8 + 4],
                "exp": blue_robot_state[i * 8 + 5],
                "hp": blue_robot_state[i * 8 + 6],
                "heat": blue_robot_state[i * 8 + 7],
            } for i, robot in enumerate(blue_robots)
        }
        return game_state_dict, red_robot_state_dict, blue_robot_state_dict

    def _get_team_state(self, team: GameTeam) -> np.ndarray:
        """获取当前状态"""
        game_state, red_robot_state, blue_robot_state = self._encode_state()
        if team == GameTeam.RED:
            return np.concatenate((game_state, red_robot_state, blue_robot_state))
        else:
            return np.concatenate((game_state, blue_robot_state, red_robot_state))

    def _apply_team_action(self, team: GameTeam, action: Dict[str, Action]):
        """应用动作"""
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)
            if robot_action.navigation is not None:
                robot.set_target(robot_action.navigation)
            if robot_action.attack and robot_action.target != RobotType.NONE:
                target_robot = self.get_robot(ROBOT_ID[opposite_team(team)][robot_action.target])
                if target_robot is not None:
                    if has_line_of_sight(robot.get_position(), target_robot.get_position(), self.obstacles):
                        robot.attack(target_robot)

    def calculate_reward(self, team: GameTeam) -> float:
        """计算奖励
        Args:
            team: 队伍
        Returns:
            float: 奖励值
        """
        game_state_dict, red_robot_state_dict, blue_robot_state_dict = self._decode_state(self.last_team_state[team], team)

        # 时间消耗惩罚
        reward_time = (self.game_state_manager.remaining_time - game_state_dict["remaining_time"]) * 1

        # 获取当前血量
        our_hp = sum([robot["hp"] for robot in red_robot_state_dict.values()])
        our_last_hp = sum([robot["hp"] for robot in red_robot_state_dict.values()])
        enemy_hp = sum([robot["hp"] for robot in blue_robot_state_dict.values()])
        enemy_last_hp = sum([robot["hp"] for robot in blue_robot_state_dict.values()])
        
        # 血量奖励
        reward_hp = (enemy_last_hp - enemy_hp) * 1.0 - (our_last_hp - our_hp) * 1.0

        # 距离奖励
        our_robot = self.get_robot("RED_3_STANDARD")
        enemy_robot = self.get_robot("BLUE_3_STANDARD")
        current_distance = calc_distance(our_robot.get_position(), enemy_robot.get_position())
        
        # 计算距离变化奖励
        last_distance = calc_distance(red_robot_state_dict["RED_3_STANDARD"]["position"], blue_robot_state_dict["BLUE_3_STANDARD"]["position"])
        current_distance = calc_distance(our_robot.get_position(), enemy_robot.get_position())
        reward_distance = (last_distance - current_distance) * 10  # 距离减小给予正奖励，距离增加给予负奖励
        
        # 游戏结束奖励
        if self.game_state_manager.state == GameState.RED_TEAM_WIN:
            reward_win = 1000.0
        elif self.game_state_manager.state == GameState.BLUE_TEAM_WIN:
            reward_win = -1000.0
        else:
            reward_win = 0.0
        
        # 更新状态记录
        self.last_team_state = {
            GameTeam.RED: self._get_team_state(GameTeam.RED),
            GameTeam.BLUE: self._get_team_state(GameTeam.BLUE)
        }

        # print(reward_time, reward_hp, reward_distance, reward_win)
        return reward_time + reward_hp + reward_distance + reward_win
    
    def get_robot(self, id: str) -> Robot:
        if id not in self.robots:
            return None
        return self.robots[id]
