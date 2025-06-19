import pymunk
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, ROBOT_ID, RobotType
from utils.grid_map import GridMap, world_to_grid
from utils.robot import Robot
from utils.obstacle import Obstacle
from utils.utils import attack_sight_clear, calc_distance, opposite_team, timer

from base.config import env_config
from base.config.robot_config import BASE_ROBOT_CONFIGS, BASE_ROBOT_TYPE_LIST
from base.game import GameStateManager

@dataclass
class Action():
    navigation: Tuple[float, float] = None
    attack: bool = False
    target: RobotType = None

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
        self._last_team_state = {
            GameTeam.RED: self._get_team_state(GameTeam.RED),
            GameTeam.BLUE: self._get_team_state(GameTeam.BLUE)
        }
        self._last_team_action = {
            GameTeam.RED: {
                robot_id: Action() for robot_id in ROBOT_ID[GameTeam.RED].values()
            },
            GameTeam.BLUE: {
                robot_id: Action() for robot_id in ROBOT_ID[GameTeam.BLUE].values()
            },
        }

        # 性能统计
        self.time_stats = defaultdict(list)

    def _create_robots(self):
        """根据配置创建机器人"""
        for config in self.robot_configs:
            robot = Robot(
                self.physics_engine,
                **config.__dict__,
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
            robot_hp = {
                robot.id: robot.hp for robot in self.robots.values()
            }
            self.game_state_manager.update(robot_hp, dt)
    
    def _encode_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码状态"""
        # 全局状态向量
        game_state = [
            self.game_state_manager.get_remaining_time() / env_config.GAME_TIME_LIMIT,
        ]
        
        # 机器人状态向量
        red_robot_state = []
        blue_robot_state = []
        for robot in self.robots.values():
            x, y = robot.get_position()
            robot_state = [
                x / env_config.FIELD_WIDTH,
                y / env_config.FIELD_HEIGHT,
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

    def calculate_reward(self, team: GameTeam, action: Dict[str, Action]) -> float:
        """计算奖励
        Args:
            team: 队伍
        Returns:
            float: 奖励值
        """
        last_game_state_dict, last_red_robot_state_dict, last_blue_robot_state_dict = self._decode_state(self._last_team_state[team], team)
        reward_list = []
        reward_weight = []

        # 时间消耗惩罚
        reward_time = (self.game_state_manager.get_remaining_time() - last_game_state_dict["remaining_time"]) * 1
        reward_list.append(reward_time)
        reward_weight.append(0)

        # 不可行导航点惩罚
        col, row = world_to_grid(action["RED_3_STANDARD"].navigation)  
        if self.robots["RED_3_STANDARD"].grid_map.is_blocked(col, row):
            reward_navigation_unmovable = -1.0
        else:
            reward_navigation_unmovable = 1.0
        reward_list.append(reward_navigation_unmovable)
        reward_weight.append(5)

        # 导航点差异惩罚
        last_navigation = self._last_team_action[team]["RED_3_STANDARD"].navigation
        if last_navigation is None:
            last_navigation = self.robots["RED_3_STANDARD"].get_position()
        current_navigation = action["RED_3_STANDARD"].navigation
        navigation_diff = calc_distance(last_navigation, current_navigation)
        reward_navigation_diff = (1 - navigation_diff ** 2) / (1 + navigation_diff ** 2)
        reward_list.append(reward_navigation_diff)
        reward_weight.append(10)

        # 获取当前血量
        our_last_hp = sum([robot["hp"] for robot in last_red_robot_state_dict.values()])
        our_hp = sum([robot.hp / robot.max_hp for robot in self.robots.values() if robot.team == team])
        enemy_last_hp = sum([robot["hp"] for robot in last_blue_robot_state_dict.values()])
        enemy_hp = sum([robot.hp / robot.max_hp for robot in self.robots.values() if robot.team == opposite_team(team)])
        
        # 血量奖励
        reward_hp = np.sign((enemy_last_hp - enemy_hp) - (our_last_hp - our_hp))
        reward_list.append(reward_hp)
        reward_weight.append(10)

        # 距离奖励
        our_robot = self.get_robot("RED_3_STANDARD")
        enemy_robot = self.get_robot("BLUE_3_STANDARD")
        last_distance = calc_distance(last_red_robot_state_dict["RED_3_STANDARD"]["position"], last_blue_robot_state_dict["BLUE_3_STANDARD"]["position"])
        current_distance = calc_distance(our_robot.get_position(), enemy_robot.get_position())
        reward_distance = np.sign(last_distance - current_distance)  # 距离减小给予正奖励，距离增加给予负奖励
        reward_list.append(reward_distance)
        reward_weight.append(5)
        
        # 游戏结束奖励
        if self.game_state_manager.state == GameState.RED_TEAM_WIN:
            reward_win = 1.0
        elif self.game_state_manager.state == GameState.BLUE_TEAM_WIN:
            reward_win = -1.0
        else:
            reward_win = 0.0
        reward_list.append(reward_win)
        reward_weight.append(20)
        
        # 更新状态记录
        self._last_team_state = {
            GameTeam.RED: self._get_team_state(GameTeam.RED),
            GameTeam.BLUE: self._get_team_state(GameTeam.BLUE)
        }
        self._last_team_action = {
            GameTeam.RED: action,
            GameTeam.BLUE: action,
        }

        reward = np.average(reward_list, weights=reward_weight)
        # print(reward_list, reward)
        return reward
    
    def get_robot(self, id: str) -> Robot:
        if id not in self.robots:
            return None
        return self.robots[id]
