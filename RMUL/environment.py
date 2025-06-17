import pymunk
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from base.environment import Environment, Action
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotConfig, RobotType, ROBOT_ID
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.robot import Robot
from utils.utils import point_in_polygon, opposite_team, has_line_of_sight

from RMUL.config import env_config
from RMUL.config.robot_config import RMUL_ROBOT_CONFIGS
from RMUL.game import GameStateManagerRMUL

@dataclass
class ActionRMUL(Action):
    navigation: Tuple[float, float] = None
    attack: bool = False
    target: RobotType = None
    purchase: bool = False

class EnvironmentRMUL(Environment):
    def __init__(
            self,
            env_config = env_config,
            obstacle_configs: Optional[List[Dict[str, Any]]] = env_config.OBSTACLES,
            robot_configs: Optional[Dict[str, RobotConfig]] = RMUL_ROBOT_CONFIGS,
        ):
        # 创建物理引擎
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力
        self.dt = 1 / env_config.FPS

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(obstacle_configs)

        # 创建增益区
        self.buff_zone = {}
        self.buff_zone["center"] = env_config.CENTER_ZONE_VERTICES
        self.buff_zone["red_start"] = env_config.RED_START_ZONE_VERTICES
        self.buff_zone["blue_start"] = env_config.BLUE_START_ZONE_VERTICES
        self.robots_in_zone = {GameTeam.RED: False, GameTeam.BLUE: False}

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self.robot_configs = robot_configs
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(env_config)

        # 创建游戏状态管理器
        self.game_state_manager = GameStateManagerRMUL()

        # 状态记录，仅用于计算奖励
        self.last_team_state = {
            GameTeam.RED: self._get_team_state(GameTeam.RED),
            GameTeam.BLUE: self._get_team_state(GameTeam.BLUE)
        }

    def step(self, dt: float, red_action: Dict[str, ActionRMUL], blue_action: Dict[str, ActionRMUL]):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)

        # 更新机器人状态
        for robot in self.robots.values():
            robot.step(dt)

        # 应用动作
        self._apply_team_action(GameTeam.RED, red_action)
        self._apply_team_action(GameTeam.BLUE, blue_action)

        # 检查中心区域占领情况
        self.robots_in_zone = {GameTeam.RED: False, GameTeam.BLUE: False}
        for robot in self.robots.values():
            if robot.is_alive and point_in_polygon(robot.get_position(), self.buff_zone["center"]):
                self.robots_in_zone[robot.team] = True
        
        # 检查补给区占领情况
        for robot in self.robots.values():
            if robot.team == GameTeam.RED and point_in_polygon(robot.get_position(), self.buff_zone["red_start"]) or \
                robot.team == GameTeam.BLUE and point_in_polygon(robot.get_position(), self.buff_zone["blue_start"]):
                # 解锁发射机构
                robot.gun_locked = False
                # 为防止血量计算中出现小数，仅在整数秒时一次性回复血量
                if 0 < math.modf(time.time())[0] < dt:
                    robot.heal(int(robot.max_hp * 0.25))

        # 更新游戏状态
        self.game_state_manager.update(self.robots_in_zone, dt)
        
    def reset(self):
        """重置环境"""
        super().reset()

    def _state_encoder(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码状态"""
        # 全局状态向量
        game_state = [
            self.game_state_manager.get_remaining_time() / env_config.GAME_TIME_LIMIT,
            self.game_state_manager.get_victory_progress(GameTeam.RED) / env_config.OCCUPATION_TARGET,
            self.game_state_manager.get_victory_progress(GameTeam.BLUE) / env_config.OCCUPATION_TARGET,
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

    def _get_team_state(self, team: GameTeam) -> np.ndarray:
        """获取当前状态"""
        game_state, red_robot_state, blue_robot_state = self._state_encoder()
        if team == GameTeam.RED:
            return np.concatenate((game_state, red_robot_state, blue_robot_state))
        else:
            return np.concatenate((game_state, blue_robot_state, red_robot_state))
    
    def _apply_team_action(self, team: GameTeam, action: Dict[str, Any]):
        """应用本方动作"""
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)

            # 导航
            if robot_action.navigation is not None:
                robot.set_target(robot_action.navigation)

            # 攻击
            if robot_action.attack and robot_action.target != RobotType.NONE:
                target_robot = self.get_robot(ROBOT_ID[opposite_team(team)][robot_action.target])
                if target_robot is not None:
                    if has_line_of_sight(robot.get_position(), target_robot.get_position(), self.obstacles):
                        if robot.attack(target_robot) and not target_robot.is_alive:
                            # 结算击杀经验
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
                            # 结算占领进度
                            self.game_state_manager.gain_victory_progress(team, 20)

            # 购买允许发弹量
            if robot_action.purchase and robot.robot_type != RobotType.SENTRY:
                if self.game_state_manager.get_economics(team) >= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM:
                    if point_in_polygon(robot.get_position(), self.buff_zone["red_start"] if team == GameTeam.RED else self.buff_zone["blue_start"]):
                        robot.ammo_allowed += robot.bullet.PURCHASE_NUM
                        self.game_state_manager.cost_economics(team, robot.bullet.PRICE * robot.bullet.PURCHASE_NUM)

    def _calculate_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 获取红方机器人
        red_robot = self.get_robot("RED_3_STANDARD")
        if red_robot is None:
            return reward
        
        # # 存活奖励
        # if red_robot.is_alive:
        #     reward += 0.1
        
        # 中心区域距离奖励
        robot_pos = red_robot.get_position()
        distance = math.sqrt((robot_pos[0] - env_config.FIELD_WIDTH / 2) ** 2 + (robot_pos[1] - env_config.FIELD_HEIGHT / 2) ** 2)
        reward_distance = 10 * (1 - distance / env_config.FIELD_WIDTH)
        reward += reward_distance
        print(f"中心区域距离奖励: {reward_distance}")
        
        # 中心区域占领奖励
        if self.robots_in_zone[GameTeam.RED]:
            reward_zone = 0.2
        else:
            reward_zone = -0.2
        reward += reward_zone
        print(f"中心区域占领奖励: {reward_zone}")        
        
        # 击杀奖励
        blue_robots = [self.get_robot(f"BLUE_{i}") for i in [3, 4, 5, 7]]
        for robot in blue_robots:
            if robot is not None and not robot.is_alive:
                reward_kill = 1.0
                reward += reward_kill
                print(f"击杀奖励: {reward_kill}")
        
        # 胜利奖励
        if self.game_state_manager.state.value == 2:  # 红方胜利
            reward_win = 10.0
            reward += reward_win
            print(f"胜利奖励: {reward_win}")
        
        return reward
