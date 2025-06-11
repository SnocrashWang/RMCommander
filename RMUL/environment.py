import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from base_game.environment import Environment
from base_game.config.robot_config import RobotConfig
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from RMUL.config import env_config
from RMUL.config.robot_config import RMUL_ROBOT_CONFIGS

@dataclass
class Action():
    navigation: Tuple[float, float]
    attack: bool
    target: int

class EnvironmentRL(Environment):
    def __init__(
            self,
            obstacle_configs: Optional[List[Dict[str, Any]]] = env_config.OBSTACLES,
            robot_configs: Optional[Dict[str, RobotConfig]] = RMUL_ROBOT_CONFIGS,
        ):
        super().__init__(obstacle_configs, robot_configs)

        # 创建中心区域
        self.center_zone_rect = env_config.CENTER_ZONE_RECT
        self.robots_in_zone = {GameTeam.RED: False, GameTeam.BLUE: False}

        # 定义状态空间大小
        self.state_size = len(self._get_team_state(GameTeam.RED))

    def step(self, dt):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)

        # 更新机器人状态
        for robot in self.robots.values():
            robot.step(dt)

        # 检查中心区域占领情况
        for robot in self.robots.values():
            if robot.is_alive:
                # 检查是否在中心区域
                pos = robot.body.position
                pixel_x = meters_to_pixels(pos.x, env_config.SCALE)
                pixel_y = meters_to_pixels(pos.y, env_config.SCALE)
                if self.center_zone_rect.collidepoint(pixel_x, pixel_y):
                    self.robots_in_zone[robot.team] = True

        # 更新游戏状态
        self.game_state_manager.update(self.robots_in_zone, dt)
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境并返回初始状态"""
        super().reset()
        return self._get_team_state(GameTeam.RED), self.get_game_state()
    
    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作并返回下一个状态、奖励、是否结束和额外信息"""
        # 应用红方动作
        self._apply_team_action(GameTeam.RED, action)
        
        # 推进环境仿真
        super().step(1 / env_config.FPS)
        
        # 获取新状态
        next_state = self._get_team_state(GameTeam.RED)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self.game_state_manager.state != GameState.PLAYING
        
        # 获取额外信息
        info = self.get_game_state()
        
        return next_state, reward, done, info
    
    def _state_encoder(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """编码状态"""
        # 全局状态向量
        game_state = [
            self.game_state_manager.remaining_time / env_config.GAME_TIME_LIMIT,
            self.game_state_manager.center_zone_progress[GameTeam.RED] / env_config.OCCUPATION_TARGET,
            self.game_state_manager.center_zone_progress[GameTeam.BLUE] / env_config.OCCUPATION_TARGET,
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
                robot.exp / (LEVEL_NEED_EXP[robot.level + 1] - LEVEL_NEED_EXP[robot.level]),
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
        robot_ours = [bot for bot in self.robots.values() if bot.team == team]
        if len(robot_ours) == 0:
            return
        
        nav_x, nav_y = action.navigation
        should_attack = action.attack
        attack_target = action.target
        # 设置导航目标
        if self.robots:  # 确保有机器人
            self.robots["RED_3_STANDARD"].set_target((nav_x, nav_y))
            print(f"红方机器人移动到: {nav_x}, {nav_y}")
            
            # 执行攻击
            if should_attack and len(self.robots) > 1:
                self.robots["RED_3_STANDARD"].attack(self.get_robot(ROBOT_ID[GameTeam.BLUE][attack_target]))
                print(f"红方机器人攻击: {ROBOT_ID[GameTeam.BLUE][attack_target]}")

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
