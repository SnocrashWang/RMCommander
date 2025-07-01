import pymunk
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from base.environment import Environment
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, RobotType, ROBOT_ID
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.robot import Robot
from utils.utils import point_in_polygon, opposite_team, has_line_of_sight, attack_sight_clear

from RMUL.config import env_config
from RMUL.config.robot_config import RMUL_ROBOT_CONFIGS


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

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.total_time = env_config.GAME_TIME_LIMIT       # 总时长
        self._remaining_time = env_config.GAME_TIME_LIMIT  # 剩余时间

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(obstacle_configs)

        # 创建增益区
        self.buff_zone = {}
        self.buff_zone["center"] = env_config.CENTER_ZONE_VERTICES
        self.buff_zone["boot_red"] = env_config.BOOT_ZONE_RED_VERTICES
        self.buff_zone["boot_blue"] = env_config.BOOT_ZONE_BLUE_VERTICES

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self.robot_configs = robot_configs
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(env_config)

        # 游戏机制
        self._economics = {GameTeam.RED: 0, GameTeam.BLUE: 0}           # 经济
        self._victory_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}    # 胜利进度
        self._laggard_bonus_taken = {                                   # 落后奖励
            "red_lag_70": False,
            "red_lag_140": False,
            "blue_lag_70": False,
            "blue_lag_140": False,
        }

    def step(self, dt: float, red_action: Dict[str, Dict[str, Any]], blue_action: Dict[str, Dict[str, Any]]):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)

        # 应用动作
        self._apply_team_action(GameTeam.RED, red_action)
        self._apply_team_action(GameTeam.BLUE, blue_action)

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
        target = env_config.OCCUPATION_TARGET

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
    
    def _apply_team_action(self, team: GameTeam, action: Dict[str, Any]):
        """应用本方动作"""
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)

            # 导航
            if robot_action["navigation_move"]:
                robot.set_target(robot_action["navigation_target"])

            # 攻击
            target_type = RobotType(robot_action["attack_target"])
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
                            # 结算胜利进度
                            self._victory_progress[team] += 20

            # 购买允许发弹量
            if robot_action["purchase"] and robot.robot_type != RobotType.SENTRY:
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