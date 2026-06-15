import pymunk
import numpy as np
import math
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from rules.base.environment import Environment
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotType, ROBOT_ID
from utils.robot import Robot
from utils.utils import opposite_team

from rules.rmul.config.action_config import ActionRMUL
from rules.rmul.config.env_config import EnvConfigRMUL
from rules.rmul.config.obstacle_config import RMUL_OBSTACLE_CONFIGS
from rules.rmul.config.robot_config import RMUL_ROBOT_CONFIGS
from rules.rmul.config.zone_config import RMUL_ZONES


class EnvironmentRMUL(Environment):
    def __init__(self, env_config, obstacle_configs, robot_configs):
        # 环境设置
        self.env_config = env_config
        self.dt = 1 / env_config.fps

        # 创建物理引擎
        self._create_physics_engine()

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.total_time = self.env_config.game_time_limit           # 总时长
        self._remaining_time = self.env_config.game_remaining_time  # 剩余时间

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(obstacle_configs)

        # 创建增益区
        self.zones = RMUL_ZONES

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self._create_robots(robot_configs)

        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(self.env_config)

        # 游戏机制
        self._economics = {
            GameTeam.RED: self.env_config.economics_red,
            GameTeam.BLUE: self.env_config.economics_blue,
        }                                                               # 经济
        self._victory_progress = {
            GameTeam.RED: self.env_config.victory_progress_red,
            GameTeam.BLUE: self.env_config.victory_progress_red,
        }    # 胜利进度
        self._laggard_bonus_taken = {                                   # 落后奖励
            "red_lag_70": False,
            "red_lag_140": False,
            "blue_lag_70": False,
            "blue_lag_140": False,
        }

    def reset(self, env_config, obstacle_configs, robot_configs):
        """重置环境"""
        self.env_config = env_config

        # 重新创建障碍物
        self._create_obstacles(obstacle_configs)

        # 销毁现有机器人
        for robot in self.robots.values():
            robot.destroy_physics_body(self.physics_engine)
        self.robots.clear()
        
        # 创建新机器人
        self._create_robots(robot_configs)
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(self.env_config)
        
        # 重置游戏状态
        self.game_state = GameState.PLAYING
        self._remaining_time = self.env_config.game_remaining_time

        # 游戏机制
        self._economics = {GameTeam.RED: 0, GameTeam.BLUE: 0}           # 经济
        self._victory_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}    # 胜利进度
        self._laggard_bonus_taken = {                                   # 落后奖励
            "red_lag_70": False,
            "red_lag_140": False,
            "blue_lag_70": False,
            "blue_lag_140": False,
        }

    def step(self, red_action: Dict[str, ActionRMUL], blue_action: Dict[str, ActionRMUL]):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(self.dt)

        # 应用动作
        # 为了使结算效果与红蓝先后解耦，我们先统一应用运动动作，再应用攻击动作
        self._apply_team_motion(red_action)
        self._apply_team_motion(blue_action)
        self._apply_team_attack(GameTeam.RED, red_action)
        self._apply_team_attack(GameTeam.BLUE, blue_action)
        self._apply_team_purchase(GameTeam.RED, red_action)
        self._apply_team_purchase(GameTeam.BLUE, blue_action)

        # 更新机器人状态
        for robot in self.robots.values():
            robot.step(self.dt, self._remaining_time)

        # 更新所有增益区
        for zone in self.zones.values():
            zone.update(self.robots)
        
        # 检查补给区占领情况
        for id in self.zones["red_boot"].occupation_robots[GameTeam.RED]:
            self.robots[id].gun_locked = False
        for id in self.zones["blue_boot"].occupation_robots[GameTeam.BLUE]:
            self.robots[id].gun_locked = False

        # 倒计时减少
        self._remaining_time = max(0, self._remaining_time - self.dt)

        # 更新经济
        if 0 < 300 - self._remaining_time < self.dt:
            self._economics[GameTeam.RED] += 200
            self._economics[GameTeam.BLUE] += 200
        elif 0 < 240 - self._remaining_time < self.dt:
            self._economics[GameTeam.RED] += 200
            self._economics[GameTeam.BLUE] += 200
        elif 0 < 180 - self._remaining_time < self.dt:
            self._economics[GameTeam.RED] += 200
            self._economics[GameTeam.BLUE] += 200
        elif 0 < 120 - self._remaining_time < self.dt:
            self._economics[GameTeam.RED] += 300
            self._economics[GameTeam.BLUE] += 300
        elif 0 < 60 - self._remaining_time < self.dt:
            self._economics[GameTeam.RED] += 300
            self._economics[GameTeam.BLUE] += 300

        # 更新来自中心区域的胜利进度
        for team, occupied in self.zones["center"].occupation_team.items():
            if occupied:
                self._victory_progress[team] += self.dt

        # 检查胜利条件
        red_progress = self._victory_progress[GameTeam.RED]
        blue_progress = self._victory_progress[GameTeam.BLUE]
        target = self.env_config.victory_target

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
            self._victory_progress[GameTeam.RED] = target
        elif blue_progress >= target > red_progress:
            self.game_state = GameState.BLUE_TEAM_WIN
            self._victory_progress[GameTeam.BLUE] = target
        # 2. 同时积满
        elif min(red_progress, blue_progress) >= target:
            self.game_state = GameState.DRAW
            self._victory_progress[GameTeam.RED] = target
            self._victory_progress[GameTeam.BLUE] = target
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
        return super()._apply_robot_attack(robot_attacker, robot_target)

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
            if not self._apply_robot_attack(robot_attacker, robot_target):
                continue

            # 被攻击目标阵亡
            if not robot_target.is_alive:
                # 结算击杀经验
                if robot_attacker.robot_type == RobotType.SENTRY:
                    killer_level = self.get_robot(ROBOT_ID[team][RobotType.STANDARD_3]).level
                    kill_exp = 50 * robot_target.level * (1 + max(0, 0.2 * (robot_target.level - killer_level)))
                    robot_alive = [robot for robot in self.robots.values() if robot.team == team and robot.is_alive]
                    # 经验分享
                    for robot in robot_alive:
                        robot.update_exp(int(kill_exp / len(robot_alive)))
                else:
                    kill_exp = 50 * robot_target.level * (1 + max(0, 0.2 * (robot_target.level - robot_attacker.level)))
                    robot_attacker.update_exp(int(kill_exp))
                # 结算胜利进度
                self._victory_progress[team] += 20

    def _apply_team_purchase(self, team: GameTeam, action: Dict[str, ActionRMUL]):
        # 购买允许发弹量
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)
            if robot_action.purchase and robot.robot_type != RobotType.SENTRY:
                if self._economics[team] >= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM:
                    if robot_id in (self.zones["red_boot"].occupation_robots[GameTeam.RED] if team == GameTeam.RED else self.zones["blue_boot"].occupation_robots[GameTeam.BLUE]):
                        robot.ammo_allowed += robot.bullet.PURCHASE_NUM
                        self._economics[team] -= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM

    def get_top_bar_info(self) -> Dict[str, Any]:
        """获取渲染顶部信息"""
        return {
            "remaining_time": self._remaining_time,
            "victory_progress": self._victory_progress,
            "economics": self._economics,
        }
