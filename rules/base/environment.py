import pymunk
import math
import numpy as np
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, ROBOT_ID, RobotType
from utils.action import Action
from utils.grid_map import GridMap
from utils.robot import Robot
from utils.obstacle import Obstacle
from utils.utils import attack_sight_clear, opposite_team, pos_norm2real, timer

from rules.base.config.action_config import ActionBase


class Environment:
    def __init__(self, env_config, obstacle_configs, robot_configs):
        # 环境设置
        self.env_config = env_config
        self.dt = 1 / env_config.fps

        # 创建物理引擎
        self._create_physics_engine()

        # 游戏状态
        self.game_state = GameState.PLAYING
        self.total_time = self.env_config.game_time_limit       # 总时长
        self._remaining_time = self.env_config.game_remaining_time  # 剩余时间

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles(obstacle_configs)

        # 创建机器人
        self.robots: Dict[str, Robot] = {}
        self._create_robots(robot_configs)
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps(self.env_config)

        # 性能统计
        self._time_stats = defaultdict(list)

    def _create_physics_engine(self):
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)                    # 无重力
        self.physics_engine.iterations = 30                     # 每个物理步里碰撞约束求解器迭代次数
        self.physics_engine.collision_slop = 0.001              # 允许保留 0.001m 的碰撞穿透容差
        self.physics_engine.collision_bias = math.pow(0.5, 60)  # 每秒修正大约 50% 的穿透误差

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
                width=env_config.field_width,       # 场地宽度
                height=env_config.field_height,     # 场地高度
                robot_radius=robot.radius
            )
            # 标记所有障碍物
            grid_map.mark_obstacles(self.obstacles)
            # 设置机器人的网格地图
            robot._grid_map = grid_map

    def _create_obstacles(self, obstacles: List[Dict[str, Any]]):
        for obstacle_config in obstacles:
            self.obstacles.append(Obstacle(obstacle_config, self.physics_engine))

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

    def step(self, red_action: Dict[str, ActionBase], blue_action: Dict[str, ActionBase]):
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

    def _apply_team_motion(self, action: Dict[str, ActionBase]):
        """应用移动和自旋姿态"""
        for robot_id, robot_action in action.items():
            robot = self.get_robot(robot_id)

            robot.set_spin(robot_action.spin == 1)

            # 设置导航点
            if robot_action.navigation_set == 1:
                navigation_target = pos_norm2real(robot_action.navigation_target_norm, self.env_config.field_size())
                robot.set_target(tuple(navigation_target))

            robot.refresh_motion_speed()

    def _apply_robot_attack(self, robot_attacker: Robot, robot_target: Robot):
        # 目标不存在
        if robot_target is None:
            return False
        # 不满足射频间隔
        if robot_attacker.last_attack_time_remain - self._remaining_time < 1 / robot_attacker.shoot_frequency:
            return False
        # 判断完整视野
        if not attack_sight_clear(robot_attacker.get_position(), robot_target.get_position(), robot_target.radius, self.obstacles, self.robots.values()):
            return False
        # 攻击
        if not robot_attacker.attack(robot_target):
            return False
        # 结算攻击时间
        robot_attacker.last_attack_time_remain = self._remaining_time
        robot_attacker.last_in_combat_time_remain = self._remaining_time
        robot_target.last_in_combat_time_remain = self._remaining_time
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

            robot_attacker.last_in_combat_time_remain = self._remaining_time
            robot_target.last_in_combat_time_remain = self._remaining_time
    
    def get_top_bar_info(self) -> Dict[str, Any]:
        """获取渲染顶部信息"""
        return {
            "remaining_time": self._remaining_time,
        }

    def get_robot(self, id: str) -> Robot:
        if id not in self.robots:
            return None
        return self.robots[id]
