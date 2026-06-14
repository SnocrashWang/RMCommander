import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Dict

from rules.base.config.action_config import ActionBase
from rules.base.config.env_config import EnvConfigBase
from rules.base.config.obstacle_config import OBSTACLE_CONFIGS
from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION, BASE_ROBOT_CONFIGS

from utils.config.exp_prop_config import *
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotConfig, RobotType, ROBOT_ID
from utils.obstacle import Obstacle
from utils.robot import Robot
from utils.utils import pos_norm2real, opposite_team, calc_distance, attack_sight_clear


class EnemyScriptControllerBase():
    """脚本化行为控制器，用于控制简单的敌方行动"""
    def __init__(
            self,
            auto_nav: bool = False,
            attack_weight: List[int] = [0, 1],
            spin_mode: int = 0
        ):
        """
        Args:
            auto_nav: 是否自动导航（一直切换不同导航目标）
            attack_weight: 自动攻击权重，按 ROBOT_TYPE + NONE 排序
            spin: 0 不自旋，1 常态自旋，2 随机
        """
        self.auto_nav = auto_nav
        self.attack_weight = attack_weight
        self.spin_mode = spin_mode

        self.attack_target_list = list(BASE_ROBOT_TYPE_ACTION) + [RobotType.NONE]
        assert len(self.attack_weight) == len(self.attack_target_list), \
            f"attack_weight({self.attack_weight}) must has the same length as self.attack_target_list({self.attack_target_list})"

        self.switch_time_interval = (1, 10)         # 动作切换时间
        self.next_switch_time = math.inf
        self.update()

    def take_action(self, time: float, robots: Dict[str, Robot]):
        actions = {}
        for id, robot in robots.items():
            if robot.team == GameTeam.BLUE:
                actions[id] = ActionBase(
                    navigation_target_norm=self.nav_target,
                    navigation_set=int(self.auto_nav),
                    attack_target=self.attack_target,
                    spin=self.spin
                )
        if time < self.next_switch_time:
            self.update()
            # 重新随机下次更新时间
            self.next_switch_time = time - random.uniform(*self.switch_time_interval)
        return actions

    def update(self):
        self.nav_target = self._random_nav_target_norm()
        self.attack_target = self._random_attack_target()
        self.spin = self._random_spin()

    def _random_nav_target_norm(self):
        return np.random.uniform(-1, 1, 2)
            
    def _random_attack_target(self):
        return random.choices(self.attack_target_list, self.attack_weight)[0].value

    def _random_spin(self):
        if self.spin_mode == 2:
            return random.choice([0, 1])
        else:
            return self.spin_mode


class CurriculumBase():
    """
    该课程作为课程基类，调用时仅用于测试
    初始观测完全随机
    对手脚本随机移动、不攻击
    """
    def __init__(self):
        self.env_config = EnvConfigBase()
        self.obstacle_configs = deepcopy(OBSTACLE_CONFIGS)
        self.robot_configs = deepcopy(BASE_ROBOT_CONFIGS)

        self._enemy_controller = EnemyScriptControllerBase()

    def random_start(self, if_env: bool, if_obstacles: bool, if_robots: bool):
        if if_env:
            self._random_env_config()
        if if_obstacles:
            self._random_obstacle_configs()
        if if_robots:
            self._random_robot_configs()
        return self.env_config, self.obstacle_configs, self.robot_configs

    def _random_env_config(self):
        self.env_config = EnvConfigBase(
            game_remaining_time=random.uniform(30, EnvConfigBase().game_time_limit)
        )

    def _random_obstacle_configs(self, std: float = 0.1):
        for obstacle in deepcopy(OBSTACLE_CONFIGS):
            obstacle["p1"] = (obstacle["p1"][0] + random.gauss(0, std), obstacle["p1"][1] + random.gauss(0, std))
            obstacle["p2"] = (obstacle["p2"][0] + random.gauss(0, std), obstacle["p2"][1] + random.gauss(0, std))
            obstacle["thickness"] += random.gauss(0, std)

    def _random_robot_configs(self):
        self.robot_configs = []
        for team in GameTeam:
            for robot_type in BASE_ROBOT_TYPE_ACTION:
                init_pos = pos_norm2real(np.random.uniform(-0.9, 0.9, 2), EnvConfigBase.field_size())       # 为了防止被随机围墙挤到地图外，此处随机初始化位置范围为 [-0.9, 0.9]
                chassis_property_type = random.choice([CHASSIS_PROPERTY_TYPE.POWER, CHASSIS_PROPERTY_TYPE.HP])
                gimbal_property_type = random.choice([GIMBAL_PROPERTY_TYPE.HEAT, GIMBAL_PROPERTY_TYPE.COOLDOWN])
                max_hp = CHASSIS_PROPERTY_STANDARD[chassis_property_type][1]["HP"]
                max_heat = GIMBAL_PROPERTY_17[gimbal_property_type][1]["HEAT"]
                self.robot_configs.append(RobotConfig(
                    team=team,
                    robot_type=robot_type,
                    init_pos=init_pos,
                    chassis_property_type=chassis_property_type,
                    gimbal_property_type=gimbal_property_type,
                    max_ammo=200,
                    ammo_allowed=200,
                    hp=random.randint(int(0.1 * max_hp), max_hp),
                    heat=random.uniform(0, max_heat),
                    enable_exp=False,
                ))

    def get_enemy_action(self, time: float, robots: Dict[str, Robot]):
        """
        Args:
            time: 倒计时
            robots: 环境中的所有机器人
        """
        return self._enemy_controller.take_action(time, robots)

    def reward(self, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        return 0


class CurriculumBaseMovement(CurriculumBase):
    """
    该课程用于训练模型的基本移动能力，包括设置导航点、靠近敌人等
    初始观测完全随机
    对手脚本随机移动、不攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True)

    def reward(self, robots: Dict[str, Robot], red_actions: Dict[str, ActionBase]):
        """计算红方奖励"""
        reward = 0
        for id, action in red_actions.items():
            if action.navigation_set != 1:
                reward += 0.01
                continue

            # 鼓励导航点合法
            navigation_target = pos_norm2real(action.navigation_target_norm, EnvConfigBase.field_size())
            reward += 0.05 if robots[id].is_valid_target(navigation_target) else -0.1

            # 鼓励导航点接近蓝方坐标
            blue_robot_pos_list = [robot.get_position() for robot in robots.values() if robot.team == opposite_team(robots[id].team)]
            if any(x < 1 for x in [calc_distance(navigation_target, pos) for pos in blue_robot_pos_list]):
                reward += 1.0
        return reward


class CurriculumBaseBattle(CurriculumBase):
    """
    该课程用于训练模型的战斗能力，包括选择攻击目标、自旋防御等
    初始观测完全随机
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

    def reward(self, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        """计算红方奖励"""
        reward = 0
        for id, action in actions.items():
            robot_attacker = robots[id]
            # 奖励攻击目标
            if RobotType(action.attack_target) in list(BASE_ROBOT_TYPE_ACTION) + [RobotType.NONE]:
                reward += 0.05

                if action.attack_target != RobotType.NONE.value:
                    robot_target = robots[ROBOT_ID[opposite_team(robot_attacker.team)][RobotType(action.attack_target)]]
                    # 奖励攻击视野
                    if attack_sight_clear(
                        robot_attacker.get_position(),
                        robot_target.get_position(),
                        robot_target.radius,
                        self.obstacle_configs,
                        robots,
                    ):
                        reward += 0.1
                    else:
                        reward -= 0.02

            # 奖励自旋防御
            robot_enemy = robots[ROBOT_ID[opposite_team(robot_attacker.team)][robot_attacker.robot_type]]
            if attack_sight_clear(
                robot_enemy.get_position(),
                robot_attacker.get_position(),
                robot_attacker.radius,
                [Obstacle(obstacle_config) for obstacle_config in self.obstacle_configs],
                robots,
            ):
                if action.spin == 1:
                    reward += 0.1
                else:
                    reward -= 0.1
        return reward


class CurriculumBaseEasy(CurriculumBase):
    """
    该课程用于训练模型的完整能力
    初始观测完全随机
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True, attack_weight=[1, 9], spin_mode=0)

    def reward(self, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        reward = 0
        for id, action in actions.items():
            robot_attacker = robots[id]
            robot_enemy = robots[ROBOT_ID[opposite_team(robot_attacker.team)][robot_attacker.robot_type]]

            # 血量奖励
            reward += 0.001 * (robot_attacker.hp - robot_enemy.hp)

            # 胜利奖励
            if not robot_attacker.is_alive:
                reward -= 100
            elif not robot_enemy.is_alive:
                reward += 100
        return reward


class CurriculumBaseMedium(CurriculumBaseEasy):
    """
    该课程用于训练模型的完整能力
    初始观测完全随机
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True, attack_weight=[1, 1], spin_mode=2)

    def _random_robot_configs(self):
        self.robot_configs = []
        for team in GameTeam:
            for robot_type in BASE_ROBOT_TYPE_ACTION:
                init_pos = pos_norm2real(np.random.uniform(-0.9, 0.9, 2), EnvConfigBase.field_size())       # 为了防止被随机围墙挤到地图外，此处随机初始化位置范围为 [-0.9, 0.9]
                chassis_property_type = random.choice([CHASSIS_PROPERTY_TYPE.POWER, CHASSIS_PROPERTY_TYPE.HP])
                gimbal_property_type = random.choice([GIMBAL_PROPERTY_TYPE.HEAT, GIMBAL_PROPERTY_TYPE.COOLDOWN])
                max_hp = CHASSIS_PROPERTY_STANDARD[chassis_property_type][1]["HP"]
                max_heat = GIMBAL_PROPERTY_17[gimbal_property_type][1]["HEAT"]
                self.robot_configs.append(RobotConfig(
                    team=team,
                    robot_type=robot_type,
                    init_pos=init_pos,
                    chassis_property_type=chassis_property_type,
                    gimbal_property_type=gimbal_property_type,
                    max_ammo=200,
                    ammo_allowed=200,
                    hp=random.randint(int(0.8 * max_hp), max_hp),
                    heat=random.uniform(0, max_heat),
                    enable_exp=False,
                ))

class CurriculumBaseHard(CurriculumBaseMedium):
    """
    该课程用于训练模型的完整能力
    初始观测完全随机
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True, attack_weight=[1, 0], spin_mode=1)
