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
from utils.config.robot_config import RobotConfig, RobotType
from utils.robot import Robot
from utils.utils import pos_norm2real, opposite_team, calc_distance


class EnemyScriptControllerBase():
    """脚本化行为控制器，用于控制简单的敌方行动"""
    def __init__(
            self,
            auto_nav: bool = False,
            attack_weight: List[int] = [0, 1],
            spin: int = 0
        ):
        """
        Args:
            auto_nav: 是否自动导航（一直切换不同导航目标）
            attack_weight: 自动攻击权重，按 ROBOT_TYPE + NONE 排序
            spin: 0 不自旋，1 常态自旋，2 随机
        """
        self.auto_nav = auto_nav
        self.attack_weight = attack_weight
        self.spin = spin

        self.switch_time_interval = (1, 10)         # 动作切换时间
        self.next_switch_time = math.inf

        self.attack_target_list = list(BASE_ROBOT_TYPE_ACTION) + [RobotType.NONE]
        assert len(attack_weight) == len(self.attack_target_list), \
            f"attack_weight({attack_weight}) must has the same length as self.attack_target_list({self.attack_target_list})"

        self.nav_target = self._random_nav_target_norm()
        self.attack_target = self._random_attack_target()

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

    def _random_nav_target_norm(self):
        return np.random.uniform(-1, 1, 2)
            
    def _random_attack_target(self):
        return random.choices(self.attack_target_list, self.attack_weight)[0].value


class CurriculumBase():
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
            game_remaining_time=random.uniform(1, EnvConfigBase().game_time_limit)
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
                init_pos = pos_norm2real(np.random.uniform(-1, 1, 2), EnvConfigBase.field_size())
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
                    hp=random.randint(50, max_hp),
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

            # 合法导航点只给很小奖励，非法点仍然明显惩罚。
            # 否则模型可以靠每步输出任意合法点获得很高回报，却完全不接敌。
            navigation_target = pos_norm2real(action.navigation_target_norm, EnvConfigBase.field_size())
            reward += 0.05 if robots[id].is_valid_target(navigation_target) else -1.0

            # 鼓励导航点接近蓝方坐标
            blue_robot_pos_list = [robot.get_position() for robot in robots.values() if robot.team == opposite_team(robots[id].team)]
            if any(x < 1 for x in [calc_distance(navigation_target, pos) for pos in blue_robot_pos_list]):
                reward += 0.05
            else:
                reward -= 0.02
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
        for id, action in actions.items:
            pass
        return reward
