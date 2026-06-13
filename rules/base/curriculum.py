import random
import numpy as np
from copy import deepcopy
from typing import Dict

from rules.base.config.env_config import EnvConfigBase
from rules.base.config.obstacle_config import OBSTACLE_CONFIGS
from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION, BASE_ROBOT_CONFIGS

from utils.config.exp_prop_config import *
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotConfig
from utils.action import Action
from utils.robot import Robot
from utils.utils import pos_norm2real


class CurriculumBase():
    def __init__(self):
        self.env_config = EnvConfigBase()
        self.obstacle_configs = deepcopy(OBSTACLE_CONFIGS)
        self.robot_configs = deepcopy(BASE_ROBOT_CONFIGS)
        # print(self.env_config)

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

    def reward(self, robots: Dict[str, Robot], actions: Dict[str, Action]):
        return 0


class CurriculumBaseMovement(CurriculumBase):
    """
    该课程用于训练模型的基本移动能力，包括设置导航点、自旋等
    初始观测完全随机
    对手脚本随机移动、不攻击
    """
    def __init__(self):
        super().__init__()
        # env_obs = ObsBaseEnv.get_space().sample()

    def reward(self, robots: Dict[str, Robot], actions: Dict[str, Action]):
        """计算红方奖励"""