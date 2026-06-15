import numpy as np
from copy import deepcopy
from collections import OrderedDict
from typing import Dict
from gymnasium import spaces
from dataclasses import dataclass, field

from utils.config.robot_config import RobotType
from utils.config.game_config import GameTeam
from utils.config.robot_config import ROBOT_ID
from utils.observation import Observation, linear_norm, sqrt_norm, reverse_sqrt_norm
from utils.robot import Robot
from utils.utils import pos_real2norm

from rules.base.config.observation_config import ObsBaseGame
from rules.rmul.config.env_config import EnvConfigRMUL


class ObsRMULEnv(Observation):
    _schema = {
        "remaining_time_norm": spaces.Box(
            low=-1.0, 
            high=1.0,
            dtype=np.float32
        ),
        "victory_progress_red_norm": spaces.Box(
            low=-1.0, 
            high=1.0,
            dtype=np.float32
        ),
        "victory_progress_blue_norm": spaces.Box(
            low=-1.0, 
            high=1.0,
            dtype=np.float32
        ),
        "economics_red_norm": spaces.Box(
            low=-1.0, 
            high=1.0,
            dtype=np.float32
        ),
        "economics_blue_norm": spaces.Box(
            low=-1.0, 
            high=1.0,
            dtype=np.float32
        ),
    }

class ObsRMULRobot(Observation):
    _schema = {
        "position_norm": spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        ),
        "level_norm": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32
        ),
        "hp_norm": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32
        ),
        "power_norm": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32
        ),
        "heat_norm": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32
        ),
        "cooldown_norm": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32
        ),
    }

    @classmethod
    def from_robot(cls, robot: Robot):
        return cls(
            position_norm=pos_real2norm(
                robot.get_position(),
                EnvConfigRMUL.field_size(),
            ),
            level_norm=linear_norm(robot.level, 1, 10),
            hp_norm=linear_norm(robot.hp, 0, 500),
            power_norm=linear_norm(robot.power, 45, 120),
            heat_norm=sqrt_norm(robot.heat, 0.5, 0, 650),
            cooldown_norm=linear_norm(robot.cooldown, 10, 120),
        )


RMUL_ROBOT_TYPE_OBS = {
    "friend": {
        RobotType.HERO: ObsRMULRobot,
        RobotType.STANDARD_3: ObsRMULRobot,
        RobotType.SENTRY: ObsRMULRobot,
    },
    "enemy": {
        RobotType.HERO: ObsRMULRobot,
        RobotType.STANDARD_3: ObsRMULRobot,
        RobotType.SENTRY: ObsRMULRobot,
    }
}


@dataclass
class ObsRMULGame(ObsBaseGame):
    env_obs: ObsRMULEnv
    robots_obs: Dict[str, ObsRMULRobot]
    robot_type_obs: Dict = field(default_factory=lambda: RMUL_ROBOT_TYPE_OBS)

    @classmethod
    def get_dict_space(
        cls,
        robot_configs,
        robot_type_obs=RMUL_ROBOT_TYPE_OBS,
        team: GameTeam = GameTeam.RED,
    ) -> spaces.Dict:
        return spaces.Dict(OrderedDict([
            ("env_obs", ObsRMULEnv.get_space()),
            ("robots_obs", spaces.Dict(OrderedDict([
                (
                    ROBOT_ID[robot_config.team][robot_config.robot_type],
                    cls._get_robot_obs_cls(
                        robot_type_obs,
                        "friend" if robot_config.team == team else "enemy",
                        robot_config.robot_type,
                    ).get_space(),
                )
                for robot_config in cls._ordered_robot_configs(robot_configs, team)
            ]))),
        ]))

    @classmethod
    def get_space(
        cls,
        robot_configs,
        robot_type_obs=RMUL_ROBOT_TYPE_OBS,
        team: GameTeam = GameTeam.RED,
    ) -> spaces.Box:
        low, high = cls._flatten_space_bounds(cls.get_dict_space(robot_configs, robot_type_obs, team))
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )
