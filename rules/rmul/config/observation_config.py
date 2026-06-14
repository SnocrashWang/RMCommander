import numpy as np
from copy import deepcopy
from collections import OrderedDict
from typing import Dict
from gymnasium import spaces
from dataclasses import dataclass, field

from utils.config.robot_config import RobotType
from utils.config.game_config import GameTeam
from utils.observation import Observation
from utils.robot import Robot
from utils.utils import pos_real2norm

from rules.base.config.observation_config import ObsBaseGame
from rules.rmul.config.env_config import EnvConfigRMUL


class ObsRMULEnv(Observation):
    _schema = {
        "remaining_time_norm": spaces.Box(
            low=0.0, 
            high=1.0,
            dtype=np.float32
        ),
        "victory_progress_red_norm": spaces.Box(
            low=0.0, 
            high=1.0,
            dtype=np.float32
        ),
        "victory_progress_blue_norm": spaces.Box(
            low=0.0, 
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
        "level": spaces.Box(
            low=0,
            high=10,
            dtype=int
        ),
        "hp": spaces.Box(
            low=0,
            high=1000,
            dtype=int
        ),
        "max_hp": spaces.Box(
            low=0,
            high=1000,
            dtype=int
        ),
        "power": spaces.Box(
            low=0,
            high=1000,
            dtype=int
        ),
        "heat": spaces.Box(
            low=0,
            high=1000,
            dtype=int
        ),
        "max_heat": spaces.Box(
            low=0,
            high=1000,
            dtype=int
        ),
        "cooldown": spaces.Box(
            low=0,
            high=1000,
            dtype=int
        ),
    }

    @classmethod
    def from_robot(cls, robot: Robot):
        return cls(
            position_norm=pos_real2norm(
                robot.get_position(),
                EnvConfigRMUL.field_size(),
            ),
            level = robot.level,
            hp=min(1000, robot.hp),
            max_hp=min(1000, robot.max_hp),
            power=min(1000, robot.power),
            heat=min(1000, robot.heat),
            max_heat=min(1000, robot.heat),
            cooldown=min(1000, robot.cooldown)
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
                    cls._robot_config_id(robot_config),
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
