import numpy as np
from copy import deepcopy
from collections import OrderedDict
from typing import Dict
from gymnasium import spaces
from dataclasses import dataclass

from utils.config.game_config import GameTeam
from utils.observation import Observation
from utils.robot import Robot
from utils.utils import pos_real2norm

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

@dataclass
class ObsRMULGame():
    env_obs: ObsRMULEnv
    robots_obs: Dict[str, ObsRMULRobot]

    @classmethod
    def get_dict_space(cls, robot_ids) -> spaces.Dict:
        return spaces.Dict(OrderedDict([
            ("env_obs", ObsRMULEnv.get_space()),
            ("robots_obs", spaces.Dict(OrderedDict([
                (robot_id, ObsRMULRobot.get_space())
                for robot_id in robot_ids
            ]))),
        ]))

    @classmethod
    def get_space(cls, robot_ids) -> spaces.Box:
        low, high = cls._flatten_space_bounds(cls.get_dict_space(robot_ids))
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    @classmethod
    def _flatten_space_bounds(cls, space):
        low = []
        high = []

        if isinstance(space, spaces.Box):
            low.extend(np.asarray(space.low, dtype=np.float32).reshape(-1))
            high.extend(np.asarray(space.high, dtype=np.float32).reshape(-1))
        elif isinstance(space, spaces.Discrete):
            low.append(0)
            high.append(space.n - 1)
        elif isinstance(space, spaces.Dict):
            for subspace in space.spaces.values():
                sub_low, sub_high = cls._flatten_space_bounds(subspace)
                low.extend(sub_low)
                high.extend(sub_high)
        else:
            raise TypeError(f"Unsupported observation space: {space}")

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    def to_array(self, team: GameTeam = GameTeam.RED) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        red_robots_obs = {robot_id: deepcopy(robot_obs) for robot_id, robot_obs in self.robots_obs.items() if robot_id.startswith("RED")}
        blue_robots_obs = {robot_id: deepcopy(robot_obs) for robot_id, robot_obs in self.robots_obs.items() if robot_id.startswith("BLUE")}
        if team == GameTeam.RED:
            # 先己方，后对方
            robots_obs = {**red_robots_obs, **blue_robots_obs}
        else:
            # 调换红蓝方的坐标方向
            for _, robot_obs in red_robots_obs.items():
                robot_obs.position_norm *= -1
            for _, robot_obs in blue_robots_obs.items():
                robot_obs.position_norm *= -1
            robots_obs = {**blue_robots_obs, **red_robots_obs}

        return np.concatenate([
            self.env_obs.to_array(),
            *[robot_obs.to_array() for _, robot_obs in robots_obs.items()]
        ])
