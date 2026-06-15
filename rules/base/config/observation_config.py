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

from rules.base.config.env_config import EnvConfigBase


class ObsBaseEnv(Observation):
    _schema = {
        "remaining_time_norm": spaces.Box(
            low=-1.0,
            high=1.0,
            dtype=np.float32
        ),
    }

class ObsBaseRobot(Observation):
    _schema = {
        "position_norm": spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
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
                EnvConfigBase.field_size(),
            ),
            hp_norm=linear_norm(robot.hp, 0, 200),
            power_norm=linear_norm(robot.power, 45, 60),
            heat_norm=sqrt_norm(robot.heat, 0.5, 0, 200),
            cooldown_norm=linear_norm(robot.cooldown, 10, 40),
        )


# 对不同机器人的观测类型，对于不同的机器人，该设定可能不同
BASE_ROBOT_TYPE_OBS = {
    "friend": {
        RobotType.STANDARD_3: ObsBaseRobot,
    },
    "enemy": {
        RobotType.STANDARD_3: ObsBaseRobot,
    }
}


@dataclass
class ObsBaseGame():
    env_obs: ObsBaseEnv
    robots_obs: Dict[str, ObsBaseRobot]
    robot_type_obs: Dict = field(default_factory=lambda: BASE_ROBOT_TYPE_OBS)

    @classmethod
    def get_dict_space(
        cls,
        robot_configs,
        robot_type_obs=BASE_ROBOT_TYPE_OBS,
        team: GameTeam = GameTeam.RED,
    ) -> spaces.Dict:
        return spaces.Dict(OrderedDict([
            ("env_obs", ObsBaseEnv.get_space()),
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
        robot_type_obs=BASE_ROBOT_TYPE_OBS,
        team: GameTeam = GameTeam.RED,
    ) -> spaces.Box:
        low, high = cls._flatten_space_bounds(cls.get_dict_space(robot_configs, robot_type_obs, team))
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

    @classmethod
    def _ordered_robot_configs(cls, robot_configs, team: GameTeam):
        friend_configs = [robot_config for robot_config in robot_configs if robot_config.team == team]
        enemy_configs = [robot_config for robot_config in robot_configs if robot_config.team != team]
        return [*friend_configs, *enemy_configs]

    @classmethod
    def _get_robot_obs_cls(cls, robot_type_obs, role: str, robot_type: RobotType):
        try:
            return robot_type_obs[role][robot_type]
        except KeyError as exc:
            raise KeyError(f"Missing observation config for role={role}, robot_type={robot_type}") from exc

    @classmethod
    def _get_storage_robot_obs_cls(cls, robot_type_obs, robot_type: RobotType):
        obs_classes = [
            role_robot_type_obs[robot_type]
            for role_robot_type_obs in robot_type_obs.values()
            if robot_type in role_robot_type_obs
        ]
        if not obs_classes:
            raise KeyError(f"Missing observation config for robot_type={robot_type}")
        return max(obs_classes, key=lambda obs_cls: obs_cls.get_array_size())

    @classmethod
    def _get_robot_id_info(cls, robot_id: str):
        for team, robot_ids in ROBOT_ID.items():
            for robot_type, candidate_id in robot_ids.items():
                if candidate_id == robot_id:
                    return team, robot_type
        raise KeyError(f"Unknown robot id: {robot_id}")

    @classmethod
    def _robot_obs_to_array(cls, robot_obs, obs_cls) -> np.ndarray:
        if isinstance(robot_obs, obs_cls):
            return robot_obs.to_array()

        kwargs = {}
        for name in obs_cls._schema:
            if not hasattr(robot_obs, name):
                raise AttributeError(f"{type(robot_obs).__name__} does not provide observation field '{name}'")
            kwargs[name] = getattr(robot_obs, name)
        return obs_cls(**kwargs).to_array()

    def to_array(self, team: GameTeam = GameTeam.RED, robot_type_obs=None) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组"""
        robot_type_obs = robot_type_obs or self.robot_type_obs
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

        robot_arrays = []
        for robot_id, robot_obs in robots_obs.items():
            robot_team, robot_type = self._get_robot_id_info(robot_id)
            role = "friend" if robot_team == team else "enemy"
            obs_cls = self._get_robot_obs_cls(robot_type_obs, role, robot_type)
            robot_arrays.append(self._robot_obs_to_array(robot_obs, obs_cls))

        return np.concatenate([self.env_obs.to_array(), *robot_arrays])
