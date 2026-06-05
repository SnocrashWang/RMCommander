from typing import List

from utils.config.exp_prop_config import CHASSIS_PROPERTY_TYPE, GIMBAL_PROPERTY_TYPE
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotConfig, RobotType


BASE_ROBOT_TYPE_LIST = [
    RobotType.STANDARD_3,
]

# Default robot config for the simple base 1v1 scene.
BASE_ROBOT_CONFIGS: List[RobotConfig] = [
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.STANDARD_3,
        init_pos=(0.5, 0.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
        ammo_allowed=200,
        enable_exp=False,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.STANDARD_3,
        init_pos=(4.5, 4.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
        ammo_allowed=200,
        enable_exp=False,
    ),
]
