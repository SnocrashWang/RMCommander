from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass
from utils.config.game_config import GameTeam
from utils.config.exp_prop_config import *

class RobotType(Enum):
    NONE = 0
    HERO = 1
    ENGINEER = 2
    STANDARD_3 = 3
    STANDARD_4 = 4
    STANDARD_5 = 5
    DRONE = 6
    SENTRY = 7

ROBOT_ID = {
    GameTeam.RED: {
        RobotType.NONE: "RED_0_NONE",
        RobotType.HERO: "RED_1_HERO",
        RobotType.ENGINEER: "RED_2_ENGINEER",
        RobotType.STANDARD_3: "RED_3_STANDARD",
        RobotType.STANDARD_4: "RED_4_STANDARD",
        RobotType.STANDARD_5: "RED_5_STANDARD",
        RobotType.DRONE: "RED_6_DRONE",
        RobotType.SENTRY: "RED_7_SENTRY",
    },
    GameTeam.BLUE: {
        RobotType.NONE: "BLUE_0_NONE",
        RobotType.HERO: "BLUE_1_HERO",
        RobotType.ENGINEER: "BLUE_2_ENGINEER",
        RobotType.STANDARD_3: "BLUE_3_STANDARD",
        RobotType.STANDARD_4: "BLUE_4_STANDARD",
        RobotType.STANDARD_5: "BLUE_5_STANDARD",
        RobotType.DRONE: "BLUE_6_DRONE",
        RobotType.SENTRY: "BLUE_7_SENTRY",
    },
}

@dataclass
class RobotConfig:
    """机器人配置类"""
    team: GameTeam
    robot_type: RobotType
    init_pos: Tuple[float, float]
    chassis_property_type: CHASSIS_PROPERTY_TYPE = CHASSIS_PROPERTY_TYPE.DEFAULT
    gimbal_property_type: GIMBAL_PROPERTY_TYPE = GIMBAL_PROPERTY_TYPE.DEFAULT
    forward_speed_efficiency: float = 0.02
    rotation_speed_efficiency: float = 2.0
    radius: float = 0.25
    max_ammo: int = 200
    ammo_allowed: int = 0
