from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass
from utils.game_config import GameTeam
from utils.exp_prop_config import *

class RobotID(Enum):
    RED_1 = "RED_1_HERO"
    RED_2 = "RED_2_ENGINEER"
    RED_3 = "RED_3_STANDARD"
    RED_4 = "RED_4_STANDARD"
    RED_5 = "RED_5_STANDARD"
    RED_6 = "RED_6_DRONE"
    RED_7 = "RED_7_SENTRY"
    
    BLUE_1 = "BLUE_1_HERO"
    BLUE_2 = "BLUE_2_ENGINEER"
    BLUE_3 = "BLUE_3_STANDARD"
    BLUE_4 = "BLUE_4_STANDARD"
    BLUE_5 = "BLUE_5_STANDARD"
    BLUE_6 = "BLUE_6_DRONE"
    BLUE_7 = "BLUE_7_SENTRY"

@dataclass
class RobotConfig:
    """机器人配置类"""
    team: GameTeam
    init_pos: Tuple[float, float]
    chassis_property_type: CHASSIS_PROPERTY_TYPE
    gimbal_property_type: GIMBAL_PROPERTY_TYPE
    forward_speed_efficiency: float = 0.02
    rotation_speed_efficiency: float = 2.0
    radius: float = 0.25

# 默认机器人配置
DEFAULT_ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    RobotID.RED_3: RobotConfig(
        team=GameTeam.RED,
        init_pos=(1.0, 1.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
    ),
    RobotID.BLUE_3: RobotConfig(
        team=GameTeam.BLUE,
        init_pos=(6.0, 4.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.HP,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.HEAT,
    ),
}