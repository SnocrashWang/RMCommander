from typing import List
from utils.config.game_config import GameTeam
from utils.config.exp_prop_config import CHASSIS_PROPERTY_TYPE, GIMBAL_PROPERTY_TYPE
from utils.config.robot_config import RobotConfig, RobotType

# 默认机器人配置
DEFAULT_ROBOT_CONFIGS: List[RobotConfig] = [
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.STANDARD_3,
        init_pos=(0.5, 0.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.STANDARD_3,
        init_pos=(4.5, 4.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.HP,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.HEAT,
    ),
]
