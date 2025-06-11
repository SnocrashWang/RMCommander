from typing import List
from utils.config.robot_config import RobotConfig, RobotType, GameTeam, CHASSIS_PROPERTY_TYPE, GIMBAL_PROPERTY_TYPE

RMUL_ROBOT_TYPE_LIST = [
    RobotType.HERO,
    RobotType.STANDARD_3,
    RobotType.SENTRY,
]

# RMUL机器人配置
RMUL_ROBOT_CONFIGS: List[RobotConfig] = [
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.HERO,
        init_pos=(1.0, 1.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.DEFAULT,
    ),
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.STANDARD_3,
        init_pos=(1.0, 1.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
    ),
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.SENTRY,
        init_pos=(1.5, 1.0),
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.HERO,
        init_pos=(11.0, 7.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.HP,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.DEFAULT,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.STANDARD_3,
        init_pos=(11.0, 6.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.HP,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.HEAT,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.SENTRY,
        init_pos=(10.5, 7.0),
    ),
]
