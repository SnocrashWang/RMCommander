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
        init_pos=(0.5, 0.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.DEFAULT,
        max_ammo=60,
    ),
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.STANDARD_3,
        init_pos=(0.5, 1.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
        max_ammo=400,
    ),
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.SENTRY,
        init_pos=(1.0, 0.5),
        max_ammo=750,
        ammo_allowed=750,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.HERO,
        init_pos=(11.5, 7.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.HP,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.DEFAULT,
        max_ammo=60,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.STANDARD_3,
        init_pos=(11.5, 7.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.HP,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.HEAT,
        max_ammo=400,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.SENTRY,
        init_pos=(11.0, 7.5),
        max_ammo=750,
        ammo_allowed=750,
    ),
]
