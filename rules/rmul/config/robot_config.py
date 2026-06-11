from typing import List

from utils.config.exp_prop_config import CHASSIS_PROPERTY_TYPE, GIMBAL_PROPERTY_TYPE
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotConfig, RobotType

from rules.rmul.config.action_config import ActionRMUL


RMUL_ROBOT_TYPE_ACTION = {
    RobotType.HERO: ActionRMUL,
    RobotType.STANDARD_3: ActionRMUL,
    RobotType.SENTRY: ActionRMUL,
}

# RMUL机器人配置
RMUL_ROBOT_CONFIGS: List[RobotConfig] = [
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.HERO,
        init_pos=(0.5, 0.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.DEFAULT,
        shoot_frequency=2,
        max_ammo=60,
    ),
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.STANDARD_3,
        init_pos=(0.5, 1.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOLDOWN,
        forward_speed_efficiency=0.05,
        max_ammo=40,
    ),
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.SENTRY,
        init_pos=(1.0, 0.5),
        forward_speed_efficiency=0.05,
        max_ammo=750,
        ammo_allowed=750,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.HERO,
        init_pos=(11.5, 7.5),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.DEFAULT,
        shoot_frequency=2,
        max_ammo=60,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.STANDARD_3,
        init_pos=(11.5, 7.0),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOLDOWN,
        forward_speed_efficiency=0.05,
        max_ammo=400,
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.SENTRY,
        forward_speed_efficiency=0.05,
        init_pos=(11.0, 7.5),
        max_ammo=750,
        ammo_allowed=750,
    ),
]
