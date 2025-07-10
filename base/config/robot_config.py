import torch
from typing import List

from config import DEVICE
from utils.config.game_config import GameTeam
from utils.config.exp_prop_config import CHASSIS_PROPERTY_TYPE, GIMBAL_PROPERTY_TYPE
from utils.config.robot_config import RobotConfig, RobotType

BASE_ROBOT_TYPE_LIST = [
    RobotType.STANDARD_3,
]

# 默认机器人配置
BASE_ROBOT_CONFIGS: List[RobotConfig] = [
    RobotConfig(
        team=GameTeam.RED,
        robot_type=RobotType.STANDARD_3,
        init_pos=torch.tensor([0.5, 0.5], dtype=torch.float, device=DEVICE),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
        ammo_allowed=torch.tensor(200, dtype=torch.int, device=DEVICE),
    ),
    RobotConfig(
        team=GameTeam.BLUE,
        robot_type=RobotType.STANDARD_3,
        init_pos=torch.tensor([4.5, 4.5], dtype=torch.float, device=DEVICE),
        chassis_property_type=CHASSIS_PROPERTY_TYPE.POWER,
        gimbal_property_type=GIMBAL_PROPERTY_TYPE.COOL_DOWN,
        ammo_allowed=torch.tensor(200, dtype=torch.int, device=DEVICE),
    ),
]
