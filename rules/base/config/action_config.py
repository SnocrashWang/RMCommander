import numpy as np
from gymnasium import spaces

from utils.config.robot_config import RobotType
from utils.action import Action


class ActionBase(Action):
    _schema={
        "navigation_target_norm": spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ),
        "navigation_set": spaces.Discrete(2),
        "attack_target": spaces.Discrete(len(RobotType)),
        "spin": spaces.Discrete(2),
    }
