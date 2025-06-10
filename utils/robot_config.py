from typing import Dict, List, Tuple
from dataclasses import dataclass
from utils.game_config import GameTeam

@dataclass
class RobotConfig:
    """机器人配置类"""
    team: GameTeam
    init_pos: Tuple[float, float]
    hp: int = 200
    speed: float = 2.0
    rotation_speed: float = 180.0
    radius: float = 0.25

# 默认机器人配置
DEFAULT_ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    "robot1": RobotConfig(
        team=GameTeam.RED,
        init_pos=(1.0, 1.0),
        hp=200,
        speed=2.0,
        rotation_speed=180.0,
        radius=0.25
    ),
    "robot2": RobotConfig(
        team=GameTeam.BLUE,
        init_pos=(6.0, 4.0),
        hp=200,
        speed=2.0,
        rotation_speed=180.0,
        radius=0.3
    )
}

# 队伍颜色配置
ROBOT_COLORS = {
    GameTeam.RED: (255, 120, 120),
    GameTeam.BLUE: (120, 120, 255)
}