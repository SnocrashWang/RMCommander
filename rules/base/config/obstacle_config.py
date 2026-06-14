from rules.base.config.env_config import EnvConfigBase
from utils.utils import opposite_obstacle

width = EnvConfigBase().field_width
height = EnvConfigBase().field_height

# 四周围墙
walls = [
    {"p1": (0.0, 0.0), "p2": (width, 0.0), "thickness": 0.1},  # 上
    {"p1": (0.0, height), "p2": (width, height), "thickness": 0.1},  # 下
    {"p1": (0.0, 0.0), "p2": (0.0, height), "thickness": 0.1},  # 左
    {"p1": (width, 0.0), "p2": (width, height), "thickness": 0.1},  # 右
]

# 红方障碍物
red_obstacle = {"p1": (2.050, 2.050), "p2": (1.200, 2.900), "thickness": 0.2}
# 蓝方障碍物
blue_obstacle = opposite_obstacle(red_obstacle, (width, height))

BASE_OBSTACLE_CONFIGS = walls + [red_obstacle, blue_obstacle]
