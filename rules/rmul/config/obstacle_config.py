from rules.rmul.config.env_config import EnvConfigRMUL
from utils.utils import opposite_obstacle

width = EnvConfigRMUL().field_width
height = EnvConfigRMUL().field_height

# 四周围墙
walls = [
    {"p1": (0.0, 0.0), "p2": (width, 0.0), "thickness": 0.0},  # 上
    {"p1": (0.0, height), "p2": (width, height), "thickness": 0.0},  # 下
    {"p1": (0.0, 0.0), "p2": (0.0, height), "thickness": 0.0},  # 左
    {"p1": (width, 0.0), "p2": (width, height), "thickness": 0.0},  # 右
]

# 红方障碍物
red_obstacles = [
    {"p1": (2.808, 3.780), "p2": (3.014, 3.780), "thickness": 0.2},
    {"p1": (2.964, 3.830), "p2": (4.638, 1.950), "thickness": 0.15},
    {"p1": (4.588, 2.000), "p2": (4.794, 2.000), "thickness": 0.2},
]
# 蓝方障碍物
blue_obstacles = [
    opposite_obstacle(o, (width, height)) for o in red_obstacles
]

RMUL_OBSTACLE_CONFIGS = walls + red_obstacles + blue_obstacles
