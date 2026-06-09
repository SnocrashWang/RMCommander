from rules.rmul.config.env_config import FIELD_WIDTH, FIELD_HEIGHT
from utils.utils import opposite_obstacle

# 四周围墙
walls = [
    {"p1": (0.0, 0.0), "p2": (FIELD_WIDTH, 0.0), "thickness": 0.0},  # 上
    {"p1": (0.0, FIELD_HEIGHT), "p2": (FIELD_WIDTH, FIELD_HEIGHT), "thickness": 0.0},  # 下
    {"p1": (0.0, 0.0), "p2": (0.0, FIELD_HEIGHT), "thickness": 0.0},  # 左
    {"p1": (FIELD_WIDTH, 0.0), "p2": (FIELD_WIDTH, FIELD_HEIGHT), "thickness": 0.0},  # 右
]

# 红方障碍物
red_obstacles = [
    {"p1": (2.808, 3.780), "p2": (3.014, 3.780), "thickness": 0.2},
    {"p1": (2.964, 3.830), "p2": (4.638, 1.950), "thickness": 0.15},
    {"p1": (4.588, 2.000), "p2": (4.794, 2.000), "thickness": 0.2},
]
# 蓝方障碍物
blue_obstacles = [
    opposite_obstacle(o, (FIELD_WIDTH, FIELD_HEIGHT)) for o in red_obstacles
]

RMUL_OBSTACLES = walls + red_obstacles + blue_obstacles