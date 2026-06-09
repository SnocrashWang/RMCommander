from rules.base.config.env_config import FIELD_WIDTH, FIELD_HEIGHT
from utils.utils import opposite_obstacle

# 四周围墙
walls = [
    {"p1": (0.0, 0.0), "p2": (FIELD_WIDTH, 0.0), "thickness": 0.1},  # 上
    {"p1": (0.0, FIELD_HEIGHT), "p2": (FIELD_WIDTH, FIELD_HEIGHT), "thickness": 0.1},  # 下
    {"p1": (0.0, 0.0), "p2": (0.0, FIELD_HEIGHT), "thickness": 0.1},  # 左
    {"p1": (FIELD_WIDTH, 0.0), "p2": (FIELD_WIDTH, FIELD_HEIGHT), "thickness": 0.1},  # 右
]

# 红方障碍物
red_obstacle = {"p1": (2.050, 2.050), "p2": (1.200, 2.900), "thickness": 0.2}
# 蓝方障碍物
blue_obstacle = opposite_obstacle(red_obstacle, (FIELD_WIDTH, FIELD_HEIGHT))

OBSTACLES = walls + [red_obstacle, blue_obstacle]
