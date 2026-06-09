from utils.utils import get_reverse_obstacle_config


ENV_NAME = "RMUL"

# 仿真配置
FPS = 60

# 场地尺寸（米）
FIELD_WIDTH = 12.0
FIELD_HEIGHT = 8.0

# 障碍物
OBSTACLES = [
    # 四周围墙
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
    get_reverse_obstacle_config(v, (FIELD_WIDTH, FIELD_HEIGHT)) for v in red_obstacles
]

OBSTACLES.extend(red_obstacles)
OBSTACLES.extend(blue_obstacles)

GAME_TIME_LIMIT = 300  # 游戏时间限制（秒）
OCCUPATION_TARGET = 200  # 占领目标进度

