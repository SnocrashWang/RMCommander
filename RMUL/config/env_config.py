from utils.utils import opposite_position, get_reverse_obstacle_config

ENV_NAME = "RMUL"

# 仿真配置
FPS = 60

# 场地尺寸（米）
FIELD_WIDTH = 12.0
FIELD_HEIGHT = 8.0

# 启动区
RED_START_ZONE_VERTICES = [
    (0.0, 0.0),  # 左上
    (1.5, 0),  # 右上
    (1.5, 2.0),  # 右下
    (0.0, 2.0),  # 左下
]
BLUE_START_ZONE_VERTICES = [
    opposite_position(v, FIELD_WIDTH, FIELD_HEIGHT) for v in RED_START_ZONE_VERTICES
]
# 中心增益区
CENTER_ZONE_VERTICES = [
    (FIELD_WIDTH / 2 - 2.0 / 2, FIELD_HEIGHT / 2 - 2.0 / 2),  # 左上
    (FIELD_WIDTH / 2 + 2.0 / 2, FIELD_HEIGHT / 2 - 2.0 / 2),  # 右上
    (FIELD_WIDTH / 2 + 2.0 / 2, FIELD_HEIGHT / 2 + 2.0 / 2),  # 右下
    (FIELD_WIDTH / 2 - 2.0 / 2, FIELD_HEIGHT / 2 + 2.0 / 2),  # 左下
]

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
    get_reverse_obstacle_config(v, FIELD_WIDTH, FIELD_HEIGHT) for v in red_obstacles
]

OBSTACLES.extend(red_obstacles)
OBSTACLES.extend(blue_obstacles)

GAME_TIME_LIMIT = 300  # 游戏时间限制（秒）
OCCUPATION_TARGET = 30  # 占领目标进度

