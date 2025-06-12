from utils.utils import get_reverse_obstacle_config

ENV_NAME = "base_game"

# 仿真配置
FPS = 60

# 场地尺寸（米）
FIELD_WIDTH = 5.0
FIELD_HEIGHT = 5.0

# 障碍物
OBSTACLES = [
    # 四周围墙
    {"x1": 0.0, "y1": 0.0, "x2": FIELD_WIDTH, "y2": 0.0, "thickness": 0.0},  # 上
    {"x1": 0.0, "y1": FIELD_HEIGHT, "x2": FIELD_WIDTH, "y2": FIELD_HEIGHT, "thickness": 0.0},  # 下
    {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": FIELD_HEIGHT, "thickness": 0.0},  # 左
    {"x1": FIELD_WIDTH, "y1": 0.0, "x2": FIELD_WIDTH, "y2": FIELD_HEIGHT, "thickness": 0.0},  # 右
]

wall_red = {"x1": 2.050, "y1": 2.050, "x2": 1.200, "y2": 2.900, "thickness": 0.2}
wall_blue = get_reverse_obstacle_config(wall_red, FIELD_WIDTH, FIELD_HEIGHT)
OBSTACLES.append(wall_red)
OBSTACLES.append(wall_blue)

GAME_TIME_LIMIT = 120  # 游戏时间限制（秒）
