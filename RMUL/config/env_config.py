import pygame
from visualization.config import render_config

ENV_NAME = "RMUL"

# 仿真配置
FPS = 60

# 场地尺寸（米）
FIELD_WIDTH = 12.0
FIELD_HEIGHT = 8.0

# 中心区域
CENTER_ZONE_SIZE = 2.0  # 米

# 计算中心区域矩形（像素坐标）
CENTER_ZONE_RECT = pygame.Rect(
    (FIELD_WIDTH - CENTER_ZONE_SIZE) * render_config.SCALE / 2,  # x
    (FIELD_HEIGHT - CENTER_ZONE_SIZE) * render_config.SCALE / 2,  # y
    CENTER_ZONE_SIZE * render_config.SCALE,  # width
    CENTER_ZONE_SIZE * render_config.SCALE   # height
)

# 障碍物
OBSTACLES = [
    # 四周围墙
    {"x1": 0.0, "y1": 0.0, "x2": FIELD_WIDTH, "y2": 0.0, "thickness": 0.0},  # 上
    {"x1": 0.0, "y1": FIELD_HEIGHT, "x2": FIELD_WIDTH, "y2": FIELD_HEIGHT, "thickness": 0.0},  # 下
    {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": FIELD_HEIGHT, "thickness": 0.0},  # 左
    {"x1": FIELD_WIDTH, "y1": 0.0, "x2": FIELD_WIDTH, "y2": FIELD_HEIGHT, "thickness": 0.0},  # 右
    # 左侧45度墙（大致从左下到中间偏左）
    {"x1": 4.0, "y1": 2.0, "x2": 2.0, "y2": 4.0, "thickness": 0.5},
    # 右侧45度墙（大致从右上到中间偏右）
    {"x1": 8.0, "y1": 6.0, "x2": 10.0, "y2": 4.0, "thickness": 0.5},
]

GAME_TIME_LIMIT = 300  # 游戏时间限制（秒）
OCCUPATION_TARGET = 30  # 占领目标进度

