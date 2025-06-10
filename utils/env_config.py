import pygame

# 仿真配置
SCALE = 100  # 1米 = 100像素
FPS = 60

# 场地尺寸（米）
FIELD_WIDTH = 12.0
FIELD_HEIGHT = 8.0

# 中心区域
CENTER_ZONE_SIZE = 2.0  # 米

# 计算中心区域矩形（像素坐标）
CENTER_ZONE_RECT = pygame.Rect(
    (FIELD_WIDTH - CENTER_ZONE_SIZE) * SCALE / 2,  # x
    (FIELD_HEIGHT - CENTER_ZONE_SIZE) * SCALE / 2,  # y
    CENTER_ZONE_SIZE * SCALE,  # width
    CENTER_ZONE_SIZE * SCALE   # height
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

# 颜色
COLOR_BACKGROUND = (240, 240, 240)  # 浅灰色背景
COLOR_WALL = (100, 100, 100)  # 深灰色墙壁
COLOR_CENTER_ZONE = (200, 255, 200)  # 浅绿色中心区域
COLOR_OBSTACLE = (150, 150, 150)  # 灰色障碍物
COLOR_TEXT = (50, 50, 50)  # 深灰色文字
COLOR_PROGRESS_BAR_BG = (200, 200, 200)  # 进度条背景色
COLOR_HEAT_BAR = (255, 120, 0)  # 热量条颜色
COLOR_EXP_BAR = (220, 80, 240)  # 经验条颜色

# 可移动栅格配置
GRID_CELL_SIZE = 0.1  # 栅格大小（米）
GRID_COLOR = (20, 20, 20)  # 栅格线颜色
GRID_ALPHA = 128  # 栅格透明度（0-255）

GAME_TIME_LIMIT = 300  # 游戏时间限制（秒）
OCCUPATION_TARGET = 30  # 占领目标进度
