# 仿真配置
SCALE = 100  # 1米 = 100像素
FPS = 120

# 场地尺寸（米）
FIELD_WIDTH = 12.0
FIELD_HEIGHT = 8.0

# 中心区域
CENTER_ZONE_SIZE = 2.0  # 米

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
BACKGROUND = (40, 44, 52)
WALL_COLOR = (86, 156, 214)
OBSTACLE_COLOR = (198, 120, 221)
CENTER_ZONE_COLOR = (152, 195, 121, 100)  # RGBA
TEXT_COLOR = (220, 220, 220)
PROGRESS_BAR_BG = (60, 60, 60)
PROGRESS_BAR1 = (237, 118, 118)
PROGRESS_BAR2 = (108, 155, 239)