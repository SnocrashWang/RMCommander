# 仿真配置
SCALE = 100  # 1米 = 100像素
FPS = 120

# 场地尺寸（米）
FIELD_WIDTH = 12.0
FIELD_HEIGHT = 8.0

# 机器人参数
TANK_RADIUS = 0.25  # 机器人半径（米）
TANK_SPEED = 0.5  # 米/秒
TANK_ROTATION_SPEED = 180  # 度/秒

# 中心区域
CENTER_ZONE_SIZE = 2.0  # 米

# 障碍物
OBSTACLES = [
    # 左侧45度墙（大致从左下到中间偏左）
    {"x1": 4.0, "y1": 2.0, "x2": 2.0, "y2": 4.0, "thickness": 0.15},
    # 右侧45度墙（大致从右上到中间偏右）
    {"x1": 8.0, "y1": 6.0, "x2": 10.0, "y2": 4.0, "thickness": 0.15}
]

# 颜色
BACKGROUND = (40, 44, 52)
WALL_COLOR = (86, 156, 214)
OBSTACLE_COLOR = (198, 120, 221)
CENTER_ZONE_COLOR = (152, 195, 121, 100)  # RGBA
robot1_COLOR = (220, 163, 163)
robot2_COLOR = (140, 170, 238)
TEXT_COLOR = (220, 220, 220)
PROGRESS_BAR_BG = (60, 60, 60)
PROGRESS_BAR1 = (237, 118, 118)
PROGRESS_BAR2 = (108, 155, 239)