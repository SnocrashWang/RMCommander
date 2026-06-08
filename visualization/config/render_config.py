from utils.config.game_config import GameTeam

# 比例尺
SCALE = 100  # 1米 = 100像素

# 颜色
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GRAY = (211, 211, 211)      # 浅灰色
COLOR_SILVER_GRAY = (192, 192, 192)     # 银灰色
COLOR_MEDIUM_GRAY = (128, 128, 128)     # 中灰色
COLOR_CHARCOAL_GRAY = (54, 54, 54)      # 碳灰色

COLOR_YELLOW = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)

# 透明度
ALPHA_HALF = 128

# 队伍颜色配置
TEAM_COLORS = {
    GameTeam.RED: (255, 0, 0),
    GameTeam.BLUE: (0, 0, 255)
}

# 机器人颜色配置
ROBOT_COLORS = {
    GameTeam.RED: (255, 120, 120),
    GameTeam.BLUE: (120, 120, 255)
}
ARMOR_COLORS = {
    GameTeam.RED: (255, 40, 40),
    GameTeam.BLUE: (40, 40, 255)
}

# 其他配置
COLOR_BACKGROUND = COLOR_WHITE              # 背景
COLOR_WALL = COLOR_MEDIUM_GRAY              # 墙壁
COLOR_OBSTACLE = COLOR_MEDIUM_GRAY          # 障碍物
COLOR_TEXT = COLOR_CHARCOAL_GRAY            # 文字
COLOR_PROGRESS_BAR_BG = COLOR_SILVER_GRAY   # 进度条背景

COLOR_CENTER_ZONE = (200, 255, 200)         # 增益区

COLOR_REVIVE_BAR = (0, 255, 0)  # 复活条颜色
COLOR_HEAT_BAR = (255, 120, 0)  # 热量条颜色
COLOR_EXP_BAR = (220, 80, 240)  # 经验条颜色

# 可移动栅格配置
COLOR_GRID = COLOR_LIGHT_GRAY  # 栅格线
ALPHA_GRID = ALPHA_HALF