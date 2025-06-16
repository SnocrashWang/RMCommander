from utils.config.game_config import GameTeam

# 比例尺
SCALE = 100  # 1米 = 100像素

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

# 其他颜色
COLOR_BACKGROUND = (240, 240, 240)  # 浅灰色背景
COLOR_WALL = (100, 100, 100)  # 深灰色墙壁
COLOR_CENTER_ZONE = (200, 255, 200)  # 浅绿色增益区
COLOR_OBSTACLE = (150, 150, 150)  # 灰色障碍物
COLOR_TEXT = (50, 50, 50)  # 深灰色文字
COLOR_PROGRESS_BAR_BG = (200, 200, 200)  # 进度条背景色
COLOR_HEAT_BAR = (255, 120, 0)  # 热量条颜色
COLOR_EXP_BAR = (220, 80, 240)  # 经验条颜色

# 可移动栅格配置
GRID_COLOR = (20, 20, 20)  # 栅格线颜色
GRID_ALPHA = 128  # 栅格透明度（0-255）