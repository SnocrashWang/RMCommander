from enum import Enum

class GameState(Enum):
    PLAYING = 0
    DRAW = 1
    RED_TEAM_WIN = 2
    BLUE_TEAM_WIN = 3

class GameTeam(Enum):
    RED = 0
    BLUE = 1

# class ControlType(Enum):
#     robot1 = 1
#     robot2 = 2

GAME_TIME_LIMIT = 300  # 游戏时间限制（秒）
OCCUPATION_TARGET = 30  # 占领目标进度