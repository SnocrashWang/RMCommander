from enum import Enum

class GameState(Enum):
    PLAYING = 0
    DRAW = 1
    RED_TEAM_WIN = 2
    BLUE_TEAM_WIN = 3

class GameTeam(Enum):
    RED = "RED"
    BLUE = "BLUE"

# 伤害
DAMAGE_PER_17 = 10
DAMAGE_PER_42 = 100

# 热量
HEAT_PER_17 = 10
HEAT_PER_42 = 100
