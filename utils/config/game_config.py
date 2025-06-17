from enum import Enum

class GameType(Enum):
    BASE = 0
    RMUL = 1
    RMUC = 2

class GameState(Enum):
    PLAYING = 0
    DRAW = 1
    RED_TEAM_WIN = 2
    BLUE_TEAM_WIN = 3

class GameTeam(Enum):
    RED = 0
    BLUE = 1
