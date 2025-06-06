from enum import Enum

class GameState(Enum):
    PLAYING = 0
    TEAM1_WIN = 1
    TEAM2_WIN = 2

class ControlType(Enum):
    robot1 = 1
    robot2 = 2