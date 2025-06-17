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

# 伤害
DAMAGE_PER_17 = 10
DAMAGE_PER_42 = 100

# 热量
HEAT_PER_17 = 10
HEAT_PER_42 = 100

# 子弹价格
PRICE_PER_17 = 1
PRICE_PER_42 = 10

# 单次购买子弹的数量
PURCHASE_NUM_17 = 10
PURCHASE_NUM_42 = 1
