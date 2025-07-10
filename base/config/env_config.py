import torch

from config import DEVICE
from utils.utils import get_reverse_obstacle_config

ENV_NAME = "solo"

# 仿真配置
FPS = torch.tensor(60, dtype=torch.int, device=DEVICE)

# 场地尺寸（米）
FIELD_WIDTH = torch.tensor(5.0, dtype=torch.float, device=DEVICE)
FIELD_HEIGHT = torch.tensor(5.0, dtype=torch.float, device=DEVICE)

# 障碍物
OBSTACLES = [
    # 四周围墙
    {"p1": (0.0, 0.0), "p2": (FIELD_WIDTH, 0.0), "thickness": 0.0},  # 上
    {"p1": (0.0, FIELD_HEIGHT), "p2": (FIELD_WIDTH, FIELD_HEIGHT), "thickness": 0.0},  # 下
    {"p1": (0.0, 0.0), "p2": (0.0, FIELD_HEIGHT), "thickness": 0.0},  # 左
    {"p1": (FIELD_WIDTH, 0.0), "p2": (FIELD_WIDTH, FIELD_HEIGHT), "thickness": 0.0},  # 右
]

red_obstacle = {"p1": (2.050, 2.050), "p2": (1.200, 2.900), "thickness": 0.2}
blue_obstacle = get_reverse_obstacle_config(red_obstacle, FIELD_WIDTH, FIELD_HEIGHT)
OBSTACLES.append(red_obstacle)
OBSTACLES.append(blue_obstacle)

GAME_TIME_LIMIT = torch.tensor(60, dtype=torch.int, device=DEVICE)  # 游戏时间限制（秒）
